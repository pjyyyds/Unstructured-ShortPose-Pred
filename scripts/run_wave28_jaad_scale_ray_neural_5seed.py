"""Wave 28B: JAAD 2.0 Anisotropic Scale-Ray PJ-IAKF + Full-17 Split AutoPose Neural Trajectory & Zero-Retrain Conformal."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.run_wave18_conflict_gated_interaction_5seed import _loc_bootstrap_ci
from scripts.run_wave23_jaad_autopose_and_isofpr_leadtime import SEEDS, _jaad_post_onset_err
from scripts.run_wave26_jaad_full17_reappear_and_conformal_fpr import build_jaad2_full17_cohorts
from src.metrics.metrics_and_stats import _binary_average_precision
from src.models.ray_kalman_and_cusum import anisotropic_ray_kalman_filter, calibrate_ray_kalman_params


class JAADScaleRaySplitTrajectoryNet(nn.Module):
    """JAAD 2.0 trajectory predictor anchored on Anisotropic Scale-Ray PJ-IAKF + Terminal-Mobility Split-Pose residual."""

    def __init__(self, d_pose: int, d_q: int, use_pose: bool, use_veh: bool, ray_cfg: dict[str, float]) -> None:
        super().__init__()
        self.use_pose = use_pose
        self.use_veh = use_veh
        self.ray_cfg = ray_cfg
        self.mot_gru = nn.GRU(4, 48, batch_first=True)
        self.ray_mlp = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, 48))
        d_in = 48
        if use_pose:
            self.pose_gru = nn.GRU(d_pose, 32, batch_first=True)
            self.dpose_mlp = nn.Sequential(nn.Linear(2 * d_pose, 32), nn.SiLU(), nn.Linear(32, 32))
            self.q_w_raw = nn.Parameter(torch.full((1, d_q), 0.70))
            self.q_b = nn.Parameter(torch.tensor([-0.15]))
            d_in += 32
        if use_veh:
            self.veh_gru = nn.GRU(3, 16, batch_first=True)
            d_in += 16

        self.slot0_knots = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 32), nn.SiLU(), nn.Linear(32, 3 * 2))
        nn.init.zeros_(self.slot0_knots[-1].weight)
        nn.init.zeros_(self.slot0_knots[-1].bias)
        self.multi_res = nn.Linear(d_in, 3 * 20 * 2)
        nn.init.normal_(self.multi_res.weight, std=0.01)
        nn.init.zeros_(self.multi_res.bias)
        self.mode_lg = nn.Linear(d_in, 4)
        self.act_head = nn.Sequential(nn.Linear(d_in, 48), nn.SiLU(), nn.Linear(48, 4))

    def forward(
        self,
        hist: torch.Tensor,
        pose5: torch.Tensor,
        q_vec: torch.Tensor,
        veh5: torch.Tensor,
        jac: torch.Tensor,
        bh_rel: torch.Tensor,
        ank_c: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bsz = hist.size(0)
        dt_s = torch.full((bsz, 4), 0.1, dtype=torch.float64, device=hist.device)
        kf = anisotropic_ray_kalman_filter(hist, dt_s, jac, bh_rel, ank_c, **self.ray_cfg)
        dp0 = kf["p_filt"] - hist[:, -1].double()
        v_f = kf["v_filt"]
        v_spd = torch.linalg.vector_norm(v_f, dim=-1)
        pl = torch.linalg.vector_norm(hist[:, 1:] - hist[:, :-1], dim=-1).sum(dim=-1).double()
        ch = torch.linalg.vector_norm(hist[:, -1] - hist[:, 0], dim=-1).double()

        vel = torch.cat((torch.zeros_like(hist[:, :1]), (hist[:, 1:] - hist[:, :-1]) / 0.1), dim=1)
        _, hm = self.mot_gru(torch.cat((hist, vel), dim=-1).float())
        r_aux = torch.stack((pl, ch, v_spd, kf["P_vv"][:, 1, 1].clamp_min(0.0).sqrt()), dim=-1).float()
        feats = [hm[0] + self.ray_mlp(r_aux)]
        if self.use_pose:
            _, hp = self.pose_gru(pose5)
            dp = torch.cat((pose5[:, -1] - pose5[:, 0], pose5[:, -1] - pose5[:, -2]), dim=-1)
            gq = torch.sigmoid(F.linear(q_vec.clamp(0.0, 1.5), F.softplus(self.q_w_raw), self.q_b))
            feats.append(gq * (hp[0] + self.dpose_mlp(dp)))
        if self.use_veh:
            _, hv = self.veh_gru(veh5)
            feats.append(hv[0])
        ctx = torch.cat(feats, dim=-1)

        t_s = torch.arange(1, 21, dtype=torch.float64, device=hist.device)[None, :] * 0.1
        kn_idx = ((t_s / 2.0) * 3.0 - 1e-9).floor().long().clamp(0, 2)
        kn_ph = ((t_s / 2.0) * 3.0 - kn_idx.double())[:, :, None]
        bi = torch.arange(bsz, device=hist.device)[:, None]

        mob_last = torch.tanh((v_spd / 0.18) ** 2)
        r_cap = (0.14 + 0.50 * torch.tanh(v_spd / 0.50)) * mob_last
        # Weight vertical (depth/scale) component 1.35x and horizontal 0.75x matching camera perspective
        aniso_w = hist.new_tensor([[[0.75, 1.35]]], dtype=torch.float64)
        k0 = r_cap[:, None, None] * aniso_w * torch.tanh(self.slot0_knots(ctx).double().view(bsz, 3, 2))
        res0 = torch.cat((torch.zeros((bsz, 1, 2), dtype=torch.float64, device=hist.device), k0), dim=1)
        ramp_t = ((t_s / 2.0).clamp(0.0, 1.0) ** 1.40)[:, :, None]
        delta0 = (res0[bi, kn_idx] * (1.0 - kn_ph) + res0[bi, kn_idx + 1] * kn_ph) * ramp_t

        base0 = dp0[:, None, :] + v_f[:, None, :] * t_s[:, :, None]
        slot0 = base0 + delta0
        m_res = self.multi_res(ctx.detach()).view(bsz, 3, 20, 2).double()
        stop_anc = dp0[:, None, :] + v_f[:, None, :] * (0.45 * (1.0 - torch.exp(-t_s[:, :, None] / 0.45)))
        p4 = torch.stack((slot0, base0 + m_res[:, 0], base0 + mob_last[:, None, None] * m_res[:, 1], stop_anc + 0.3 * m_res[:, 2]), dim=1)
        return {"paths": p4, "logits": self.mode_lg(ctx), "act_logits": self.act_head(ctx), "kin_path": base0}


def eval_jaad_wave28_scale_ray_neural_5seed(jd: dict, seeds: list[int], device: torch.device) -> dict:
    sp = jd["split"]
    tr, va, te = np.flatnonzero(sp == "train"), np.flatnonzero(sp == "val"), np.flatnonzero(sp == "test")
    trva = np.concatenate([tr, va])
    n = len(sp)
    ht = torch.from_numpy(jd["hist"]).to(device)
    tt = torch.from_numpy(jd["truth"]).to(device)
    dt_all = torch.full((n, 4), 0.1, dtype=torch.float64, device=device)
    ft_all = torch.arange(1, 21, dtype=torch.float64, device=device)[None].expand(n, -1) * 0.1
    jac = torch.zeros((n, 5, 2, 2), dtype=torch.float64, device=device)
    jac[:, :, 0, 0] = 0.010
    jac[:, :, 1, 1] = 0.028
    dh_rel = torch.from_numpy(jd["veh5"][:, :, 1]).double().to(device) * 10.0
    bh_rel = 100.0 + torch.cumsum(dh_rel, dim=1)
    ank_c = torch.from_numpy(jd["auto_q4"][:, 1:2]).double().expand(n, 5).to(device)

    trva_g, tr_t, va_t, te_t = (torch.from_numpy(x).to(device) for x in (trva, tr, va, te))
    rcfg = calibrate_ray_kalman_params(
        ht[trva_g], dt_all[trva_g], ft_all[trva_g], tt[trva_g],
        jac_src=jac[trva_g], bbox_h_src=bh_rel[trva_g], ank_conf_src=ank_c[trva_g], d_ref=0.25,
    )
    t_s = np.arange(1, 21, dtype=np.float64)[None, :, None] * 0.1
    cv_all = ((jd["hist"][:, -1:] - jd["hist"][:, -2:-1]) / 0.1) * t_s
    err_cv = _jaad_post_onset_err(cv_all[te], jd["truth"][te], jd["onset"][te])
    with torch.inference_mode():
        ork = anisotropic_ray_kalman_filter(ht[te_t], dt_all[te_t], jac[te_t], bh_rel[te_t], ank_c[te_t], **rcfg)
        prk = (ork["p_filt"] - ht[te_t, -1]).cpu().numpy()[:, None, :] + ork["v_filt"].cpu().numpy()[:, None, :] * t_s
    err_rk = _jaad_post_onset_err(prk, jd["truth"][te], jd["onset"][te])

    vt = torch.from_numpy(jd["veh5"]).to(device)
    yat, vat = torch.from_numpy(jd["y_act"]).to(device), torch.from_numpy(jd["v_act"]).to(device)
    pw_c = ((vat[tr].float().sum(0) - yat[tr].float().sum(0)) / yat[tr].float().sum(0).clamp_min(1.0)).clamp(1.0, 12.0)

    arms = [
        ("J28_M0_ScaleRay", jd["auto_pose20"], jd["auto_q4"], False, False),
        ("J28_AutoPose12_ScaleRay", jd["auto_pose12"], jd["auto_q2"], True, False),
        ("J28_AutoPose17_ScaleRay", jd["auto_pose20"], jd["auto_q4"], True, False),
        ("J28_M2_ScaleRay_Veh", jd["auto_pose20"], jd["auto_q4"], False, True),
        ("J28_AutoPose17_ScaleRay_Veh", jd["auto_pose20"], jd["auto_q4"], True, True),
        ("J28_Full_Human_ScaleRay_Veh", jd["human_attr5"], np.ones((n, 2), dtype=np.float32), True, True),
    ]
    err_s = {nm: np.zeros((len(seeds), len(te), 4), dtype=np.float64) for nm, *_ in arms}
    ap_s = {nm: np.zeros((len(seeds), len(te), 4), dtype=np.float64) for nm, *_ in arms}
    va_p4_sum = np.zeros((len(va), 4, 20, 2), dtype=np.float64)
    te_p4_sum = np.zeros((len(te), 4, 20, 2), dtype=np.float64)

    for si, seed in enumerate(seeds):
        for nm, p_arr, q_arr, up, uv in arms:
            pt, qt = torch.from_numpy(p_arr).to(device), torch.from_numpy(q_arr).to(device)
            torch.manual_seed(seed + 37)
            net = JAADScaleRaySplitTrajectoryNet(p_arr.shape[-1], q_arr.shape[-1], up, uv, rcfg).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=2.2e-3, weight_decay=2.5e-4)
            gen = torch.Generator().manual_seed(seed + 149)
            best_sd, best_vl = None, 1e9
            for _ in range(30):
                net.train()
                for b in torch.randperm(len(tr_t), generator=gen).split(64):
                    ib = tr_t[b.to(device)]
                    opt.zero_grad(set_to_none=True)
                    o = net(ht[ib], pt[ib], qt[ib], vt[ib], jac[ib], bh_rel[ib], ank_c[ib])
                    d_k = torch.linalg.vector_norm(o["paths"] - tt[ib][:, None], dim=-1)
                    l0 = (torch.sqrt(d_k[:, 0].square() + 0.0025) - 0.05).mean()
                    lm = d_k[:, 1:].mean(dim=-1).amin(dim=1).mean()
                    bce = F.binary_cross_entropy_with_logits(o["act_logits"], yat[ib].float(), pos_weight=pw_c, reduction="none")
                    lac = ((bce * vat[ib].float()).sum(1) / vat[ib].float().sum(1).clamp_min(1.0)).mean().double()
                    (l0 + 0.35 * lm + 0.25 * lac).backward()
                    opt.step()
                net.eval()
                with torch.inference_mode():
                    ov = net(ht[va_t], pt[va_t], qt[va_t], vt[va_t], jac[va_t], bh_rel[va_t], ank_c[va_t])
                    vl = float(torch.linalg.vector_norm(ov["paths"][:, 0] - tt[va_t], dim=-1).mean().item())
                    if vl < best_vl:
                        best_vl = vl
                        best_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if best_sd is not None:
                net.load_state_dict(best_sd)
            net.eval()
            with torch.inference_mode():
                ovf = net(ht[va_t], pt[va_t], qt[va_t], vt[va_t], jac[va_t], bh_rel[va_t], ank_c[va_t])
                otf = net(ht[te_t], pt[te_t], qt[te_t], vt[te_t], jac[te_t], bh_rel[te_t], ank_c[te_t])
                err_s[nm][si] = _jaad_post_onset_err(otf["paths"][:, 0].cpu().numpy(), jd["truth"][te], jd["onset"][te])
                ap_s[nm][si] = torch.sigmoid(otf["act_logits"]).cpu().numpy()
                if nm == "J28_AutoPose17_ScaleRay_Veh":
                    va_p4_sum += ovf["paths"].cpu().numpy() / len(seeds)
                    te_p4_sum += otf["paths"].cpu().numpy() / len(seeds)

    em = {nm: err_s[nm].mean(axis=0) for nm, *_ in arms}
    v_te = jd["vid"][te]
    strata = {
        "all_locked_test_953": np.ones(len(te), dtype=bool),
        "strict_5frame_deoccluded_168": jd["strict_5f_deocc"][te],
        "strict_5f_deocc_lowdisp_lt0p25_bbox_h_84": jd["strict_deocc_lowdisp_lt0p25_bh"][te],
        "broad_occ_or_late_union_406": jd["broad_occ_406"][te],
    }

    def _r4(a4: np.ndarray, m: np.ndarray) -> dict[str, float]:
        v = a4[m].mean(0)
        return {"ade_2s": float(v[0]), "fde_2s": float(v[1]), "post_0p5s": float(v[2]), "post_1p0s": float(v[3])}

    st_out = {}
    for sk, sm in strata.items():
        md = {"Last2_CV": _r4(err_cv, sm), "J28_AnisoScaleRay_KinOnly": _r4(err_rk, sm)}
        for nm, *_ in arms:
            md[nm] = _r4(em[nm], sm)
        st_out[sk] = {
            "n_windows": int(sm.sum()),
            "n_videos": len(set(v_te[sm].tolist())),
            "models": md,
            "paired_video_ci": {
                "aniso_scale_ray_minus_CV_2s": _loc_bootstrap_ci(err_rk[sm, 0:2] - err_cv[sm, 0:2], v_te[sm]),
                "autopose17_minus_m0_scaleray_2s": _loc_bootstrap_ci(em["J28_AutoPose17_ScaleRay"][sm, 0:2] - em["J28_M0_ScaleRay"][sm, 0:2], v_te[sm]),
                "autopose17_minus_autopose12_scaleray_2s": _loc_bootstrap_ci(em["J28_AutoPose17_ScaleRay"][sm, 0:2] - em["J28_AutoPose12_ScaleRay"][sm, 0:2], v_te[sm]),
                "autopose17_veh_minus_m2_veh_2s": _loc_bootstrap_ci(em["J28_AutoPose17_ScaleRay_Veh"][sm, 0:2] - em["J28_M2_ScaleRay_Veh"][sm, 0:2], v_te[sm]),
                "autopose17_veh_minus_CV_2s": _loc_bootstrap_ci(em["J28_AutoPose17_ScaleRay_Veh"][sm, 0:2] - err_cv[sm, 0:2], v_te[sm]),
            },
        }

    # Zero-retrain val -> test split-conformal safety tubes (90% & 95%, Top-1 & Union-4)
    t_sc = np.linspace(0.25, 1.0, 20, dtype=np.float64)[None, :]
    vspd_va = np.linalg.norm((jd["hist"][va, -1] - jd["hist"][va, -2]) / 0.1, axis=-1)
    vspd_te = np.linalg.norm((jd["hist"][te, -1] - jd["hist"][te, -2]) / 0.1, axis=-1)
    sig_va = (0.12 + 0.45 * vspd_va)[:, None] * t_sc
    sig_te = (0.12 + 0.45 * vspd_te)[:, None] * t_sc
    s1_va = (np.linalg.norm(va_p4_sum[:, 0] - jd["truth"][va], axis=-1) / sig_va).max(axis=1)
    su_va = (np.linalg.norm(va_p4_sum - jd["truth"][va, None], axis=-1).min(axis=1) / sig_va).max(axis=1)
    sc_va = (np.linalg.norm(cv_all[va] - jd["truth"][va], axis=-1) / sig_va).max(axis=1)
    e1_te = np.linalg.norm(te_p4_sum[:, 0] - jd["truth"][te], axis=-1)
    eu_te = np.linalg.norm(te_p4_sum - jd["truth"][te, None], axis=-1).min(axis=1)
    ec_te = np.linalg.norm(cv_all[te] - jd["truth"][te], axis=-1)
    conf_j = {}
    mc = len(va)
    for sk, sm in strata.items():
        conf_j[sk] = {}
        for a_nm, alpha in (("90", 0.10), ("95", 0.05)):
            kq = min(max(int(math.ceil((mc + 1) * (1.0 - alpha))) - 1, 0), mc - 1)
            q1, qu, qc = float(np.sort(s1_va)[kq]), float(np.sort(su_va)[kq]), float(np.sort(sc_va)[kq])
            conf_j[sk][f"nominal_{a_nm}pct"] = {
                "j28_top1_joint_cov": float(np.all(e1_te[sm] <= q1 * sig_te[sm], axis=1).mean()),
                "j28_union4_joint_cov": float(np.all(eu_te[sm] <= qu * sig_te[sm], axis=1).mean()),
                "cv_joint_cov": float(np.all(ec_te[sm] <= qc * sig_te[sm], axis=1).mean()),
                "j28_top1_mean_radius_bh": float((q1 * sig_te[sm]).mean()),
                "j28_union4_mean_radius_bh": float((qu * sig_te[sm]).mean()),
                "cv_mean_radius_bh": float((qc * sig_te[sm]).mean()),
            }

    ap_out = {}
    for sub_nm, sub_m in (("strict_5frame_deoccluded_168", jd["strict_5f_deocc"][te]), ("all_locked_test_953", np.ones(len(te), dtype=bool))):
        ap_out[sub_nm] = {
            nm: {
                cn: {"pos": int(jd["y_act"][te, ci][sub_m & jd["v_act"][te, ci]].sum()),
                     "ap": float(_binary_average_precision(jd["y_act"][te, ci][sub_m & jd["v_act"][te, ci]].astype(np.float64), ap_s[nm].mean(0)[sub_m & jd["v_act"][te, ci], ci]))}
                for ci, cn in enumerate(("Start", "Stop", "Cross"))
            }
            for nm, *_ in arms
        }
    return {"calibrated_scale_ray_params": rcfg, "test_strata": st_out, "zero_retrain_val_to_test_conformal": conf_j, "maneuver_ap": ap_out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch-dir", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, default=Path("wave28_jaad_scale_ray_neural_5seed.zip"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    if not torch.cuda.is_available():
        torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    traj_ds, _ = build_jaad2_full17_cohorts(args.scratch_dir)
    print(f"[Wave 28B] Built JAAD 2.0 trajectory cohort on {device} in {time.perf_counter()-t0:.1f}s", flush=True)
    res = eval_jaad_wave28_scale_ray_neural_5seed(traj_ds, list(SEEDS), device)
    summary = {"wave": "wave28_jaad_scale_ray_neural_and_conformal_5seed", "jaad2_scale_ray_neural_5seed": res}
    raw_b = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sha_b = (json.dumps({"summary.json": hashlib.sha256(raw_b).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw_b)
        zf.writestr("sha256.json", sha_b)
    print(f"Saved {args.zip_path} ({args.zip_path.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
