"""Wave 27B: JAAD 2.0 Symmetric 5-Frame Intra-Window SPRT Precursor Net, AutoVis Gate, NP Conformal, & Anisotropic Scale-Ray Kalman."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.run_wave18_conflict_gated_interaction_5seed import _loc_bootstrap_ci
from scripts.run_wave23_jaad_autopose_and_isofpr_leadtime import SEEDS, _jaad_post_onset_err
from scripts.run_wave25_jaad_reappearance_per_event_audit import (
    audit_per_event_timestamps_and_visible_alarms,
    extract_jaad_slide_reappearance_metadata,
)
from scripts.run_wave26_jaad_full17_reappear_and_conformal_fpr import (
    audit_np_conformal_fpr_stability,
    build_jaad2_full17_cohorts,
)
from src.metrics.metrics_and_stats import _binary_average_precision
from src.models.jaad_full17_reappear_models import JAADCoherenceSplitTrajectoryNet
from src.models.ray_kalman_and_cusum import (
    anisotropic_ray_kalman_filter,
    calibrate_ray_kalman_params,
)


class JAADIntra5FSPRTPrecursorNet(nn.Module):
    """Symmetric 5-frame intra-window causal SPRT evidence accumulator + split co-visibility precursor network."""

    def __init__(self, d_kin: int, d_pose: int, d_q: int, d_veh: int, use_pose: bool = True, use_veh: bool = True) -> None:
        super().__init__()
        self.use_pose = use_pose
        self.use_veh = use_veh
        self.kin_gru = nn.GRU(d_kin, 32, batch_first=True)
        self.step_kin_ev = nn.Linear(32, 3)
        if use_pose:
            self.pose_gru = nn.GRU(d_pose, 32, batch_first=True)
            self.delta_mlp = nn.Sequential(nn.Linear(d_pose * 2, 32), nn.SiLU(), nn.Linear(32, 24))
            self.q_gate = nn.Linear(d_q, 2)
            self.step_pose_ev = nn.Linear(32, 3)
            nn.init.zeros_(self.q_gate.bias)
        if use_veh:
            self.veh_mlp = nn.Sequential(nn.Linear(d_veh, 16), nn.SiLU(), nn.Linear(16, 16))
        d_tot = 32 + (32 + 24 if use_pose else 0) + (16 if use_veh else 0)
        self.head = nn.Sequential(nn.LayerNorm(d_tot), nn.Linear(d_tot, 48), nn.SiLU(), nn.Dropout(0.10))
        self.cls_out = nn.Linear(48, 3)
        self.dt_out = nn.Linear(48, 3)
        self.sprt_decay_raw = nn.Parameter(torch.tensor(0.65))
        self.sprt_mix_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, kin: torch.Tensor, pose: torch.Tensor, q: torch.Tensor, veh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hk_seq, hk = self.kin_gru(kin)
        # Stepwise 5-frame intra-window SPRT log-evidence (identically applied to every 5-frame positive & negative window!)
        dec = torch.sigmoid(self.sprt_decay_raw) * 0.5 + 0.45  # in [0.45, 0.95]
        exps = torch.arange(kin.size(1) - 1, -1, -1, device=kin.device, dtype=kin.dtype)
        w_t = (dec ** exps)[None, :, None]
        w_t = w_t / w_t.sum(dim=1, keepdim=True).clamp_min(1e-6)
        ev_sprt = (w_t * self.step_kin_ev(hk_seq)).sum(dim=1)

        feats = [hk[0]]
        if self.use_pose:
            hp_seq, hp = self.pose_gru(pose)
            g2 = torch.sigmoid(self.q_gate(q) + 0.40)
            dp = self.delta_mlp(torch.cat((pose[:, -1] - pose[:, 0], pose[:, -1] - pose[:, 2]), dim=-1))
            feats.extend([g2[:, 0:1] * hp[0], g2[:, 1:2] * dp])
            ev_sprt = ev_sprt + g2[:, 0:1] * (w_t * self.step_pose_ev(hp_seq)).sum(dim=1)
        if self.use_veh:
            feats.append(self.veh_mlp(veh))
        h = self.head(torch.cat(feats, dim=-1))
        mix = torch.sigmoid(self.sprt_mix_raw)
        logits = (1.0 - 0.45 * mix) * self.cls_out(h) + (0.45 * mix) * ev_sprt
        return logits, F.softplus(self.dt_out(h))


def fit_and_predict_wave27_sprt_arms(
    kin: np.ndarray,
    arms_spec: list[tuple[str, np.ndarray, np.ndarray, bool, bool]],
    veh: np.ndarray,
    y: np.ndarray,
    tau_s: np.ndarray,
    is_neg: np.ndarray,
    split: np.ndarray,
    seeds: list[int],
    device: torch.device,
    epochs: int = 28,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    tr = np.flatnonzero(split == "train")
    va = np.flatnonzero(split == "val")
    te = np.flatnonzero(split == "test")
    kt = torch.from_numpy(kin).to(device)
    vt = torch.from_numpy(veh).to(device)
    yt = torch.from_numpy(y).to(device)
    dt_tgt = torch.from_numpy(np.clip(-tau_s, 0.0, 2.0).astype(np.float32)).to(device)
    nt = torch.from_numpy(is_neg).to(device)
    tr_t, va_t, te_t = torch.from_numpy(tr).to(device), torch.from_numpy(va).to(device), torch.from_numpy(te).to(device)
    pw = ((len(tr) - yt[tr_t].sum(0)) / yt[tr_t].sum(0).clamp_min(1.0)).clamp(1.0, 12.0)

    probs_tr = {nm: np.zeros((len(seeds), len(tr), 3), dtype=np.float64) for nm, *_ in arms_spec}
    probs_va = {nm: np.zeros((len(seeds), len(va), 3), dtype=np.float64) for nm, *_ in arms_spec}
    probs_te = {nm: np.zeros((len(seeds), len(te), 3), dtype=np.float64) for nm, *_ in arms_spec}

    for si, seed in enumerate(seeds):
        for nm, p_arr, q_arr, up, uv in arms_spec:
            pt, qt = torch.from_numpy(p_arr).to(device), torch.from_numpy(q_arr).to(device)
            q_obs = (qt[:, :2].mean(dim=1).clamp(0.0, 1.0)) if up else torch.ones(len(kin), device=device)
            w_vis = torch.where(nt, torch.ones_like(q_obs), (0.35 + 0.65 * torch.exp(-dt_tgt / 0.85)) * (0.45 + 0.55 * q_obs))
            torch.manual_seed(seed + 53)
            net = JAADIntra5FSPRTPrecursorNet(kin.shape[-1], p_arr.shape[-1], q_arr.shape[-1], veh.shape[-1], use_pose=up, use_veh=uv).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=2.2e-3, weight_decay=2.5e-4)
            gen = torch.Generator().manual_seed(seed + 251)
            best_sd, best_vl = None, 1e9
            for _ in range(epochs):
                net.train()
                for b in torch.randperm(len(tr_t), generator=gen).split(128):
                    ib = tr_t[b.to(device)]
                    opt.zero_grad(set_to_none=True)
                    lg, dl = net(kt[ib], pt[ib], qt[ib], vt[ib])
                    bce = F.binary_cross_entropy_with_logits(lg, yt[ib], pos_weight=pw, reduction="none").mean(dim=1)
                    l_cls = (bce * w_vis[ib]).mean()
                    pos_m = ~nt[ib]
                    l_dl = F.smooth_l1_loss((dl[pos_m] * yt[ib][pos_m]).sum(dim=1), dt_tgt[ib][pos_m]) if pos_m.any() else lg.new_tensor(0.0)
                    (l_cls + 0.15 * l_dl).backward()
                    opt.step()
                net.eval()
                with torch.inference_mode():
                    lv, _ = net(kt[va_t], pt[va_t], qt[va_t], vt[va_t])
                    vl = float(F.binary_cross_entropy_with_logits(lv, yt[va_t], pos_weight=pw).item())
                    if vl < best_vl:
                        best_vl = vl
                        best_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if best_sd is not None:
                net.load_state_dict(best_sd)
            net.eval()
            with torch.inference_mode():
                probs_tr[nm][si] = torch.sigmoid(net(kt[tr_t], pt[tr_t], qt[tr_t], vt[tr_t])[0]).cpu().numpy()
                probs_va[nm][si] = torch.sigmoid(net(kt[va_t], pt[va_t], qt[va_t], vt[va_t])[0]).cpu().numpy()
                probs_te[nm][si] = torch.sigmoid(net(kt[te_t], pt[te_t], qt[te_t], vt[te_t])[0]).cpu().numpy()

    return (
        {k: v.mean(0) for k, v in probs_tr.items()},
        {k: v.mean(0) for k, v in probs_va.items()},
        {k: v.mean(0) for k, v in probs_te.items()},
    )


def apply_symmetric_visibility_gate(
    p_dict: dict[str, np.ndarray],
    vis_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Symmetrically damp non-visible windows across ALL positive and negative 5-frame windows."""
    m = np.asarray(vis_mask, dtype=bool)[:, None]
    return {k: np.where(m, v, 1e-5 * v) for k, v in p_dict.items()}


def eval_jaad_wave27_aniso_scale_ray_traj(jd: dict, device: torch.device) -> dict:
    """Evaluate JAAD 2.0 Dashcam Anisotropic Scale-Ray PJ-IAKF on 953 / 168 / 84 strata."""
    sp = jd["split"]
    tr, va, te = np.flatnonzero(sp == "train"), np.flatnonzero(sp == "val"), np.flatnonzero(sp == "test")
    trva = np.concatenate([tr, va])
    n = len(sp)
    ht = torch.from_numpy(jd["hist"]).to(device)
    tt = torch.from_numpy(jd["truth"]).to(device)
    dt_all = torch.full((n, 4), 0.1, dtype=torch.float64, device=device)
    ft_all = torch.arange(1, 21, dtype=torch.float64, device=device)[None].expand(n, -1) * 0.1
    # Dashcam normalized bbox plane: horizontal fcx (0.010 bbox-H/px) vs vertical bottom fby (0.028 bbox-H/px)
    jac = torch.zeros((n, 5, 2, 2), dtype=torch.float64, device=device)
    jac[:, :, 0, 0] = 0.010
    jac[:, :, 1, 1] = 0.028
    # Reconstruct relative bbox height curve from veh5[:, :, 1] = (h_t - h_{t-1}) / h_ref * 10
    dh_rel = torch.from_numpy(jd["veh5"][:, :, 1]).double().to(device) * 10.0  # ~px units on 100px ref
    bh_rel = 100.0 + torch.cumsum(dh_rel, dim=1)
    ank_c = torch.from_numpy(jd["auto_q4"][:, 1:2]).double().expand(n, 5).to(device)

    trva_g, te_t = torch.from_numpy(trva).to(device), torch.from_numpy(te).to(device)
    rcfg = calibrate_ray_kalman_params(
        ht[trva_g], dt_all[trva_g], ft_all[trva_g], tt[trva_g],
        jac_src=jac[trva_g], bbox_h_src=bh_rel[trva_g], ank_conf_src=ank_c[trva_g], d_ref=0.25,
    )
    icfg = calibrate_ray_kalman_params(
        ht[trva_g], dt_all[trva_g], ft_all[trva_g], tt[trva_g],
        jac_src=None, bbox_h_src=bh_rel[trva_g], ank_conf_src=ank_c[trva_g], d_ref=0.25,
    )
    t_s = np.arange(1, 21, dtype=np.float64)[None, :, None] * 0.1
    cv_te = ((jd["hist"][te, -1:] - jd["hist"][te, -2:-1]) / 0.1) * t_s
    err_cv = _jaad_post_onset_err(cv_te, jd["truth"][te], jd["onset"][te])
    with torch.inference_mode():
        oi = anisotropic_ray_kalman_filter(ht[te_t], dt_all[te_t], None, bh_rel[te_t], ank_c[te_t], **icfg)
        pi = (oi["p_filt"] - ht[te_t, -1]).cpu().numpy()[:, None, :] + oi["v_filt"].cpu().numpy()[:, None, :] * t_s
        or_k = anisotropic_ray_kalman_filter(ht[te_t], dt_all[te_t], jac[te_t], bh_rel[te_t], ank_c[te_t], **rcfg)
        pr = (or_k["p_filt"] - ht[te_t, -1]).cpu().numpy()[:, None, :] + or_k["v_filt"].cpu().numpy()[:, None, :] * t_s
    err_iso = _jaad_post_onset_err(pi, jd["truth"][te], jd["onset"][te])
    err_ray = _jaad_post_onset_err(pr, jd["truth"][te], jd["onset"][te])

    v_te = jd["vid"][te]
    strata = {
        "all_locked_test_953": np.ones(len(te), dtype=bool),
        "strict_5frame_deoccluded_168": jd["strict_5f_deocc"][te],
        "strict_5f_deocc_lowdisp_lt0p25_bbox_h_84": jd["strict_deocc_lowdisp_lt0p25_bh"][te],
    }
    out_st = {}
    for sk, sm in strata.items():
        out_st[sk] = {
            "n_windows": int(sm.sum()),
            "models": {
                "Last2_CV": {"ade_2s": float(err_cv[sm, 0].mean()), "fde_2s": float(err_cv[sm, 1].mean())},
                "J27_IsoKalman_Only": {"ade_2s": float(err_iso[sm, 0].mean()), "fde_2s": float(err_iso[sm, 1].mean())},
                "J27_AnisoScaleRay_PJ_IAKF": {"ade_2s": float(err_ray[sm, 0].mean()), "fde_2s": float(err_ray[sm, 1].mean())},
            },
            "paired_video_ci": {
                "aniso_scale_ray_minus_CV": _loc_bootstrap_ci(err_ray[sm, 0:2] - err_cv[sm, 0:2], v_te[sm]),
                "aniso_scale_ray_minus_iso": _loc_bootstrap_ci(err_ray[sm, 0:2] - err_iso[sm, 0:2], v_te[sm]),
            },
        }
    return {"calibrated_aniso_params": rcfg, "calibrated_iso_params": icfg, "strata": out_st}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch-dir", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, default=Path("wave27_jaad_cusum_and_ray_traj.zip"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    traj_ds, slide_ds = build_jaad2_full17_cohorts(args.scratch_dir)
    meta = extract_jaad_slide_reappearance_metadata((args.scratch_dir / "jaad2.zip").read_bytes())
    print(f"[Wave 27B] Built JAAD 2.0 cohorts on {device} in {time.perf_counter()-t0:.1f}s", flush=True)

    lead_arms = [
        ("Lead_M0_kin_only", slide_ds["auto_pose20"], slide_ds["auto_q4"], False, False),
        ("Lead_AutoPose_CovisSafe", slide_ds["auto_pose12"], slide_ds["auto_q2"], True, False),
        ("Lead_AutoPose_rtmpose17", slide_ds["auto_pose20"], slide_ds["auto_q4"], True, False),
        ("Lead_M2_kin_veh", slide_ds["auto_pose20"], slide_ds["auto_q4"], False, True),
        ("Lead_AutoPose_veh", slide_ds["auto_pose20"], slide_ds["auto_q4"], True, True),
        ("Lead_HumanAttr_orient_gaze", slide_ds["human_attr"], slide_ds["human_q"], True, False),
        ("Lead_Full_human_veh", slide_ds["human_attr"], slide_ds["human_q"], True, True),
    ]
    p_tr, p_va, p_te = fit_and_predict_wave27_sprt_arms(
        slide_ds["kin"], lead_arms, slide_ds["veh"], slide_ds["y"], slide_ds["tau_s"],
        slide_ds["is_neg"], slide_ds["split"], list(SEEDS), device,
    )
    sp = slide_ds["split"]
    tr_m, va_m, te_m = sp == "train", sp == "val", sp == "test"
    q_vis_all = 0.65 * slide_ds["auto_q4"][:, 0] + 0.35 * slide_ds["auto_q4"][:, 1] >= 0.32
    occ_vis_all = meta["occ_at_t0"] == "none"

    p_tr_av = apply_symmetric_visibility_gate(p_tr, q_vis_all[tr_m])
    p_va_av = apply_symmetric_visibility_gate(p_va, q_vis_all[va_m])
    p_te_av = apply_symmetric_visibility_gate(p_te, q_vis_all[te_m])

    p_tr_sv = apply_symmetric_visibility_gate(p_tr, occ_vis_all[tr_m])
    p_va_sv = apply_symmetric_visibility_gate(p_va, occ_vis_all[va_m])
    p_te_sv = apply_symmetric_visibility_gate(p_te, occ_vis_all[te_m])

    summary = {
        "wave": "wave27_jaad_intra5f_sprt_autovis_and_scale_ray_5seed",
        "ungated_intra5f_sprt": {
            "event_audit": audit_per_event_timestamps_and_visible_alarms(p_va, p_te, slide_ds, meta),
            "np_conformal_fpr": audit_np_conformal_fpr_stability(p_tr, p_va, p_te, slide_ds),
        },
        "autovis_camera_gated_intra5f_sprt": {
            "event_audit": audit_per_event_timestamps_and_visible_alarms(p_va_av, p_te_av, slide_ds, meta),
            "np_conformal_fpr": audit_np_conformal_fpr_stability(p_tr_av, p_va_av, p_te_av, slide_ds),
        },
        "strict_t0_visible_gated_intra5f_sprt": {
            "event_audit": audit_per_event_timestamps_and_visible_alarms(p_va_sv, p_te_sv, slide_ds, meta),
            "np_conformal_fpr": audit_np_conformal_fpr_stability(p_tr_sv, p_va_sv, p_te_sv, slide_ds),
        },
        "jaad2_aniso_scale_ray_trajectory": eval_jaad_wave27_aniso_scale_ray_traj(traj_ds, device),
    }
    raw_b = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sha_b = (json.dumps({"summary.json": hashlib.sha256(raw_b).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw_b)
        zf.writestr("sha256.json", sha_b)
    print(f"Saved {args.zip_path} ({args.zip_path.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
