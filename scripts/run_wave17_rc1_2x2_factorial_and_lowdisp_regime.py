"""Wave 17 GPU0: Exact 2x2 (q x speed-damping) factorial on PINNS & JRDB + zero-leakage <0.5m low-displacement regime test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scripts.ppt_mainline_o2_models import PPTMainlineO2Forecaster, local_frenet_with_pose
from scripts.run_jrdb_rc1_pose_quality_and_initiation_5seed import _loc_bootstrap_ci
from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import _stage_gpu_dict, fast_person_bootstrap_ci, load_jrdb_10scene_rc2_rc3
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_matched_o2_multiduration import SITES, paired_site_clip_ci
from scripts.run_pinns_o2_joint_actions_5seed import SEEDS
from scripts.run_pinns_short_disp_hardq_anchor_5seed import load_short_disp_cohort
from src.models.bioeq_o2_joint import o2_joint_loss

FACT_ARMS = ("M0", "M1a", "M1b", "M1_q", "M1_v", "M1_qv")


class FactorialPoseO2Net(PPTMainlineO2Forecaster):
    """Extends PPTMainlineO2Forecaster with clean 2x2 {no-q, +q} x {no-v, +v} gate arms."""

    def __init__(self, fact_arm: str, v_ref: float = 0.85) -> None:
        base_arm = "M0" if fact_arm == "M0" else ("M1a" if fact_arm == "M1a" else ("M1b" if fact_arm in ("M1b", "M1_v") else "M1"))
        super().__init__(arm=base_arm, v_ref=v_ref)
        self.fact_arm = fact_arm
        self.use_v_damp = fact_arm in ("M1_v", "M1_qv")

    def forward(self, history: torch.Tensor, bio_even: torch.Tensor | None = None, pose_vec_world: torch.Tensor | None = None,
                signed_knee: torch.Tensor | None = None, q_vec: torch.Tensor | None = None, m_pose: torch.Tensor | None = None,
                obs_dt: torch.Tensor | None = None, fut_times_s: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        bsz, t_obs, _ = history.shape
        use_p_bkp = self.use_pose and (m_pose is not None)
        pvw_m = (pose_vec_world * m_pose[:, None, None, None].to(pose_vec_world.dtype)) if (pose_vec_world is not None and m_pose is not None) else None
        origin, basis, valid = local_frenet_with_pose(history, pvw_m, use_pose_backup=use_p_bkp)
        body = (history - origin) @ basis
        dt_s = history.new_full((bsz, t_obs - 1), 0.1) if obs_dt is None else obs_dt.to(history.dtype).clamp_min(1e-4)
        vel = torch.cat((torch.zeros_like(body[:, :1]), (body[:, 1:] - body[:, :-1]) / dt_s[:, :, None]), dim=1)
        fp = torch.cat((body, vel, torch.linalg.vector_norm(vel, dim=2, keepdim=True)), dim=2).float()
        fn = fp * fp.new_tensor([1.0, -1.0, 1.0, -1.0, 1.0])
        _, hm = self.motion(torch.cat((fp, fn), dim=0))
        cb = self.ctx_proj(hm[0])
        cp, cn = cb[:bsz], cb[bsz:]
        last_v = (torch.linalg.vector_norm(history[:, -1] - history[:, -2], dim=-1) / dt_s[:, -1]).float()
        disp_all = torch.linalg.vector_norm(history[:, 1:] - history[:, :-1], dim=-1).sum(dim=-1).float()
        gate = last_v.new_zeros(bsz)
        if self.use_pose and bio_even is not None and pose_vec_world is not None and signed_knee is not None and m_pose is not None:
            pvl = torch.einsum("btvk,bks->btvs", pose_vec_world.double(), basis).float()
            fc, fl = pvl[:, :, 0], pvl[:, :, 1]
            flp = fc.new_tensor([1.0, -1.0])
            pp = torch.cat((bio_even, fc, fl, signed_knee), dim=-1)
            pn = torch.cat((bio_even, fc * flp, fl * flp, -signed_knee), dim=-1)
            _, hpb = self.pose_gru(F.silu(self.pose_in(torch.cat((pp, pn), dim=0))))
            hpp, hpn = hpb[0][:bsz], hpb[0][bsz:]
            hpe, ce0 = 0.5 * (hpp + hpn), 0.5 * (cp + cn)
            if self.fact_arm == "M1a":
                gate = m_pose
            elif self.fact_arm in ("M1b", "M1_v"):
                gate = m_pose * torch.sigmoid(self.gate_no_q(torch.cat((ce0, hpe), dim=-1)).squeeze(-1))
            else:
                qv = q_vec if q_vec is not None else last_v.new_zeros((bsz, 4))
                gate = m_pose * torch.sigmoid(self.gate_with_q(torch.cat((ce0, hpe, qv), dim=-1)).squeeze(-1))
            if self.use_v_damp:
                gate = gate * torch.exp(-((last_v / self.v_ref) ** 2))
            cp = cp + gate[:, None] * self.pose_proj(hpp)
            cn = cn + gate[:, None] * self.pose_proj(hpn)
        ce, co = 0.5 * (cp + cn), 0.5 * (cp - cn)
        ld = torch.tanh((disp_all / 0.25).clamp_min(0.05))[:, None, None]
        ck = torch.stack((ld * self.central_dx(ce).view(bsz, 3, 3), ld * self.central_dy(co).view(bsz, 3, 3)), dim=-1)
        pp_k, pn_k = ld * self.pair_knots(cp).view(bsz, 3, 2), ld * self.pair_knots(cn).view(bsz, 3, 2)
        m4 = torch.stack((pn_k[..., 0], -pn_k[..., 1]), dim=-1)[:, None]
        knots5 = torch.cat((ck[:, :2], pp_k[:, None], m4, ck[:, 2:]), dim=1).double()
        knots5 = torch.cat((torch.zeros_like(knots5[:, :, :1]), knots5), dim=2)
        csc = self.central_scores(ce)
        c0 = csc[:, :1] + self.cv_low_bonus * torch.exp(-((disp_all / 0.25) ** 2))[:, None]
        logits = torch.where(valid[:, None], torch.cat((c0, csc[:, 1:3], self.pair_score(cp), self.pair_score(cn), csc[:, 3:]), dim=1), 0.0)
        ts = torch.arange(1, 31, dtype=history.dtype, device=history.device)[None, :] * 0.1 if fut_times_s is None else fut_times_s.to(history.dtype)
        idx = (ts - 1e-9).floor().long().clamp(0, 2)
        ph = (ts - idx.to(history.dtype))[:, None, :, None]
        bi, mi = torch.arange(bsz, device=history.device)[:, None, None], torch.arange(5, device=history.device)[None, :, None]
        delta = knots5[bi, mi, idx[:, None, :]] * (1.0 - ph) + knots5[bi, mi, (idx + 1)[:, None, :]] * ph
        cv = origin + ((history[:, -1:] - history[:, -2:-1]) / dt_s[:, -1:, None]) * ts[:, :, None]
        return {"paths": torch.cat((cv[:, None], cv[:, None] + torch.einsum("bkts,bws->bktw", delta, basis)), dim=1), "logits": logits, "valid": valid, "gate": gate}


def fit_eval_2x2_factorial_fold(
    seed: int, loc_or_site: np.ndarray, gd: dict, tr_ids: np.ndarray, te_ids: np.ndarray,
    device: torch.device, is_jrdb: bool,
) -> dict[str, np.ndarray]:
    w_tr = make_site_weights(loc_or_site, tr_ids).to(device)
    tr_t, te_t = torch.from_numpy(tr_ids).to(device), torch.from_numpy(te_ids).to(device)
    truth_tr = gd["truth"][tr_t]
    truth_te = gd["truth"][te_t].cpu().numpy()
    bsz_sp = 256 if is_jrdb else 64
    lr0, lr1 = (2.5e-3, 1.5e-3) if is_jrdb else (2.0e-3, 1.2e-3)

    def _fwd(net: FactorialPoseO2Net, ids: torch.Tensor) -> dict:
        return net(
            gd["history"][ids], gd["bio_even"][ids], gd["pose_vec_world"][ids],
            gd["signed_knee"][ids], gd["quality"][ids], gd["m_pose"][ids],
            obs_dt=gd["obs_dt"][ids] if is_jrdb else None,
            fut_times_s=gd["fut_s"][ids] if is_jrdb else None,
        )

    def _train(net: FactorialPoseO2Net, prefs: tuple[str, ...], ep: int, lr: float, wd: float, soff: int) -> FactorialPoseO2Net:
        for nm, p in net.named_parameters():
            p.requires_grad = nm.startswith(prefs) or nm in prefs
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(seed + soff)
        net.train()
        for _ in range(ep):
            for ch in torch.randperm(len(tr_ids), generator=gen).split(bsz_sp):
                cg = ch.to(device)
                loss = o2_joint_loss(_fwd(net, tr_t[cg]), truth_tr[cg], w_tr[cg], act_weight=0.0)
                if loss is not None:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return net.eval()

    torch.manual_seed(seed)
    m0 = _train(FactorialPoseO2Net("M0").to(device), ("motion.", "ctx_proj.", "cv_low_bonus", "central_", "pair_"), 45, lr0, 1e-4, 0)
    mods: dict[str, FactorialPoseO2Net] = {"M0": m0}
    for i, arm in enumerate(FACT_ARMS[1:]):
        torch.manual_seed(seed)
        net = FactorialPoseO2Net(arm).to(device)
        sd = net.state_dict()
        for k, v in m0.state_dict().items():
            if k in sd and sd[k].shape == v.shape:
                sd[k].copy_(v)
        net.load_state_dict(sd)
        mods[arm] = _train(net, ("pose_", "gate_", "central_", "pair_", "cv_low_bonus"), 26, lr1, 3e-4, 10 + i)

    out_m: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for arm, net in mods.items():
            o = _fwd(net, te_t)
            p, q = o["paths"].cpu().numpy(), o["logits"].softmax(dim=1).cpu().numpy()
            dst = np.linalg.norm(p - truth_te[:, None], axis=-1)
            ade_k, fde_k = dst.mean(2), dst[:, :, -1]
            top = q.argmax(1); ii = np.arange(len(te_ids))
            out_m[arm] = np.column_stack((ade_k[ii, top], fde_k[ii, top], ade_k.min(1), fde_k.min(1), o["gate"].cpu().numpy()))
    return out_m


def evaluate_lowdisp_regime_loso(data_p: dict) -> dict:
    """Zero-leakage O(2) low-displacement (<0.5m) specialized kinematic vs ground/joint pose test on PINNS."""
    hist = data_p["history"].double()
    truth = data_p["truth"].double()
    pvw = data_p["pose_vec_world"].double()
    be = data_p["bio_even"].double()
    sk = data_p["signed_knee"].double()
    qmask = data_p["qmask_clear_partial"].double()
    n = len(hist)
    disp5 = torch.linalg.vector_norm(hist[:, 1:] - hist[:, :-1], dim=-1).sum(dim=-1).numpy()
    low_all = disp5 < 0.50

    _, basis_m, valid_m = local_frenet_with_pose(hist, None, use_pose_backup=False)
    _, basis_p, _ = local_frenet_with_pose(hist, pvw * qmask[:, None, None, None], use_pose_backup=True)
    h_loc = torch.einsum("btw,bws->bts", hist, basis_m).numpy()
    t_w = np.arange(-4, 1, dtype=np.float64)
    v_ols_w = ((hist.numpy() * t_w[None, :, None]).sum(1) / (t_w ** 2).sum()) / 0.1
    v_last_w = (hist[:, -1] - hist[:, -2]).numpy() / 0.1
    ts = np.arange(1, 31, dtype=np.float64)[None, :, None] * 0.1
    truth_np = truth.numpy()

    p_hold = np.zeros_like(truth_np)
    p_ols5 = v_ols_w[:, None, :] * ts
    p_cv2 = v_last_w[:, None, :] * ts

    v_l_loc = np.einsum("bw,bws->bs", v_last_w, basis_m.numpy())
    v_o_loc = np.einsum("bw,bws->bs", v_ols_w, basis_m.numpy())
    fc_w = (pvw[:, -1, 0] * qmask[:, None]).numpy()
    dfc_w = ((pvw[:, -1, 0] - pvw[:, 0, 0]) * qmask[:, None]).numpy()
    fl_w = (pvw[:, -1, 1] * qmask[:, None]).numpy()
    fc_loc = np.einsum("bw,bws->bs", fc_w, basis_p.numpy())
    dfc_loc = np.einsum("bw,bws->bs", dfc_w, basis_p.numpy())
    fl_loc = np.einsum("bw,bws->bs", fl_w, basis_p.numpy())
    dbe = ((be[:, -1] - be[:, 0]) * qmask[:, None]).numpy()
    sk_l = (sk[:, -1, 0] * qmask).numpy()

    # Parity-preserving O(2) features: longitudinal uses even scalars, lateral uses odd lateral scalars
    x_m_even = np.column_stack((v_l_loc[:, 0], v_o_loc[:, 0], disp5, h_loc[:, 0, 0]))
    x_m_odd = np.column_stack((v_l_loc[:, 1], v_o_loc[:, 1], h_loc[:, 0, 1]))
    x_g_even = np.column_stack((x_m_even, fc_loc[:, 0], dfc_loc[:, 0], fl_loc[:, 0]))
    x_g_odd = np.column_stack((x_m_odd, fc_loc[:, 1], dfc_loc[:, 1], fl_loc[:, 1]))
    x_f_even = np.column_stack((x_g_even, dbe))
    x_f_odd = np.column_stack((x_g_odd, sk_l))
    y_res_m = np.einsum("btw,bws->bts", truth_np - p_cv2, basis_m.numpy())
    y_res_p = np.einsum("btw,bws->bts", truth_np - p_cv2, basis_p.numpy())

    preds = {"Zero_Hold": p_hold, "OLS5_CV": p_ols5, "Last2_CV": p_cv2,
             "Regime_M0": p_cv2.copy(), "Regime_M1_ground": p_cv2.copy(), "Regime_M1_full": p_cv2.copy()}
    alphas = (1.0, 5.0, 20.0, 80.0, 300.0, 1e6)
    site_arr, clip_arr = data_p["site"], data_p["clip"]
    for s in SITES:
        tr = (site_arr != s) & low_all
        te = (site_arr == s) & low_all
        src = [u for u in SITES if u != s]
        for nm, xe, xo, yr, b_mat in (
            ("Regime_M0", x_m_even, x_m_odd, y_res_m, basis_m.numpy()),
            ("Regime_M1_ground", x_g_even, x_g_odd, y_res_p, basis_p.numpy()),
            ("Regime_M1_full", x_f_even, x_f_odd, y_res_p, basis_p.numpy()),
        ):
            a_err = np.zeros(len(alphas))
            for vs in src:
                fi, vi = tr & (site_arr != vs), tr & (site_arr == vs)
                for ai, al in enumerate(alphas):
                    we = np.linalg.solve(xe[fi].T @ xe[fi] + al * np.eye(xe.shape[1]), xe[fi].T @ yr[fi, :, 0])
                    wo = np.linalg.solve(xo[fi].T @ xo[fi] + al * np.eye(xo.shape[1]), xo[fi].T @ yr[fi, :, 1])
                    pr_v = p_cv2[vi] + np.einsum("bts,bws->btw", np.stack((xe[vi] @ we, xo[vi] @ wo), axis=-1), b_mat[vi])
                    a_err[ai] += float(np.linalg.norm(pr_v - truth_np[vi], axis=-1).mean())
            best_a = alphas[int(a_err.argmin())]
            we = np.linalg.solve(xe[tr].T @ xe[tr] + best_a * np.eye(xe.shape[1]), xe[tr].T @ yr[tr, :, 0])
            wo = np.linalg.solve(xo[tr].T @ xo[tr] + best_a * np.eye(xo.shape[1]), xo[tr].T @ yr[tr, :, 1])
            preds[nm][te] = p_cv2[te] + np.einsum("bts,bws->btw", np.stack((xe[te] @ we, xo[te] @ wo), axis=-1), b_mat[te])

    err_map = {k: np.column_stack((np.linalg.norm(v - truth_np, axis=-1).mean(1), np.linalg.norm(v[:, -1] - truth_np[:, -1], axis=-1))) for k, v in preds.items()}
    out_st = {}
    for sn, sm in (("first_short_323", data_p["strata"]["first_short_323"]), ("all_lowdisp_lt05", low_all), ("first_short_clear_81", data_p["strata"]["first_short_clear"])):
        ss, sc = site_arr[sm], clip_arr[sm]
        md = {k: {"site_equal": {"ade1_m": float(np.mean([v[sm][ss == u, 0].mean() for u in SITES if (ss == u).any()])),
                                 "fde1_m": float(np.mean([v[sm][ss == u, 1].mean() for u in SITES if (ss == u).any()]))}} for k, v in err_map.items()}
        cs = (("Zero_Hold_minus_Last2_CV", "Zero_Hold", "Last2_CV"), ("OLS5_minus_Last2_CV", "OLS5_CV", "Last2_CV"),
              ("Regime_M0_minus_Last2_CV", "Regime_M0", "Last2_CV"), ("Regime_M1_ground_minus_Last2_CV", "Regime_M1_ground", "Last2_CV"),
              ("Regime_M1_ground_minus_Regime_M0", "Regime_M1_ground", "Regime_M0"), ("Regime_M1_full_minus_Regime_M0", "Regime_M1_full", "Regime_M0"))
        out_st[sn] = {"n_windows": int(sm.sum()), "models": md, "paired_clip_ci": {cn: paired_site_clip_ci(err_map[ma][sm] - err_map[mb][sm], ss, sc) for cn, ma, mb in cs}}
    return out_st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--output-zip", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = load_short_disp_cohort(args.meta_dir, args.pose_npz)
    dj = load_jrdb_10scene_rc2_rc3(args.repo_root)
    gdp = {k: dp[k].to(dev) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose")}
    gdj = _stage_gpu_dict(dj, dev)

    np_w, nj_w = len(dp["site"]), len(dj["scene"])
    acc_p = {a: np.zeros((len(args.seeds), np_w, 5), dtype=np.float64) for a in FACT_ARMS}
    acc_j = {a: np.zeros((len(args.seeds), nj_w, 5), dtype=np.float64) for a in FACT_ARMS}
    j_locs = sorted(set(dj["location"].tolist()))

    for si, seed in enumerate(args.seeds):
        for s in SITES:
            tr, te = np.flatnonzero(dp["site"] != s), np.flatnonzero(dp["site"] == s)
            op = fit_eval_2x2_factorial_fold(seed, dp["site"], gdp, tr, te, dev, is_jrdb=False)
            for a in FACT_ARMS:
                acc_p[a][si, te] = op[a]
        for l in j_locs:
            tr, te = np.flatnonzero(dj["location"] != l), np.flatnonzero(dj["location"] == l)
            oj = fit_eval_2x2_factorial_fold(seed, dj["location"], gdj, tr, te, dev, is_jrdb=True)
            for a in FACT_ARMS:
                acc_j[a][si, te] = oj[a]
        print(f"Finished Wave17 2x2 factorial PINNS+JRDB seed={seed}", flush=True)

    mp = {a: acc_p[a].mean(0) for a in FACT_ARMS}
    mj = {a: acc_j[a].mean(0) for a in FACT_ARMS}
    cols = ("ade1_m", "fde1_m", "min_ade6_m", "min_fde6_m", "mean_gate")
    f_cs = (("q_at_no_v_M1q_minus_M1b", "M1_q", "M1b"), ("q_at_v_M1qv_minus_M1v", "M1_qv", "M1_v"),
            ("v_at_no_q_M1v_minus_M1b", "M1_v", "M1b"), ("v_at_q_M1qv_minus_M1q", "M1_qv", "M1_q"),
            ("M1qv_minus_M1b_joint", "M1_qv", "M1b"), ("M1qv_minus_M0", "M1_qv", "M0"))

    p_out, j_out = {}, {}
    for sn, sm in (("all_1482", np.ones(np_w, dtype=bool)), ("clear_pose", dp["strata"]["clear_pose"]), ("heavy_pose", dp["strata"]["heavy_pose"]), ("first_short_323", dp["strata"]["first_short_323"])):
        ss, sc = dp["site"][sm], dp["clip"][sm]
        p_out[sn] = {
            "n_windows": int(sm.sum()),
            "models": {a: {"site_equal": {c: float(np.mean([mp[a][sm][ss == u, k].mean() for u in SITES if (ss == u).any()])) for k, c in enumerate(cols)}} for a in FACT_ARMS},
            "paired_clip_ci": {cn: paired_site_clip_ci(mp[ma][sm, :2] - mp[mb][sm, :2], ss, sc) for cn, ma, mb in f_cs},
        }
    for sn, sm in (("all_10scenes", dj["strata"]["all_10scenes"]), ("clear_pose", dj["strata"]["clear_pose"]), ("partial_pose", dj["strata"]["partial_pose"]), ("short_disp_lt05", dj["strata"]["short_disp_lt05"])):
        sl = dj["location"][sm]; ul = sorted(set(sl.tolist()))
        j_out[sn] = {
            "n_windows": int(sm.sum()),
            "models": {a: {"location_equal": {c: float(np.mean([mj[a][sm][sl == u, k].mean() for u in ul])) for k, c in enumerate(cols)}} for a in FACT_ARMS},
            "paired_location_ci": {cn: _loc_bootstrap_ci(mj[ma][sm, :2] - mj[mb][sm, :2], sl) for cn, ma, mb in f_cs},
            "paired_person_ci": {cn: fast_person_bootstrap_ci(mj[ma][:, :2] - mj[mb][:, :2], sm, dj["actors"]) for cn, ma, mb in f_cs},
        }

    lowdisp_reg = evaluate_lowdisp_regime_loso(dp)
    report = {"experiment": "wave17_rc1_2x2_factorial_and_lowdisp_regime", "seeds": list(args.seeds),
              "pinns_2x2_factorial": p_out, "jrdb_2x2_factorial": j_out, "pinns_lowdisp_regime_loso": lowdisp_reg}
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = json.dumps({"wave17_rc1_2x2_factorial_and_lowdisp_regime.json": hashlib.sha256(raw).hexdigest()}, indent=2).encode("utf-8")
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("wave17_rc1_2x2_factorial_and_lowdisp_regime.json", raw)
        zf.writestr("sha256.json", sh)
    print("Saved:", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
