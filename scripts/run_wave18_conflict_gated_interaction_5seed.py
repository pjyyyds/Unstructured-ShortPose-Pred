"""Wave 18 GPU0 (Slides 6, 8, 10, 18): Conflict-gated O(2) neighbor interaction & TTC/vehicle stratification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from scripts.ppt_mainline_o2_models import PPTMainlineO2Forecaster, local_frenet_with_pose
from scripts.run_dair_o2_bioeq_external import SEEDS
from scripts.run_jrdb_rc1_pose_quality_and_initiation_5seed import _loc_bootstrap_ci
from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import _stage_gpu_dict, fast_person_bootstrap_ci, load_jrdb_10scene_rc2_rc3
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_matched_o2_multiduration import SITES, paired_site_clip_ci
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import _copy_matching, load_ppt_cohort_with_causal_maps
from src.models.bioeq_o2_joint import o2_joint_loss

W18_INT_ARMS = ("M1", "M2b_frozen_head", "M2b_coadapt_head", "M2b_ttc_conflict_gate")


def compute_causal_conflict_risk(edges: torch.Tensor, nbr_mask: torch.Tensor) -> torch.Tensor:
    """Slide 6 & 10 physical radial-approaching risk gate in [0, 1]; strictly 0 when nbr_mask == 0."""
    m = (nbr_mask > 0.5).float()
    dist = edges[:, :, 4].clamp_min(0.0)
    v_rad = edges[:, :, 5] * (edges[:, :, 6] > 0.5).float()
    per_nbr = m * torch.sigmoid((v_rad - 0.20) / 0.25) * torch.exp(-dist / 5.0)
    return per_nbr.amax(dim=1)


def build_interaction_strata(edges_np: np.ndarray, is_car_np: np.ndarray, nbr_mask_np: np.ndarray, has_cars: bool) -> dict[str, np.ndarray]:
    m = nbr_mask_np > 0.5
    dist = np.where(m, edges_np[:, :, 4], 1e6)
    v_rad = np.where(m & (edges_np[:, :, 6] > 0.5), edges_np[:, :, 5], -1e6)
    has_nbr = m.any(axis=1)
    close_app = (m & (dist <= 6.0) & (v_rad > 0.25)).any(axis=1)
    ttc_lt4 = (m & (dist <= 8.0) & (v_rad > 0.25) & (dist / np.maximum(v_rad, 1e-6) <= 4.0)).any(axis=1)
    out = {
        "all": np.ones(len(edges_np), dtype=bool),
        "S0_isolated": ~has_nbr,
        "S1_non_approaching": has_nbr & (~close_app),
        "S2_close_approaching": close_app,
        "S3_ttc_lt4s": ttc_lt4,
    }
    if has_cars:
        car_m = is_car_np > 0.5
        out["S4_approaching_car"] = (m & car_m & (v_rad > 0.25)).any(axis=1)
        out["S5_approaching_ped"] = (m & (~car_m) & (v_rad > 0.25)).any(axis=1)
    return out


class ConflictGatedM2bForecaster(PPTMainlineO2Forecaster):
    """O2-M2b with optional causal radial-approaching conflict gate w_risk and scalar scale."""

    def __init__(self, use_conflict_gate: bool = True) -> None:
        super().__init__(arm="M2b")
        self.use_conflict_gate = use_conflict_gate
        self.graph_scale = 1.0

    def _scaled_graph_step(self, ctx_p: torch.Tensor, ctx_n: torch.Tensor, basis: torch.Tensor, edges: torch.Tensor, is_car: torch.Tensor, nbr_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dp_l = torch.einsum("bmk,bks->bms", edges[:, :, 0:2].double(), basis).float()
        dv_l = torch.einsum("bmk,bks->bms", edges[:, :, 2:4].double(), basis).float()
        inv = torch.stack((edges[:, :, 4] / 10.0, edges[:, :, 5], edges[:, :, 6], is_car), dim=-1).float()
        ep = torch.cat((dp_l, dv_l, inv), dim=-1)
        en = torch.cat((dp_l * ep.new_tensor([1.0, -1.0]), dv_l * ep.new_tensor([1.0, -1.0]), inv), dim=-1)
        w_r = compute_causal_conflict_risk(edges, nbr_mask) if self.use_conflict_gate else edges.new_ones(len(ctx_p))
        sc = float(self.graph_scale) * w_r[:, None]
        return ctx_p + sc * self._graph_delta(ctx_p, ep, nbr_mask), ctx_n + sc * self._graph_delta(ctx_n, en, nbr_mask), w_r

    def forward(self, history: torch.Tensor, bio_even: torch.Tensor | None = None, pose_vec_world: torch.Tensor | None = None,
                signed_knee: torch.Tensor | None = None, q_vec: torch.Tensor | None = None, m_pose: torch.Tensor | None = None,
                p_env: torch.Tensor | None = None, m_map: torch.Tensor | None = None, edges: torch.Tensor | None = None,
                is_car: torch.Tensor | None = None, nbr_mask: torch.Tensor | None = None, obs_dt: torch.Tensor | None = None,
                fut_times_s: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        bsz, t_obs, _ = history.shape
        pvw_m = (pose_vec_world * m_pose[:, None, None, None].to(pose_vec_world.dtype)) if (pose_vec_world is not None and m_pose is not None) else None
        origin, basis, valid = local_frenet_with_pose(history, pvw_m, use_pose_backup=(m_pose is not None))
        body = (history - origin) @ basis
        dt_s = history.new_full((bsz, t_obs - 1), 0.1) if obs_dt is None else obs_dt.to(history.dtype).clamp_min(1e-4)
        vel = torch.cat((torch.zeros_like(body[:, :1]), (body[:, 1:] - body[:, :-1]) / dt_s[:, :, None]), dim=1)
        fp = torch.cat((body, vel, torch.linalg.vector_norm(vel, dim=2, keepdim=True)), dim=2).float()
        _, hm = self.motion(torch.cat((fp, fp * fp.new_tensor([1.0, -1.0, 1.0, -1.0, 1.0])), dim=0))
        cb = self.ctx_proj(hm[0])
        cp, cn = cb[:bsz], cb[bsz:]
        disp_all = torch.linalg.vector_norm(history[:, 1:] - history[:, :-1], dim=-1).sum(dim=-1).float()
        gate = disp_all.new_zeros(bsz)
        if bio_even is not None and pose_vec_world is not None and signed_knee is not None and m_pose is not None:
            pvl = torch.einsum("btvk,bks->btvs", pose_vec_world.double(), basis).float()
            fc, fl = pvl[:, :, 0], pvl[:, :, 1]
            flp = fc.new_tensor([1.0, -1.0])
            _, hpb = self.pose_gru(F.silu(self.pose_in(torch.cat((torch.cat((bio_even, fc, fl, signed_knee), -1), torch.cat((bio_even, fc * flp, fl * flp, -signed_knee), -1)), 0))))
            hpp, hpn = hpb[0][:bsz], hpb[0][bsz:]
            qv = q_vec if q_vec is not None else disp_all.new_zeros((bsz, 4))
            gate = m_pose * torch.sigmoid(self.gate_with_q(torch.cat((0.5 * (cp + cn), 0.5 * (hpp + hpn), qv), dim=-1)).squeeze(-1))
            cp, cn = cp + gate[:, None] * self.pose_proj(hpp), cn + gate[:, None] * self.pose_proj(hpn)
        w_risk = disp_all.new_zeros(bsz)
        if edges is not None and is_car is not None and nbr_mask is not None and self.graph_scale > 0.0:
            cp, cn, w_risk = self._scaled_graph_step(cp, cn, basis, edges, is_car, nbr_mask)
        ce, co = 0.5 * (cp + cn), 0.5 * (cp - cn)
        ld = torch.tanh((disp_all / 0.25).clamp_min(0.05))[:, None, None]
        ck = torch.stack((ld * self.central_dx(ce).view(bsz, 3, 3), ld * self.central_dy(co).view(bsz, 3, 3)), dim=-1)
        pp_k, pn_k = ld * self.pair_knots(cp).view(bsz, 3, 2), ld * self.pair_knots(cn).view(bsz, 3, 2)
        knots5 = torch.cat((ck[:, :2], pp_k[:, None], torch.stack((pn_k[..., 0], -pn_k[..., 1]), -1)[:, None], ck[:, 2:]), dim=1).double()
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
        return {"paths": torch.cat((cv[:, None], cv[:, None] + torch.einsum("bkts,bws->bktw", delta, basis)), dim=1), "logits": logits, "valid": valid, "gate": gate, "w_risk": w_risk}


def _fwd_conflict(net: PPTMainlineO2Forecaster, gd: dict, ids_g: torch.Tensor) -> dict[str, torch.Tensor]:
    return net(
        gd["history"][ids_g], gd["bio_even"][ids_g], gd["pose_vec_world"][ids_g], gd["signed_knee"][ids_g],
        gd["quality"][ids_g], gd["m_pose"][ids_g], gd["p_env"][ids_g], gd["m_map"][ids_g],
        gd["edges"][ids_g], gd["is_car"][ids_g], gd["nbr_mask"][ids_g],
        obs_dt=gd["obs_dt"][ids_g], fut_times_s=gd["fut_s"][ids_g],
    )


def _fit_base_and_conflict(seed: int, grp_arr: np.ndarray, gd: dict, tr_ids: np.ndarray, bsz: int, ep_s: tuple[int, int, int], lr_s: tuple[float, float, float], device: torch.device) -> dict[str, PPTMainlineO2Forecaster]:
    w_tr = make_site_weights(grp_arr, tr_ids).to(device)
    tr_t = torch.from_numpy(tr_ids).to(device)
    truth_tr = gd["truth"][tr_t]

    def _tr(net: PPTMainlineO2Forecaster, prefs: tuple[str, ...], ep: int, lr: float, wd: float, soff: int) -> PPTMainlineO2Forecaster:
        for nm, p in net.named_parameters():
            p.requires_grad = nm.startswith(prefs) or nm in prefs
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(seed + soff)
        net.train()
        for _ in range(ep):
            for ch in torch.randperm(len(tr_ids), generator=gen).split(bsz):
                cg = ch.to(device)
                loss = o2_joint_loss(_fwd_conflict(net, gd, tr_t[cg]), truth_tr[cg], w_tr[cg], act_weight=0.0)
                if loss is not None:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return net.eval()

    torch.manual_seed(seed)
    m0 = _tr(PPTMainlineO2Forecaster("M0").to(device), ("motion.", "ctx_proj.", "cv_low_bonus", "central_", "pair_"), ep_s[0], lr_s[0], 1e-4, 0)
    torch.manual_seed(seed); m1 = PPTMainlineO2Forecaster("M1").to(device); _copy_matching(m1, m0)
    m1 = _tr(m1, ("pose_", "gate_", "central_", "pair_", "cv_low_bonus"), ep_s[1], lr_s[1], 3e-4, 10)

    torch.manual_seed(seed); m2_frz = ConflictGatedM2bForecaster(use_conflict_gate=False).to(device); _copy_matching(m2_frz, m1)
    m2_frz = _tr(m2_frz, ("edge_", "graph_proj", "null_logit"), ep_s[2], lr_s[2], 3e-4, 21)

    torch.manual_seed(seed); m2_co = ConflictGatedM2bForecaster(use_conflict_gate=False).to(device); _copy_matching(m2_co, m1)
    m2_co = _tr(m2_co, ("edge_", "graph_proj", "null_logit", "central_", "pair_", "cv_low_bonus"), ep_s[2], lr_s[2], 3e-4, 22)

    torch.manual_seed(seed); m2_cg = ConflictGatedM2bForecaster(use_conflict_gate=True).to(device); _copy_matching(m2_cg, m1)
    m2_cg = _tr(m2_cg, ("edge_", "graph_proj", "null_logit", "central_", "pair_", "cv_low_bonus"), ep_s[2], lr_s[2], 3e-4, 23)
    return {"M1": m1, "M2b_frozen_head": m2_frz, "M2b_coadapt_head": m2_co, "M2b_ttc_conflict_gate": m2_cg}


def select_inner_conflict_scale(seed: int, grp_arr: np.ndarray, gd: dict, tr_ids: np.ndarray, bsz: int, ep_s: tuple[int, int, int], lr_s: tuple[float, float, float], device: torch.device) -> float:
    src_grps = sorted(set(grp_arr[tr_ids].tolist()))
    scales = (0.0, 0.25, 0.5, 1.0)
    oof = {sc: [] for sc in scales}
    for vg in src_grps[:4]:  # up to 4 inner validation folds (exact all-3 on PINNS, fast 4-loc sub-CV on JRDB)
        in_tr = tr_ids[grp_arr[tr_ids] != vg]
        in_va = tr_ids[grp_arr[tr_ids] == vg]
        va_g = torch.from_numpy(in_va).to(device)
        mods = _fit_base_and_conflict(seed, grp_arr, gd, in_tr, bsz, ep_s, lr_s, device)
        cg = mods["M2b_ttc_conflict_gate"]
        assert isinstance(cg, ConflictGatedM2bForecaster)
        with torch.inference_mode():
            for sc in scales:
                cg.graph_scale = sc
                oo = _fwd_conflict(cg, gd, va_g)
                top_p = oo["paths"][torch.arange(len(in_va), device=device), oo["logits"].argmax(1)]
                oof[sc].append(float(torch.linalg.vector_norm(top_p - gd["truth"][va_g], dim=-1).mean()))
    return float(min(scales, key=lambda sc: np.mean(oof[sc]) + (1e-7 if sc > 0 else 0.0)))


def safe_site_clip_ci(diff_sub: np.ndarray, site_sub: np.ndarray, clip_sub: np.ndarray, seed: int = 2026) -> dict:
    p_sites = sorted(set(site_sub.tolist()))
    if not p_sites:
        return {"mean": {"ade_m": 0.0, "fde_m": 0.0}, "ci95": {"ade_m": [0.0, 0.0], "fde_m": [0.0, 0.0]}, "n_sites_present": 0}
    rng = np.random.default_rng(seed)
    site_clips = {s: sorted(set(clip_sub[site_sub == s].tolist())) for s in p_sites}
    clip_rows = {c: np.flatnonzero(clip_sub == c) for c in set(clip_sub.tolist())}
    pt = np.mean([diff_sub[site_sub == s].mean(axis=0) for s in p_sites], axis=0)
    draws = np.empty((10000, 2), dtype=np.float64)
    for r in range(10000):
        vals = []
        for s in p_sites:
            cs = site_clips[s]
            picked = [cs[j] for j in rng.integers(0, len(cs), size=len(cs))]
            vals.append(diff_sub[np.concatenate([clip_rows[c] for c in picked])].mean(axis=0))
        draws[r] = np.mean(vals, axis=0)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return {"mean": {"ade_m": float(pt[0]), "fde_m": float(pt[1])}, "ci95": {"ade_m": [float(lo[0]), float(hi[0])], "fde_m": [float(lo[1]), float(hi[1])]}, "n_sites_present": len(p_sites)}


def run_conflict_benchmark(
    tag: str, grp_arr: np.ndarray, unit_arr: np.ndarray, is_pinns: bool, gd: dict, truth_np: np.ndarray,
    cv_err: np.ndarray, int_strata: dict[str, np.ndarray], bsz: int, ep_s: tuple[int, int, int], lr_s: tuple[float, float, float], device: torch.device,
) -> tuple[dict, dict[str, np.ndarray]]:
    n = len(truth_np)
    groups = sorted(set(grp_arr.tolist()))
    arms = list(W18_INT_ARMS) + ["M2b_ttc_conflict_unscaled"]
    top_s = {a: np.zeros((len(SEEDS), n, 2)) for a in arms}
    min_s = {a: np.zeros((len(SEEDS), n, 2)) for a in arms}
    sel_scales: dict[str, list[float]] = {g: [] for g in groups}

    for si, seed in enumerate(SEEDS):
        for g in groups:
            tr_ids = np.flatnonzero(grp_arr != g)
            te_ids = np.flatnonzero(grp_arr == g)
            sc_star = select_inner_conflict_scale(seed, grp_arr, gd, tr_ids, bsz, ep_s, lr_s, device)
            sel_scales[str(g)].append(sc_star)
            mods = _fit_base_and_conflict(seed, grp_arr, gd, tr_ids, bsz, ep_s, lr_s, device)
            te_g = torch.from_numpy(te_ids).to(device)
            with torch.inference_mode():
                for a in W18_INT_ARMS:
                    net = mods[a]
                    if a == "M2b_ttc_conflict_gate" and isinstance(net, ConflictGatedM2bForecaster):
                        net.graph_scale = 1.0
                        oo1 = _fwd_conflict(net, gd, te_g)
                        p1, q1 = oo1["paths"].cpu().numpy(), oo1["logits"].softmax(1).cpu().numpy()
                        d1 = np.linalg.norm(p1 - truth_np[te_ids][:, None], axis=-1)
                        tp1, ii = q1.argmax(1), np.arange(len(te_ids))
                        top_s["M2b_ttc_conflict_unscaled"][si, te_ids] = np.column_stack((d1[ii, tp1].mean(1), d1[ii, tp1, -1]))
                        min_s["M2b_ttc_conflict_unscaled"][si, te_ids] = np.column_stack((d1.mean(2).min(1), d1[:, :, -1].min(1)))
                        net.graph_scale = sc_star
                    oo = _fwd_conflict(net, gd, te_g)
                    pp, qq = oo["paths"].cpu().numpy(), oo["logits"].softmax(1).cpu().numpy()
                    dd = np.linalg.norm(pp - truth_np[te_ids][:, None], axis=-1)
                    tp, ii = qq.argmax(1), np.arange(len(te_ids))
                    top_s[a][si, te_ids] = np.column_stack((dd[ii, tp].mean(1), dd[ii, tp, -1]))
                    min_s[a][si, te_ids] = np.column_stack((dd.mean(2).min(1), dd[:, :, -1].min(1)))
        print(f"[W18-INT {tag}] seed={seed} M1={top_s['M1'][si].mean(0)[0]:.4f} M2b_frz={top_s['M2b_frozen_head'][si].mean(0)[0]:.4f} M2b_cg={top_s['M2b_ttc_conflict_gate'][si].mean(0)[0]:.4f}", flush=True)

    tm = {a: top_s[a].mean(0) for a in arms}
    mm = {a: min_s[a].mean(0) for a in arms}

    def _pair_ci(d2: np.ndarray, msk: np.ndarray) -> dict:
        if int(msk.sum()) < 5:
            return {"n": int(msk.sum())}
        if is_pinns:
            return safe_site_clip_ci(d2[msk], grp_arr[msk], unit_arr[msk])
        p_locs = sorted(set(grp_arr[msk].tolist()))
        loc_ci = _loc_bootstrap_ci(d2[msk], grp_arr[msk]) if len(p_locs) >= 2 else {}
        return {"location_equal": loc_ci, **fast_person_bootstrap_ci(d2, msk, unit_arr, n_boot=10000)}

    strata_out: dict[str, object] = {}
    for sn, msk in int_strata.items():
        if not msk.any():
            continue
        ug = sorted(set(grp_arr[msk].tolist()))
        models_d: dict[str, object] = {
            "Last2_CV": {
                "group_equal": {"ade1_m": float(np.mean([cv_err[msk & (grp_arr == g), 0].mean() for g in ug])), "fde1_m": float(np.mean([cv_err[msk & (grp_arr == g), 1].mean() for g in ug]))},
                "window_equal": {"ade1_m": float(cv_err[msk, 0].mean()), "fde1_m": float(cv_err[msk, 1].mean())},
            }
        }
        for a in arms:
            models_d[a] = {
                "group_equal": {
                    "ade1_m": float(np.mean([tm[a][msk & (grp_arr == g), 0].mean() for g in ug])),
                    "fde1_m": float(np.mean([tm[a][msk & (grp_arr == g), 1].mean() for g in ug])),
                    "min_ade6_m": float(np.mean([mm[a][msk & (grp_arr == g), 0].mean() for g in ug])),
                    "min_fde6_m": float(np.mean([mm[a][msk & (grp_arr == g), 1].mean() for g in ug])),
                },
                "window_equal": {"ade1_m": float(tm[a][msk, 0].mean()), "fde1_m": float(tm[a][msk, 1].mean()), "min_ade6_m": float(mm[a][msk, 0].mean()), "min_fde6_m": float(mm[a][msk, 1].mean())},
            }
        strata_out[sn] = {
            "n_windows": int(msk.sum()),
            "n_groups_present": len(ug),
            "models": models_d,
            "paired_top1_ci": {
                "M2b_frozen_minus_M1": _pair_ci(tm["M2b_frozen_head"] - tm["M1"], msk),
                "M2b_coadapt_minus_M1": _pair_ci(tm["M2b_coadapt_head"] - tm["M1"], msk),
                "M2b_conflict_gate_minus_M1": _pair_ci(tm["M2b_ttc_conflict_gate"] - tm["M1"], msk),
                "M2b_conflict_gate_minus_M2b_frozen": _pair_ci(tm["M2b_ttc_conflict_gate"] - tm["M2b_frozen_head"], msk),
                "M2b_conflict_unscaled_minus_M2b_coadapt": _pair_ci(tm["M2b_ttc_conflict_unscaled"] - tm["M2b_coadapt_head"], msk),
            },
        }
    w_r = compute_causal_conflict_risk(gd["edges"], gd["nbr_mask"]).cpu().numpy()
    npz = {f"{tag}_{a}_top1": tm[a] for a in arms} | {f"{tag}_w_risk": w_r}
    return {"n_windows": n, "inner_selected_scales": sel_scales, "mean_w_risk_by_stratum": {sn: float(w_r[m].mean()) for sn, m in int_strata.items() if m.any()}, "strata": strata_out}, npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, required=True)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dp = load_ppt_cohort_with_causal_maps(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin)
    np_w = len(dp["history"])
    gdp = {k: dp[k].to(dev) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
    gdp["obs_dt"] = torch.full((np_w, 4), 0.1, dtype=torch.float64, device=dev)
    gdp["fut_s"] = torch.arange(1, 31, dtype=torch.float64, device=dev)[None].expand(np_w, -1) * 0.1
    gdp["p_env"] = torch.zeros((np_w, 16, 10, 11), dtype=torch.float32, device=dev)
    gdp["m_map"] = torch.zeros(np_w, dtype=torch.float32, device=dev)
    tp_np = dp["truth"].numpy()
    cv_p = ((dp["history"][:, -1:] - dp["history"][:, -2:-1]).numpy() / 0.1) * (np.arange(1, 31, dtype=np.float64)[None, :, None] * 0.1)
    d_cvp = np.linalg.norm(cv_p - tp_np, axis=-1)
    p_str = build_interaction_strata(dp["edges"].numpy(), dp["is_car"].numpy(), dp["nbr_mask"].numpy(), has_cars=True)
    p_res, p_npz = run_conflict_benchmark("pinns", dp["site"], dp["clip"], True, gdp, tp_np, np.column_stack((d_cvp.mean(1), d_cvp[:, -1])), p_str, 64, (45, 25, 26), (2e-3, 1.2e-3, 1.5e-3), dev)

    dj = load_jrdb_10scene_rc2_rc3(args.repo_root)
    gdj = _stage_gpu_dict(dj, dev)
    tj_np = dj["truth"].numpy()
    cv_j = ((dj["history"][:, 1:2] - dj["history"][:, 0:1]).numpy() / 0.4) * (np.arange(1, 8, dtype=np.float64)[None, :, None] * 0.4)
    d_cvj = np.linalg.norm(cv_j - tj_np, axis=-1)
    j_str = build_interaction_strata(dj["edges"].numpy(), dj["is_car"].numpy(), dj["nbr_mask"].numpy(), has_cars=False)
    j_res, j_npz = run_conflict_benchmark("jrdb", dj["location"], dj["actors"], False, gdj, tj_np, np.column_stack((d_cvj.mean(1), d_cvj[:, -1])), j_str, 512, (45, 28, 28), (2.5e-3, 1.5e-3, 1.8e-3), dev)

    summary = {"experiment": "wave18_conflict_gated_interaction_5seed", "device": str(dev), "seeds": list(SEEDS), "pinns_4site_loso_1482": p_res, "jrdb_9location_lolo_5836": j_res}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(args.out_dir / "per_window_conflict_5seed.npz", **p_npz, **j_npz)
    dig = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out_dir.iterdir()) if p.is_file()}
    (args.out_dir / "sha256.json").write_text(json.dumps(dig, indent=2), encoding="utf-8")
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(args.out_dir.iterdir()):
            zf.write(p, arcname=p.name)
    print(f"DONE: {args.zip_path} sha256={hashlib.sha256(args.zip_path.read_bytes()).hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
