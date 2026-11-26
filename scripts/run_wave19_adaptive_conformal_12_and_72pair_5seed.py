"""Wave 19b GPU1: Exact zero-retrain locally-weighted split-conformal (Constant vs Speed vs Speed+Spread) on PINNS (12-pair) & JRDB (72-pair)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.optimize import nnls
import torch

from scripts.run_dair_o2_bioeq_external import SEEDS
from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import _stage_gpu_dict, load_jrdb_10scene_rc2_rc3
from scripts.run_pinns_conformal_and_events import conformal_quantile
from scripts.run_pinns_matched_o2_multiduration import SITES
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import load_ppt_cohort_with_causal_maps
from scripts.run_wave19_unified_clean_m0_to_m3_5seed import UnifiedCleanO2Forecaster, _fwd_u, fit_unified_clean_fold


def fit_positive_scale_model(err_oof: np.ndarray, v0_oof: np.ndarray, spread_oof: np.ndarray | None) -> np.ndarray:
    """Fit positive scale sigma(x) = w0 + w1*v0 (+ w2*spread) strictly on source-only OOF residuals via NNLS."""
    cols = [np.ones_like(v0_oof), v0_oof] if spread_oof is None else [np.ones_like(v0_oof), v0_oof, spread_oof]
    x_mat = np.column_stack(cols).astype(np.float64)
    w, _ = nnls(x_mat, np.maximum(err_oof.astype(np.float64), 0.05))
    w[0] = max(float(w[0]), 0.15)
    return w


def eval_scale_model(w: np.ndarray, v0: np.ndarray, spread: np.ndarray | None) -> np.ndarray:
    cols = [np.ones_like(v0), v0] if spread is None else [np.ones_like(v0), v0, spread]
    return np.maximum(np.column_stack(cols) @ w, 0.15)


@torch.inference_mode()
def predict_tubes_and_spread(net: UnifiedCleanO2Forecaster, gd: dict, truth_np: np.ndarray, ids: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    oo = _fwd_u(net, gd, torch.from_numpy(ids).to(device))
    pw = oo["paths"].cpu().numpy()
    top = oo["logits"].argmax(1).cpu().numpy()
    dst = np.linalg.norm(pw - truth_np[ids, None], axis=-1)
    ii = np.arange(len(ids))
    tubes = np.column_stack((dst[ii, top].max(1), dst.max(2).min(1)))
    end_pts = pw[:, :, -1, :]
    spread = np.linalg.norm(end_pts - end_pts.mean(axis=1, keepdims=True), axis=-1).mean(axis=1)
    return tubes, spread


def source_oof_scale_weights(
    seed: int, grp_arr: np.ndarray, gd: dict, truth_np: np.ndarray, v0_np: np.ndarray,
    tr_ids: np.ndarray, bsz: int, ep_s: tuple[int, int, int, int], lr_s: tuple[float, float, float, float],
    device: torch.device, max_inner_folds: int = 2,
) -> dict[str, np.ndarray]:
    all_src = sorted(set(grp_arr[tr_ids].tolist()))
    oof_m = np.zeros(len(grp_arr), dtype=bool)
    oof_tb = np.zeros((len(grp_arr), 2), dtype=np.float64)
    oof_sp = np.zeros(len(grp_arr), dtype=np.float64)
    if len(all_src) >= 3:
        splits = [(tr_ids[grp_arr[tr_ids] != vg], tr_ids[grp_arr[tr_ids] == vg]) for vg in all_src[:max_inner_folds]]
    else:
        splits = [(tr_ids[1::2], tr_ids[::2]), (tr_ids[::2], tr_ids[1::2])][:max_inner_folds]
    for in_tr, in_va in splits:
        m3 = fit_unified_clean_fold(seed, grp_arr, gd, in_tr, bsz, ep_s, lr_s, device, g_scale=0.5)["U_M3_mono_conflict"]
        tb, sp = predict_tubes_and_spread(m3, gd, truth_np, in_va, device)
        oof_m[in_va] = True
        oof_tb[in_va] = tb
        oof_sp[in_va] = sp
    om = np.flatnonzero(oof_m)
    return {
        "t1_spd": fit_positive_scale_model(oof_tb[om, 0], v0_np[om], None),
        "u6_spd": fit_positive_scale_model(oof_tb[om, 1], v0_np[om], None),
        "t1_hyb": fit_positive_scale_model(oof_tb[om, 0], v0_np[om], oof_sp[om]),
        "u6_hyb": fit_positive_scale_model(oof_tb[om, 1], v0_np[om], oof_sp[om]),
    }


def score_cal_test_split(
    tb_cal: np.ndarray, sp_cal: np.ndarray, v0_cal: np.ndarray, cv_cal: np.ndarray,
    tb_te: np.ndarray, sp_te: np.ndarray, v0_te: np.ndarray, cv_te: np.ndarray,
    w_dict: dict[str, np.ndarray], alpha: float,
) -> dict[str, dict[str, float]]:
    rcv = conformal_quantile(cv_cal, alpha)
    spd_masks = (v0_te < 0.30, (v0_te >= 0.30) & (v0_te < 1.00), v0_te >= 1.00)
    nom = 1.0 - alpha
    out: dict[str, dict[str, float]] = {
        "Last2_CV_const": {"cov": float((cv_te <= rcv).mean()), "mean_rad_m": float(rcv)}
    }
    specs = (
        ("M3_constant", np.ones_like(v0_cal), np.ones_like(v0_te), np.ones_like(v0_cal), np.ones_like(v0_te)),
        ("M3_speed_scaled", eval_scale_model(w_dict["t1_spd"], v0_cal, None), eval_scale_model(w_dict["t1_spd"], v0_te, None),
         eval_scale_model(w_dict["u6_spd"], v0_cal, None), eval_scale_model(w_dict["u6_spd"], v0_te, None)),
        ("M3_speed_spread_scaled", eval_scale_model(w_dict["t1_hyb"], v0_cal, sp_cal), eval_scale_model(w_dict["t1_hyb"], v0_te, sp_te),
         eval_scale_model(w_dict["u6_hyb"], v0_cal, sp_cal), eval_scale_model(w_dict["u6_hyb"], v0_te, sp_te)),
    )
    for name, s1_c, s1_t, s6_c, s6_t in specs:
        q1 = conformal_quantile(tb_cal[:, 0] / s1_c, alpha)
        q6 = conformal_quantile(tb_cal[:, 1] / s6_c, alpha)
        r1, r6 = q1 * s1_t, q6 * s6_t
        hit6 = tb_te[:, 1] <= r6
        scovs = [float(hit6[m].mean()) for m in spd_masks if int(m.sum()) >= 5]
        out[name] = {
            "top1_cov": float((tb_te[:, 0] <= r1).mean()),
            "u6_cov": float(hit6.mean()),
            "mean_u6_rad_m": float(r6.mean()),
            "slow_u6_cov": float(hit6[spd_masks[0]].mean()) if spd_masks[0].any() else float("nan"),
            "fast_u6_cov": float(hit6[spd_masks[2]].mean()) if spd_masks[2].any() else float("nan"),
            "slow_mean_rad_m": float(r6[spd_masks[0]].mean()) if spd_masks[0].any() else float("nan"),
            "fast_mean_rad_m": float(r6[spd_masks[2]].mean()) if spd_masks[2].any() else float("nan"),
            "worst_speed_stratum_abs_gap": float(max(abs(c - nom) for c in scovs)) if scovs else 0.0,
        }
    return out


def _agg_runs(run_list: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for m_name in run_list[0]:
        out[m_name] = {}
        for k in run_list[0][m_name]:
            arr = np.array([r[m_name][k] for r in run_list], dtype=np.float64)
            fin = arr[np.isfinite(arr)]
            out[m_name][k] = float(fin.mean()) if len(fin) else 0.0
    return out


def run_adaptive_conformal_benchmark(
    tag: str, grp_arr: np.ndarray, gd: dict, truth_np: np.ndarray, v0_np: np.ndarray, cv_tube: np.ndarray,
    bsz: int, ep_s: tuple[int, int, int, int], lr_s: tuple[float, float, float, float],
    seeds: list[int], cross_seeds: list[int], device: torch.device,
) -> dict:
    groups = sorted(set(grp_arr.tolist()))
    within_runs = {"90": [], "95": []}
    for seed in seeds:
        for g in groups:
            tr = np.flatnonzero(grp_arr != g)
            te = np.flatnonzero(grp_arr == g)
            if len(te) < 16:
                continue
            w_d = source_oof_scale_weights(seed, grp_arr, gd, truth_np, v0_np, tr, bsz, ep_s, lr_s, device)
            m3 = fit_unified_clean_fold(seed, grp_arr, gd, tr, bsz, ep_s, lr_s, device, g_scale=0.5)["U_M3_mono_conflict"]
            tb, sp = predict_tubes_and_spread(m3, gd, truth_np, te, device)
            mid = len(te) // 2
            c_sl, t_sl = slice(0, mid), slice(mid, len(te))
            for lvl, al in (("90", 0.10), ("95", 0.05)):
                within_runs[lvl].append(score_cal_test_split(
                    tb[c_sl], sp[c_sl], v0_np[te[c_sl]], cv_tube[te[c_sl]],
                    tb[t_sl], sp[t_sl], v0_np[te[t_sl]], cv_tube[te[t_sl]], w_d, al,
                ))
        print(f"[W19b-GPU1 {tag}] Finished within-group conformal seed={seed}", flush=True)

    cross_runs = {"90": [], "95": []}
    pairs = [(groups[i], groups[j]) for i in range(len(groups)) for j in range(i + 1, len(groups))]
    max_in = 2 if len(groups) <= 5 else 1
    for seed in cross_seeds:
        for ga, gb in pairs:
            tr_rem = np.flatnonzero((grp_arr != ga) & (grp_arr != gb))
            ia, ib = np.flatnonzero(grp_arr == ga), np.flatnonzero(grp_arr == gb)
            w_d = source_oof_scale_weights(seed, grp_arr, gd, truth_np, v0_np, tr_rem, bsz, ep_s, lr_s, device, max_inner_folds=max_in)
            m3 = fit_unified_clean_fold(seed, grp_arr, gd, tr_rem, bsz, ep_s, lr_s, device, g_scale=0.5)["U_M3_mono_conflict"]
            tba, spa = predict_tubes_and_spread(m3, gd, truth_np, ia, device)
            tbb, spb = predict_tubes_and_spread(m3, gd, truth_np, ib, device)
            for (tc, sc, vc, cc, tt, st, vt, ct) in (
                (tba, spa, v0_np[ia], cv_tube[ia], tbb, spb, v0_np[ib], cv_tube[ib]),
                (tbb, spb, v0_np[ib], cv_tube[ib], tba, spa, v0_np[ia], cv_tube[ia]),
            ):
                for lvl, al in (("90", 0.10), ("95", 0.05)):
                    cross_runs[lvl].append(score_cal_test_split(tc, sc, vc, cc, tt, st, vt, ct, w_d, al))
        print(f"[W19b-GPU1 {tag}] Finished {2 * len(pairs)} ordered cross-pairs seed={seed}", flush=True)

    return {
        "n_windows": len(grp_arr),
        "n_groups": len(groups),
        "n_ordered_cross_pairs": 2 * len(pairs),
        "within_group_50_50_fixed_predictor": {l: _agg_runs(within_runs[l]) for l in ("90", "95")},
        "cross_group_ordered_pairs_fixed_predictor_no_retrain": {l: _agg_runs(cross_runs[l]) for l in ("90", "95")},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = load_ppt_cohort_with_causal_maps(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin)
    np_w = len(dp["history"])
    gdp = {k: dp[k].to(dev) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask", "y_act", "valid_act")}
    gdp["obs_dt"] = torch.full((np_w, 4), 0.1, dtype=torch.float64, device=dev)
    gdp["fut_s"] = torch.arange(1, 31, dtype=torch.float64, device=dev)[None].expand(np_w, -1) * 0.1
    hp_np, tp_np = dp["history"].numpy(), dp["truth"].numpy()
    v0_p = np.linalg.norm(hp_np[:, -1] - hp_np[:, -2], axis=-1) / 0.1
    cv_tp = np.linalg.norm((hp_np[:, -1:] + ((hp_np[:, -1:] - hp_np[:, -2:-1]) / 0.1) * (np.arange(1, 31)[None, :, None] * 0.1)) - tp_np, axis=-1).max(1)
    p_res = run_adaptive_conformal_benchmark("pinns", np.asarray(dp["site"], dtype=str), gdp, tp_np, v0_p, cv_tp, 64, (45, 25, 26, 28), (2e-3, 1.2e-3, 1.5e-3, 1.2e-3), list(args.seeds), list(args.seeds), dev)

    dj = load_jrdb_10scene_rc2_rc3(args.repo_root)
    gdj = _stage_gpu_dict(dj, dev)
    hj_np, tj_np = dj["history"].numpy(), dj["truth"].numpy()
    v0_j = np.linalg.norm(hj_np[:, 1] - hj_np[:, 0], axis=-1) / 0.4
    cv_tj = np.linalg.norm((hj_np[:, -1:] + ((hj_np[:, 1:2] - hj_np[:, 0:1]) / 0.4) * (np.arange(1, 8)[None, :, None] * 0.4)) - tj_np, axis=-1).max(1)
    j_res = run_adaptive_conformal_benchmark("jrdb", np.asarray(dj["location"], dtype=str), gdj, tj_np, v0_j, cv_tj, 512, (45, 28, 28, 30), (2.5e-3, 1.5e-3, 1.8e-3, 1.5e-3), list(args.seeds), list(args.seeds)[:3], dev)

    summary = {"experiment": "wave19_adaptive_conformal_12_and_72pair_5seed", "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
               "seeds": list(args.seeds), "pinns_12pair_adaptive_conformal": p_res, "jrdb_72pair_adaptive_conformal": j_res}
    raw = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = (json.dumps({"summary.json": hashlib.sha256(raw).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw)
        zf.writestr("sha256.json", sh)
    print("Saved:", args.zip_path, flush=True)


if __name__ == "__main__":
    main()
