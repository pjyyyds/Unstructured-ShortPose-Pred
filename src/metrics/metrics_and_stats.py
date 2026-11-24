"""Section 5.3 metrics (ADE_1, FDE_1, minADE_6, minFDE_6, NLL, Brier, Macro-F1, mAP) and stats."""

from typing import Dict, List, Optional, Sequence
import math
import numpy as np

try:
    from scipy import stats as _scipy_stats
except ImportError:
    _scipy_stats = None


def _fallback_normality_and_paired_p(a: np.ndarray, b: np.ndarray) -> tuple:
    """Dependency-free Jarque-Bera normality + paired t / Wilcoxon fallback when scipy is absent."""
    diff = a - b
    n = len(diff)
    mu = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) + 1e-12
    z = (diff - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)
    jb = (n / 6.0) * (skew ** 2 + 0.25 * (kurt ** 2))
    norm_p = math.exp(-0.5 * jb)  # Exact chi2(df=2) survival function exp(-x/2)!
    if norm_p >= 0.05:
        t_stat = abs(mu) / (sd / math.sqrt(n))
        # Normal/Cornish-Fisher tail approximation for two-sided paired t
        z_eq = t_stat * (1.0 - 1.0 / (4.0 * max(n - 1, 1)))
        p_val = math.erfc(z_eq / math.sqrt(2.0))
        return norm_p, p_val, "paired_t"
    nz = diff[np.abs(diff) > 1e-12]
    m = len(nz)
    if m < 3:
        return norm_p, 1.0, "wilcoxon_degenerate"
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
    w_plus = float(np.sum(ranks[nz > 0]))
    mean_w = m * (m + 1.0) / 4.0
    std_w = math.sqrt(m * (m + 1.0) * (2.0 * m + 1.0) / 24.0) + 1e-12
    z_w = max(0.0, abs(w_plus - mean_w) - 0.5) / std_w
    return norm_p, math.erfc(z_w / math.sqrt(2.0)), "wilcoxon"


def _binary_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact Precision-Recall Average Precision without external sklearn dependency."""
    pos_total = int(np.sum(y_true > 0.5))
    if pos_total == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    yt = (y_true[order] > 0.5).astype(np.float64)
    tp = np.cumsum(yt)
    prec = tp / np.arange(1, len(yt) + 1, dtype=np.float64)
    return float(np.sum(prec * yt) / pos_total)


def compute_action_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    m_act: np.ndarray,
    thresh: float = 0.50,
) -> Dict[str, float]:
    """Compute 5-class Macro-F1 and mAP strictly over valid m_act == 1 samples."""
    valid = np.asarray(m_act).reshape(-1) > 0.5
    if int(np.sum(valid)) == 0:
        return {"macro_f1": float("nan"), "mAP": float("nan"), "valid_act_n": 0.0}

    yt = np.asarray(y_true)[valid]  # [Nv, 5]
    yp = np.asarray(y_prob)[valid]  # [Nv, 5]
    f1_list, ap_list = [], []
    for c in range(yt.shape[1]):
        tc = yt[:, c] > 0.5
        if int(np.sum(tc)) == 0:
            continue
        pc = yp[:, c] >= thresh
        tp = float(np.sum(tc & pc))
        fp = float(np.sum((~tc) & pc))
        fn = float(np.sum(tc & (~pc)))
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        f1_list.append(f1)
        ap_list.append(_binary_average_precision(yt[:, c], yp[:, c]))

    return {
        "macro_f1": float(np.mean(f1_list)) if f1_list else 0.0,
        "mAP": float(np.nanmean(ap_list)) if ap_list else 0.0,
        "valid_act_n": float(np.sum(valid)),
    }


def compute_sample_trajectory_metrics(
    pred_trajs: np.ndarray,
    mode_probs: np.ndarray,
    fut_xy: np.ndarray,
    fut_mask: np.ndarray,
    p_env: Optional[np.ndarray] = None,
    clearance_m: float = 0.25,
) -> Dict[str, np.ndarray]:
    """Return per-sample arrays [N] for ADE_1, FDE_1, minADE_6, minFDE_6, NLL, Brier, BoundaryViol."""
    n_samp, num_k, fut_t, _ = pred_trajs.shape
    f_mask = (np.asarray(fut_mask) > 0.5).astype(np.float64)  # [N, T_f]
    v_cnt = np.maximum(f_mask.sum(axis=-1, keepdims=True), 1.0)  # [N, 1]
    gt = np.nan_to_num(fut_xy, nan=0.0)[:, None, :, :]  # [N, 1, T_f, 2]

    dists = np.linalg.norm(pred_trajs - gt, axis=-1) * f_mask[:, None, :]  # [N, K, T_f]
    ade_all = dists.sum(axis=-1) / v_cnt  # [N, K]

    # Last valid frame index per sample (avoids reading padded 0 at T_f-1)
    idx_grid = np.arange(fut_t)[None, :] * (f_mask > 0.5)
    last_idx = np.argmax(idx_grid, axis=-1)  # [N]
    row_idx = np.arange(n_samp)
    fde_all = np.linalg.norm(
        pred_trajs[row_idx, :, last_idx, :] - np.nan_to_num(fut_xy, nan=0.0)[row_idx, last_idx, :][:, None, :],
        axis=-1,
    )  # [N, K]

    top1_k = np.argmax(mode_probs, axis=-1)  # [N]
    best_k = np.argmin(ade_all, axis=-1)  # [N]

    ade_1 = ade_all[row_idx, top1_k]
    fde_1 = fde_all[row_idx, top1_k]
    min_ade_6 = ade_all[row_idx, best_k]
    min_fde_6 = np.min(fde_all, axis=-1)

    probs = np.clip(mode_probs, 1e-7, 1.0)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    nll = -np.log(probs[row_idx, best_k])
    target_oh = np.eye(num_k)[best_k]
    brier = np.sum((probs - target_oh) ** 2, axis=-1)

    viol = np.zeros(n_samp, dtype=np.float64)
    if p_env is not None:
        top1_traj = pred_trajs[row_idx, top1_k]  # [N, T_f, 2]
        # Channels 2..7 correspond to structural edges/walls/barriers/cones/parked-cars (attr[0..4]);
        # gt_min_d >= clearance_m ensures legal crosswalk curb crossings are NOT counted as violations!
        obs_flag = (p_env[:, :, :, 10] > 0.5) & (p_env[:, :, :, 2:7].sum(axis=-1) > 0.5)
        for i in range(n_samp):
            if not np.any(obs_flag[i]):
                continue
            obs_pts = p_env[i, :, :, :2][obs_flag[i]]  # [M_obs, 2]
            v_steps = top1_traj[i][f_mask[i] > 0.5]
            gt_steps = np.nan_to_num(fut_xy[i], nan=0.0)[f_mask[i] > 0.5]
            if len(v_steps) > 0:
                pred_min_d = np.min(np.linalg.norm(v_steps[:, None, :] - obs_pts[None, :, :], axis=-1))
                gt_min_d = np.min(np.linalg.norm(gt_steps[:, None, :] - obs_pts[None, :, :], axis=-1))
                viol[i] = 1.0 if (pred_min_d < clearance_m and gt_min_d >= clearance_m) else 0.0

    return {
        "ADE_1": ade_1,
        "FDE_1": fde_1,
        "minADE_6": min_ade_6,
        "minFDE_6": min_fde_6,
        "NLL": nll,
        "Brier": brier,
        "BoundaryViol": viol,
    }


def summarize_bucket_slices(
    sample_metrics: Dict[str, np.ndarray],
    buckets: np.ndarray,
    y_act: Optional[np.ndarray] = None,
    m_act: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate mean metrics overall, across Clear/Partial/Heavy buckets, and on transition subset."""
    bucket_names = {0: "Clear", 1: "Partial", 2: "Heavy"}
    b_arr = np.asarray(buckets, dtype=np.int64)
    out: Dict[str, Dict[str, float]] = {}

    def _agg(mask: np.ndarray) -> Dict[str, float]:
        cnt = int(np.sum(mask))
        if cnt == 0:
            return {"count": 0.0}
        res = {k: float(np.mean(v[mask])) for k, v in sample_metrics.items()}
        res["count"] = float(cnt)
        return res

    out["Overall"] = _agg(np.ones(len(b_arr), dtype=bool))
    for b_id, b_name in bucket_names.items():
        out[b_name] = _agg(b_arr == b_id)

    if y_act is not None and m_act is not None:
        trans_mask = (np.asarray(m_act).reshape(-1) > 0.5) & (np.asarray(y_act)[:, :4].sum(axis=-1) > 0.5)
        out["Transition"] = _agg(trans_mask)
    return out


def paired_hypothesis_test(
    err_model: np.ndarray,
    err_ref: np.ndarray,
    rng_seed: int = 42,
) -> Dict[str, object]:
    """Section 5.3 paired statistical test: Shapiro-Wilk -> Paired t-test or Wilcoxon Signed-Rank."""
    a = np.asarray(err_model, dtype=np.float64).reshape(-1)
    b = np.asarray(err_ref, dtype=np.float64).reshape(-1)
    assert len(a) == len(b) and len(a) >= 3, "Need >=3 paired samples"

    diff = a - b
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    abs_delta = float(np.mean(diff))
    rel_pct = 100.0 * abs_delta / (abs(mean_b) + 1e-8)

    if np.allclose(diff, 0.0):
        return {
            "n": len(a),
            "mean_model": mean_a,
            "mean_ref": mean_b,
            "abs_delta": 0.0,
            "rel_pct": 0.0,
            "test_used": "identical",
            "shapiro_p": 1.0,
            "raw_p": 1.0,
            "mpses_met": False,
        }

    if _scipy_stats is not None:
        if len(diff) > 5000:
            rng = np.random.default_rng(rng_seed)
            sub_diff = rng.choice(diff, size=5000, replace=False)
        else:
            sub_diff = diff
        _, shapiro_p = _scipy_stats.shapiro(sub_diff)
        if shapiro_p >= 0.05:
            _, p_val = _scipy_stats.ttest_rel(a, b, alternative="two-sided")
            test_name = "paired_t"
        else:
            nz = diff[np.abs(diff) > 1e-12]
            if len(nz) < 3:
                p_val, test_name = 1.0, "wilcoxon_degenerate"
            else:
                _, p_val = _scipy_stats.wilcoxon(nz, alternative="two-sided")
                test_name = "wilcoxon"
    else:
        shapiro_p, p_val, test_name = _fallback_normality_and_paired_p(a, b)

    mpses_met = (rel_pct <= -5.0) or (abs_delta <= -0.04)
    return {
        "n": int(len(a)),
        "mean_model": mean_a,
        "mean_ref": mean_b,
        "abs_delta": abs_delta,
        "rel_pct": rel_pct,
        "test_used": test_name,
        "shapiro_p": float(shapiro_p),
        "raw_p": float(p_val),
        "mpses_met": bool(mpses_met),
    }


def holm_bonferroni_adjust(results: List[Dict[str, object]], alpha: float = 0.05) -> List[Dict[str, object]]:
    """Apply Holm-Bonferroni step-down family-wise error correction in-place and return list."""
    m = len(results)
    if m == 0:
        return results
    order = np.argsort([float(r["raw_p"]) for r in results])
    running_max = 0.0
    reject_still_allowed = True
    for rank, idx in enumerate(order):
        mult = m - rank
        adj_p = min(1.0, float(results[idx]["raw_p"]) * mult)
        running_max = max(running_max, adj_p)
        results[idx]["holm_p"] = running_max
        if running_max >= alpha:
            reject_still_allowed = False
        results[idx]["significant_005"] = bool(reject_still_allowed)
    return results
