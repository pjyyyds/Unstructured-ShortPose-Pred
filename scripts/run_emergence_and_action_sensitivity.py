"""Five-seed LOSO track-emergence strata (449/261/323/1033) & action threshold sensitivity (0.5x-1.5x)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.audit_pinns_observations import source_frames
from scripts.benchmark_heldout_sites import HISTORY, WINDOW, load_site_windows
from scripts.run_pinns_hybrid_action_events_5seed import _best_f1_thresh, score_action_matrix
from scripts.run_pinns_matched_o2_multiduration import SITES, paired_site_clip_ci
from scripts.run_pinns_o2_joint_actions_5seed import ACTIONS, SEEDS
from scripts.run_unified_o2_full_gpu_benchmark import (
    fit_unified_fold,
    load_unified_cohort,
    select_unified_scales,
)
from src.data.actions import compute_action_labels


def build_emergence_masks(meta_dir: Path, data: dict) -> dict[str, np.ndarray]:
    groups, _, _ = load_site_windows(meta_dir)
    rows = [r for s in sorted(groups) for r in groups[s]]
    if len(rows) != len(data["site"]):
        raise ValueError("Window count mismatch against load_site_windows")
    cache_frames: dict[str, tuple] = {}
    first_contig = np.zeros(len(rows), dtype=bool)
    first_raw_arr = np.zeros(len(rows), dtype=bool)
    prior_35 = np.zeros(len(rows), dtype=bool)
    for idx, row in enumerate(rows):
        if row.clip not in cache_frames:
            cache_frames[row.clip] = source_frames(meta_dir / "labels" / f"{row.clip}.txt")
        frames, first_raw, _, _ = cache_frames[row.clip]
        start = row.t0 - HISTORY + 1
        times = frames[row.person_id]
        prev = 0
        while start - prev - 1 in times:
            prev += 1
        first_contig[idx] = prev == 0
        first_raw_arr[idx] = (3 * start + 1) == first_raw[row.person_id]
        prior_35[idx] = prev >= WINDOW
    disp5 = np.linalg.norm(np.diff(data["history"].numpy(), axis=1), axis=-1).sum(axis=1)
    first_short = first_contig & (disp5 < 0.50)
    if (int(first_contig.sum()), int(first_raw_arr.sum()), int(first_short.sum()), int(prior_35.sum())) != (449, 261, 323, 1033):
        raise ValueError("Emergence counts diverged from verified audit (449, 261, 323, 1033)")
    return {
        "all_1482": np.ones(len(rows), dtype=bool),
        "first_contiguous_449": first_contig,
        "first_raw_261": first_raw_arr,
        "first_short_323": first_short,
        "prior_35_1033": prior_35,
    }


def scaled_labels_and_valid(hist_np: np.ndarray, truth_np: np.ndarray, v_hist_mean: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(hist_np)
    hm = np.ones(5, dtype=np.float32)
    fm = np.ones(30, dtype=np.float32)
    ya = np.zeros((n, 5), dtype=bool)
    va = np.zeros((n, 5), dtype=bool)
    v_low, v_high = 0.20 * scale, 0.60 * scale
    for i in range(n):
        yi, mi = compute_action_labels(hist_np[i], hm, truth_np[i], fm, dt=0.1, speed_scale=scale, angle_scale=scale)
        if mi > 0.5:
            ya[i] = yi > 0.5
            va[i, :] = True
            va[i, 0] = bool(v_hist_mean[i] <= v_low)
            va[i, 1] = bool(v_hist_mean[i] >= v_high)
    return ya, va


def implied_probs_scaled(hist_np: np.ndarray, paths_np: np.ndarray, probs_np: np.ndarray, vh: np.ndarray, scale: float) -> np.ndarray:
    n, k_modes, _, _ = paths_np.shape
    hm = np.ones(5, dtype=np.float32)
    fm = np.ones(30, dtype=np.float32)
    out = np.zeros((n, 5), dtype=np.float64)
    for i in range(n):
        hp = hist_np[i]
        for k in range(k_modes):
            yk, _ = compute_action_labels(hp, hm, paths_np[i, k], fm, dt=0.1, speed_scale=scale, angle_scale=scale)
            out[i] += probs_np[i, k] * yk
    out = np.clip(out, 0.0, 1.0)
    out[:, 0] = np.where(vh <= 0.20 * scale, out[:, 0], 0.0)
    out[:, 1] = np.where(vh >= 0.60 * scale, out[:, 1], 0.0)
    return out


def summarize_stratum(err_dict: dict[str, np.ndarray], mask: np.ndarray, site_arr: np.ndarray, clip_arr: np.ndarray) -> dict:
    sub_site = site_arr[mask]
    sub_clip = clip_arr[mask]
    res: dict[str, dict] = {"n_windows": int(mask.sum()), "by_site_counts": {s: int((sub_site == s).sum()) for s in SITES}, "models": {}}
    for m_name, arr in err_dict.items():
        sub = arr[mask]
        by_s = {}
        for s in SITES:
            sm = sub_site == s
            if sm.any():
                by_s[s] = {"ade_m": float(sub[sm, 0].mean()), "fde_m": float(sub[sm, 1].mean()), "min_fde6_m": float(sub[sm, 2].mean())}
        res["models"][m_name] = {
            "site_equal_ade_m": float(np.mean([v["ade_m"] for v in by_s.values()])),
            "site_equal_fde_m": float(np.mean([v["fde_m"] for v in by_s.values()])),
            "site_equal_min_fde6_m": float(np.mean([v["min_fde6_m"] for v in by_s.values()])),
            "pooled_ade_m": float(sub[:, 0].mean()),
            "pooled_fde_m": float(sub[:, 1].mean()),
            "by_site": by_s,
        }
    res["paired_site_clip_ci"] = {
        "full_minus_cv": paired_site_clip_ci(err_dict["full_bayes"][mask, :2] - err_dict["cv"][mask, :2], sub_site, sub_clip),
        "full_minus_u0": paired_site_clip_ci(err_dict["full_bayes"][mask, :2] - err_dict["u0_bayes"][mask, :2], sub_site, sub_clip),
        "native_t5_minus_naive_pad_t20": paired_site_clip_ci(
            err_dict["full_bayes"][mask, :2] - err_dict["naive_pad_t20_bayes"][mask, :2], sub_site, sub_clip
        ),
    }
    return res


def _eval_unified_mode(net, data: dict, ids: np.ndarray, bs: float, gs: float, use_br: bool, pad_t20: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.inference_mode():
        h = data["history"][ids]
        b = data["bio_seq"][ids]
        if pad_t20:
            h = torch.cat([h[:, :1].expand(-1, 15, -1), h], dim=1)
            b = torch.cat([b[:, :1].expand(-1, 15, -1), b], dim=1)
        out = net(h, b, data["q_bio"][ids], data["edges"][ids], data["is_car"][ids], data["nbr_mask"][ids], bio_scale=bs, graph_scale=gs, use_action_bridge=use_br)
        paths = out["paths"].cpu().numpy()
        probs = out["logits"].softmax(1).cpu().numpy()
        act_p = torch.sigmoid(out["act_logits"]).cpu().numpy().astype(np.float64)
    truth = data["truth"][ids].numpy()
    bay = np.einsum("nk,nktw->ntw", probs, paths)
    d_bay = np.linalg.norm(bay - truth, axis=-1)
    d_all = np.linalg.norm(paths - truth[:, None], axis=-1)
    err = np.stack((d_bay.mean(1), d_bay[:, -1], d_all[:, :, -1].min(1)), axis=-1)
    return err, paths, probs, act_p


def run_suite(meta_dir: Path, pose_npz: Path, joint_zip: Path, out_dir: Path, seeds: tuple[int, ...] = SEEDS) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_unified_cohort(meta_dir, pose_npz, joint_zip)
    masks = build_emergence_masks(meta_dir, data)
    n_tot = len(data["site"])
    hist_np, truth_np, vh = data["history"].numpy(), data["truth"].numpy(), data["v_hist_mean"]

    v_last = hist_np[:, -1:] - hist_np[:, -2:-1]
    cv_pred = v_last * np.arange(1, 31, dtype=np.float64)[None, :, None]
    d_cv = np.linalg.norm(cv_pred - truth_np, axis=-1)
    cv_err = np.stack((d_cv.mean(1), d_cv[:, -1], d_cv[:, -1]), axis=-1)

    arms = ("u0_bayes", "u1_bayes", "u2_bayes", "full_bayes", "naive_pad_t20_bayes")
    err_sum = {k: np.zeros((n_tot, 3), dtype=np.float64) for k in arms}
    head_sum = np.zeros((n_tot, 5), dtype=np.float64)
    paths_ens = np.zeros((n_tot, 6, 30, 2), dtype=np.float64)
    probs_ens = np.zeros((n_tot, 6), dtype=np.float64)
    inner_oof_by_fold: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for s_hold in SITES:
        tr_ids = np.flatnonzero(data["site"] != s_hold)
        te_ids = np.flatnonzero(data["site"] == s_hold)
        oof_h = np.zeros((n_tot, 5), dtype=np.float64)
        oof_pa = np.zeros((n_tot, 6, 30, 2), dtype=np.float64)
        oof_pr = np.zeros((n_tot, 6), dtype=np.float64)
        for seed in seeds:
            bs, gs = select_unified_scales(seed, data, tr_ids, device)
            net = fit_unified_fold(seed, data, tr_ids, device)
            for tag, b_arg, g_arg, br_arg, p20 in (
                ("u0_bayes", 0.0, 0.0, False, False),
                ("u1_bayes", bs, 0.0, False, False),
                ("u2_bayes", bs, gs, False, False),
                ("full_bayes", bs, gs, True, False),
                ("naive_pad_t20_bayes", bs, gs, True, True),
            ):
                e_te, p_te, pr_te, ap_te = _eval_unified_mode(net, data, te_ids, b_arg, g_arg, br_arg, p20)
                err_sum[tag][te_ids] += e_te / len(seeds)
                if tag == "full_bayes":
                    head_sum[te_ids] += ap_te / len(seeds)
                    paths_ens[te_ids] += p_te / len(seeds)
                    probs_ens[te_ids] += pr_te / len(seeds)
            for vs in sorted(set(data["site"][tr_ids].tolist())):
                fi = tr_ids[data["site"][tr_ids] != vs]
                vi = tr_ids[data["site"][tr_ids] == vs]
                nv = fit_unified_fold(seed + 17, data, fi, device)
                _, p_v, pr_v, ap_v = _eval_unified_mode(nv, data, vi, bs, gs, True, False)
                oof_h[vi] += ap_v / len(seeds)
                oof_pa[vi] += p_v / len(seeds)
                oof_pr[vi] += pr_v / len(seeds)
        inner_oof_by_fold[s_hold] = (oof_h, oof_pa, oof_pr)
    return _finalize_suite(data, masks, cv_err, err_sum, head_sum, paths_ens, probs_ens, inner_oof_by_fold, out_dir, seeds)


def _calibrate_hybrid_fold(ya: np.ndarray, va: np.ndarray, h_oof: np.ndarray, imp_oof: np.ndarray, tr_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alphas, thrs = np.zeros(5), np.full(5, 0.5)
    for c_i in range(5):
        vm = va[tr_ids, c_i]
        yt = ya[tr_ids, c_i][vm]
        if yt.sum() == 0:
            continue
        best_f1, best_a, best_t = -1.0, 0.0, 0.5
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            sc = (1.0 - a) * h_oof[tr_ids, c_i][vm] + a * imp_oof[tr_ids, c_i][vm]
            t = _best_f1_thresh(yt, sc)
            pr = sc >= t
            tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
            f1 = (2.0 * tp) / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best_f1, best_a, best_t = f1, a, t
        alphas[c_i], thrs[c_i] = best_a, best_t
    return alphas, thrs


def _finalize_suite(data, masks, cv_err, err_sum, head_sum, paths_ens, probs_ens, inner_oof_by_fold, out_dir: Path, seeds) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_errs = {"cv": cv_err, **err_sum}
    strata_out = {k: summarize_stratum(all_errs, m, data["site"], data["clip"]) for k, m in masks.items()}
    hist_np, truth_np, vh = data["history"].numpy(), data["truth"].numpy(), data["v_hist_mean"]

    sens_out: dict[str, dict] = {}
    for sc in (0.50, 0.75, 1.00, 1.25, 1.50):
        ya, va = scaled_labels_and_valid(hist_np, truth_np, vh, sc)
        imp_te = implied_probs_scaled(hist_np, paths_ens, probs_ens, vh, sc)
        h_elig = head_sum.copy()
        h_elig[:, 0] = np.where(vh <= 0.20 * sc, h_elig[:, 0], 0.0)
        h_elig[:, 1] = np.where(vh >= 0.60 * sc, h_elig[:, 1], 0.0)
        hyb_pred = np.zeros_like(h_elig)
        hyb_bin = np.zeros_like(ya)
        for s_hold in SITES:
            tr_ids = np.flatnonzero(data["site"] != s_hold)
            te_ids = np.flatnonzero(data["site"] == s_hold)
            oof_h, oof_pa, oof_pr = inner_oof_by_fold[s_hold]
            oof_he = oof_h.copy()
            oof_he[:, 0] = np.where(vh <= 0.20 * sc, oof_he[:, 0], 0.0)
            oof_he[:, 1] = np.where(vh >= 0.60 * sc, oof_he[:, 1], 0.0)
            imp_oof = implied_probs_scaled(hist_np[tr_ids], oof_pa[tr_ids], oof_pr[tr_ids], vh[tr_ids], sc)
            imp_full = np.zeros_like(oof_he)
            imp_full[tr_ids] = imp_oof
            al, th = _calibrate_hybrid_fold(ya, va, oof_he, imp_full, tr_ids)
            hyb_pred[te_ids] = (1.0 - al)[None, :] * h_elig[te_ids] + al[None, :] * imp_te[te_ids]
            hyb_bin[te_ids] = hyb_pred[te_ids] >= th[None, :]
        key = f"scale_{sc:.2f}x"
        sens_out[key] = {
            "scale": sc,
            "neural_head_fixed05": _score_prob_and_bin(ya, va, h_elig, h_elig >= 0.5, data["site"]),
            "hybrid_inner_calibrated": _score_prob_and_bin(ya, va, hyb_pred, hyb_bin, data["site"]),
        }
        if abs(sc - 1.0) < 1e-9:
            for sk in ("first_contiguous_449", "prior_35_1033"):
                sm = masks[sk]
                strata_out[sk]["action_hybrid_1x"] = _score_prob_and_bin(
                    ya[sm], va[sm], hyb_pred[sm], hyb_bin[sm], data["site"][sm]
                )["site_equal"]
    return _pack_outputs(strata_out, sens_out, out_dir, seeds)


def _score_prob_and_bin(ya: np.ndarray, va: np.ndarray, probs: np.ndarray, pred_bin: np.ndarray, site_arr: np.ndarray) -> dict:
    from src.metrics.metrics_and_stats import _binary_average_precision
    pooled = {}
    for c_i, c_name in enumerate(ACTIONS):
        vm = va[:, c_i]
        yt, ps, pr = ya[vm, c_i], probs[vm, c_i], pred_bin[vm, c_i]
        tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
        ap = float(_binary_average_precision(yt.astype(np.float64), ps)) if yt.sum() > 0 else 0.0
        br = float(((ps - yt.astype(np.float64)) ** 2).mean()) if len(yt) else 0.0
        pooled[c_name] = {"pos": int(yt.sum()), "ap": ap, "f1": float((2 * tp) / max(2 * tp + fp + fn, 1)), "tp": tp, "fp": fp, "brier": br}
    s_ap, s_f1, s_br = [], [], []
    for s in SITES:
        sm = site_arr == s
        aps, f1s, brs = [], [], []
        for c_i in range(5):
            vm = sm & va[:, c_i]
            if not vm.any():
                continue
            yt, ps, pr = ya[vm, c_i], probs[vm, c_i], pred_bin[vm, c_i]
            if yt.sum() > 0:
                aps.append(float(_binary_average_precision(yt.astype(np.float64), ps)))
            tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
            f1s.append((2.0 * tp) / max(2 * tp + fp + fn, 1))
            brs.append(float(((ps - yt.astype(np.float64)) ** 2).mean()))
        if aps:
            s_ap.append(float(np.mean(aps)))
        if f1s:
            s_f1.append(float(np.mean(f1s)))
            s_br.append(float(np.mean(brs)))
    return {"site_equal": {"macro_ap": float(np.mean(s_ap)), "macro_f1": float(np.mean(s_f1)), "mean_brier": float(np.mean(s_br))}, "pooled_classes": pooled}


def _pack_outputs(strata_out: dict, sens_out: dict, out_dir: Path, seeds: tuple[int, ...]) -> dict:
    agg = {"study": "Track-emergence strata (449/261/323/1033) & action threshold sensitivity (0.50x-1.50x) for Ours_O2_Full", "seeds": list(seeds), "emergence_strata": strata_out, "action_threshold_sensitivity": sens_out}
    agg_p, md_p, man_p = out_dir / "report.json", out_dir / "RESULTS.md", out_dir / "sha256.json"
    agg_p.write_text(json.dumps(agg, indent=2) + "\n")
    s449 = strata_out["first_contiguous_449"]["models"]["full_bayes"]
    scv = strata_out["first_contiguous_449"]["models"]["cv"]
    spad = strata_out["first_contiguous_449"]["models"]["naive_pad_t20_bayes"]
    md_p.write_text(
        f"# Track-Emergence Strata & Action Threshold Sensitivity (5-Seed Ours_O2_Full)\n\n"
        f"- **449 First-Contiguous Emergence Windows**: `Ours_O2_Full` `{s449['site_equal_ade_m']:.4f} / {s449['site_equal_fde_m']:.4f} m` "
        f"vs `CV` `{scv['site_equal_ade_m']:.4f} / {scv['site_equal_fde_m']:.4f} m` and naive `T=20` pad `{spad['site_equal_ade_m']:.4f} / {spad['site_equal_fde_m']:.4f} m`.\n"
    )
    mani = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (agg_p, md_p)}
    man_p.write_text(json.dumps(mani, indent=2) + "\n")
    z_p = out_dir / "emergence_and_action_sensitivity_5seed.zip"
    with zipfile.ZipFile(z_p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in (agg_p, md_p, man_p):
            zf.write(p, arcname=p.name)
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    run_suite(args.meta_dir, args.pose_npz, args.joint_zip, args.out_dir)
