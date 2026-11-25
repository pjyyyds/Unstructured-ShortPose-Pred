"""Zero-shot clock-aware O(2) site-balanced & ridge transfer on 3,066 DUT campus windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.benchmark_heldout_sites import load_site_windows
from scripts.run_dut_equivariant_external import ANGLES, forward_dut
from scripts.run_heldout_seed_sensitivity import split_indices
from scripts.run_pinns_bioeq_benchmark import prepare_cohort
from scripts.run_pinns_bioeq_v2_benchmark import fit_backbone
from src.models.biomech_equivariant import _pred_even, _pred_odd, causal_frenet_frame, fit_biomech_equivariant


def parity_features_clock_aware(history: np.ndarray, ticks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    origin, basis, valid = causal_frenet_frame(history)
    rel = np.einsum("ntw,nws->nts", history - origin, basis)
    dt = np.maximum((ticks[:, 1:5] - ticks[:, :4]) / 23976.0, 1e-6)
    vel = np.diff(rel, axis=1) / dt[:, :, None]
    dt2 = 0.5 * (dt[:, 1:] + dt[:, :-1])
    acc = np.diff(vel, axis=1) / dt2[:, :, None]
    last_speed = np.linalg.norm(history[:, -1] - history[:, -2], axis=-1) / dt[:, -1]
    path_len = np.linalg.norm(np.diff(history, axis=1), axis=-1).sum(axis=1)
    even = np.column_stack((rel[:, :-1, 0], np.abs(rel[:, :-1, 1]), vel[:, :, 0], np.abs(vel[:, :, 1]), acc[:, :, 0], last_speed, path_len))
    odd = np.column_stack((rel[:, :-1, 1], vel[:, :, 1], acc[:, :, 1]))
    return np.where(valid[:, None], even, 0.0), np.where(valid[:, None], odd, 0.0)


def predict_o2_ridge_clock_aware(model: dict, history: np.ndarray, ticks: np.ndarray, cv_meas: np.ndarray) -> np.ndarray:
    _, basis, valid = causal_frenet_frame(history)
    even, odd = parity_features_clock_aware(history, ticks)
    par = np.where(valid[:, None], model["g_mot"] * _pred_even(model["fe"], even), 0.0)
    perp = np.where(valid[:, None], model["g_mot"] * _pred_odd(model["fo"], odd), 0.0)
    tau = np.clip((ticks[:, 5:] - ticks[:, 4:5]) / 23976.0, 0.0, 3.0)
    idx_f = np.clip(tau / 0.1 - 1.0, 0.0, 29.0)
    lo = np.floor(idx_f).astype(np.int64)
    hi = np.minimum(lo + 1, 29)
    w = idx_f - lo
    rows = np.arange(len(history))[:, None]
    par_i = par[rows, lo] * (1.0 - w) + par[rows, hi] * w
    perp_i = perp[rows, lo] * (1.0 - w) + perp[rows, hi] * w
    loc = np.stack((par_i, perp_i), axis=-1)
    return cv_meas + np.einsum("nts,nws->ntw", loc, basis)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinns-data", type=Path, required=True)
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--dut-code", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import sys as _sys
    _sys.path.insert(0, str(args.dut_code))
    import motion_cloud as mc

    keys_dut, _, ticks_np, xy_dut = mc.build(args.dut_code / "dut_raw")
    tensors, idx, keys_p, _ = prepare_cohort(args.pinns_data, args.pose)
    site_arr = np.empty(len(keys_p), dtype="<U24")
    for s, r in idx.items():
        site_arr[r] = s
    hist_p, truth_p = tensors["history"].numpy(), tensors["truth"].numpy()

    ridge_folds, neural_v2_folds = {}, {}
    for s in sorted(idx):
        _, tr_l, _ = split_indices(idx, keys_p, s)
        tr = np.array(tr_l)
        ridge_folds[s] = fit_biomech_equivariant(hist_p[tr], truth_p[tr], site_arr[tr])
        neural_v2_folds[s] = fit_backbone(42, tensors, tr, site_arr, torch.device("cpu"), arm="motion")

    hist_base = xy_dut[:, :5] - xy_dut[:, 4:5]
    truth = xy_dut[:, 5:] - xy_dut[:, 4:5]
    diff = hist_base[:, -1] - hist_base[:, -2]
    elapsed = ticks_np[:, 5:] - ticks_np[:, 4, None]
    cv_meas = diff[:, None] * elapsed[:, :, None] / (ticks_np[:, 4] - ticks_np[:, 3])[:, None, None]
    cv_err = np.column_stack((np.linalg.norm(cv_meas - truth, axis=-1).mean(1), np.linalg.norm(cv_meas - truth, axis=-1)[:, -1]))
    ticks_t = torch.from_numpy(ticks_np)
    all_mask = np.ones(len(keys_dut), dtype=bool)

    report = {"windows": len(keys_dut), "ridge_g_mot_by_fold": {s: ridge_folds[s]["g_mot"] for s in sorted(ridge_folds)}, "angles": {}}
    for deg, (c, s_ang) in zip(ANGLES, ((1, 0), (0, 1), (-1, 0), (0, -1))):
        rot = np.array([[c, -s_ang], [s_ang, c]], dtype=np.float64)
        h_rot = hist_base @ rot.T
        cv_rot = cv_meas @ rot.T
        r_errs, n_top_errs, n_bay_errs = [], [], []
        for site in sorted(idx):
            pr_r = predict_o2_ridge_clock_aware(ridge_folds[site], h_rot, ticks_np, cv_rot) @ rot
            r_errs.append(np.column_stack((np.linalg.norm(pr_r - truth, axis=-1).mean(1), np.linalg.norm(pr_r - truth, axis=-1)[:, -1])))
            p_n, q_n = forward_dut(neural_v2_folds[site], torch.from_numpy(h_rot), ticks_t, clock_aware=True)
            pw = p_n @ rot
            tp = pw[np.arange(len(keys_dut)), q_n.argmax(1)]
            bp = np.einsum("nk,nktw->ntw", q_n, pw)
            n_top_errs.append(np.column_stack((np.linalg.norm(tp - truth, axis=-1).mean(1), np.linalg.norm(tp - truth, axis=-1)[:, -1])))
            n_bay_errs.append(np.column_stack((np.linalg.norm(bp - truth, axis=-1).mean(1), np.linalg.norm(bp - truth, axis=-1)[:, -1])))
        report["angles"][str(deg)] = {
            "cv_measured": mc.grouped(cv_err, all_mask, keys_dut),
            "o2_ridge_clock_aware": mc.grouped(np.mean(r_errs, axis=0), all_mask, keys_dut),
            "bioeq_v2_site_balanced_top1": mc.grouped(np.mean(n_top_errs, axis=0), all_mask, keys_dut),
            "bioeq_v2_site_balanced_bayes": mc.grouped(np.mean(n_bay_errs, axis=0), all_mask, keys_dut),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("dut_o2_site_balanced_transfer.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["angles"]["0"], indent=2))


if __name__ == "__main__":
    main()
