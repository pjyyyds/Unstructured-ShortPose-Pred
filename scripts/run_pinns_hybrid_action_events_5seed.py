"""Five-seed implied-kinematic + parity-head hybrid action scoring & inner-LOSO calibration (1,482 windows)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.run_pinns_matched_o2_multiduration import SITES
from scripts.run_pinns_o2_joint_actions_5seed import ACTIONS, SEEDS, fit_joint_arm, load_joint_cohort
from scripts.run_pinns_o2_joint_v2_5seed import fit_stage2_joint
from src.data.actions import compute_action_labels
from src.metrics.metrics_and_stats import _binary_average_precision


def implied_event_probs(hist_np: np.ndarray, paths_np: np.ndarray, probs_np: np.ndarray) -> np.ndarray:
    n, k_modes, _, _ = paths_np.shape
    hm = np.ones(5, dtype=np.float32)
    fm = np.ones(30, dtype=np.float32)
    out = np.zeros((n, 5), dtype=np.float64)
    for i in range(n):
        hp = hist_np[i]
        for k in range(k_modes):
            yk, _ = compute_action_labels(hp, hm, paths_np[i, k], fm, dt=0.1)
            out[i] += probs_np[i, k] * yk
    return np.clip(out, 0.0, 1.0)


@torch.inference_mode()
def predict_head_and_implied(net, data: dict, ids: np.ndarray, sc_t: float, sc_a: float) -> tuple[np.ndarray, np.ndarray]:
    ot = net(data["history"][ids], data["bio_seq"][ids], data["q_bio"][ids] * float(sc_t))
    oa = net(data["history"][ids], data["bio_seq"][ids], data["q_bio"][ids] * float(sc_a))
    p_head = torch.sigmoid(oa["act_logits"]).cpu().numpy().astype(np.float64)
    # Apply strict causal speed eligibility gate for Start (<=0.20 m/s) and Stop (>=0.60 m/s)
    vh = data["v_hist_mean"][ids]
    p_elig = p_head.copy()
    p_elig[:, 0] = np.where(vh <= 0.20, p_elig[:, 0], 0.0)
    p_elig[:, 1] = np.where(vh >= 0.60, p_elig[:, 1], 0.0)
    p_imp = implied_event_probs(data["history"][ids].numpy(), ot["paths"].cpu().numpy(), ot["logits"].softmax(1).cpu().numpy())
    p_imp[:, 0] = np.where(vh <= 0.20, p_imp[:, 0], 0.0)
    p_imp[:, 1] = np.where(vh >= 0.60, p_imp[:, 1], 0.0)
    return p_elig, p_imp


def _best_f1_thresh(yt: np.ndarray, sc: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.08, 0.88, 41):
        pr = sc >= t
        tp = int((pr & yt).sum())
        fp = int((pr & (~yt)).sum())
        fn = int(((~pr) & yt).sum())
        f1 = (2.0 * tp) / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def score_action_matrix(y_true: np.ndarray, v_mask: np.ndarray, probs: np.ndarray, site_arr: np.ndarray, thresh: np.ndarray) -> dict:
    pooled_cls = {}
    for c_i, c_name in enumerate(ACTIONS):
        vm = v_mask[:, c_i]
        yt = y_true[vm, c_i]
        ps = probs[vm, c_i]
        pr = ps >= thresh[c_i]
        tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
        f1 = (2.0 * tp) / max(2 * tp + fp + fn, 1)
        ap = float(_binary_average_precision(yt.astype(np.float64), ps))
        br = float(((ps - yt.astype(np.float64)) ** 2).mean())
        pooled_cls[c_name] = {"pos": int(yt.sum()), "ap": ap, "f1": float(f1), "tp": tp, "fp": fp, "brier": br}

    site_aps, site_f1s, site_brs = [], [], []
    for s in SITES:
        sm = site_arr == s
        aps, f1s, brs = [], [], []
        for c_i in range(5):
            vm = sm & v_mask[:, c_i]
            yt = y_true[vm, c_i]
            ps = probs[vm, c_i]
            if yt.sum() > 0:
                aps.append(float(_binary_average_precision(yt.astype(np.float64), ps)))
            pr = ps >= thresh[c_i]
            tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
            f1s.append((2.0 * tp) / max(2 * tp + fp + fn, 1))
            brs.append(float(((ps - yt.astype(np.float64)) ** 2).mean()))
        site_aps.append(float(np.mean(aps)))
        site_f1s.append(float(np.mean(f1s)))
        site_brs.append(float(np.mean(brs)))
    return {
        "site_equal": {"macro_ap": float(np.mean(site_aps)), "macro_f1": float(np.mean(site_f1s)), "mean_brier": float(np.mean(site_brs))},
        "pooled_classes": pooled_cls,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("joint_zip", type=Path)
    ap.add_argument("pose_npz", type=Path)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    torch.set_num_threads(4)
    data = load_joint_cohort(args.joint_zip, args.pose_npz)
    n = len(data["site"])
    site_arr = data["site"]
    yt_all = data["y_act"].numpy().astype(bool)
    vm_all = data["valid_act"].numpy().astype(bool)

    head_seeds, imp_seeds, hyb_seeds, cal_pred_seeds = [], [], [], []
    for seed in args.seeds:
        p_h = np.zeros((n, 5), dtype=np.float64)
        p_i = np.zeros((n, 5), dtype=np.float64)
        p_b = np.zeros((n, 5), dtype=np.float64)
        cal_pr = np.zeros((n, 5), dtype=bool)
        for s in SITES:
            tr_ids = np.flatnonzero(site_arr != s)
            te_ids = np.flatnonzero(site_arr == s)
            # Inner out-of-site calibration across the 3 source sites
            src_sites = sorted(set(site_arr[tr_ids].tolist()))
            oof_h = np.zeros((len(tr_ids), 5), dtype=np.float64)
            oof_i = np.zeros((len(tr_ids), 5), dtype=np.float64)
            for vs in src_sites:
                m_val = site_arr[tr_ids] == vs
                fi = tr_ids[~m_val]
                vi = tr_ids[m_val]
                mot_in = fit_joint_arm(seed, "motion", "stop_grad", data, fi)
                bio_in, sct_in, sca_in = fit_stage2_joint(mot_in, seed, data, fi)
                oof_h[m_val], oof_i[m_val] = predict_head_and_implied(bio_in, data, vi, sct_in, sca_in)

            lam = np.zeros(5, dtype=np.float64)
            tau = np.full(5, 0.5, dtype=np.float64)
            for c_i in range(5):
                vm = vm_all[tr_ids, c_i]
                yt = yt_all[tr_ids, c_i][vm]
                grid_l = (0.0, 0.25, 0.5, 0.75, 1.0)
                aps = [float(_binary_average_precision(yt.astype(np.float64), (1.0 - gl) * oof_h[vm, c_i] + gl * oof_i[vm, c_i])) for gl in grid_l]
                lam[c_i] = grid_l[int(np.nanargmax(aps))]
                blended_oof = (1.0 - lam[c_i]) * oof_h[vm, c_i] + lam[c_i] * oof_i[vm, c_i]
                tau[c_i] = _best_f1_thresh(yt, blended_oof)

            mot = fit_joint_arm(seed, "motion", "stop_grad", data, tr_ids)
            bio, sct, sca = fit_stage2_joint(mot, seed, data, tr_ids)
            th_te, ti_te = predict_head_and_implied(bio, data, te_ids, sct, sca)
            tb_te = (1.0 - lam[None, :]) * th_te + lam[None, :] * ti_te
            p_h[te_ids], p_i[te_ids], p_b[te_ids] = th_te, ti_te, tb_te
            cal_pr[te_ids] = tb_te >= tau[None, :]
        head_seeds.append(p_h); imp_seeds.append(p_i); hyb_seeds.append(p_b); cal_pred_seeds.append(cal_pr.astype(np.float64))
        print(f"Done hybrid action scoring seed={seed}", flush=True)

    ph_avg, pi_avg, pb_avg = np.mean(head_seeds, axis=0), np.mean(imp_seeds, axis=0), np.mean(hyb_seeds, axis=0)
    cp_maj = np.mean(cal_pred_seeds, axis=0) >= 0.5
    t05 = np.full(5, 0.5)
    # Also evaluate majority-vote inner-LOSO calibrated binary predictions
    cal_cls = {}
    for c_i, c_name in enumerate(ACTIONS):
        vm = vm_all[:, c_i]; yt = yt_all[vm, c_i]; pr = cp_maj[vm, c_i]
        tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
        cal_cls[c_name] = {"pos": int(yt.sum()), "f1": float((2.0 * tp) / max(2 * tp + fp + fn, 1)), "tp": tp, "fp": fp}
    s_f1s = []
    for s in SITES:
        sm = site_arr == s; f1s = []
        for c_i in range(5):
            vm = sm & vm_all[:, c_i]; yt = yt_all[vm, c_i]; pr = cp_maj[vm, c_i]
            tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
            f1s.append((2.0 * tp) / max(2 * tp + fp + fn, 1))
        s_f1s.append(float(np.mean(f1s)))

    report = {
        "windows": n,
        "seeds": list(args.seeds),
        "parity_head_eligible_t05": score_action_matrix(yt_all, vm_all, ph_avg, site_arr, t05),
        "implied_kinematic_only_t05": score_action_matrix(yt_all, vm_all, pi_avg, site_arr, t05),
        "hybrid_head_plus_kinematic_t05": score_action_matrix(yt_all, vm_all, pb_avg, site_arr, t05),
        "hybrid_inner_loso_calibrated_f1": {"site_equal_macro_f1": float(np.mean(s_f1s)), "pooled_classes": cal_cls},
    }
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    members = {"report.json": (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()}
    members["sha256.json"] = (json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in members.items()}, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in members.items():
            zf.writestr(k, v)
    print("Saved", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
