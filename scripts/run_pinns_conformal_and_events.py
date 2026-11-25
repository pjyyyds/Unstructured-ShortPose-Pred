"""Leave-one-site-out split-conformal safety tubes (90%/95%) and multimodal calibration for O(2) BioEq-Act v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.run_pinns_matched_o2_multiduration import SITES
from scripts.run_pinns_o2_joint_actions_5seed import SEEDS, fit_joint_arm, load_joint_cohort
from scripts.run_pinns_o2_joint_v2_5seed import fit_stage2_joint


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    rank = min(max(math.ceil((n + 1) * (1.0 - alpha)) - 1, 0), n - 1)
    return float(np.sort(scores)[rank])


@torch.inference_mode()
def predict_paths_and_probs(net, data: dict, ids: np.ndarray, bio_scale: float) -> tuple[np.ndarray, np.ndarray]:
    out = net(data["history"][ids], data["bio_seq"][ids], data["q_bio"][ids] * float(bio_scale))
    return out["paths"].cpu().numpy(), out["logits"].softmax(dim=1).cpu().numpy()


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
    truth = data["truth"].numpy()
    hist = data["history"].numpy()
    steps = np.arange(1, 31, dtype=np.float64)[None, :, None]
    cv_paths = (hist[:, -1:] - hist[:, -2:-1]) * steps

    # Conformal score: max_t ||y_t - pred_t|| (3.0s tube radius) and end-point FDE radius; multimodal union = min_k max_t ||y_t - pred_{k,t}||
    cv_tube_err = np.linalg.norm(cv_paths - truth, axis=-1).max(axis=1)
    cv_fde_err = np.linalg.norm(cv_paths[:, -1] - truth[:, -1], axis=-1)

    cov_acc = {lvl: {"cv_single": [], "o2_top1_tube": [], "o2_union6_tube": [], "o2_union6_fde": []} for lvl in ("90", "95")}
    rad_acc = {lvl: {"cv_single": [], "o2_top1_tube": [], "o2_union6_tube": [], "o2_union6_fde": []} for lvl in ("90", "95")}
    calib_acc = {"nll": [], "brier": [], "oracle_hit_rate": []}

    for seed in args.seeds:
        seed_nll, seed_br, seed_hit = np.zeros(n), np.zeros(n), np.zeros(n)
        for s in SITES:
            tr_all = np.flatnonzero(site_arr != s)
            te_ids = np.flatnonzero(site_arr == s)
            # Split source sites: hold out one source site for split-conformal calibration (exchangeable site/window calibration)
            src_sites = sorted(set(site_arr[tr_all].tolist()))
            cal_site = src_sites[0]
            fit_ids = tr_all[site_arr[tr_all] != cal_site]
            cal_ids = tr_all[site_arr[tr_all] == cal_site]

            mot_cal = fit_joint_arm(seed, "motion", "stop_grad", data, fit_ids)
            bio_cal, sc_t_cal, _ = fit_stage2_joint(mot_cal, seed, data, fit_ids)
            p_cal, q_cal = predict_paths_and_probs(bio_cal, data, cal_ids, sc_t_cal)

            mot_full = fit_joint_arm(seed, "motion", "stop_grad", data, tr_all)
            bio_full, sc_t_full, _ = fit_stage2_joint(mot_full, seed, data, tr_all)
            p_te, q_te = predict_paths_and_probs(bio_full, data, te_ids, sc_t_full)

            # Mode calibration metrics on held-out site
            dst_te = np.linalg.norm(p_te - truth[te_ids, None], axis=-1)
            ade_te = dst_te.mean(axis=2)
            oracle = ade_te.argmin(axis=1)
            top = q_te.argmax(axis=1)
            ii = np.arange(len(te_ids))
            seed_hit[te_ids] = (top == oracle).astype(np.float64)
            seed_nll[te_ids] = -np.log(np.clip(q_te[ii, oracle], 1e-12, 1.0))
            oh = np.eye(6)[oracle]
            seed_br[te_ids] = ((q_te - oh) ** 2).sum(axis=1)

            # Split-conformal calibration scores on cal_ids -> evaluate coverage & radius on te_ids
            dst_cal = np.linalg.norm(p_cal - truth[cal_ids, None], axis=-1)
            cal_top_tube = dst_cal[np.arange(len(cal_ids)), q_cal.argmax(1)].max(axis=1)
            cal_u6_tube = dst_cal.max(axis=2).min(axis=1)
            cal_u6_fde = dst_cal[:, :, -1].min(axis=1)

            te_top_tube = dst_te[ii, top].max(axis=1)
            te_u6_tube = dst_te.max(axis=2).min(axis=1)
            te_u6_fde = dst_te[:, :, -1].min(axis=1)

            for lvl, alpha in (("90", 0.10), ("95", 0.05)):
                r_cv = conformal_quantile(cv_tube_err[cal_ids], alpha)
                r_top = conformal_quantile(cal_top_tube, alpha)
                r_u6 = conformal_quantile(cal_u6_tube, alpha)
                r_f6 = conformal_quantile(cal_u6_fde, alpha)
                rad_acc[lvl]["cv_single"].append(r_cv)
                rad_acc[lvl]["o2_top1_tube"].append(r_top)
                rad_acc[lvl]["o2_union6_tube"].append(r_u6)
                rad_acc[lvl]["o2_union6_fde"].append(r_f6)
                cov_acc[lvl]["cv_single"].append(float((cv_tube_err[te_ids] <= r_cv).mean()))
                cov_acc[lvl]["o2_top1_tube"].append(float((te_top_tube <= r_top).mean()))
                cov_acc[lvl]["o2_union6_tube"].append(float((te_u6_tube <= r_u6).mean()))
                cov_acc[lvl]["o2_union6_fde"].append(float((te_u6_fde <= r_f6).mean()))

        calib_acc["nll"].append(float(np.mean([seed_nll[site_arr == s].mean() for s in SITES])))
        calib_acc["brier"].append(float(np.mean([seed_br[site_arr == s].mean() for s in SITES])))
        calib_acc["oracle_hit_rate"].append(float(np.mean([seed_hit[site_arr == s].mean() for s in SITES])))
        print(f"Done conformal & calibration seed={seed}", flush=True)

    report = {
        "windows": n,
        "seeds": list(args.seeds),
        "mode_calibration_site_equal_5seed": {k: float(np.mean(v)) for k, v in calib_acc.items()},
        "archived_global_M1_reference": {"nll": 1.5175, "brier": 0.7346, "oracle_hit_rate": 0.3539},
        "split_conformal_loso": {
            lvl: {
                k: {
                    "mean_radius_m": float(np.mean(rad_acc[lvl][k])),
                    "empirical_site_equal_coverage": float(np.mean(cov_acc[lvl][k])),
                }
                for k in rad_acc[lvl]
            }
            for lvl in ("90", "95")
        },
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
