"""Five-seed horizon (0.5s..3.0s) and causal speed-regime stratification for O(2) BioEq-Act v2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.run_pinns_matched_o2_multiduration import SITES, paired_site_clip_ci
from scripts.run_pinns_o2_joint_actions_5seed import SEEDS, fit_joint_arm, load_joint_cohort
from scripts.run_pinns_o2_joint_v2_5seed import fit_stage2_joint

HORIZONS = (("0.5s", 4), ("1.0s", 9), ("1.5s", 14), ("2.0s", 19), ("2.5s", 24), ("3.0s", 29))
SPEED_BINS = (
    ("quasi_static_lt_0.25", 0.0, 0.25),
    ("slow_transition_0.25_0.80", 0.25, 0.80),
    ("normal_walk_0.80_1.40", 0.80, 1.40),
    ("brisk_ge_1.40", 1.40, 1e9),
)


@torch.inference_mode()
def bayes_path_errors(net, data: dict, ids: np.ndarray, bio_scale: float) -> tuple[np.ndarray, np.ndarray]:
    out = net(data["history"][ids], data["bio_seq"][ids], data["q_bio"][ids] * float(bio_scale))
    q = out["logits"].softmax(dim=1)
    bay = torch.einsum("bk,bktw->btw", q.double(), out["paths"]).cpu().numpy()
    err30 = np.linalg.norm(bay - data["truth"][ids].numpy(), axis=-1)
    return err30, out["regime"].cpu().numpy()


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
    site_arr, clip_arr = data["site"], data["clip"]
    hist, truth = data["history"].numpy(), data["truth"].numpy()
    cv_paths = (hist[:, -1:] - hist[:, -2:-1]) * np.arange(1, 31, dtype=np.float64)[None, :, None]
    cv_err30 = np.linalg.norm(cv_paths - truth, axis=-1)
    v_last = np.linalg.norm(hist[:, -1] - hist[:, -2], axis=-1) / 0.1

    mot_seeds, bio_seeds, reg_seeds = [], [], []
    for seed in args.seeds:
        em = np.zeros((n, 30), dtype=np.float64)
        eb = np.zeros((n, 30), dtype=np.float64)
        rg = np.zeros(n, dtype=np.float64)
        for s in SITES:
            tr_ids = np.flatnonzero(site_arr != s)
            te_ids = np.flatnonzero(site_arr == s)
            mot = fit_joint_arm(seed, "motion", "stop_grad", data, tr_ids)
            bio, sc_t, _ = fit_stage2_joint(mot, seed, data, tr_ids)
            em[te_ids], _ = bayes_path_errors(mot, data, te_ids, 0.0)
            eb[te_ids], rg[te_ids] = bayes_path_errors(bio, data, te_ids, sc_t)
        mot_seeds.append(em); bio_seeds.append(eb); reg_seeds.append(rg)
        print(f"Done horizon & speed stratification seed={seed}", flush=True)

    m30, b30, rg_avg = np.mean(mot_seeds, axis=0), np.mean(bio_seeds, axis=0), np.mean(reg_seeds, axis=0)
    horizons_out = {}
    for label, k in HORIZONS:
        def hstat(e30: np.ndarray) -> dict:
            ade = float(np.mean([e30[site_arr == s, : k + 1].mean() for s in SITES]))
            fde = float(np.mean([e30[site_arr == s, k].mean() for s in SITES]))
            return {"cum_ade_m": ade, "step_err_m": fde}
        horizons_out[label] = {"cv_last2": hstat(cv_err30), "o2_motion_bayes": hstat(m30), "o2_bioeq_v2_bayes": hstat(b30)}

    strata_out = {}
    for name, lo, hi in SPEED_BINS:
        msk = (v_last >= lo) & (v_last < hi)
        cv_w = np.column_stack((cv_err30.mean(1), cv_err30[:, -1]))
        mw = np.column_stack((m30.mean(1), m30[:, -1]))
        bw = np.column_stack((b30.mean(1), b30[:, -1]))
        strata_out[name] = {
            "windows": int(msk.sum()),
            "mean_causal_v_last_mps": float(v_last[msk].mean()),
            "mean_bio_gate_weight": float(rg_avg[msk].mean()),
            "cv_window_mean": {"ade_m": float(cv_w[msk, 0].mean()), "fde_m": float(cv_w[msk, 1].mean())},
            "o2_motion_window_mean": {"ade_m": float(mw[msk, 0].mean()), "fde_m": float(mw[msk, 1].mean())},
            "o2_bioeq_v2_window_mean": {"ade_m": float(bw[msk, 0].mean()), "fde_m": float(bw[msk, 1].mean())},
            "paired_bioeq_minus_motion_mm": {
                "ade_mm": float((bw[msk, 0] - mw[msk, 0]).mean() * 1000.0),
                "fde_mm": float((bw[msk, 1] - mw[msk, 1]).mean() * 1000.0),
            },
            "paired_bioeq_minus_cv_cm": {
                "ade_cm": float((bw[msk, 0] - cv_w[msk, 0]).mean() * 100.0),
                "fde_cm": float((bw[msk, 1] - cv_w[msk, 1]).mean() * 100.0),
            },
        }
    report = {"windows": n, "seeds": list(args.seeds), "horizons": horizons_out, "causal_speed_strata": strata_out}
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    members = {"report.json": (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()}
    members["sha256.json"] = (json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in members.items()}, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in members.items():
            zf.writestr(k, v)
    print("Saved", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
