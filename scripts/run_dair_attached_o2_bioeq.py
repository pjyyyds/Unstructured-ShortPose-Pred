"""Zero-shot 5-seed O(2) BioEq evaluation on attached DAIR-V2X 54-window Beijing roadside archive."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.run_pinns_bioeq_benchmark import prepare_cohort
from scripts.run_pinns_matched_o2_multiduration import SITES, fit_motion_and_bio
from scripts.run_dair_o2_bioeq_external import COCO_SWAP, SEEDS, person_bootstrap_ci, reduce_dair
from src.models.bioeq_neural import biomech_sequence_torch
from src.models.bioeq_o2_joint import O2BioEqJointForecaster

DAIR_ARCHIVE_ZIP_NAME = "dair_roadside_third_cohort.zip"


def load_attached_dair_54(dair_zip: Path) -> tuple[dict, dict]:
    with zipfile.ZipFile(dair_zip) as zf:
        mani = json.loads(zf.read("sha256.json"))
        for k, h in mani.items():
            if hashlib.sha256(zf.read(k)).hexdigest() != h:
                raise ValueError(f"DAIR archive member hash mismatch: {k}")
        agg_prev = json.loads(zf.read("aggregate.json"))
        with np.load(io.BytesIO(zf.read("private.npz")), allow_pickle=False) as ar:
            npz = {k: ar[k] for k in ar.files}
    if len(npz["actors"]) != 54:
        raise ValueError("Expected 54 DAIR roadside windows")
    delta_last = npz["nominal_cv"][:, 0, :].astype(np.float64)
    hist2 = np.stack((-delta_last, np.zeros_like(delta_last)), axis=1)
    h_px = np.clip(npz["quality"][:, 2:3].astype(np.float32) * 1080.0, 2.0, 1080.0)
    bbox = np.zeros((54, 5, 4), dtype=np.float32)
    bbox[:, :, 3] = h_px
    # Archived M0/M1 4-fold mean errors across rotations (54, 4_angles, 4_folds, 2_models, 2_ade_fde)
    m0_deg = npz["top1_errors_m"][:, :, :, 0, :].mean(axis=2)
    m1_deg = npz["top1_errors_m"][:, :, :, 1, :].mean(axis=2)
    keys = json.loads(str(npz["keys_json"]))
    seq_to_inter = {seq: info["intersection"] for seq, info in agg_prev["coverage"]["seqs"].items()}
    sites = np.array([seq_to_inter[str(k[0])] for k in keys], dtype=str)
    return {
        "hist2": hist2,
        "truth": npz["gt_relative"].astype(np.float64),
        "kpts": npz["rtmpose_kpts"].astype(np.float32),
        "bbox": bbox,
        "ts_us": npz["timestamps_us"].astype(np.int64),
        "actors": npz["actors"].astype(str),
        "sites": sites,
        "m0_by_rot": m0_deg,
        "m1_by_rot": m1_deg,
    }, agg_prev


@torch.inference_mode()
def score_dair_transform(
    dair: dict,
    ensemble: list[tuple[O2BioEqJointForecaster, O2BioEqJointForecaster, float]],
    op: np.ndarray,
    reflect: bool,
    rot_idx: int | None,
) -> dict[str, np.ndarray]:
    hist_g = dair["hist2"] @ op.T
    truth = dair["truth"]
    kp_g = dair["kpts"].copy()
    if reflect:
        kp_g[..., 0] = 1920.0 - kp_g[..., 0]
        kp_g = kp_g[:, :, COCO_SWAP, :]
    bs_t, qb_t = biomech_sequence_torch(torch.from_numpy(kp_g), torch.from_numpy(dair["bbox"]))
    hist_t = torch.from_numpy(hist_g)

    ts = dair["ts_us"]
    obs_dt = ((ts[:, 4:5] - ts[:, 3:4]) / 1e6).astype(np.float64)
    fut_s = ((ts[:, 5:] - ts[:, 4:5]) / 1e6).astype(np.float64)
    obs_dt_t = torch.from_numpy(obs_dt)
    fut_s_t = torch.from_numpy(fut_s)

    v_last_meas = (hist_g[:, -1:] - hist_g[:, -2:-1]) / obs_dt[:, :, None]
    cv_meas = (v_last_meas * fut_s[:, :, None]) @ op
    cv_nom = ((hist_g[:, -1:] - hist_g[:, -2:-1]) * np.arange(1, 31, dtype=np.float64)[None, :, None]) @ op

    acc = {k: [] for k in ("mot_nom_top1", "bio_nom_top1", "mot_clk_top1", "bio_clk_top1", "mot_clk_bayes", "bio_clk_bayes")}
    for mot, bio, sc in ensemble:
        for prefix, dt_arg, ft_arg in (("nom", None, None), ("clk", obs_dt_t, fut_s_t)):
            for mtag, net, q_mult in (("mot", mot, 0.0), ("bio", bio, sc)):
                out = net(hist_t, bs_t, qb_t * q_mult, obs_dt=dt_arg, fut_times_s=ft_arg)
                paths_w = out["paths"].numpy() @ op
                pr = out["logits"].softmax(dim=1).numpy()
                top = paths_w[np.arange(len(hist_g)), pr.argmax(1)]
                d_top = np.linalg.norm(top - truth, axis=-1)
                acc[f"{mtag}_{prefix}_top1"].append(np.stack((d_top.mean(1), d_top[:, -1]), axis=-1))
                if prefix == "clk":
                    bay = np.einsum("nk,nktw->ntw", pr, paths_w)
                    d_bay = np.linalg.norm(bay - truth, axis=-1)
                    acc[f"{mtag}_clk_bayes"].append(np.stack((d_bay.mean(1), d_bay[:, -1]), axis=-1))

    def err2(pred: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(pred - truth, axis=-1)
        return np.stack((d.mean(1), d[:, -1]), axis=-1)

    res = {
        "cv_measured": err2(cv_meas),
        "cv_nominal": err2(cv_nom),
        **{k: np.mean(v, axis=0) for k, v in acc.items()},
    }
    if rot_idx is not None:
        res["archived_M0"] = dair["m0_by_rot"][:, rot_idx, :]
        res["archived_M1"] = dair["m1_by_rot"][:, rot_idx, :]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pinns_metadata", type=Path)
    ap.add_argument("pose_npz", type=Path)
    ap.add_argument("dair_zip", type=Path)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    dair, _ = load_attached_dair_54(args.dair_zip)
    if args.check_only:
        print("Checked DAIR 54 windows:", len(dair["actors"]), sorted(set(dair["sites"].tolist())), flush=True)
        return

    torch.set_num_threads(4)
    tensors, idx_by_site, keys, _ = prepare_cohort(args.pinns_metadata, args.pose_npz)
    site_arr = np.empty(len(keys), dtype="<U24")
    for s, ids in idx_by_site.items():
        site_arr[ids] = s
    pinns_data = {
        "site": site_arr,
        "hist20": tensors["history"][:, -2:],
        "truth30": tensors["truth"],
        "bio_seq": tensors["bio_seq"],
        "q_bio": tensors["q_bio"],
    }
    ensemble = []
    for seed in args.seeds:
        for s in SITES:
            tr_ids = np.flatnonzero(site_arr != s)
            ensemble.append(fit_motion_and_bio(seed, 2, pinns_data, tr_ids))
        print(f"Fitted 2-point motion + 5-frame bio O(2) BioEq for seed={seed}", flush=True)

    all_m = np.ones(len(dair["actors"]), dtype=bool)
    strict_m = np.max(np.abs(np.diff(dair["ts_us"][:, :5], axis=1) - 100000), axis=1) <= 1000
    transforms = [
        ("0", np.array([[1.0, 0.0], [0.0, 1.0]]), False, 0),
        ("90", np.array([[0.0, -1.0], [1.0, 0.0]]), False, 1),
        ("180", np.array([[-1.0, 0.0], [0.0, -1.0]]), False, 2),
        ("270", np.array([[0.0, 1.0], [-1.0, 0.0]]), False, 3),
        ("reflect_y", np.array([[1.0, 0.0], [0.0, -1.0]]), True, None),
    ]
    by_tf, zero_errs = {}, None
    for name, op, ref, ridx in transforms:
        errs = score_dair_transform(dair, ensemble, op, ref, ridx)
        if name == "0":
            zero_errs = errs
        by_tf[name] = {
            "all_54": {k: reduce_dair(v, all_m, dair["actors"]) for k, v in errs.items()},
            "strict_history_1ms_31": {k: reduce_dair(v, strict_m, dair["actors"]) for k, v in errs.items()},
        }
    o2_swing = float(max(
        abs(by_tf[a]["all_54"]["bio_clk_top1"]["window_equal"]["ade_m"] - by_tf["0"]["all_54"]["bio_clk_top1"]["window_equal"]["ade_m"])
        for a, _, _, _ in transforms
    ))
    m1_swing = float(max(
        abs(by_tf[a]["all_54"]["archived_M1"]["window_equal"]["ade_m"] - by_tf["0"]["all_54"]["archived_M1"]["window_equal"]["ade_m"])
        for a in ("0", "90", "180", "270")
    ))
    report = {
        "windows": int(len(dair["actors"])),
        "strict_history_1ms_windows": int(strict_m.sum()),
        "seeds": list(args.seeds),
        "max_o2_transform_ade_swing_m": o2_swing,
        "max_archived_M1_rotation_ade_swing_m": m1_swing,
        "transforms": by_tf,
        "by_intersection_deg0": {loc: {k: reduce_dair(v, dair["sites"] == loc, dair["actors"]) for k, v in zero_errs.items()} for loc in sorted(set(dair["sites"].tolist()))},
        "paired_bootstrap_all54": {
            "bio_clk_minus_mot_clk_top1": person_bootstrap_ci(zero_errs["bio_clk_top1"] - zero_errs["mot_clk_top1"], all_m, dair["actors"]),
            "bio_clk_minus_archived_M1": person_bootstrap_ci(zero_errs["bio_clk_top1"] - zero_errs["archived_M1"], all_m, dair["actors"]),
            "bio_clk_bayes_minus_archived_M1": person_bootstrap_ci(zero_errs["bio_clk_bayes"] - zero_errs["archived_M1"], all_m, dair["actors"]),
            "bio_clk_minus_cv_measured_top1": person_bootstrap_ci(zero_errs["bio_clk_top1"] - zero_errs["cv_measured"], all_m, dair["actors"]),
            "bio_clk_bayes_minus_cv_measured": person_bootstrap_ci(zero_errs["bio_clk_bayes"] - zero_errs["cv_measured"], all_m, dair["actors"]),
            "bio_clk_minus_bio_nom_top1": person_bootstrap_ci(zero_errs["bio_clk_top1"] - zero_errs["bio_nom_top1"], all_m, dair["actors"]),
        },
        "paired_bootstrap_strict31": {
            "bio_clk_minus_mot_clk_top1": person_bootstrap_ci(zero_errs["bio_clk_top1"] - zero_errs["mot_clk_top1"], strict_m, dair["actors"]),
            "bio_clk_minus_archived_M1": person_bootstrap_ci(zero_errs["bio_clk_top1"] - zero_errs["archived_M1"], strict_m, dair["actors"]),
            "bio_clk_minus_cv_measured_top1": person_bootstrap_ci(zero_errs["bio_clk_top1"] - zero_errs["cv_measured"], strict_m, dair["actors"]),
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
