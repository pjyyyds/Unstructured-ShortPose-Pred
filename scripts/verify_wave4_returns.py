"""Independent local SHA-256 and CPU checkpoint replay verifier for Wave 4 & GPU timing archives."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch


def check_zip_sha256(bundle: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(bundle) as zf:
        if zf.testzip() is not None:
            raise ValueError(f"Corrupt ZIP: {bundle}")
        members = {n: zf.read(n) for n in zf.namelist()}
    if "sha256.json" in members:
        manifest = json.loads(members["sha256.json"])
        for k, expected in manifest.items():
            actual = hashlib.sha256(members[k]).hexdigest()
            if actual != expected:
                raise ValueError(f"SHA-256 mismatch on {k}: {actual} != {expected}")
    return members


def verify_matched_multiduration(bundle: Path, matched_npz: Path, pose_npz: Path) -> dict:
    from scripts.run_pinns_matched_o2_multiduration import DURATIONS, SITES, eval_o2, load_matched_623
    from src.models.bioeq_o2_joint import O2BioEqJointForecaster

    members = check_zip_sha256(bundle)
    report = json.loads(members["report.json"])
    data = load_matched_623(matched_npz, pose_npz)
    replayed = {}
    for dur in DURATIONS:
        m_arr = np.zeros((len(data["site"]), 6), dtype=np.float64)
        b_arr = np.zeros((len(data["site"]), 6), dtype=np.float64)
        for s in SITES:
            te_ids = np.flatnonzero(data["site"] == s)
            for tag, arr in (("motion", m_arr), ("bio_pose", b_arr)):
                ck = torch.load(io.BytesIO(members[f"T{dur}/{s}/{tag}.pt"]), map_location="cpu", weights_only=True)
                net = O2BioEqJointForecaster(pose_mode=tag, action_mode="none").eval()
                net.load_state_dict(ck["state_dict"], strict=True)
                sc = 0.0 if tag == "motion" else float(ck["bio_scale"])
                arr[te_ids] = eval_o2(net, data, te_ids, dur, sc)
        replayed[str(dur)] = {
            "seed42_motion_ade_m": float(np.mean([m_arr[data["site"] == s, 0].mean() for s in SITES])),
            "seed42_bioeq_ade_m": float(np.mean([b_arr[data["site"] == s, 0].mean() for s in SITES])),
        }
    return {
        "archive_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "checked_checkpoints": 24,
        "seed42_replayed": replayed,
        "five_seed_summary": report["durations"],
        "duration_scaling_T20_minus_T5": report["duration_scaling_T20_minus_T5"],
    }


def verify_joint_actions(bundle: Path, joint_zip: Path, pose_npz: Path) -> dict:
    from scripts.run_pinns_o2_joint_actions_5seed import SITES, eval_joint_arm, load_joint_cohort
    from src.models.bioeq_o2_joint import O2BioEqJointForecaster

    members = check_zip_sha256(bundle)
    report = json.loads(members["report.json"])
    data = load_joint_cohort(joint_zip, pose_npz)
    replayed = {}
    for tag in ("bio_none", "bio_auxiliary", "bio_end_to_end", "bio_stop_grad", "motion_stop_grad"):
        t_arr = np.zeros((len(data["site"]), 6), dtype=np.float64)
        for s in SITES:
            te_ids = np.flatnonzero(data["site"] == s)
            ck = torch.load(io.BytesIO(members[f"{s}/{tag}.pt"]), map_location="cpu", weights_only=True)
            net = O2BioEqJointForecaster(pose_mode=ck["pose_mode"], action_mode=ck["action_mode"]).eval()
            net.load_state_dict(ck["state_dict"], strict=True)
            t_arr[te_ids], _ = eval_joint_arm(net, data, te_ids)
        replayed[tag] = float(np.mean([t_arr[data["site"] == s, 0].mean() for s in SITES]))
    return {
        "archive_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "checked_checkpoints": 20,
        "seed42_replayed_ade_m": replayed,
        "five_seed_arms": report["arms"],
        "paired_comparisons": report["paired_comparisons"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=("manifest_only", "multiduration", "joint_actions"))
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--aux", type=Path, default=None)
    ap.add_argument("--pose", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.kind == "manifest_only":
        m = check_zip_sha256(args.bundle)
        res = {"archive_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(), "files": sorted(m.keys())}
    elif args.kind == "multiduration":
        res = verify_matched_multiduration(args.bundle, args.aux, args.pose)
    else:
        res = verify_joint_actions(args.bundle, args.aux, args.pose)
    args.output.write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
    print("VERIFIED:", args.kind, res["archive_sha256"], flush=True)
