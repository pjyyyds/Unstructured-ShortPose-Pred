"""Independently replay saved seed-42 checkpoints and verify 5-seed BioEq report."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.run_pinns_bioeq_benchmark import ARMS, FIELDS, prepare_cohort
from src.models.bioeq_neural import BioEqGroundForecaster


def verify_bioeq_archive(archive: Path, data_dir: Path, pose_path: Path) -> dict:
    raw = archive.read_bytes()
    archive_sha = hashlib.sha256(raw).hexdigest()
    tensors, idx, keys, subhalf = prepare_cohort(data_dir, pose_path)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert z.testzip() is None
        report = json.loads(z.read("bioeq_5seed_report.json"))
        assert report["windows"] == 1482 and report["subhalf_windows"] == int(subhalf.sum()) == 323
        replayed = {a: np.zeros((len(keys), len(FIELDS)), dtype=np.float64) for a in ARMS}
        ckpt_shas = {}
        for site, te_ids in idx.items():
            for arm in ARMS:
                member = f"{site}/{arm}.pt"
                pt_bytes = z.read(member)
                ckpt_shas[member] = hashlib.sha256(pt_bytes).hexdigest()
                state = torch.load(io.BytesIO(pt_bytes), map_location="cpu", weights_only=True)
                net = BioEqGroundForecaster(arm).eval()
                net.load_state_dict(state, strict=True)
                with torch.inference_mode():
                    p, lg, _ = net(
                        tensors["history"][te_ids], tensors["pose"][te_ids],
                        tensors["quality"][te_ids], tensors["bio_seq"][te_ids], tensors["q_bio"][te_ids],
                    )
                paths_np, probs_np = p.numpy(), lg.softmax(dim=1).numpy()
                truth_np = tensors["truth"][te_ids].numpy()
                dst = np.linalg.norm(paths_np - truth_np[:, None], axis=3)
                ade_k, fde_k = dst.mean(axis=2), dst[:, :, -1]
                top = probs_np.argmax(axis=1)
                ii = np.arange(len(paths_np))
                bayes = np.einsum("nk,nktw->ntw", probs_np, paths_np)
                bd = np.linalg.norm(bayes - truth_np, axis=2)
                replayed[arm][te_ids] = np.column_stack((
                    ade_k[ii, top], fde_k[ii, top], ade_k.min(1), fde_k.min(1),
                    (top == 0).astype(np.float64), bd.mean(1), bd[:, -1],
                ))
    seed42_macro = {
        a: dict(zip(FIELDS, np.mean([replayed[a][idx[s]].mean(0) for s in sorted(idx)], axis=0).tolist()))
        for a in ARMS
    }
    return {
        "archive_sha256": archive_sha,
        "verified_checkpoints": len(ckpt_shas),
        "checkpoint_sha256": ckpt_shas,
        "seed42_replayed_site_equal": seed42_macro,
        "five_seed_site_equal": report["all_1482"]["site_equal"],
        "five_seed_subhalf_323": report["subhalf_323"]["site_equal"],
        "paired_5seed": report["paired_5seed"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("pose_path", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    out = verify_bioeq_archive(a.archive, a.data_dir, a.pose_path)
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["five_seed_site_equal"], indent=2))
