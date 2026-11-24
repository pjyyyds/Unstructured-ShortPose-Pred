"""Replay six-mode forecasts and score their coverage and probabilities on the verified PINNS folds."""
import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.run_heldout_seed_sensitivity import FIXED_SEEDS, MODELS, pilot_inputs
from scripts.train_pinns_heldout_sites import ROOT, file_hash, save_json
from scripts.verify_heldout_bundle import ATOL_M, RTOL, validate_zip
from src.metrics.metrics_and_stats import compute_sample_trajectory_metrics
from src.models.predictor import UnstructuredShortPosePredictor


COLUMNS = ("ADE_1", "FDE_1", "minADE_6", "minFDE_6", "NLL", "Brier", "top1_is_minade", "top1_probability", "top1_ade_gap_m")


def score_modes(state, name, tensors, ids):
    model = UnstructuredShortPosePredictor(name).cpu().eval()
    model.load_state_dict(state, strict=True)
    inputs = ("x_m", "ego_type_oh") if name == "M0" else ("x_m", "ego_type_oh", "x_p", "q_vec")
    parts = []
    with torch.inference_mode():
        for group in torch.as_tensor(ids).split(64):
            out = model({k: tensors[k][group] for k in inputs})
            trajectories = out["trajs"].numpy().astype(np.float64)
            probs = out["mode_probs"].numpy().astype(np.float64)
            target = tensors["fut_xy"][group].numpy().astype(np.float64)
            if trajectories.shape != (len(group), 6, 30, 2) or probs.shape != (len(group), 6):
                raise ValueError("Unexpected trajectory or probability shape")
            if not np.isfinite(trajectories).all() or not np.isfinite(probs).all() or (probs < 0).any():
                raise ValueError("Invalid positions or mode probabilities")
            np.testing.assert_allclose(probs.sum(axis=1), 1., atol=1e-6, rtol=0)
            mask = tensors["fut_mask"][group].numpy()
            if not np.all(mask == 1):
                raise ValueError("The matched benchmark requires 30 observed future targets")
            distances = np.linalg.norm(trajectories - target[:, None], axis=-1)
            row = np.arange(len(group))
            top, best = probs.argmax(axis=1), distances.mean(axis=2).argmin(axis=1)
            scores = compute_sample_trajectory_metrics(trajectories, probs, target, mask)
            scores.update(top1_is_minade=(top == best), top1_probability=probs[row, top],
                          top1_ade_gap_m=scores["ADE_1"]-scores["minADE_6"])
            parts.append(np.column_stack([scores[k] for k in COLUMNS]))
    return np.concatenate(parts)


def metric_means(rows):
    return dict(zip(COLUMNS, rows.mean(axis=0).tolist()))


def run(bundle, data_root, pilot):
    accepted = json.loads((ROOT / "results/pinns_heldout_five_seed_replay.json").read_text())
    if file_hash(bundle) != accepted["archive_sha256"]:
        raise ValueError("Expected the previously replayed five-seed archive")
    tensors, index, keys, _, _, _ = pilot_inputs(data_root, pilot)
    by_seed_site, macro, effects = {}, {}, defaultdict(list)
    with zipfile.ZipFile(bundle) as z:
        validate_zip(z)
        for seed in FIXED_SEEDS:
            by_seed_site[seed] = {}
            for site, ids in index.items():
                values = {}
                for model in MODELS:
                    checkpoint = torch.load(io.BytesIO(z.read(f"seed_{seed}/{site}/{model}.pt")),
                                            weights_only=True, map_location="cpu")
                    values[model] = score_modes(checkpoint["model_state"], model, tensors, ids)
                    np.testing.assert_allclose(values[model][:, :2], checkpoint["errors"].numpy(), rtol=RTOL, atol=ATOL_M)
                values["M1_minus_M0"] = values["M1"] - values["M0"]
                by_seed_site[seed][site] = {name: metric_means(rows) for name, rows in values.items()}
            macro[seed] = {}
            for name in (*MODELS, "M1_minus_M0"):
                sites = [metrics[name] for metrics in by_seed_site[seed].values()]
                macro[seed][name] = {k: float(np.mean([s[k] for s in sites])) for k in COLUMNS}
                for k in COLUMNS:
                    effects[(name, k)].append(macro[seed][name][k])
    return {"archive_sha256": file_hash(bundle), "pose_sha256": accepted["pose_sha256"],
            "cpu_torch": str(torch.__version__), "unique_windows": len(keys), "seeds": list(FIXED_SEEDS),
            "scope": "Descriptive CPU replay of 40 accepted fits. Average windows within held-out site, then sites and seeds equally. Four previously examined physical sites. The minimum-ADE mode and minimum-FDE mode can differ.",
            "definition": "Each model supplies six candidate paths and probabilities. Top mode is the highest-probability candidate. minADE and minFDE independently take the smallest error among the six after ground truth is available. NLL and summed six-class Brier target the one-hot index of the smallest-ADE candidate within that same model, using the repository probability clipping at 1e-7. Ties use first index. These scores describe a model's ranking of its own modes. Top1 gap = ADE_1 minus minADE_6, in metres.",
            "uniform_probability_reference": {"NLL": float(np.log(6)), "Brier": 5./6},
            "four_site_five_seed_means": {name: {k: float(np.mean(effects[name,k])) for k in COLUMNS}
                                         for name in (*MODELS, "M1_minus_M0")},
            "site_macro_by_seed": macro, "heldout_site_by_seed": by_seed_site}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    save_json(args.output, run(args.bundle, args.data_root, args.pilot_dir))
