"""Score a fixed last-two-observation constant-velocity selector on 40 frozen PINNS models."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.analyze_heldout_pairs import BOOTSTRAP_SEED, REPLICATES, check_keys, intervals, paired_clip_draws
from scripts.run_heldout_seed_sensitivity import FIXED_SEEDS, MODELS, SITE_COUNTS, pilot_inputs
from scripts.score_pinns_six_modes import COLUMNS, score_modes
from scripts.train_pinns_heldout_sites import ROOT, Protocol, atomic_write, file_hash, save_json
from scripts.verify_heldout_bundle import ATOL_M, RTOL, validate_zip
from src.data.constants import DT, T_FUT
from src.models.predictor import UnstructuredShortPosePredictor


METHODS = ("network_top1", "velocity_selected", "last_two_velocity", "oracle_minade_mode")
METRICS = ("ade_m", "fde_m")


def last_two_future(history):
    """Constant-velocity path in the same local coordinates as the five observations."""
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 3 or history.shape[1:] != (5, 2) or not np.isfinite(history).all():
        raise ValueError("Expected five finite observed xy positions per window")
    velocity = (history[:, -1] - history[:, -2]) / DT
    time = DT * np.arange(1, T_FUT + 1, dtype=np.float64)
    return history[:, -1:] + velocity[:, None] * time[None, :, None]


def select_velocity_mode(history, candidates, probabilities):
    """Select with observed history and model outputs; the interface has no target input."""
    paths = np.asarray(candidates, dtype=np.float64)
    weights = np.asarray(probabilities, dtype=np.float64)
    reference = last_two_future(history)
    if (paths.shape != (len(reference), 6, T_FUT, 2) or weights.shape != (len(reference), 6)
            or not np.isfinite(paths).all() or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError("Expected six finite 30-step candidate paths and nonnegative probabilities")
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-6, rtol=0)
    cost = np.linalg.norm(paths - reference[:, None], axis=-1).mean(axis=-1)
    min_cost = cost.min(axis=1, keepdims=True)
    # argmax returns the lowest index among modes tied on both cost and probability.
    return np.where(cost == min_cost, weights, -np.inf).argmax(axis=1)


def score_forecasts(history, candidates, probabilities, targets):
    paths = np.asarray(candidates, dtype=np.float64)
    selected = select_velocity_mode(history, paths, probabilities)
    top = np.asarray(probabilities).argmax(axis=1)
    truth = np.asarray(targets, dtype=np.float64)
    if truth.shape != (len(paths), T_FUT, 2) or not np.isfinite(truth).all():
        raise ValueError("Expected 30 finite evaluation targets for every selected window")
    distances = np.linalg.norm(paths - truth[:, None], axis=-1)
    oracle = distances.mean(axis=-1).argmin(axis=1)
    rows = np.arange(len(truth))
    paths_errors = (distances[rows, top], distances[rows, selected],
                    np.linalg.norm(last_two_future(history) - truth, axis=-1), distances[rows, oracle])
    scores = np.stack([np.column_stack((err.mean(axis=1), err[:, -1])) for err in paths_errors], axis=1)
    return scores, np.column_stack((top, selected, oracle))


def verify_inputs(bundle, data_root, pilot_dir):
    accepted = json.loads((ROOT / "results/pinns_heldout_five_seed_replay.json").read_text())
    mode_reference = json.loads((ROOT / "results/pinns_heldout_mode_scores.json").read_text())
    archive_hash = file_hash(bundle)
    if archive_hash != accepted["archive_sha256"] or archive_hash != mode_reference["archive_sha256"]:
        raise ValueError("Expected the previously verified five-seed model archive")
    tensors, index, keys, _, _, _ = pilot_inputs(data_root, pilot_dir)
    expected = {f"seed_{seed}/{site}/{model}.pt" for seed in FIXED_SEEDS for site in index for model in MODELS}
    with zipfile.ZipFile(bundle) as archive:
        validate_zip(archive)
        manifest = json.loads(archive.read("sha256.json"))
        actual = {n for n in archive.namelist() if n.endswith(".pt")}
        if actual != expected:
            raise ValueError("Missing or unexpected model checkpoint")
        for name in expected:
            if hashlib.sha256(archive.read(name)).hexdigest() != manifest[name]:
                raise ValueError(f"Checkpoint checksum differs: {name}")
    with np.load(pilot_dir / "pose.npz", allow_pickle=False) as cache:
        ids = list(zip(cache["clip"].tolist(), cache["person_id"].tolist(), cache["t0"].tolist()))
        history = cache["hist_pos_relative"][check_keys(ids, keys)]
    if not np.all(history[:, -1] == 0) or not torch.all(tensors["fut_mask"] == 1):
        raise ValueError("Incomplete windows or unexpected origin")
    np.testing.assert_array_equal(history.astype(np.float32), tensors["x_m"].numpy()[:, :, :2])
    return archive_hash, mode_reference, tensors, index, keys, history


@torch.inference_mode()
def replay_one(model_name, state, tensors, ids, history):
    predictor = UnstructuredShortPosePredictor(model_name).cpu().eval()
    predictor.load_state_dict(state, strict=True)
    names = ("x_m", "ego_type_oh") if model_name == "M0" else ("x_m", "ego_type_oh", "x_p", "q_vec")
    errors, indices = [], []
    for batch in torch.as_tensor(ids).split(Protocol().batch_size):
        pred = predictor({k: tensors[k][batch] for k in names})
        scores, modes = score_forecasts(history[batch.numpy()], pred["trajs"].numpy(), pred["mode_probs"].numpy(),
                                       tensors["fut_xy"][batch].numpy())
        errors.append(scores)
        indices.append(modes)
    return np.concatenate(errors), np.concatenate(indices)


def summarize(errors, modes):
    switched = modes[..., 0] != modes[..., 1]
    means = errors.mean(axis=0)
    return {"methods": {m: dict(zip(METRICS, means[i].tolist())) for i, m in enumerate(METHODS)},
            "reranked_minus_network_m": dict(zip(METRICS, (means[1] - means[0]).tolist())),
            "changed_mode_rate": float(switched.mean())}


def average(summaries):
    first = summaries[0]
    if isinstance(first, dict):
        return {k: average([row[k] for row in summaries]) for k in first}
    return float(np.mean(summaries))


def aggregate(errors, modes, index, clips):
    # Axes: seed, model, ordered window, method, metric. All seeds reuse each identical window.
    by_seed, macro, by_site = {}, {}, {}
    for s, seed in enumerate(FIXED_SEEDS):
        by_seed[seed] = {site: {model: summarize(errors[s, m, ids], modes[s, m, ids])
                                 for m, model in enumerate(MODELS)} for site, ids in index.items()}
        macro[seed] = average(list(by_seed[seed].values()))
    for site in index:
        by_site[site] = average([by_seed[seed][site] for seed in FIXED_SEEDS])
    cluster = {}
    for m, model in enumerate(MODELS):
        delta = (errors[:, m, :, 1] - errors[:, m, :, 0]).mean(axis=0)
        rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        draws = {site: paired_clip_draws(delta[ids], clips[ids], rng) for site, ids in sorted(index.items())}
        cluster[model] = {"by_site": {site: intervals(v) for site, v in draws.items()},
                          "four_site_equal": intervals(np.mean(list(draws.values()), axis=0))}
    return {"by_seed_site": by_seed, "four_site_means_by_seed": macro, "five_seed_means_by_site": by_site,
            "four_site_five_seed_means": average(list(macro.values())), "paired_clip_percentile95_m": cluster}


def run(args):
    digest, reference, tensors, index, keys, history = verify_inputs(args.bundle, args.data_root, args.pilot_dir)
    if args.check_only:
        print(f"Verified source, five-point observations, {len(keys)} keys and 40 archived weights. No forecast was evaluated.")
        return
    scores = np.full((len(FIXED_SEEDS), len(MODELS), len(keys), len(METHODS), 2), np.nan)
    modes = np.full(scores.shape[:3] + (3,), -1, dtype=np.int8)
    velocity_reference = json.loads((ROOT / "results/pinns_velocity_estimators.json").read_text())
    with zipfile.ZipFile(args.bundle) as archive:
        for s, seed in enumerate(FIXED_SEEDS):
            for site, ids in index.items():
                for m, model in enumerate(MODELS):
                    name = f"seed_{seed}/{site}/{model}.pt"
                    ck = torch.load(io.BytesIO(archive.read(name)), map_location="cpu", weights_only=True)
                    if [tuple(k) for k in ck["window_keys"]] != [keys[i] for i in ids]:
                        raise ValueError(f"Test key order mismatch for {name}")
                    values, indices = replay_one(model, ck["model_state"], tensors, ids, history)
                    np.testing.assert_allclose(values[:, 0], ck["errors"].numpy(), rtol=RTOL, atol=ATOL_M)
                    # Independently replay the existing six-mode scorer without editing its frozen source.
                    archived_scorer = score_modes(ck["model_state"], model, tensors, ids)
                    columns = [COLUMNS.index(k) for k in ("ADE_1","FDE_1","minADE_6")]
                    direct = np.column_stack((values[:, 0], values[:, 3, 0]))
                    np.testing.assert_allclose(direct, archived_scorer[:, columns], rtol=RTOL, atol=ATOL_M)
                    old = reference["heldout_site_by_seed"][str(seed)][site][model]
                    np.testing.assert_allclose(direct.mean(axis=0), [old[COLUMNS[i]] for i in columns], rtol=RTOL, atol=ATOL_M)
                    vref = velocity_reference["sites"][site]["metrics"]["last_two"]
                    np.testing.assert_allclose(values[:, 2].mean(0), [vref[k] for k in METRICS], rtol=RTOL, atol=ATOL_M)
                    scores[s, m, ids] = values
                    modes[s, m, ids] = indices
                    print(f"Scored and checked {name}", flush=True)
    if not np.isfinite(scores).all() or not np.isin(modes, np.arange(6)).all():
        raise ValueError("Incomplete model/seed/site coverage")
    clips = np.array([k[0] for k in keys])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    errors_path = out / "reranked_paired_windows.npz"
    def write_npz(path):
        with path.open("wb") as f:
            np.savez_compressed(f, errors=scores, mode_indices=modes, clip=clips,
                                person_id=np.array([k[1] for k in keys]), t0=np.array([k[2] for k in keys]),
                                seeds=np.array(FIXED_SEEDS), models=np.array(MODELS), methods=np.array(METHODS),
                                metrics=np.array(METRICS), index_columns=np.array(["network_top1","velocity_selected","oracle_minade_mode"]))
    atomic_write(errors_path, write_npz)
    result = {"archive_sha256": digest, "paired_windows_sha256": file_hash(errors_path), "script_sha256": file_hash(Path(__file__)),
              "seeds": list(FIXED_SEEDS), "site_window_counts": SITE_COUNTS, "cpu_torch": str(torch.__version__),
              "rule": "The last two of five measured xy points at 10 Hz define an unscaled constant velocity. "
                      "Minimize mean Euclidean distance to its 30-step extrapolation; exact ties choose higher network "
                      "probability then lower index. Each method scores the mean error and step-30 error of its one path.",
              "oracle_definition": "Future-informed reference: select the candidate with minimum true ADE (lowest index "
                                   "on ties); FDE is for that same candidate. The direct velocity path is unrestricted.",
              "development_scope": "Exploratory choice informed by earlier CV and candidate-coverage results on these same "
                                   "four sites. All 40 fits frozen. Each site averages its own windows, then sites and "
                                   "the five seeds receive equal weight. New-site performance has not been measured.",
              "interval": {"replicates": REPLICATES, "rng_seed": BOOTSTRAP_SEED,
                           "method": "Average paired reranked-minus-network errors over the same five seeds per window, "
                                     "then jointly resample whole clips within each site. Four-site means give equal weight "
                                     "to sites. Percentile intervals conditional on these sites and fits."},
              **aggregate(scores, modes, index, clips)}
    report_path = out / "reranked_mode_scores.json"
    save_json(report_path, result)
    def pack(path):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in (report_path, errors_path):
                z.write(f, f.name)
    atomic_write(out / "reranked_modes.zip", pack)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("data_root", type=Path)
    p.add_argument("pilot_dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/shortpose_reranked"))
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    run(args)
