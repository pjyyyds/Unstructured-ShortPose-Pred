"""Paired PINNS development experiment on strictly earlier image and occupancy polylines.

python -m scripts.evaluate_pinns_causal_geometry DATA_ROOT MAP_DIR --public SUMMARY.json --private REPLAY.npz
Use --design-only --public DESIGN.json to validate inputs, split identities and observation support without a fit.
MAP_DIR contains the frozen maps.npz and manifest.json produced by scripts.build_pinns_causal_maps.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_pinns_map_sources import digest
from scripts.train_pinns_heldout_sites import atomic_write, save_json
from src.data.constants import DT, T_FUT


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("motion", "availability", "image_geometry", "occupancy_geometry")
SPLITS = {"train": 5686, "val": 1100, "test": 497}
ALPHA = 0.1
SEED = 2026
REPLICATES = 20000
ARRAY_DIGEST = "2b1873f056dfba34b79fd939da4cf6aa89c352d0d11f78988068072db14b5e90"
SOURCE_DIGEST = "87983d8e23a0853ef6b7bbe68959c7d3e604b5319d486fc663d21424890408bd"
METRICS = ("ade_m", "fde_last_valid_m", "fde_exact_3s_m")
GEOM = ("nearest_dx_m", "nearest_dy_m", "distance_m", "cos_2theta", "sin_2theta", "approach_mps")


def mask_velocity(history, observed):
    """OLS velocity over available points at their true timestamps, anchored at the last observation for prediction."""
    if not observed[:, -1].all() or (observed.sum(axis=1) < 2).any():
        raise ValueError("Expected t0 and at least two historical samples")
    times = DT * np.arange(history.shape[1])
    mean = (observed * times).sum(axis=1) / observed.sum(axis=1)
    centered = (times - mean[:, None]) * observed
    return np.einsum("nt,ntd->nd", centered, history) / np.square(centered).sum(axis=1)[:, None]


def geometry_features(environment, velocity):
    """Nearest point on a nonzero-length segment, using the actual local 10-point sampled lines."""
    outputs = np.zeros((len(environment), len(GEOM)))
    valid = np.zeros(len(environment), dtype=bool)
    for i, lines in enumerate(environment):
        mask = lines[:, :, 10].astype(bool)
        a = lines[:, :-1, :2].astype(float)
        segment = lines[:, 1:, :2].astype(float) - a
        norm2 = np.square(segment).sum(axis=2)
        keep = mask[:, :-1] & mask[:, 1:] & (norm2 > 0)
        if not keep.any():
            continue
        # Row-major selection makes exact ties prefer the earliest stored line and segment.
        a, segment, norm2 = a[keep], segment[keep], norm2[keep]
        fraction = np.clip(-np.sum(a * segment, axis=1) / norm2, 0., 1.)
        closest = a + fraction[:, None] * segment
        j = int(np.argmin(np.square(closest).sum(axis=1)))
        delta, tangent = closest[j], segment[j] / np.sqrt(norm2[j])
        distance = np.linalg.norm(delta)
        tx, ty = tangent
        approach = float(velocity[i] @ delta / distance) if distance > 0 else 0.
        outputs[i] = [*delta, distance, tx*tx - ty*ty, 2*tx*ty, approach]
        valid[i] = True
    return outputs, valid


def array_digest(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = arrays[key]
        h.update(key.encode() + b"\0" + a.dtype.str.encode() + b"\0" + json.dumps(a.shape).encode() + b"\0")
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


@dataclass
class Cohort:
    keys: list
    sites: np.ndarray
    clips: np.ndarray
    splits: np.ndarray
    history_mask: np.ndarray
    future: np.ndarray
    future_mask: np.ndarray
    velocity: np.ndarray
    availability: np.ndarray
    features: dict
    identity: dict


def load_inputs(data, map_dir):
    from scripts.build_pinns_causal_maps import load_motion_windows
    manifest = json.loads((map_dir / "manifest.json").read_text())
    npz_path = map_dir / "maps.npz"
    source_digest = hashlib.sha256(json.dumps(manifest["sources"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (manifest["protocol"] != "causal_same_clip_training_reference_v1" or source_digest != SOURCE_DIGEST
            or manifest["npz_sha256"] != digest(npz_path)):
        raise ValueError("Unexpected causal source manifest or map payload checksum")
    with np.load(npz_path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    if array_digest(arrays) != ARRAY_DIGEST:
        raise ValueError("The map tensors differ from the predeclared source. Use the frozen inputs or the original decoder environment")
    records = load_motion_windows(data)
    keys = list(zip(arrays["clip"].tolist(), arrays["person_id"].tolist(), arrays["t0"].tolist()))
    clips, splits = arrays["clip"], arrays["split"]
    if len(keys) != len(set(keys)) or set(keys) != set(records) or Counter(splits.tolist()) != SPLITS:
        raise ValueError("Map keys and official 7,283-window cohort differ")
    for clip, info in manifest["sources"].items():
        condition = clip.rsplit("_", 1)[0]
        if (digest(data / "labels" / (clip + ".txt")) != info["label_sha256"]
                or digest(data / "videos" / condition / "calib.json") != info["calibration_sha256"]):
            raise ValueError(f"Official label/calibration identity differs for {clip}")
        local = clips == clip
        reference = info["reference_10hz"]
        expected = arrays["t0"][local] > max(reference) if reference else np.zeros(int(local.sum()), dtype=bool)
        np.testing.assert_array_equal(arrays["source_eligible"][local], expected)
    site, official_splits, hist, hist_mask, future, future_mask = zip(*(records[k] for k in keys))
    np.testing.assert_array_equal(splits, official_splits)
    sites = np.array(site)
    if not all(manifest["sources"][c]["site"] == s for c, s in zip(clips, sites)):
        raise ValueError("Map site assignments differ from the official cohorts")
    h, hm = np.stack(hist).astype(float), np.stack(hist_mask).astype(bool)
    fut, fm = np.stack(future).astype(float), np.stack(future_mask).astype(bool)
    if np.any(fm.sum(axis=1) < 15) or not np.isfinite(h[hm]).all() or not np.isfinite(fut[fm]).all():
        raise ValueError("Historical or future annotation support differs from the archived eligibility rule")
    h = np.where(hm[:, :, None], h, 0.)
    fut = np.where(fm[:, :, None], fut, 0.)
    v = mask_velocity(h, hm)
    motion = np.column_stack([h[:, :-1].reshape(len(keys), -1), hm[:, :-1], v])
    geom, present = {}, {}
    for name in ("image", "hull"):
        result, valid = geometry_features(arrays["p_env_" + name], v)
        if not np.array_equal(valid, arrays["m_map_" + name].astype(bool)):
            raise ValueError(f"Source map availability differs from usable line segments in {name}")
        geom[name], present[name] = result, valid
    available = np.column_stack([arrays["source_eligible"], present["image"], present["hull"]]).astype(bool)
    if np.any(available[:, 1:] & ~available[:, :1]):
        raise ValueError("Local geometry without a causal reference")
    base = np.column_stack([motion, available])
    features = dict(zip(METHODS, (motion, base, np.column_stack([base, geom["image"]]), np.column_stack([base, geom["hull"]]))))
    identity = {"map_arrays_sha256": ARRAY_DIGEST, "source_manifest_sha256": SOURCE_DIGEST,
                "maps_npz_sha256": manifest["npz_sha256"], "sidecar_builder_sha256": manifest["script_sha256"],
                "evaluator_sha256": digest(Path(__file__)), "numpy": np.__version__,
                "ordered_key_sha256": hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()}
    return Cohort(keys, sites, clips, splits, hm, fut, fm, v, available, features, identity)


def fit_horizons(x, residual, valid, train):
    """One training-only transform, intercept and ridge per annotated forecast time."""
    n_time, n_features = residual.shape[1], x.shape[1]
    mean = np.empty((n_time, n_features))
    scale = np.empty_like(mean)
    bias = np.empty((n_time, 2))
    weights = np.empty((n_time, n_features, 2))
    for t in range(n_time):
        keep = train & valid[:, t]
        if not keep.any():
            raise ValueError(f"No training observations for lead {t+1}")
        xx, y = x[keep], residual[keep, t]
        mean[t] = xx.mean(axis=0)
        std = xx.std(axis=0)
        scale[t] = np.where(std > 1e-10, std, 1.)
        bias[t] = y.mean(axis=0)
        z = (xx - mean[t]) / scale[t]
        weights[t] = np.linalg.solve(z.T @ z + len(xx)*ALPHA*np.eye(n_features), z.T @ (y - bias[t]))
    return {"mean": mean, "scale": scale, "intercept": bias, "weights": weights}


def forecast(x, cv, model):
    predicted = cv.copy()
    for t in range(cv.shape[1]):
        z = (x-model["mean"][t])/model["scale"][t]
        predicted[:,t] += model["intercept"][t] + z @ model["weights"][t]
    return predicted


def errors(pred, truth, valid):
    distances = np.linalg.norm(pred - truth, axis=2)
    last = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    ade = np.where(valid, distances, 0.).sum(axis=1) / valid.sum(axis=1)
    return np.column_stack([ade, distances[np.arange(len(last)), last], np.where(valid[:, -1], distances[:, -1], np.nan)])


def metric_means(values):
    return {name: float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else None for name,v in zip(METRICS,values.T)}


def support(c, mask):
    hist, fut, avail = c.history_mask[mask], c.future_mask[mask], c.availability[mask]
    last_valid = fut.shape[1] - np.argmax(fut[:, ::-1], axis=1)
    return {"windows": int(mask.sum()), "clips": len(set(c.clips[mask])), "complete_5_history": int(hist.all(1).sum()),
            "complete_30_future": int(fut.all(1).sum()), "fde_last_valid_windows": int(mask.sum()),
            "fde_exact_3s_windows": int(fut[:, -1].sum()), "fde_exact_3s_clips": len(set(c.clips[mask][fut[:, -1]])),
            "last_valid_lead_s_range": [round(DT*int(last_valid.min()), 1), round(DT*int(last_valid.max()), 1)] if len(last_valid) else None,
            "future_scored_steps_by_lead": fut.sum(axis=0).tolist(),
            "source_eligible": int(avail[:,0].sum()), "image": int(avail[:,1].sum()), "hull": int(avail[:,2].sum()),
            "both": int(avail[:,1:].all(1).sum())}


def summarize(c, mask, scores):
    result = support(c, mask)
    if scores and mask.any():
        result["means"] = {k: metric_means(v[mask]) for k,v in scores.items()}
        result["paired_differences"] = {
            f"{m}_minus_{b}": metric_means(scores[m][mask]-scores[b][mask])
            for m,b in (("image_geometry","availability"),("occupancy_geometry","availability"),("availability","motion"),
                        ("image_geometry","motion"),("occupancy_geometry","motion"),("motion","ols_five"))}
    return result


def clip_interval(deltas, sites, clips):
    """Sample whole test clips within each fixed site; preserve the site's observed mixture weight."""
    rng = np.random.default_rng(SEED)
    draws = np.zeros(REPLICATES)
    for site in sorted(set(sites)):
        local = sites == site
        unique, ix = np.unique(clips[local], return_inverse=True)
        sizes = np.bincount(ix); totals = np.bincount(ix, weights=deltas[local])
        chosen = rng.integers(0,len(unique),size=(REPLICATES,len(unique)))
        draws += (local.sum()/len(sites)) * totals[chosen].sum(axis=1)/sizes[chosen].sum(axis=1)
    return {"estimate_m": float(deltas.mean()), "percentile_95_m": np.quantile(draws,[.025,.975]).tolist(),
            "seed":SEED,"replicates":REPLICATES}


PROTOCOL = {
  "cohort": "Original previously reused PINNS development split: 5,686 train / 1,100 val / 497 test, 47 clips, four sites. "
            "Five 10 Hz observed slots (0.4 s span), anchor observed, at least two valid historical slots and 15/30 future points. "
            "All methods share windows; missing history is zero with mask. Validation and test are scores only.",
  "source": "Same-clip training-only reference starts at that clip's earliest training trigger and covers at most six seconds. "
            "Every source time precedes the target anchor and the earliest held window's 19-frame history. At least seven "
            "distinct training timestamps are required. Median of seven BEV frames provides generic Canny image contours. "
            "The convex hull uses only other annotated pedestrians in the reference; the target's whole ID is excluded. "
            "No available reference or local geometry is not evidence of empty/open physical space. Neither input is a verified curb or walkable boundary.",
  "models": "OLS velocity from available historical points at actual timestamps, extrapolated from t0. The four independent "
            "direct residual ridge fits use: (1) four historical xy offsets, four masks and two OLS velocity components; "
            "(2) same motion plus three flags: historical reference exists, locally usable image segments, locally usable hull segments; "
            "(3) same flags and image-only continuous geometry; (4) same flags and hull-only continuous geometry. "
            "Subsequent comparisons with availability thus condition on the same information about availability in both branches. "
            "Select nearest point on any segment between consecutive valid sampled polyline vertices; deterministic line/segment order breaks ties. "
            "Encode relative xy, distance, axial orientation (cos 2theta, sin 2theta, unchanged if reversed), and ego velocity toward that point "
            "(0 at coincident points). Missing geometry receives six zeros plus the availability flags.",
  "training": "Each of 30 leads is fitted separately on that lead's ground-truth-available training rows with equal weights. "
              "Recompute column mean/population SD on these rows (SD<=1e-10 uses 1). Predict residual xy to OLS, unpenalized "
              "intercept, fixed mean squared loss + 0.1 times coefficient squared norm. No tuning or cross-split normalization.",
  "score": "Window ADE averages Euclidean error over available future points, then averages windows. Original FDE uses the last "
           "valid target, regardless of lead; additionally report exact 3.0 s FDE only for rows with the 30th target. "
           "Validation and test are scored separately; common-map support is a secondary descriptive subgroup.",
  "primary": "Test pooled window-weighted ADE difference (image geometry minus availability only), lower is better. "
             "Within each of the four fixed locations sample its whole evaluation clips with replacement, retaining all their windows; "
             "keep each site's original test-window weight when averaging the four bootstrap site means. 20,000 draws, seed 2026, "
             "percentile 95% interval conditional on fitted models. Related recordings and shared training may induce dependence. "
             "Hull comparisons, last-valid/exact FDE, per-site and common-support scores are secondary. These four reused sites give development evidence.",
}


def evaluate(c, design_only):
    report = {"protocol":PROTOCOL, "design_only":design_only, "identity":c.identity,
              "feature_dimensions":{k:x.shape[1] for k,x in c.features.items()},
              "splits":{s:support(c,c.splits==s) for s in SPLITS}}
    scores = {}
    private = {"clip":c.clips,"person_id":np.array([k[1] for k in c.keys]),"t0":np.array([k[2] for k in c.keys]),
               "split":c.splits,"site":c.sites,"historical_mask":c.history_mask,"future_mask":c.future_mask,
               "availability":c.availability,"truth_relative":c.future,"velocity":c.velocity}
    if not design_only:
        cv = c.velocity[:,None] * (DT*np.arange(1,T_FUT+1))[None,:,None]
        predictions = {"ols_five":cv}
        train = c.splits=="train"
        for method,x in c.features.items():
            params = fit_horizons(x,c.future-cv,c.future_mask,train)
            predictions[method] = forecast(x,cv,params)
            private["inputs_"+method] = x
            private.update({method+"_"+name:v for name,v in params.items()})
        for name,pred in predictions.items():
            if not np.isfinite(pred).all():
                raise ValueError(f"Nonfinite predictions: {name}")
            scores[name] = errors(pred,c.future,c.future_mask)
            private["prediction_"+name] = pred
        test = c.splits=="test"
        primary = (scores["image_geometry"] - scores["availability"])[test,0]
        report["primary_test_paired_ade"] = clip_interval(primary,c.sites[test],c.clips[test])
        private["identity_json"] = np.array(json.dumps(c.identity,sort_keys=True))
    for sp in ("val","test"):
        m = c.splits==sp
        common = m & c.availability[:,1:].all(axis=1)
        report[sp] = {"all":summarize(c,m,scores), "by_site":{s:summarize(c,m&(c.sites==s),scores) for s in sorted(set(c.sites))},
                      "both_maps":summarize(c,common,scores),
                      "both_maps_by_site":{s:summarize(c,common&(c.sites==s),scores) for s in sorted(set(c.sites))}}
    return report, private if not design_only else None


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root",type=Path)
    parser.add_argument("map_dir",type=Path)
    parser.add_argument("--public",type=Path,required=True)
    parser.add_argument("--private",type=Path,help="Non-pickle NPZ with weights, scaling and per-window 30-step predictions")
    parser.add_argument("--design-only",action="store_true")
    args=parser.parse_args()
    if not args.design_only and args.private is None:
        parser.error("Pass --private for full evaluation")
    targets=[p.resolve() for p in (args.public,args.private) if p is not None]
    if (len(set(targets))!=len(targets) or any(root.resolve() in p.parents for p in targets for root in (args.map_dir,args.data_root))
            or (args.design_only and args.private is not None)):
        parser.error("Use distinct output paths outside data inputs; --design-only needs only --public")
    report,private=evaluate(load_inputs(args.data_root,args.map_dir),args.design_only)
    if private is not None:
        def write_private(path):
            with path.open("wb") as stream:
                np.savez_compressed(stream,**private)
        atomic_write(args.private,write_private)
    save_json(args.public,report)
    print("Verified matched windows:",{s:r["windows"] for s,r in report["splits"].items()},flush=True)
    if not args.design_only: print("Primary test paired ADE:",report["primary_test_paired_ade"],flush=True)
