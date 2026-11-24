"""Nested site-held-out development probes for five-frame PINNS start/stop events.

python -m scripts.evaluate_pinns_nested_start_stop DATA_ROOT POSE.npz --public SUMMARY.json --private FITS_AND_SCORES.json
Use --design-only --public DESIGN.json to validate inputs and site memberships without fitting.
"""
import argparse
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit

from scripts.evaluate_heldout_implied_actions import class_scores
from scripts.evaluate_pinns_action_probes import C, NAMES, PRECONDITIONS, load_inputs
from scripts.train_pinns_heldout_sites import file_hash, save_json


EVENTS = PRECONDITIONS
DUMMY = "all_history_eligible"
METHODS = (DUMMY, *NAMES)
POLICIES = ("fixed_0_5", "inner_f1")
PROTOCOL = {
    "scope": "Development at four PINNS locations already examined. Each outer fold holds out one complete physical "
             "site; its labels and features never enter any model, feature scaling or threshold selection for that fold. "
             "Thresholds are specific to the outer training sites, event and feature family. No independent new-site validation.",
    "events": "Over the next 30 samples (3 s), apply the archived coordinate-derived start and stop definitions. "
              "Five historical samples span 0.4 s. Start eligibility requires mean pre-event speed <=0.20 m/s, "
              "stop >=0.60 m/s, using only the last three causal smoothed historical speeds. "
              "Scores are 0 and decisions negative outside the event's eligibility mask, including at threshold 0.",
    "methods": "All-eligible dummy scores one for an eligible row. Three feature families are fixed from the published "
               "five-frame probe: motion, motion plus the last normalized pose, and both plus four ordered masked joint "
               "changes. Fit each event separately, on eligible rows only. Equal window weights, no resampling or "
               "class weights. Training-only population mean and SD, divisor 1 for constant features. Fixed C=0.1; "
               "summed logistic loss + squared coefficient norm/(2C), unpenalized intercept. An eligible training set "
               "containing one class predicts its observed frequency; an empty eligible training set predicts zero.",
    "threshold": "Within the three outer training sites, hold out each site in turn and fit on the other two. "
                 "Pool the three inner OOF scores over eligible training windows. Among distinct observed OOF scores "
                 "choose the highest threshold maximizing pooled eligible F1; prediction is score >= threshold. "
                 "No eligible inner rows or no inner positives: use fixed 0.5. Refit on all three outer training sites, "
                 "then evaluate once on the fourth. Compare to fixed 0.5 on the identical outer probabilities.",
    "metrics": "TP, FP and FN use all outer windows and the eligibility decision gate. AP is non-interpolated, grouped "
               "at score ties, and uses scores independently of threshold. Also report AP within history-eligible rows. "
               "If the evaluation group has no positives, AP/F1 are null and all confusion counts remain. "
               "Four-site macro averages AP/F1 equally over sites with actual positives for that event, giving the "
               "denominator; confusion counts are also averaged over all four sites and pooled sums are provided. "
               "OOF fits share development locations and observations within a clip may be correlated.",
}


def fit_binary(x, y):
    """Return serializable training-only transform and classifier for a single event."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    n, dim = x.shape
    positives = int(y.sum())
    record = {"training_eligible": n, "training_positive": positives}
    if n == 0 or positives in (0, n):
        return {**record, "kind": "constant", "constant": positives / n if n else 0.,
                "reason": "no eligible training rows" if n == 0 else "one eligible training class"}
    mean, sd = x.mean(axis=0), x.std(axis=0)
    scale = np.where(sd > 0, sd, 1.)
    z = (x - mean) / scale
    truth = y.astype(float)

    def objective(theta):
        w, intercept = theta[:-1], theta[-1]
        logits = z @ w + intercept
        residual = expit(logits) - truth
        loss = np.logaddexp(0., (1 - 2 * truth) * logits).mean() + (w @ w) / (2 * C * n)
        return loss, np.r_[(z.T @ residual + w / C) / n, residual.mean()]

    initial = np.zeros(dim + 1)
    initial[-1] = np.log(positives / (n - positives))
    result = minimize(objective, initial, method="L-BFGS-B", jac=True,
                      options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Eligible binary logistic fit failed: {result.message}")
    return {**record, "kind": "logistic", "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": result.x[:-1].tolist(), "intercept": float(result.x[-1]),
            "iterations": int(result.nit)}


def predict(fit, x, eligible):
    scores = np.zeros(len(x), dtype=float)
    if fit["kind"] == "constant":
        scores[eligible] = fit["constant"]
    else:
        z = (x[eligible] - np.asarray(fit["mean"])) / np.asarray(fit["scale"])
        scores[eligible] = expit(z @ np.asarray(fit["coefficients"]) + fit["intercept"])
    return scores


def select_threshold(scores, truth, eligible):
    s = np.asarray(scores)[eligible]
    y = np.asarray(truth, dtype=bool)[eligible]
    if not len(s) or not y.any():
        return .5, "no eligible inner windows" if not len(s) else "no positive inner windows"
    order = np.argsort(-s, kind="stable")
    sorted_s = s[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_s) != 0) + 1, len(s)]
    tp = np.cumsum(y[order])[ends - 1]
    positives = int(y.sum())
    best_num, best_den, threshold = -1, 1, None
    for end, hits in zip(ends, tp):
        # F1 = 2*TP/(predicted positives + actual positives). Descending search keeps the higher exact tie.
        num, den = 2 * int(hits), int(end) + positives
        if num * best_den > best_num * den:
            best_num, best_den, threshold = num, den, float(sorted_s[end - 1])
    return threshold, "highest observed eligible inner OOF score attaining maximal F1"


def event_metrics(truth, eligible, scores, fixed, selected):
    actual = int(truth.sum())
    ap = class_scores(truth, scores)["ap"]
    eligible_ap = class_scores(truth[eligible], scores[eligible])["ap"]
    result = {"positive": actual, "eligible": int(eligible.sum()), "ap": ap, "eligible_only_ap": eligible_ap}
    for name, binary in zip(POLICIES, (fixed, selected)):
        if binary[~eligible].any():
            raise ValueError("An ineligible window received a positive prediction")
        tp = int((truth & binary).sum())
        fp = int((~truth & binary).sum())
        fn = actual - tp
        result[name] = {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / (2 * tp + fp + fn) if actual else None}
    return result


def macro_sites(site_reports, method, event):
    values = [row["methods"][method][event] for row in site_reports.values()]
    ok = [r for r in values if r["positive"] > 0]
    result = {"sites_total": len(values), "sites_with_positives": len(ok)}
    for metric in ("ap", "eligible_only_ap"):
        result[metric] = float(np.mean([v[metric] for v in ok])) if ok else None
    for policy in POLICIES:
        result[policy] = {m + "_mean_per_site": float(np.mean([r[policy][m] for r in values])) for m in ("tp", "fp", "fn")}
        result[policy]["f1"] = float(np.mean([r[policy]["f1"] for r in ok])) if ok else None
    return result


def evaluate(inputs, design_only=False):
    keys, sites, clips, _, features, eligible, truth, identity = inputs
    truth = np.asarray(truth[:, :2], dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    distinct = tuple(sorted(set(sites.tolist())))
    if len(distinct) != 4 or np.any(truth & ~eligible) or not all(len(set(sites[clips == c])) == 1 for c in set(clips)):
        raise ValueError("Invalid physical site split or eligibility support")
    report = {"protocol": PROTOCOL, "design_only": design_only, "verification": {
                  **identity, "nested_script_sha256": file_hash(Path(__file__))},
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "feature_dimensions": {name: features[name].shape[1] for name in NAMES}, "sites": {}}
    private = {"window_key_fields": ["clip", "person_id", "t0"], "window_keys": keys, "site": sites.tolist(),
               "event_order": EVENTS, "eligible": eligible.tolist(), "truth": truth.tolist(), "fits": {}, "outer": {}}
    storage_shape = (len(keys), len(EVENTS))
    scores = {name: np.full(storage_shape, np.nan) for name in METHODS}
    decisions = {name: {p: np.zeros(storage_shape, dtype=bool) for p in POLICIES} for name in METHODS}

    def eligible_fit(training_sites, name, event_index):
        tag = "/".join([*training_sites, name, EVENTS[event_index]])
        if tag not in private["fits"]:
            rows = np.flatnonzero(np.isin(sites, training_sites) & eligible[:, event_index])
            model = fit_binary(features[name][rows], truth[rows, event_index])
            private["fits"][tag] = {**model, "training_sites": list(training_sites), "training_rows": rows.tolist()}
        return tag

    for held in distinct:
        training = tuple(s for s in distinct if s != held)
        outer_rows = np.flatnonzero(sites == held)
        train_rows = np.flatnonzero(sites != held)
        supports = {event: {"positive": int(truth[outer_rows, j].sum()),
                            "eligible": int(eligible[outer_rows, j].sum())} for j, event in enumerate(EVENTS)}
        site_info = {"windows": len(outer_rows), "clips": len(set(clips[outer_rows])), "training_sites": training,
                     "event_support": supports}
        report["sites"][held] = site_info
        if design_only:
            continue
        details = {"outer_test_rows": outer_rows.tolist(), "outer_train_rows": train_rows.tolist(),
                   "inner_validation_sites_in_oof_order": sites[train_rows].tolist(), "methods": {}}
        private["outer"][held] = details
        site_info["methods"] = {}
        for name in METHODS:
            details["methods"][name], site_info["methods"][name] = {}, {}
            for j, event in enumerate(EVENTS):
                train_e, outer_e = eligible[train_rows, j], eligible[outer_rows, j]
                inner, inner_fits, final_fit = np.full(len(train_rows), np.nan), {}, None
                if name == DUMMY:
                    inner = train_e.astype(float)
                    outer_scores = outer_e.astype(float)
                else:
                    for inner_held in training:
                        inner_training = tuple(s for s in training if s != inner_held)
                        tag = eligible_fit(inner_training, name, j)
                        inner_fits[inner_held] = tag
                        local = sites[train_rows] == inner_held
                        inner[local] = predict(private["fits"][tag], features[name][train_rows[local]], train_e[local])
                    final_fit = eligible_fit(training, name, j)
                    outer_scores = predict(private["fits"][final_fit], features[name][outer_rows], outer_e)
                if not np.isfinite(inner).all() or not np.isfinite(outer_scores).all():
                    raise ValueError("Nonfinite or missing eligible predictions")
                threshold, reason = select_threshold(inner, truth[train_rows, j], train_e)
                fixed = outer_e & (outer_scores >= .5)
                selected = outer_e & (outer_scores >= threshold)
                scores[name][outer_rows, j] = outer_scores
                decisions[name][POLICIES[0]][outer_rows, j] = fixed
                decisions[name][POLICIES[1]][outer_rows, j] = selected
                site_info["methods"][name][event] = event_metrics(truth[outer_rows, j], outer_e, outer_scores, fixed, selected)
                details["methods"][name][event] = {"inner_fit_by_validation_site": inner_fits,
                    "final_fit": final_fit, "inner_oof_scores_outer_train_order": inner.tolist(),
                    "inner_f1_threshold": threshold, "threshold_rule": reason, "outer_scores": outer_scores.tolist(),
                    "fixed_decisions": fixed.tolist(), "inner_threshold_decisions": selected.tolist()}
    if design_only:
        return report, None
    report["pooled"], report["equal_site_macro"] = {}, {}
    for name in METHODS:
        if not np.isfinite(scores[name]).all():
            raise ValueError("A window did not receive an outer score")
        report["pooled"][name], report["equal_site_macro"][name] = {}, {}
        for j, event in enumerate(EVENTS):
            report["pooled"][name][event] = event_metrics(truth[:, j], eligible[:, j], scores[name][:, j],
                decisions[name][POLICIES[0]][:, j], decisions[name][POLICIES[1]][:, j])
            report["equal_site_macro"][name][event] = macro_sites(report["sites"], name, event)
    private["verification"] = report["verification"]
    return report, private


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("pose_npz", type=Path)
    parser.add_argument("--public", type=Path, required=True, help="Aggregate metrics and support counts")
    parser.add_argument("--private", type=Path, help="Private window scores, transforms, fitted coefficients and thresholds")
    parser.add_argument("--design-only", action="store_true")
    args = parser.parse_args()
    if not args.design_only and args.private is None:
        parser.error("--private is required for an actual experiment")
    if args.private is not None and (args.private.resolve() == args.public.resolve() or args.design_only):
        parser.error("Pass distinct destinations; the design-only pass only writes --public")
    targets = [p.resolve() for p in (args.public, args.private) if p is not None]
    if args.pose_npz.resolve() in targets or any(args.data_root.resolve() in p.parents for p in targets):
        parser.error("Write result files outside the source data and pose cache")
    public, private = evaluate(load_inputs(args.data_root, args.pose_npz), args.design_only)
    if private is not None:
        save_json(args.private, private)
    save_json(args.public, public)
    print("Validated outer sites:", ", ".join(public["sites"]), flush=True)
