"""Check saved CUDA top-mode outputs against local source labels, without constructing any predictive model."""

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch


FUSION = "5b5c9aa989f32f15c1c3b888be7a5ef8e8cf20b08255265a5b02682342367a78"
BASE = "be4e6ad8fdc29db9b0d404a103171ce8cd942b1d69260bbe48faf8e1e850d0e6"
HISTORY = "caa8f0959f28f2d69a6f16cc9df115ab6222d0b6eb3297edd575615fef81b94c"
CACHE = "59e59e3dfa6a2b5583ce4c3fbcb010866e793a3595e43cf691dbefbfcddc64f1"
ROSTER = "11ec9713c26fdd3db6a9a887b4ac8b79fbbf13c6389b5924fcd61c315a38aa34"
REVISION = "a2eeaf01568d74f7a01a0f74a077632e33e41b47"
DURATIONS = (5, 10, 20)
FUSION_ARMS = ("M1b", "M1-concat")
BASE_ARMS = ("M0", "M1")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def jhash(data):
    return sha(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())


def keys(z):
    return list(zip(*(z[k].tolist() for k in ("clip", "person_id", "t0"))))


def reference(histories, metadata, base_archive):
    check(sha(histories.read_bytes()) == HISTORY, "Unexpected strict-cohort history manifest")
    check(sha(base_archive.read_bytes()) == BASE, "Unexpected base archive")
    with zipfile.ZipFile(base_archive) as base:
        pose = base.read("matched_observations.npz")
    check(sha(pose) == CACHE, "Unexpected cached input in base archive")
    with np.load(histories, allow_pickle=False) as z:
        whole = keys(z)
        chosen = np.flatnonzero(z["strict_disjoint_20"])
        ordered = [whole[i] for i in chosen]
        truth = z["future_30_xy"][chosen]
        history = z["history_20_xy"][chosen]
        site = z["site"][chosen]
        check(z["history_20_mask"][chosen].all(), "Complete history")
    with np.load(io.BytesIO(pose), allow_pickle=False) as z:
        canonical = keys(z)
        row = {k: i for i, k in enumerate(canonical)}
        check(
            canonical == sorted(ordered) and len(set(ordered)) == 623 and jhash(canonical) == ROSTER,
            "Cohort",
        )
        alignment = [row[k] for k in ordered]
        check(z["pose_inferred"].all(), "Complete pose")
        np.testing.assert_array_equal(z["xy"][alignment], np.concatenate((history, truth), axis=1))

    lookup = defaultdict(list)
    for i, (clip, person, t0) in enumerate(ordered):
        lookup[clip].append((i, person, t0))
    for clip, group in lookup.items():
        path = metadata / "labels" / (clip + ".txt")
        positions = {}
        for line in path.read_bytes().splitlines():
            fields = line.split()
            if len(fields) >= 11 and fields[1].lower() == b"person":
                raw = int(fields[0]) - 1
                if raw >= 0 and raw % 3 == 0:
                    k = (int(fields[2]), raw // 3)
                    check(k not in positions, "Duplicate source position")
                    positions[k] = float(fields[8]), float(fields[9])
        for i, person, t0 in group:
            points = np.asarray([positions[person, t] for t in range(t0 - 19, t0 + 31)])
            np.testing.assert_array_equal(points - points[19], np.vstack((history[i], truth[i])))
    return ordered, site, truth.astype("f4")


def read_archive(path, wanted, arms, sites):
    check(sha(path.read_bytes()) == wanted, "Unexpected archive bytes")
    z = zipfile.ZipFile(path)
    manifest = json.loads(z.read("sha256.json"))
    expected = {f"T{t}/{s}/{a}.pt" for t in DURATIONS for s in sites for a in arms} | {"report.json"}
    if arms == BASE_ARMS:
        expected.add("matched_observations.npz")
    check(
        z.testzip() is None
        and set(z.namelist()) == expected | {"sha256.json"}
        and len(z.namelist()) == len(expected) + 1
        and set(manifest) == expected,
        "Archive roster/CRC",
    )
    for n, h in manifest.items():
        check(sha(z.read(n)) == h, "Archive member digest")
    return z, json.loads(z.read("report.json"))


def state_hash(state):
    h = hashlib.sha256()
    for name, array in sorted(state.items()):
        a = array.numpy()
        check(np.isfinite(a).all(), "Nonfinite parameter")
        h.update(
            json.dumps((name, str(a.dtype), list(a.shape)), separators=(",", ":")).encode()
            + b"\0"
            + a.tobytes()
        )
    return h.hexdigest()


def metrics(vector):
    return {k: float(val) for k, val in zip(("ade_m", "fde_m"), vector)}


def contrast(delta, site, clips):
    result = np.zeros((20000, 2))
    by_site = {}
    rng = np.random.Generator(np.random.PCG64(2026))
    for s in sorted(set(site)):
        mask = site == s
        local_clips = sorted(set(clips[mask]))
        counts = np.array([np.count_nonzero(mask & (clips == c)) for c in local_clips])
        totals = np.array([delta[mask & (clips == c)].sum(0) for c in local_clips])
        draws = rng.integers(len(counts), size=(len(result), len(counts)))
        result += 0.25 * totals[draws].sum(1) / counts[draws].sum(1)[:, None]
        by_site[s] = delta[mask].mean(0)
    ci = np.quantile(result, [0.025, 0.975], axis=0).T
    return {
        "difference": metrics(np.mean(list(by_site.values()), axis=0)),
        "clip_bootstrap_95_m": dict(zip(("ade", "fde"), ci.tolist())),
        "by_site": {s: metrics(v) for s, v in by_site.items()},
    }


def run(args):
    torch.set_num_threads(2)
    order, sites, truth = reference(args.histories, args.metadata, args.base)
    clips = np.array([k[0] for k in order])
    names = sorted(set(sites))
    check(len(names) == 4 and len(set(clips)) == 47, "Sites and clips")
    counts = {s: int(np.sum(sites == s)) for s in names}
    totals, states, means, by_site = {}, set(), {}, {}
    max_point_gap = 0.0
    max_prob_gap = 0.0
    for path, archive_sha, arms in [(args.fusion, FUSION, FUSION_ARMS), (args.base, BASE, BASE_ARMS)]:
        z, r = read_archive(path, archive_sha, arms, names)
        is_fusion = arms == FUSION_ARMS
        check(r["windows"] == 623 and r["site_counts"] == counts, "Summary sizes")
        if is_fusion:
            ident = r["identity"]
            check(
                ident["training_torch"] == "2.11.0+cu130"
                and ident["ordered_keys_sha256"] == ROSTER
                and ident["matched_pose_sha256"] == CACHE,
                "Fusion identity",
            )
            for n, h in {
                "scripts/run_pinns_matched_fusion.py": ident["runner_and_concat_sha256"],
                **ident["dependency_sha256"],
            }.items():
                raw = (args.source_tree / n).read_bytes()
                check(sha(raw) == h, "Frozen training program source")
            production_check = {
                "reported_replayed_checkpoints": r["cpu_replayed"],
                "reported_worst_cpu_vs_cloud_error_m": r["max_cpu_replay_abs_m"],
            }
            arch_params = r["parameters"]
            protocol = ident["protocol"]
            check(
                protocol
                == dict(
                    seed=42,
                    epochs=28,
                    batch_size=64,
                    learning_rate=0.002,
                    min_lr_ratio=0.05,
                    weight_decay=0.0001,
                    gradient_clip=2.0,
                ),
                "Unexpected training configuration",
            )
        else:
            check(
                r["protocol"] == protocol and r["cache_sha256"] == CACHE and r["keys_sha256"] == ROSTER,
                "Base protocol",
            )
        for t in DURATIONS:
            for model in arms:
                errors = np.empty((len(order), 2))
                for s in names:
                    ix = np.flatnonzero(sites == s)
                    training = [x for other in names if other != s for x in np.flatnonzero(sites == other)]
                    ck = torch.load(
                        io.BytesIO(z.read(f"T{t}/{s}/{model}.pt")), map_location="cpu", weights_only=True
                    )
                    checkpoint = ck["identity"]
                    test_keys = [order[i] for i in ix]
                    check(
                        checkpoint["test_ids_sha256"] == jhash(test_keys)
                        and checkpoint["train_ids_sha256"] == jhash([order[i] for i in training]),
                        "Fold identities",
                    )
                    check(checkpoint["duration"] == t and checkpoint["protocol"] == protocol, "Protocol")
                    check(
                        checkpoint["training_sites" if is_fusion else "train_sites"]
                        == [site for site in names if site != s]
                        and checkpoint["held_site" if is_fusion else "site"] == s
                        and checkpoint["variant" if is_fusion else "model"] == model,
                        "Fold assignment",
                    )
                    st = ck["model_state"]
                    digest = state_hash(st)
                    check(digest not in states, "Duplicate parameter state")
                    states.add(digest)
                    if is_fusion:
                        check(
                            ck["test_window_keys"] == [list(k) for k in test_keys]
                            and all(checkpoint[k] == v for k, v in ident.items()),
                            "Fusion file identity",
                        )
                        check(sum(v.numel() for v in st.values()) == arch_params[model], "State size")
                        if model == "M1-concat":
                            check(
                                tuple(st["quality_gate.weight"].shape) == (128, 256)
                                and "quality_gate.bias" not in st,
                                "Concat projection",
                            )
                        else:
                            check(
                                "quality_gate.q_mlp.0.weight" in st and "quality_gate.weight" not in st,
                                "Implicit gate",
                            )
                        pred = ck["predictions"]
                        p = pred["trajs"].numpy()
                        q = pred["probs"].numpy()
                        e = pred["errors"].numpy()
                        check(
                            p.shape == (len(ix), 6, 30, 2)
                            and q.shape == (len(ix), 6)
                            and e.shape == (len(ix), 2)
                            and np.isfinite(p).all()
                            and np.isfinite(q).all()
                            and (q >= 0).all(),
                            "Forecast shape and values",
                        )
                        gap = float(abs(q.sum(1) - 1).max())
                        max_prob_gap = max(max_prob_gap, gap)
                        check(gap < 2e-6, "Normalization")
                        winner = p[np.arange(len(ix)), q.argmax(1)]
                        distances = np.linalg.norm(winner.astype(float) - truth[ix].astype(float), axis=2)
                        rebuilt = np.column_stack((distances.mean(1), distances[:, -1]))
                        max_point_gap = max(max_point_gap, float(abs(rebuilt - e).max()))
                        np.testing.assert_allclose(rebuilt, e, atol=5e-6, rtol=1e-6)
                    else:
                        e = ck["errors"].numpy()
                    check(e.shape == (len(ix), 2) and np.isfinite(e).all(), "All errors present")
                    errors[ix] = e
                    expected = r["site_metrics"][str(t)][s][model]
                    np.testing.assert_allclose(
                        errors[ix].mean(0), [expected[k] for k in ("ade_m", "fde_m")], rtol=0, atol=1e-12
                    )
                totals[t, model] = errors
                site_values = {s: errors[sites == s].mean(0) for s in names}
                by_site.setdefault(str(t), {})[model] = {s: metrics(v) for s, v in site_values.items()}
                macro = np.mean(list(site_values.values()), axis=0)
                means.setdefault(str(t), {})[model] = metrics(macro)
                source = r["four_site_equal" if is_fusion else "four_site_macro"][str(t)][model]
                np.testing.assert_allclose(
                    macro, [source[k] for k in ("ade_m", "fde_m")], rtol=0, atol=1e-12
                )
        z.close()
    contrasts = {}
    for t in DURATIONS:
        contrasts[str(t)] = {
            a + "_minus_" + b: contrast(totals[t, a] - totals[t, b], sites, clips)
            for a, b in (
                ("M1b", "M1-concat"),
                ("M1b", "M1"),
                ("M1-concat", "M1"),
                ("M1b", "M0"),
                ("M1-concat", "M0"),
                ("M1", "M0"),
            )
        }
    public = {
        "archive_identity": {
            "fusion": FUSION,
            "base": BASE,
            "matched_pose": CACHE,
            "roster": ROSTER,
            "history": HISTORY,
        },
        "versions": {"fusion": 373591607, "base": 373585040},
        "frozen_fusion_code_revision": REVISION,
        "protocol": protocol,
        "allocated_backend": "Separate Kaggle jobs; frozen programs require CUDA and torch 2.11.0+cu130, train on cuda:0. Model-specific GPU UUID and worker identity are not embedded in the result ZIP.",
        "cohort": {
            "windows": len(order),
            "clips": len(set(clips)),
            "sites": counts,
            "site_clips": {s: len(set(clips[sites == s])) for s in names},
        },
        "audit": {
            "fusion_checkpoint_states": 24,
            "total_distinct_parameter_states_in_two_archives": len(states),
            "fusion_prediction_windows": len(order) * len(FUSION_ARMS) * len(DURATIONS),
            "fusion_candidate_trajectories": len(order) * len(FUSION_ARMS) * len(DURATIONS) * 6,
            "source_raw_position_pairs_checked": len(order) * 50,
            "saved_vs_rebuilt_top_mode_error_max_m": max_point_gap,
            "probability_row_sum_max_abs": max_prob_gap,
            "upstream_cloud_report": production_check,
            "local_forward_or_fit": False,
        },
        "four_site_equal_m": means,
        "by_site_m": by_site,
        "paired_contrasts": contrasts,
        "interval": "Each model is fixed at seed 42; within each of the four fixed sites resample all its original clips with replacement, keep all selected windows and average sites equally; PCG64 seed 2026, 20,000 replicates, percentile 95%.",
        "comparison_scope": "M1b vs M1-concat is within the fusion job. M0 vs M1 is within the prior base job. Comparisons across the two archives are paired on the same labeled windows and use identical training split IDs, duration and protocol, but architecture differences cannot be isolated from run/worker differences. No new evaluation sites.",
    }
    encoded = json.dumps(public, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion", type=Path, required=True, help="Original matched_fusion.zip")
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Original matched_durations.zip including matched_observations.npz",
    )
    parser.add_argument(
        "--histories", type=Path, required=True, help="Frozen strict-cohort coordinate manifest NPZ"
    )
    parser.add_argument("--metadata", type=Path, required=True, help="PINNS root containing labels/")
    parser.add_argument(
        "--source-tree",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository tree containing the six original training scripts",
    )
    parser.add_argument("--output", type=Path, help="Write only aggregate metrics (default: stdout)")
    run(parser.parse_args())
