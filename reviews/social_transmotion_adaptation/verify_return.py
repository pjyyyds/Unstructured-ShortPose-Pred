"""Independent read-time audit of stored scores. Requires the official labels, previously checked windows and returned ZIP."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from scripts.benchmark_heldout_sites import load_site_windows


def digest(x):
    return hashlib.sha256(x).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def check(works, reason):
    if not works:
        raise ValueError(reason)


ARCHIVE_SHA = "38f1777735602623006a27ebbe2c041c79d9339af89b3992d5912a8681dc7d94"


def verify(archive, labels, prior):
    source = Path(__file__).resolve().parent
    archive_sha = digest(archive.read_bytes())
    check(archive_sha == ARCHIVE_SHA, "Expected the registered Kaggle version 373628213 archive")
    ax = ("ade_m", "fde_m")
    pairs = (("traj_box", "last_two"), ("traj", "last_two"), ("traj_box", "traj"))
    groups, _, label_sha = load_site_windows(labels)
    keys = {s: [(r.clip, r.person_id, r.t0) for r in rows] for s, rows in groups.items()}
    site_names = sorted(keys)
    with np.load(prior, allow_pickle=False) as f:
        known = dict(
            zip(
                zip(f["clip"].tolist(), f["person_id"].tolist(), f["t0"].tolist()),
                zip(f["history_20_xy"][:, -5:], f["future_30_xy"]),
            )
        )
    enroll = [k for s in site_names for k in keys[s]]
    check(
        len(known) == len(enroll) == len(set(enroll)) == 1482 and set(enroll) == set(known),
        "two-source enrollment",
    )
    with ZipFile(archive) as z:
        scores = json.loads(z.read("scores.json"))
        run = json.loads(z.read("run.json"))
        plan = run["plan"]
        members = json.loads(z.read("sha256.json"))
        check(z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == 19, "CRC/roster")
        check(
            set(z.namelist()) == set(members) | {"sha256.json"}
            and all(digest(z.read(n)) == v for n, v in members.items()),
            "member hashes",
        )
        check(plan == json.loads((source / "candidate_plan.json").read_bytes()), "frozen plan")
        check(
            all(z.read(n) == (source / n).read_bytes() for n in plan["code_sha256"]),
            "driver and notice bytes",
        )
        check(
            plan["keys_sha256"] == canonical(enroll) and plan["data_sha256"] == label_sha,
            "label and window hashes",
        )
        check(
            canonical(plan) == scores["plan_sha256"]
            and scores["seed"] == 42
            and scores["fit_count"] == 8
            and plan["code_revision"] == scores["source_revision"],
            "registered fit count / model adaptation",
        )
        e = {}
        clips = {}
        max_err = 0.0
        max_summary = 0.0
        inits = set()
        header = None
        model_digests = set()

        def equal_numeric(a, b):
            nonlocal max_summary
            diff = np.max(np.abs(np.asarray(a) - np.asarray(b)))
            max_summary = max(max_summary, float(diff))
            np.testing.assert_allclose(a, b, rtol=0, atol=2e-11)

        for s in site_names:
            site_keys = keys[s]
            train = [k for t in site_names if t != s for k in keys[t]]
            fold = plan["folds"][s]
            check(
                fold["train_keys"] == canonical(train) and fold["test_keys"] == canonical(site_keys),
                "training/test identities",
            )
            check(
                fold["test_windows"] == len(site_keys) and fold["train_windows"] == len(train), "fold sizes"
            )
            check(
                not ({k[0] for k in train} & {k[0] for k in site_keys}), "training/test clip independence"
            )
            check(scores["site_windows"][s] == len(site_keys), "test-window count")
            ground = np.stack([r.xy for r in groups[s]])
            history, future = ground[:, :5], ground[:, 5:]
            # The second file is an independent local construction in float64; both source paths must match.
            np.testing.assert_array_equal(history, np.stack([known[k][0] for k in site_keys]))
            np.testing.assert_array_equal(future, np.stack([known[k][1] for k in site_keys]))
            with np.load(io.BytesIO(z.read(f"{s}/private_windows.npz")), allow_pickle=False) as f:
                check(
                    list(zip(f["clip"].tolist(), f["person_id"].tolist(), f["t0"].tolist())) == site_keys,
                    "fold key order",
                )
                clips[s] = f["clip"]
                e[s] = {}
                check(len(set(clips[s])) == fold["test_clips"], "test-clip count")
                for method in ("traj_box", "traj", "last_two"):
                    if method == "last_two":
                        v = (history[:, 4] - history[:, 3]) / 0.1
                        pred = history[:, 4, None] + v[:, None] * (0.1 * np.arange(1, 31)[:, None])
                    else:
                        pred = f[method + "_paths"].astype(np.float64)
                        ck = torch.load(
                            io.BytesIO(z.read(f"{s}/{method}.pt")), weights_only=True, map_location="cpu"
                        )
                        check(
                            ck["identity"]
                            == dict(
                                plan_sha256=canonical(plan),
                                worker=run["worker"],
                                site=s,
                                arm=method,
                                split=fold,
                            ),
                            "checkpoint registration",
                        )
                        check(len(ck["epochs"]) == 28, "28 complete epochs")
                        for j, log in enumerate(ck["epochs"]):
                            check(
                                log["epoch"] == j + 1
                                and log["windows"] == len(train)
                                and np.isfinite(log["objective"])
                                and log["objective"] >= 0,
                                "epoch and training loss",
                            )
                            check(log["lr"] == (1e-4 if j < 23 else 1e-5), "learning rate")
                        inits.add(ck["initial_sha256"])
                        blobs = ck["model_state"]
                        sch = [(k, tuple(t.shape), str(t.dtype)) for k, t in sorted(blobs.items())]
                        if header is None:
                            header = sch
                        check(
                            sch == header and all(torch.isfinite(t).all() for t in blobs.values()),
                            "finite state and common architecture",
                        )
                        state_hash = hashlib.sha256()
                        for name, tensor in sorted(blobs.items()):
                            state_hash.update(name.encode() + b"\0")
                            state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                        model_digests.add(state_hash.hexdigest())
                    check(pred.shape == (len(site_keys), 30, 2) and np.isfinite(pred).all(), "test paths")
                    d = np.hypot(pred[..., 0] - future[..., 0], pred[..., 1] - future[..., 1])
                    computed = np.stack([d.sum(1) / 30, d[:, 29]], axis=-1)
                    actual = f[method + "_errors"]
                    np.testing.assert_allclose(computed, actual, rtol=0, atol=1e-11)
                    max_err = max(max_err, float(np.max(abs(computed - actual))))
                    e[s][method] = computed
                    equal_numeric(computed.mean(0), [scores["by_site"][s][method][m] for m in ax])
        check(
            len(model_digests) == 8 and inits == {scores["initial_sha256"]},
            "eight distinct fitted states and common recorded initialization",
        )
        for method in ["last_two", "traj_box", "traj"]:
            macro = np.mean([e[s][method].mean(0) for s in site_names], 0)
            equal_numeric(macro, [scores["four_site_equal"][method][m] for m in ax])
        for a, b in pairs:
            rng = np.random.Generator(np.random.PCG64(2026))
            site_draw = []
            points = []
            expected = scores["paired"][a + "_minus_" + b]
            for s in site_names:
                diff = e[s][a] - e[s][b]
                names = sorted(set(clips[s]))
                per = [diff[clips[s] == n] for n in names]
                ids = rng.integers(len(names), size=(20000, len(names)))
                weights = np.stack([(ids == k).sum(1) for k in range(len(names))], axis=1)
                block = weights @ np.stack([m.sum(0) for m in per])
                size = weights @ np.array([len(m) for m in per])
                draw = block / size[:, None]
                points.append(diff.mean(0))
                site_draw.append(draw)
                equal_numeric(diff.mean(0), expected["by_site"][s]["delta_ade_fde_m"])
                for j, m in enumerate(ax):
                    equal_numeric(
                        np.percentile(draw[:, j], (2.5, 97.5)), expected["by_site"][s]["interval95_m"][m]
                    )
            equal_numeric(np.mean(points, 0), expected["site_macro_delta_ade_fde_m"])
            for j, m in enumerate(ax):
                equal_numeric(
                    np.percentile(np.mean(site_draw, 0)[:, j], (2.5, 97.5)),
                    expected["site_macro_interval95_m"][m],
                )
    return dict(
        archive_sha256=archive_sha,
        version=373628213,
        raw_labels_sha256=label_sha,
        prior_keyed_windows_sha256=digest(prior.read_bytes()),
        registered_plan_sha256=canonical(plan),
        source_revision=plan["code_revision"],
        fit_count=8,
        epochs_per_fit=28,
        worker_device=run["worker"]["device"],
        inference="Stored test predictions compared against source labels; no model forwards.",
        sites={s: dict(windows=len(keys[s]), clips=len(set(clips[s]))) for s in site_names},
        checked_window_error_rows=1482 * 3,
        common_initial_sha256=scores["initial_sha256"],
        max_absolute_stored_error_difference_m=max_err,
        max_absolute_report_or_interval_difference_m=max_summary,
        official_aggregate_sha256=members["scores.json"],
        four_site_equal=scores["four_site_equal"],
        paired_macro={
            k: dict(delta=v["site_macro_delta_ade_fde_m"], interval95_m=v["site_macro_interval95_m"])
            for k, v in scores["paired"].items()
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ["archive", "labels", "prior", "output"]:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    r = verify(a.archive, a.labels, a.prior)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r))
