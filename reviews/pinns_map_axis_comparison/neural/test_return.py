"""Build tiny private-format synthetic returns and check the offline verifier. No model runs."""

import copy
import hashlib
import io
import json
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch


ARMS = ("zero_xy", "real_xy")
METRICS = ("ade_m", "fde_last_valid_m", "fde_exact_3s_m")
SITES = ("america_crossroad", "italy_park", "japan_crossroad", "thailand_street")
DIFFERENCES = (1, 1, 1, 5, -1, -3, 2, 2)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def npz(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def packet(members):
    stream = io.BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        for k, v in members.items():
            archive.writestr(k, v)
        archive.writestr("sha256.json", encode({k: sha(v) for k, v in members.items()}))
    return stream.getvalue()


def fixture_fields():
    sites = np.array([SITES[i] for i in (0, 0, 0, 0, 1, 2, 3, 3)])
    clips = np.array([f"{s}_synthetic_{int(i == 3)}" for i, s in enumerate(sites)])
    mask = np.zeros((8, 30), dtype=bool)
    truths = np.zeros((8, 30, 2), dtype="<f4")
    for i, end in enumerate((30, 16, 28, 30, 30, 24, 17, 30)):
        mask[i, :end] = True
        mask[i, 2] = False  # An internal observation gap differs from trailing padding.
        for j in np.flatnonzero(mask[i]):
            truths[i, j] = (j * 0.5, i)
    has_map = np.arange(8) != 2
    points = np.zeros((8, 16, 10), dtype=bool)
    points[has_map, 0, :] = True
    return {
        "clip": clips,
        "person_id": np.arange(8, dtype="<i8"),
        "t0": np.arange(15, 23, dtype="<i8"),
        "site": sites,
        "hist_mask": np.ones((8, 5), bool),
        "geometry_source_available": has_map,
        "geometry_point_mask": points,
        "fut_xy": truths,
        "fut_mask": mask,
    }


def synthetic_group(rows, errors, f):
    summary = {}
    for arm, values in errors.items():
        quantities = {}
        for j, metric in enumerate(METRICS):
            data = [values[i][j] for i in rows if values[i][j] is not None]
            quantities[metric] = {
                "sum_m": sum(data),
                "windows": len(data),
                "mean_m": sum(data) / len(data) if data else None,
            }
        summary[arm] = quantities
    return {
        "windows": len(rows),
        "clips": len(set(f["clip"][rows])),
        "no_source": int((~f["geometry_source_available"][rows]).sum()),
        "no_local_image": sum(not f["geometry_point_mask"][i].any() for i in rows),
        "scored_future_steps": int(f["fut_mask"][rows].sum()),
        "errors": summary,
    }


@lru_cache(maxsize=2)
def artificial_checkpoint(arm_index):
    from src.models.predictor import UnstructuredShortPosePredictor

    # Only take parameter names, shape and dtype. No model predictions or fitted weights.
    with torch.device("meta"):
        template = UnstructuredShortPosePredictor("M2a")
    state = {k: torch.full(v.shape, arm_index, dtype=v.dtype) for k, v in template.state_dict().items()}
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        arr = np.ascontiguousarray(value.numpy())
        digest.update(encode([name, arr.dtype.str, list(arr.shape)]) + b"\0" + arr.tobytes())
    raw = io.BytesIO()
    torch.save(state, raw)
    return raw.getvalue(), digest.hexdigest(), sum(p.numel() for p in template.parameters())


def design():
    f = fixture_fields()
    rows = list(range(len(f["clip"])))
    support = {}
    for split, total, clips, count in (
        ("train", 5686, 25, 3499),
        ("val", 1100, 40, 965),
        ("test", 8, len(set(f["clip"])), 7),
    ):
        support[split] = dict(
            windows=total,
            clips=clips,
            historical_reference=count,
            local_image=count,
            no_reference=total - count,
            no_local_image=total - count,
        )
    key_sha = {sp: sha(sp.encode()) for sp in support}
    key_sha["test"] = sha(
        encode([[str(f["clip"][i]), int(f["person_id"][i]), int(f["t0"][i])] for i in rows])
    )
    recipe = dict(
        model_name="M2a",
        seed=42,
        device_str="cuda:0",
        epochs=28,
        batch_size=512,
        lr=0.002,
        dt=0.1,
        fut_steps=30,
        balanced_buckets=True,
    )
    expected = {
        k: sha(k.encode())
        for k in (
            "protocol_sha256",
            "source_map_sha256",
            "observed_image_payload_sha256",
            "source_motion_sha256",
            "full_source_motion_sha256",
            "ordered_window_keys_sha256",
            "runner_sha256",
        )
    }
    expected.update(
        source_sha256={"models/example.py": sha(b"fixture")}, recipe=recipe, split_key_sha256=key_sha
    )
    identity = copy.deepcopy(expected)
    identity.update(
        tensor_sha256={sp: sha((sp + "_tensors").encode()) for sp in support},
        execution=dict(
            gpu="Synthetic CUDA device",
            capability=[7, 5],
            device_index=0,
            worker_device_sha256=sha(b"fixture worker"),
            gpu_bytes=1_000_000,
            torch="2.10.0",
            cuda="12.8",
            cudnn=91002,
            numpy="2.1.0",
            python="3.12.0",
            deterministic_algorithms=True,
            tf32=False,
            cublas_workspace=":4096:8",
        ),
    )
    expected["base_support"] = copy.deepcopy(support)
    for item in support.values():
        item.update(
            windows_with_any_valid_pose_joint=item["windows"],
            frames_with_any_valid_pose_joint=item["windows"] * 5,
        )

    fits, outputs, errors, weights = {}, {}, {}, {}
    for ai, arm in enumerate(ARMS):
        weights[arm], fitted_sha, parameters = artificial_checkpoint(ai)
        validation = [1.25, 1.0, 0.75, 0.5, 0.5, 0.8, 1.0]  # First minimum is at epoch 16.
        fits[arm] = dict(
            arm=arm,
            identity_sha256=sha(encode(identity)),
            initial_state_sha256=sha(b"initial"),
            sampled_indices_sha256=sha(b"paired draws"),
            sampled_windows=5686 * 28,
            minibatches=12 * 28,
            validation_ade_by_checkpoint_m=validation,
            best_epoch=16,
            best_val_ade_m=0.5,
            parameters=parameters,
            train_seconds=1.0,
            best_state_sha256=fitted_sha,
        )
        paths = np.full((8, 6, 30, 2), 512.0, dtype="<f4")
        probs = np.zeros((8, 6), dtype="<f4")
        errors[arm] = []
        for i in rows:
            # Modes change by arm and row. On row zero the lower-index tied candidate is correct.
            chosen = (i + ai) % 6
            probs[i, chosen] = 1.0
            if i == 0:
                probs[i, [chosen, 5]] = 0.5
            distances = []
            for j in np.flatnonzero(f["fut_mask"][i]):
                scale = float(2 + i + j / 4)
                distance = 5 * scale + ai * DIFFERENCES[i]
                offset = (distance, 0.0) if ai else (3 * scale, 4 * scale)
                paths[i, chosen, j] = f["fut_xy"][i, j] + offset
                distances.append(distance)
            errors[arm].append(
                (
                    sum(distances) / len(distances),
                    distances[-1],
                    distances[-1] if f["fut_mask"][i, -1] else None,
                )
            )
        outputs[arm] = dict(paths=paths, probabilities=probs)
    errors["real_minus_zero"] = [
        (float(d), float(d), float(d) if f["fut_mask"][i, -1] else None) for i, d in enumerate(DIFFERENCES)
    ]
    primary = dict(estimate_m=1.0, clip_bootstrap_percentile_95_m=[0.5, 2.5], replicates=20000, seed=2026)
    public = dict(
        identity_sha256=sha(encode(identity)),
        protocol_sha256=expected["protocol_sha256"],
        fits=fits,
        backend=identity["execution"],
        registered_sample_support=support,
        primary_test_paired_ade=primary,
        pooled=synthetic_group(rows, errors, f),
    )
    for field in ("site", "clip"):
        public["by_" + field] = {
            s: synthetic_group([i for i in rows if f[field][i] == s], errors, f)
            for s in sorted(set(f[field]))
        }
    return expected, identity, public, f, fits, weights, outputs


def build_return(path, change=None):
    expected, identity, public, f, fits, weights, outputs = design()
    if change:
        change(public, f, fits, outputs)
    outer = {"identity.json": encode(identity), "aggregate.json": encode(public), "test_rows.npz": npz(f)}
    for arm in ARMS:
        outer[arm + ".zip"] = packet(
            {
                "fit.json": encode(fits[arm]),
                "best_state.pt": weights[arm],
                "predictions.npz": npz(outputs[arm]),
            }
        )
    path.write_bytes(packet(outer))
    return expected


class SyntheticReturnTest(unittest.TestCase):
    def test_accepts_golden_and_rejects_rehashed_semantic_errors(self):
        from . import verify_return as verifier

        def permuted_keys(public, fields, fits, outputs):
            fields["person_id"][[0, 1]] = fields["person_id"][[1, 0]]

        def wrong_winner(public, fields, fits, outputs):
            outputs["real_xy"]["probabilities"][0] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        def bad_normalization(public, fields, fits, outputs):
            outputs["zero_xy"]["probabilities"][1] *= 0.75

        def exact_endpoint_denominator(public, fields, fits, outputs):
            public["pooled"]["errors"]["real_xy"]["fde_exact_3s_m"]["windows"] = 8

        def last_valid_replaced_by_exact_endpoint(public, fields, fits, outputs):
            stats = public["pooled"]["errors"]["zero_xy"]
            stats["fde_last_valid_m"] = dict(stats["fde_exact_3s_m"])

        def wrong_geometry_source(public, fields, fits, outputs):
            fields["geometry_source_available"][2] = True

        def use_site_macro(public, fields, fits, outputs):
            public["primary_test_paired_ade"]["estimate_m"] = 0.0

        def use_window_interval(public, fields, fits, outputs):
            public["primary_test_paired_ade"]["clip_bootstrap_percentile_95_m"] = [-0.5, 3.0]

        def later_tied_epoch(public, fields, fits, outputs):
            fits["real_xy"]["best_epoch"] = 20

        def missing_checkpoint(public, fields, fits, outputs):
            fits["zero_xy"]["validation_ade_by_checkpoint_m"].pop()

        def unpaired_draws(public, fields, fits, outputs):
            fits["real_xy"]["sampled_indices_sha256"] = sha(b"other draws")

        def fake_equal_parameter_count(public, fields, fits, outputs):
            for arm in ARMS:
                fits[arm]["parameters"] += 1

        with TemporaryDirectory() as tmp, ExitStack() as patches:
            root = Path(tmp)
            path = root / "synthetic.zip"
            e = build_return(path)
            # Supply independent eight-window ground truth at the registration boundary only.
            # Archive integrity, weight schema/values, model selection and all score checks stay active.
            (root / "input_identity").mkdir()
            (root / "input_check.json").write_bytes(
                encode(
                    {
                        "source_label_maps_sha256": e["source_map_sha256"],
                        "ordered_keys_sha256": e["ordered_window_keys_sha256"],
                        "motion_and_masks_sha256": e["source_motion_sha256"],
                        "splits": e["base_support"],
                        "geometry_and_source_availability_sha256": e["observed_image_payload_sha256"],
                    }
                )
            )
            (root / "input_identity/verified.json").write_bytes(
                encode({"full_20_frame_absolute_sha256": e["full_source_motion_sha256"]})
            )
            for attr, value in (
                ("STUDY", root),
                ("PROTOCOL", e["protocol_sha256"]),
                ("RELEASE_RUNNER", e["runner_sha256"]),
                ("RELEASE_SOURCES", sha(encode(e["source_sha256"]))),
            ):
                patches.enter_context(patch.object(verifier, attr, value))
            patches.enter_context(
                patch.object(
                    verifier, "truth_and_maps", return_value=(fixture_fields(), e["split_key_sha256"])
                )
            )

            def verify():
                return verifier.verify_return(path, root / "synthetic_ridge", root / "synthetic_maps")

            result = verify()
            self.assertAlmostEqual(result["primary_test_paired_ade"]["estimate_m"], 1.0)
            self.assertEqual((result["test_windows"], result["exact_3s_windows"]), (8, 4))
            np.testing.assert_allclose(
                result["primary_test_paired_ade"]["clip_bootstrap_percentile_95_m"], [0.5, 2.5]
            )
            for edit, reason in (
                (permuted_keys, "person_id"),
                (wrong_winner, "primary ADE"),
                (bad_normalization, "sum to one"),
                (exact_endpoint_denominator, "fde_exact_3s_m.windows"),
                (last_valid_replaced_by_exact_endpoint, "fde_last_valid_m"),
                (wrong_geometry_source, "geometry_source_available"),
                (use_site_macro, "primary ADE.estimate_m"),
                (use_window_interval, "primary ADE.clip_bootstrap"),
                (later_tied_epoch, "first best validation"),
                (missing_checkpoint, "seven validation"),
                (unpaired_draws, "unpaired sampled_indices"),
                (fake_equal_parameter_count, "schema and parameter count"),
            ):
                with self.subTest(edit=edit.__name__):
                    build_return(path, edit)
                    with self.assertRaisesRegex(ValueError, reason):
                        verify()


if __name__ == "__main__":
    unittest.main()
