"""Verify the seed-42 ShortPose four-site cloud result ZIP. See --help."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import torch


SOURCE_SHA = "e61a2310793e900c6188588c750e47c44e23fbf41f429fb94ccf8b2ea88fd094"
LABELS_SHA = "0b276b34ab1b01dae91f67c56bfaea5cc931511b05b4fc1c92686a891e6f6801"
IDS_SHA = "f5ee511a54ad818499d3dde6213f7f3b49332943ff2fc93d93aed87d316dbb91"
POSE_MODEL_SHA = "5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c"
SITE_COUNTS = {"america_crossroad": 816, "italy_park": 84, "japan_crossroad": 432, "thailand_street": 150}
METRICS = ("ade_m", "fde_m")
ATOL_M, RTOL = 0.0002, 0.00002
DEFAULT_REPO = Path(__file__).resolve().parents[1]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def same_array(actual, expected, name, atol=0., rtol=0.):
    a, b = np.asarray(actual), np.asarray(expected)
    check(a.shape == b.shape, f"{name}: expected shape {b.shape}, got {a.shape}")
    check(np.isfinite(a).all() and np.isfinite(b).all(), f"{name}: non-finite value")
    good = np.isclose(a, b, rtol=rtol, atol=atol)
    if not good.all():
        first = tuple(np.argwhere(~good)[0])
        raise ValueError(f"{name}: {np.count_nonzero(~good)} differences outside tolerance; "
                         f"first at {first}: replay={a[first]} saved={b[first]} "
                         f"(atol={atol}, rtol={rtol})")
    return float(np.max(np.abs(a - b)))


def validate_zip(z):
    check(len(z.namelist()) == len(set(z.namelist())), "ZIP has duplicate member names")
    bad = z.testzip()
    check(bad is None, f"ZIP CRC failed on {bad}")


@contextmanager
def open_bundle(path):
    with zipfile.ZipFile(path) as outer:
        validate_zip(outer)
        if "folds/report.json" in outer.namelist():
            yield outer
        else:
            members = [k for k in outer.namelist() if Path(k).name == "shortpose_heldout.zip"]
            check(len(members) == 1, "Expected the result ZIP or a Kaggle download containing one shortpose_heldout.zip")
            with outer.open(members[0]) as src, tempfile.TemporaryFile() as dst:
                shutil.copyfileobj(src, dst)
                dst.seek(0)
                with zipfile.ZipFile(dst) as inner:
                    validate_zip(inner)
                    yield inner


def errors_by_window(prediction, truth):
    diff = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    distance = np.linalg.norm(diff, axis=-1)
    return np.column_stack((distance.mean(axis=1), distance[:, -1]))


def run_cpu_forward(predictor, name, state, data, test, batch_size):
    model = predictor(name).cpu().eval()
    model.load_state_dict(state, strict=True)
    check(all(torch.isfinite(t).all().item() for t in state.values()), f"{name}: non-finite state")
    inputs = ("x_m", "ego_type_oh") if name == "M0" else ("x_m", "ego_type_oh", "x_p", "q_vec")
    errors = []
    with torch.inference_mode():
        for idx in torch.as_tensor(test).split(batch_size):
            out = model({k: data[k][idx] for k in inputs})
            rank = out["mode_probs"].argmax(dim=-1)
            top1 = out["trajs"][torch.arange(len(idx)), rank]
            errors.append(errors_by_window(top1.numpy(), data["fut_xy"][idx].numpy()))
    return np.concatenate(errors)


def parameter_digest(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        array = tensor.detach().contiguous().numpy()
        digest.update(json.dumps([key, str(array.dtype), array.shape]).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def metrics(values):
    return dict(zip(METRICS, np.mean(values, axis=0).tolist()))


def compare_metric_dict(actual, expected, label):
    return same_array([actual[k] for k in METRICS], [expected[k] for k in METRICS], label, atol=1e-10)


def verify(args):
    sys.path.insert(0, str(args.repo.resolve()))
    from scripts.benchmark_heldout_sites import HISTORY, load_site_windows
    from scripts.train_pinns_heldout_sites import Protocol, file_hash, source_hash, tensorize
    from src.models.predictor import UnstructuredShortPosePredictor

    check(source_hash() == SOURCE_SHA, "The selected repo does not contain the frozen model/reader from the cloud run")
    groups, clip_counts, digest = load_site_windows(args.data_root)
    check(digest == LABELS_SHA and {k: len(v) for k,v in groups.items()} == SITE_COUNTS,
          "Raw labels, calibration, or site counts differ from the registered dataset")
    motion = json.loads((args.repo / "results/pinns_heldout_sites.json").read_text())
    velocity = json.loads((args.repo / "results/pinns_velocity_estimators.json").read_text())
    report_out = {"bundle_sha256": file_hash(args.bundle), "label_and_calibration_sha256": digest,
                  "frozen_source_sha256": SOURCE_SHA, "cpu_torch": str(torch.__version__),
                  "cpu_error_tolerance": {"atol_m": ATOL_M, "rtol": RTOL}, "sites": {}}
    torch.set_num_threads(args.threads)
    with open_bundle(args.bundle) as archive, tempfile.TemporaryDirectory() as tmp:
        read_json = lambda name: json.loads(archive.read(name))
        required = {"pose.npz", "folds/report.json", "logs/summary.log", "logs/pose.log"}
        for site in groups:
            required.add(f"logs/{site}.log")
            required.update(f"folds/{site}/{p}" for p in ("M0.pt", "M1.pt", "fold.json"))
        missing = required.difference(archive.namelist())
        check(not missing, f"Missing ZIP entries: {sorted(missing)}")
        cloud = read_json("folds/report.json")
        check(cloud["total_windows"] == 1482 and set(cloud["sites"]) == set(SITE_COUNTS),
              "Cloud report omits registered test sites/windows")
        pose = Path(tmp)/"pose.npz"
        with archive.open("pose.npz") as src, pose.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        with zipfile.ZipFile(pose) as npz:
            validate_zip(npz)
        data, indices, keys, _, coverage, metadata = tensorize(groups, pose)
        ids_hash = hashlib.sha256(json.dumps(keys,separators=(",",":")).encode()).hexdigest()
        check(ids_hash == IDS_SHA and len(set(keys)) == len(keys) == 1482, "Registered window order or identity changed")
        check(metadata == cloud["pose_metadata"] and metadata["model_sha256"] == POSE_MODEL_SHA
              and metadata["label_and_calibration_digest_sha256"] == LABELS_SHA
              and metadata["site_counts"] == SITE_COUNTS, "Invalid pose model, data provenance, or site counts")
        clips = {row.clip for rows in groups.values() for row in rows}
        check(set(metadata["video_sha256"]) == clips, "Unexpected video digest inventory")
        n_videos = 0
        for clip, expected_digest in metadata["video_sha256"].items():
            condition, number = clip.rsplit("_", 1)
            video = args.data_root/"videos"/condition/"videos"/f"ori_{number}.mp4"
            if video.is_file():
                check(file_hash(video) == expected_digest, f"Video digest mismatch: {clip}")
                n_videos += 1
        identity = cloud["identity"]
        independently_known = {"protocol": asdict(Protocol()), "data_sha256": digest, "source_sha256": SOURCE_SHA,
                               "ordered_ids_sha256": ids_hash, "pose_sha256": file_hash(pose)}
        check(all(identity[k] == v for k,v in independently_known.items()), "Cloud archive run signature mismatch")
        report_out.update(cloud_torch=identity["torch_version"], pose_sha256=identity["pose_sha256"],
                          video_digests_checked=n_videos, videos_with_reported_digest=len(clips),
                          ordered_window_ids_sha256=ids_hash)
        previous_states = set()
        for site, test in indices.items():
            train_sites = sorted(set(groups) - {site})
            test_keys = [list(keys[i]) for i in test]
            fold = read_json(f"folds/{site}/fold.json")
            site_cloud = cloud["sites"][site]
            check(fold["window_keys"] == test_keys and fold["identity"] == identity
                  and fold["error_columns"] == ["ADE1_m", "FDE1_m"], f"{site}: test key order, identity or error columns differ")
            expected = {**clip_counts[site], "train_sites": train_sites, "train_windows": 1482-len(test),
                        "test_windows": len(test), "coverage": coverage[site]}
            check(all(site_cloud[k] == v == fold[k] for k,v in expected.items()),
                  f"{site}: split, count, or coverage mismatch")
            check(site_cloud["metrics"] == fold["metrics"], f"{site}: cloud fold and report differ")
            xy = np.stack([r.xy for r in groups[site]])
            # Five equally spaced samples yield the centered least-squares slope weights [-2,-1,0,1,2]/10.
            displacements = np.einsum("t,ntd->nd", np.arange(-2.,3.) / 10., xy[:,:HISTORY])
            last_step = xy[:,HISTORY-1] - xy[:,HISTORY-2]
            horizon = np.arange(1, 31)[None,:,None]
            future = xy[:,HISTORY:]
            ols = errors_by_window(displacements[:,None,:]*horizon, future)
            last2 = errors_by_window(last_step[:,None,:]*horizon, future)
            saved_rows = fold["paired_window_errors"]
            same_array(ols, saved_rows["constant_velocity"], site+" five-point CV", atol=1e-10)
            compare_metric_dict(metrics(ols), motion["sites"][site]["constant_velocity"], site+" archived OLS")
            compare_metric_dict(metrics(last2), velocity["sites"][site]["metrics"]["last_two"], site+" archived two-point CV")
            all_means = {"constant_velocity":metrics(ols), "last_two":metrics(last2)}
            max_error, weights, parameters, used_device = {}, {}, {}, {}
            for name in ("M0", "M1"):
                raw = archive.read(f"folds/{site}/{name}.pt")
                with zipfile.ZipFile(io.BytesIO(raw)) as pt_zip:
                    validate_zip(pt_zip)
                weights[name] = hashlib.sha256(raw).hexdigest()
                checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                check(checkpoint["identity"] == {**identity,"model":name,"test_site":site,"training_sites":train_sites}
                      and checkpoint["window_keys"] == test_keys, f"{site}/{name}: incorrect split/IDs in checkpoint")
                parameters[name] = parameter_digest(checkpoint["model_state"])
                check(parameters[name] not in previous_states, f"{site}/{name}: repeated network parameters across folds")
                previous_states.add(parameters[name])
                used_device[name] = checkpoint["device"]
                check(used_device[name] == fold["training_devices"][name] == site_cloud["training_devices"][name],
                      f"{site}/{name}: training device metadata differs")
                saved = np.asarray(saved_rows[name], dtype=np.float64)
                same_array(checkpoint["errors"].numpy(), saved, f"{site}/{name} checkpoint vs fold")
                replay = run_cpu_forward(UnstructuredShortPosePredictor, name, checkpoint["model_state"], data, test, Protocol().batch_size)
                max_error[name] = same_array(replay, saved, f"{site}/{name} CPU vs cloud", atol=ATOL_M, rtol=RTOL)
                all_means[name] = metrics(saved)
            for name, value in fold["metrics"].items():
                compare_metric_dict(all_means[name], value, f"{site}/{name} fold means")
            report_out["sites"][site] = {**expected,"metrics_m":all_means,"checkpoint_sha256":weights,
                                        "cpu_vs_cloud_max_abs_m":max_error,"training_devices":used_device,
                                        "network_parameters_sha256":parameters}
            print(f"{site}: {len(test)} window IDs, two checkpoints, all paired ADE/FDE verified",file=sys.stderr,flush=True)

        by_site = report_out["sites"]
        names = ("constant_velocity", "last_two", "M0", "M1")
        macro = {m:{k:float(np.mean([row["metrics_m"][m][k] for row in by_site.values()]))
                    for k in METRICS} for m in names}
        for name in ("constant_velocity","M0","M1"):
            compare_metric_dict(macro[name],cloud["macro"][name],name+" cloud macro")
        compare_metric_dict(macro["last_two"], velocity["macro"]["last_two"], "archived last-two macro")
        for row in [*by_site.values(), report_out]:
            m = row.get("metrics_m", macro)
            row["differences_m"] = {
                f"{a}_minus_{b}": {k:m[a][k]-m[b][k] for k in METRICS}
                for a,b in [("M0","last_two"),("M1","last_two"),("M1","M0")]
            }
        report_out.update(macro_m=macro, verified_windows=1482, verified_checkpoints=8,
                          verification="All saved ADE and FDE rows reproduced by CPU inference from each held-out-site "
                                       "checkpoint. Fold design checked against frozen code, labels and recorded metadata.")
    return report_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle",type=Path,help="shortpose_heldout.zip or Kaggle outer download ZIP")
    parser.add_argument("data_root",type=Path,help="Original labels/, videos/<condition>/calib.json; optional videos or archive_members.txt")
    parser.add_argument("--repo",type=Path,default=DEFAULT_REPO,help="Checkout of the frozen predictor, revision a40fdd0")
    parser.add_argument("--output",type=Path,help="Write JSON report (default: stdout)")
    parser.add_argument("--threads",type=int,default=2,help="CPU inference threads (default: 2)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    output = json.dumps(verify(args),indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output,end="")
