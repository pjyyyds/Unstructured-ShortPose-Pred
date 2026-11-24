"""Locked IMPTC sample motion protocol. The default verifies source, clocks, identities, and weights only."""
import argparse
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np

SAMPLE_MD5 = "d2bf9579b127b74927150bd3666a034f"
BUNDLE_SHA = "0bf2745ab2d5ddcf5f5fbb5d330e6c6cc800625cec301f4c7105d031bdb49d2d"
FROZEN_SOURCE = "e61a2310793e900c6188588c750e47c44e23fbf41f429fb94ccf8b2ea88fd094"
SEQS = ("0000_20230322_081506", "0001_20230523_105516", "0002_20230523_111106", "0003_20230523_111514")
FOLDS = {"america_crossroad": 816, "italy_park": 84, "japan_crossroad": 432, "thailand_street": 150}
KNOWN_PERSON_COUNT = (3, 22, 22, 14)
STEP_US = 40_000
NATIVE_TICKS = 86
NOMINAL_FROM_ANCHOR = np.arange(-4, 31, dtype=np.int64)*100_000
METRICS = ("ade_m", "fde_m")


def file_digest(path, method="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, method).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


@dataclass
class Window:
    seq: str
    track: str
    ticks: tuple
    native_first: int
    native_last: int


def selected_windows(master, accepted, seq, person, reasons):
    windows = []
    end_previous = -1
    # Every fifth native master tick defines the phase. Tick +10 is the observed trigger.
    for first in range(0, len(master)-NATIVE_TICKS+1, 5):
        source = master[first:first+NATIVE_TICKS]
        if source[0] <= end_previous:
            reasons["overlaps_previously_retained"] += 1
            continue
        if not all(t in accepted for t in source):
            reasons["incomplete_accepted_native_support"] += 1
            continue
        anchor = source[10]
        chosen = []
        for off in NOMINAL_FROM_ANCHOR:
            want = anchor + int(off)
            i = bisect_left(master, want)
            neighbours = master[max(0,i-1):i+1]
            chosen.append(min(neighbours,key=lambda t:(abs(t-want),t)))
        ticks = np.asarray(chosen,dtype=np.int64)
        if (len(set(chosen)) != len(chosen) or chosen[0] != source[0] or chosen[-1] != source[-1]
                or chosen[4] != anchor or np.max(np.abs(ticks-anchor-NOMINAL_FROM_ANCHOR)) > 20_000):
            reasons["timing_tolerance_or_support"] += 1
            continue
        end_previous = source[-1]
        windows.append(Window(seq,person,tuple(chosen),source[0],source[-1]))
    return windows


def read_sample(path, coordinate_ticks=None):
    """Registration uses timestamp and status only. XY is accessed solely in explicitly requested passes."""
    if path.stat().st_size != 347692691 or file_digest(path, "md5") != SAMPLE_MD5:
        raise ValueError("Changed official four-sequence archive")
    clocks, records = {}, {}
    tagged = Counter()
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) == 1:
                continue
            if parts[0] not in SEQS:
                raise ValueError("Unexpected sample sequence")
            if parts[1:] == ("context", "master_timestamp_sync.json"):
                raw = json.load(archive.extractfile(member))
                ticks = sorted(map(int, raw))
                if [raw[str(t)]["id"] for t in ticks] != list(range(len(ticks))):
                    raise ValueError("Sequence master indices are inconsistent")
                clocks[parts[0]] = ticks
            if len(parts) != 4 or parts[1] != "vrus" or parts[3] != "track.json":
                continue
            track = json.load(archive.extractfile(member))
            tag = track["overview"]
            is_person = tag["class_id"] == 0
            if is_person != (tag["class_name"].strip().lower() in ("pedestrian", "person")):
                raise ValueError("Inconsistent pedestrian class code and name")
            if not is_person:
                continue
            tagged[parts[0]] += 1
            key = (parts[0], parts[2])
            if key in records:
                raise ValueError("Duplicate track ID in sequence")
            sample = {}
            for r in track["track_data"].values():
                tick = int(r["ts"])
                if tick in sample:
                    raise ValueError("Duplicate timestamp within track")
                requested = coordinate_ticks is not None and tick in coordinate_ticks.get(key, ())
                sample[tick] = (r["status"], r["coordinates"][:2] if requested else None)
            records[key] = sample
    if set(clocks) != set(SEQS) or tuple(tagged[s] for s in SEQS) != KNOWN_PERSON_COUNT:
        raise ValueError("Source sequences or pedestrian tracks changed")
    return clocks, records


def prepare(path):
    clocks, tracks = read_sample(path)
    windows, counts = [], {}
    for seq in SEQS:
        master = clocks[seq]
        gaps = np.diff(master)
        if np.any(gaps <= 0) or np.any(np.abs(gaps - STEP_US) >= STEP_US//2):
            raise ValueError("Gap or clock reset in sequence master")
        reasons = Counter()
        track_count = 0
        win_count = 0
        for (s, person), rows in sorted(tracks.items()):
            if s != seq:
                continue
            if not set(rows) <= set(master):
                raise ValueError("An annotated timestamp is outside the sequence master")
            accepted = {t for t, (status, _) in rows.items() if status == 2}
            retained = selected_windows(master,accepted,seq,person,reasons)
            track_count += bool(retained)
            win_count += len(retained)
            windows.extend(retained)
        counts[seq] = {"explicit_pedestrian_tracks": KNOWN_PERSON_COUNT[SEQS.index(seq)],
                       "tracks_with_windows": track_count, "windows": win_count, "master_ticks": len(master),
                       "master_gap_microseconds_min_max":[int(gaps.min()),int(gaps.max())],
                       "excluded_candidate_start_counts_in_guard_order":dict(sorted(reasons.items()))}
    if any(c["windows"]==0 for c in counts.values()):
        raise ValueError(f"A predefined demo sequence has no eligible windows: {counts}")
    keys = [[w.seq,w.track,w.ticks[4]] for w in windows]
    covered = set()
    for w in windows:
        ids = {(w.seq,w.track,t) for t in clocks[w.seq] if w.native_first<=t<=w.native_last}
        if len(ids) != NATIVE_TICKS or ids & covered:
            raise ValueError("Incomplete or repeated native timestamps in this person/sequence")
        covered.update(ids)
    table=np.asarray([w.ticks for w in windows],dtype=np.int64)
    delta=table-table[:,4:5]-NOMINAL_FROM_ANCHOR
    def value_range(x):
        return [int(np.min(x)),int(np.max(x))]
    return windows, {"per_sequence": counts, "all_windows": len(windows),
                     "ordered_window_keys_sha256": object_digest(keys),
                     "all_selected_timestamps_sha256":object_digest(table.tolist()),
                     "distinct_track_window_keys":len(set(map(tuple,keys))),
                     "all_source_ticks_per_window_accepted": NATIVE_TICKS, "raw_timestamp_overlap_within_track": 0,
                     "all_timing_errors_us_min_max":value_range(delta),
                     "mean_absolute_timing_error_us":float(np.mean(np.abs(delta))),
                     "max_absolute_timing_error_us":int(np.max(np.abs(delta))),
                     "exact_timestamps":int(np.count_nonzero(delta==0)), "selected_timestamps":int(delta.size),
                     "history_span_us_min_max":value_range(table[:,4]-table[:,0]),
                     "last_history_gap_us_min_max":value_range(table[:,4]-table[:,3]),
                     "endpoint_after_anchor_us_min_max":value_range(table[:,-1]-table[:,4])}


def load_m0(bundle):
    import torch
    from scripts.train_pinns_heldout_sites import source_hash
    from src.models.predictor import UnstructuredShortPosePredictor
    if source_hash() != FROZEN_SOURCE or file_digest(bundle) != BUNDLE_SHA:
        raise ValueError("Unexpected PINNS predictor source or five-seed weight archive")
    loaded, hashes = {}, {}
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("sha256.json"))
        all_keys = set()
        for site, count in FOLDS.items():
            name = f"seed_42/{site}/M0.pt"
            raw = archive.read(name)
            hashes[site] = hashlib.sha256(raw).hexdigest()
            if hashes[site] != manifest[name]:
                raise ValueError("Frozen weight checksum mismatch")
            ck = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            identity = ck["identity"]
            keys = set(map(tuple, ck["window_keys"]))
            if (identity["model"] != "M0" or identity["protocol"]["seed"] != 42 or identity["protocol"]["epochs"] != 28
                    or identity["source_sha256"] != FROZEN_SOURCE or identity["test_site"] != site
                    or set(identity["training_sites"]) != set(FOLDS)-{site} or len(keys) != count or all_keys & keys
                    or any(clip in SEQS or not clip.startswith(site+"_") for clip,_,_ in keys)):
                raise ValueError(f"Unexpected fold or training identity {site}")
            all_keys.update(keys)
            model = UnstructuredShortPosePredictor("M0").cpu().eval()
            model.load_state_dict(ck["model_state"], strict=True)
            loaded[site] = model
    if len(all_keys) != 1482:
        raise ValueError("Expected four disjoint registered PINNS test folds")
    return loaded, hashes


def observed_velocity(history_xy, history_us):
    """Fit only the last observed groundspeed. Inputs have no future coordinates or clock."""
    gap_us = history_us[:,-1] - history_us[:,-2]
    if history_xy.shape[1:] != (5,2) or history_us.shape != history_xy.shape[:2] or np.any(gap_us <= 0):
        raise ValueError("Expected five timed observations")
    return (history_xy[:,-1]-history_xy[:,-2]) / (gap_us[:,None] / 1_000_000.0)


def m0_inputs(history_xy):
    import torch
    from src.data.features import build_motion_features
    # The frozen feature builder uses one scalar dt, and its decoder emits a fixed 10 Hz grid.
    relative = history_xy - history_xy[:,-1:]
    motion = np.stack([build_motion_features(h, np.ones(5, np.float32), dt=.1) for h in relative])
    type_index = np.where(motion[:,-1,4] < .20, 2, 0)
    ego = np.eye(8, dtype=np.float32)[type_index]
    return {"x_m": torch.from_numpy(motion), "ego_type_oh": torch.from_numpy(ego)}


def forecasts(history_xy, models):
    """Lock all forecasts from observed positions and clocks only; return local ground displacements."""
    import torch
    if history_xy.shape[1:] != (5,2) or not np.isfinite(history_xy).all():
        raise ValueError("Five finite measured historical XY inputs required")
    outputs = {}
    x = m0_inputs(history_xy)
    with torch.inference_mode():
        for site, model in models.items():
            parts = []
            for start in range(0,len(history_xy),64):
                batch = {k:v[start:start+64] for k,v in x.items()}
                p = model(batch)
                rows = torch.arange(len(p["mode_probs"]))
                parts.append(p["trajs"][rows,p["mode_probs"].argmax(1)].numpy().astype(np.float64))
            outputs[site] = np.concatenate(parts)
    if not all(np.isfinite(v).all() and v.shape == (len(history_xy),30,2) for v in outputs.values()):
        raise ValueError("Unusable forecast")
    return outputs


def coordinates_for(sample, windows, indices):
    wanted = {}
    for w in windows:
        wanted.setdefault((w.seq, w.track), set()).update(t for t in w.ticks[indices])
    _, tracks = read_sample(sample, wanted)
    rows = []
    for w in windows:
        row = [tracks[w.seq,w.track][t][1] for t in w.ticks[indices]]
        rows.append(row)
    values = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("A registered coordinate is missing; do not silently alter the locked cohort")
    return values


def evaluate(sample, windows, models):
    hist = coordinates_for(sample,windows,slice(0,5))
    times = np.asarray([w.ticks for w in windows],dtype=np.int64)
    velocity = observed_velocity(hist, times[:,:5])
    pred = forecasts(hist, models)
    # The scoring clock queries the already specified continuous CV line at actual target timestamps.
    pred["actual_time_cv"] = velocity[:,None,:] * ((times[:,5:]-times[:,4:5])/1_000_000.)[:,:,None]
    truth = coordinates_for(sample,windows,slice(5,35))-hist[:,-1:]
    errors = {}
    for name, values in pred.items():
        distance = np.linalg.norm(values-truth,axis=2)
        errors[name] = np.column_stack((distance.mean(axis=1),distance[:,-1]))
    errors["M0_four_fixed_fold_mean_error"] = np.mean([errors[s] for s in FOLDS],axis=0)
    def summary(select):
        return {m:dict(zip(METRICS, e[select].mean(0).tolist())) for m,e in errors.items()}
    site = np.asarray([w.seq for w in windows])
    by_sequence = {s:summary(site==s) for s in SEQS}
    macro = {m:{k: float(np.mean([r[m][k] for r in by_sequence.values()])) for k in METRICS} for m in errors}
    return {"equal_four_demo_sequences_one_site":macro, "by_sequence":by_sequence,
            "pooled_windows":summary(np.arange(len(windows))),"runtime_device":"cpu"}


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample",type=Path)
    parser.add_argument("weights_bundle",type=Path)
    parser.add_argument("--lock",type=Path,required=True)
    parser.add_argument("--score",action="store_true",help="Explicitly evaluate a previously registered check-only cohort")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    windows,cohort=prepare(args.sample)
    models,hashes=load_m0(args.weights_bundle)
    identity={"schema":1,"sample_md5":SAMPLE_MD5,"weights_archive_sha256":BUNDLE_SHA,
              "source_sha256":FROZEN_SOURCE,"m0_checkpoints_sha256":hashes,
              "evaluator_sha256":file_digest(Path(__file__)),"cohort":cohort}
    if args.lock.exists():
        if json.loads(args.lock.read_text()) != identity:
            raise ValueError("Saved cohort/source/weights identity differs")
    elif args.score:
        raise ValueError("A previously saved check-only cohort lock is required")
    else:
        args.lock.write_text(json.dumps(identity,indent=2,allow_nan=False)+"\n")
    if args.score:
        if not args.output or args.output.exists() or args.output==args.lock:
            raise ValueError("Specify a new score output path")
        report={"identity":identity,"scores":evaluate(args.sample,windows,models)}
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    else:
        print(f"Verified {len(windows)} disjoint track windows, four demo sequences, and four seed-42 weights; no model forward or coordinate-field inspection.")
if __name__=="__main__":
    main()
