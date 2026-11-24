"""Verify a single development recording. Only explicit training members are opened.

Usage: python3 check_training_coordinates.py [--labels-3d /path/to/stlc-111-2019-04-19_0.json]
Optional ground truth is the JRDB 2022 train_dataset/labels/labels_3d member for this run.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import struct
import urllib.request
import zipfile
import zlib

import numpy as np


SCENE = "stlc-111-2019-04-19_0"
ARTIFACTS = {
    "odometry": (
        "https://storage.googleapis.com/gresearch/human_scene_transformer/odometry.zip",
        1078340, f"odometry/train/{SCENE}.json",
    ),
    "tracks": (
        "https://github.com/vita-epfl/social-transmotion/releases/download/ckpt_data/releases.zip",
        305870808, f"jrdb/data/train/{SCENE}_shift_0.ndjson",
    ),
}
OUT = Path(__file__).resolve().parent


class TrainingArchive:
    def __init__(self, url, size):
        self.url = url
        self.size = size
        tail = self.read(size-22, size-1)
        if tail[:4] != b"PK\x05\x06" or tail[-2:] != b"\x00\x00":
            raise ValueError("Expected the known classic ZIP with no archive comment")
        length, self.offset = struct.unpack_from("<II", tail, 12)
        with zipfile.ZipFile(io.BytesIO(self.read(self.offset,self.offset+length-1)+tail)) as directory:
            self.training = {e.filename:e for e in directory.infolist() if "/train/" in e.filename and not e.is_dir()}

    def read(self, start, stop):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{stop}", "User-Agent": "JRDB-development-check"})
        with urllib.request.urlopen(req, timeout=30) as stream:
            if stream.headers.get("Content-Range") != f"bytes {start}-{stop}/{self.size}":
                raise ValueError("Server did not honor exact byte range")
            data = stream.read()
            if len(data) != stop-start+1:
                raise ValueError("Short range")
            return data

    def member(self, member):
        entry = self.training[member]
        start = entry.header_offset + self.offset
        header = self.read(start, start+29)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("Invalid member offset")
        filename, extra = struct.unpack_from("<HH", header, 26)
        payload = start + 30 + filename + extra
        data = self.read(payload, payload+entry.compress_size-1)
        if entry.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(data, -15)
        elif entry.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Unsupported compression")
        if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
            raise ValueError("Member integrity check failed")
        return data


def training_member(url, size, member):
    return TrainingArchive(url, size).member(member)


def rotation(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def verify_raw_labels(path, tracks, ts_index, relative_r, relative_p):
    if path.name != f"{SCENE}.json":
        raise ValueError("Use only the named training recording")
    boxes = {}
    for name, objects in json.loads(path.read_text())["labels"].items():
        frame = int(name.split(".")[0])
        for obj in objects:
            obj_id = obj["label_id"]
            if obj_id.startswith("pedestrian:"):
                box = obj["box"]
                boxes[(frame, int(obj_id.split(":")[-1]))] = [box["cx"], box["cy"], box["cz"]]
    shared = sorted(set(tracks) & set(boxes))
    if not shared:
        raise ValueError("No shared original frame and pedestrian IDs")
    src = np.array([boxes[key] for key in shared])
    time_index = [ts_index[key[0]] for key in shared]
    local = np.einsum("nij,nj->ni", relative_r[time_index], src) + relative_p[time_index]
    st = np.array([tracks[key] for key in shared])
    return {"pairs": len(shared), "release_pairs": len(tracks),
            "release_minus_raw_m_p50_p95_max": np.quantile(np.linalg.norm(st-src[:,:2], axis=1), [.5,.95,1]).tolist(),
            "release_minus_compensated_m_p50_p95_max": np.quantile(np.linalg.norm(st-local[:,:2], axis=1), [.5,.95,1]).tolist(),
            "release_to_raw_norm_scale": float(np.sqrt((st**2).sum() / (src[:,:2]**2).sum()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-3d", type=Path)
    args = parser.parse_args()
    blobs = {key: training_member(*artifact) for key, artifact in ARTIFACTS.items()}
    odom = {int(k.split(".")[0]): pose for k,pose in json.loads(blobs["odometry"])["odometry"].items()}
    frames = sorted(odom)
    positions = np.array([[odom[t]["position"][d] for d in "xyz"] for t in frames])
    qs = np.array([[odom[t]["orientation"][d] for d in "xyzw"] for t in frames])
    rot = np.stack([rotation(q) for q in qs])
    relative_r = rot[0].T @ rot
    relative_p = np.einsum("ni,ij->nj", positions-positions[0], rot[0])
    scenes = []; tracks = {}
    for line in blobs["tracks"].splitlines():
        row = json.loads(line)
        if "scene" in row:
            scenes.append(row["scene"])
        if "track" in row:
            record = row["track"]; key = (record["f"], record["p"])
            xy = (record["x"], record["y"])
            if key in tracks and xy != tracks[key]:
                raise ValueError("Conflicting duplicate track identity")
            tracks[key] = xy
    ticks = sorted({t for t, _ in tracks})
    if not set(ticks).issubset(odom) or frames[0] != 0:
        raise ValueError("Missing odometry or initial frame")
    ts_index = {t:i for i,t in enumerate(frames)}
    # HST's T0^{-1} * Tt applied to (x,y,z=0); the 2D Social-Transmotion data omit z.
    correction = []
    for (t, _), xy in tracks.items():
        i=ts_index[t]; p=np.array([*xy,0.])
        out=relative_r[i] @ p + relative_p[i]
        correction.append(np.linalg.norm(out[:2]-p[:2]))
    angular = np.rad2deg(np.arccos(np.clip((np.trace(relative_r,axis1=1,axis2=2)-1)/2,-1,1)))
    report = {
        "development_recording":SCENE, "sources":{
            k:{"url":v[0],"member":v[2],"uncompressed_bytes":len(blobs[k]),
               "member_sha256":hashlib.sha256(blobs[k]).hexdigest()} for k,v in ARTIFACTS.items()},
        "all_odometry_frames":len(frames), "odometry_frame_range":[frames[0],frames[-1]],
        "odometry_gap_histogram":dict(Counter(np.diff(frames).tolist())),
        "track_people":len({p for _,p in tracks}), "unique_frame_person_positions":len(tracks),
        "track_native_frame_range":[ticks[0],ticks[-1]], "track_distinct_frames":len(ticks),
        "track_frames_with_odometry":len(set(ticks)&set(odom)), "track_scene_fps":sorted({s["fps"] for s in scenes}),
        "sampled_frame_gap_histogram":dict(Counter(np.diff(ticks).tolist())),
        "first_xyzw_quaternion":qs[0].tolist(), "first_xyz_position_m":positions[0].tolist(),
        "max_relative_z_translation_m":float(np.abs(relative_p[:,2]).max()),
        "max_xy_translation_from_initial_m":float(np.linalg.norm(relative_p[:,:2],axis=1).max()),
        "max_full_orientation_change_deg":float(angular.max()),
        "max_z_coupling_into_horizontal":float(np.linalg.norm(relative_r[:,:2,2],axis=1).max()),
        "z_zero_plane_xy_odometry_correction_m_p50_p95_max":np.quantile(correction,[.5,.95,1]).tolist(),
        "independent_original_box_check": verify_raw_labels(args.labels_3d,tracks,ts_index,relative_r,relative_p) if args.labels_3d else None,
        "interpretation":"The local odometry frame is anchored at frame 0. No geodetic map, raw RGB, retained test labels or forecast scores were used.",
    }
    (OUT/"training_coordinate_check.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
