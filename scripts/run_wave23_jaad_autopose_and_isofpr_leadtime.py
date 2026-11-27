"""Wave 23B: JAAD 2.0 automatic RTMPose-m video skeleton extraction, strict 5-frame de-occlusion audit, & iso-Test-FPR lead-time."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from pathlib import PurePosixPath
import struct
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None

from scripts.run_heldout_cloud import MODEL_SHA, MODEL_URL, download_zip
from scripts.run_wave18_conflict_gated_interaction_5seed import _loc_bootstrap_ci
from scripts.run_wave21_jaad_deocclusion_and_eth_ucy_6dataset import JAAD_ZIP_URL, VEH_ACT_MAP, _fetch_url_bytes
from scripts.run_wave21_jrdb_and_jta_emergence_maneuvers import ST_RELEASE_URL, _download_verified_zip
from scripts.run_wave21_pinns_true_emergence_and_6slot_diag import SEEDS
from scripts.run_wave22_apa_leadtime_and_delayed_onset_4dataset import TAUS_10HZ, extract_jta_sliding_leadtime_events
from src.data.pinns_extractor import RTMPoseONNXExtractor
from src.metrics.metrics_and_stats import _binary_average_precision
from src.models.apa_precursor_leadtime import (
    CausalAPAPrecursorLeadNet,
    calibrate_matched_fpr_thresholds,
    evaluate_iso_test_fpr_and_unconflated_leadtime,
    paired_iso_fpr_comparison,
)

JAAD_CLIPS_URL = "https://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip"
JAAD_CLIPS_BYTES = 3084055074


def _ensure_onnxruntime() -> None:
    if importlib.util.find_spec("onnxruntime") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "onnxruntime"], check=True)


def _jaad_range_bytes(start: int, stop: int) -> bytes:
    for att in range(5):
        try:
            req = urllib.request.Request(
                JAAD_CLIPS_URL,
                headers={"Range": f"bytes={start}-{stop - 1}", "Accept-Encoding": "identity", "User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) != stop - start:
                raise OSError(f"Short JAAD range read: {len(data)} != {stop - start}")
            return data
        except Exception:
            if att == 4:
                raise
            time.sleep(1.5 * (att + 1))
    raise RuntimeError("Unreachable")


def _load_rtmpose_extractor(scratch_dir: Path) -> RTMPoseONNXExtractor:
    _ensure_onnxruntime()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    m_zip = scratch_dir / "rtmpose_m.zip"
    onnx_p = scratch_dir / "end2end.onnx"
    if not onnx_p.is_file():
        download_zip(MODEL_URL, m_zip, 50799818)
        with zipfile.ZipFile(m_zip) as zf:
            nm = [k for k in zf.namelist() if PurePosixPath(k).name == "end2end.onnx" and "__MACOSX" not in PurePosixPath(k).parts][0]
            onnx_p.write_bytes(zf.read(nm))
    assert hashlib.sha256(onnx_p.read_bytes()).hexdigest() == MODEL_SHA, "RTMPose ONNX SHA-256 mismatch"
    ext = RTMPoseONNXExtractor(onnx_path=onnx_p, batch_size=256, device_id=0)
    if "MockFallback" in getattr(ext, "active_providers", []):
        raise RuntimeError("Real RTMPose ONNX session required; MockFallback forbidden")
    return ext


def _fetch_jaad_zip_entries() -> dict[str, zipfile.ZipInfo]:
    tail = _jaad_range_bytes(JAAD_CLIPS_BYTES - 65536, JAAD_CLIPS_BYTES)
    eocd = tail.rfind(b"PK\x05\x06")
    _, cd_off = struct.unpack_from("<II", tail, eocd + 12)
    cd_blob = _jaad_range_bytes(cd_off, JAAD_CLIPS_BYTES)

    class _TailIO(io.RawIOBase):
        def __init__(self) -> None:
            self.p = 0
        def seekable(self) -> bool:
            return True
        def tell(self) -> int:
            return self.p
        def seek(self, off: int, wh: int = 0) -> int:
            self.p = (0, self.p, JAAD_CLIPS_BYTES)[wh] + off
            return self.p
        def read(self, n: int = -1) -> bytes:
            end = JAAD_CLIPS_BYTES if n < 0 else min(JAAD_CLIPS_BYTES, self.p + n)
            b = cd_blob[self.p - cd_off : end - cd_off]
            self.p = end
            return b

    with zipfile.ZipFile(_TailIO()) as zf:
        return {Path(i.filename).stem: i for i in zf.infolist() if i.filename.endswith(".mp4")}


def _download_jaad_member(info: zipfile.ZipInfo) -> bytes:
    pos = info.header_offset
    hdr = _jaad_range_bytes(pos, pos + 30)
    n_len, ex_len = struct.unpack_from("<HH", hdr, 26)
    comp = _jaad_range_bytes(pos + 30 + n_len + ex_len, pos + 30 + n_len + ex_len + info.compress_size)
    raw = zlib.decompress(comp, -15) if info.compress_type == zipfile.ZIP_DEFLATED else comp
    if len(raw) != info.file_size or (zlib.crc32(raw) & 0xFFFFFFFF) != (info.CRC & 0xFFFFFFFF):
        raise ValueError(f"CRC/size mismatch for {info.filename}")
    return raw


def extract_jaad_rtmpose_keypoints(
    vid_frame_boxes: dict[str, dict[int, list[tuple[str, tuple[float, float, float, float]]]]],
    scratch_dir: Path,
) -> dict[tuple[str, str, int], np.ndarray]:
    """Extract real 17-joint RTMPose-m [17, 3] keypoints for every required (vid, pid, frame)."""
    cache_npz = scratch_dir / "jaad_rtmpose_cache.npz"
    if cache_npz.is_file():
        with np.load(cache_npz, allow_pickle=False) as cz:
            keys = json.loads(str(cz["keys_json"]))
            arr = cz["kpts"]
            return {(v, p, int(f)): arr[i] for i, (v, p, f) in enumerate(keys)}

    ext = _load_rtmpose_extractor(scratch_dir)
    info_map = _fetch_jaad_zip_entries()
    vids = sorted(v for v in vid_frame_boxes if v in info_map)
    kpt_map: dict[tuple[str, str, int], np.ndarray] = {}
    crops_b, cnts_b, scls_b, meta_b = [], [], [], []

    def _flush() -> None:
        if not crops_b:
            return
        uvc = ext.infer_warped_crops(np.concatenate(crops_b, 0), np.concatenate(cnts_b, 0), np.concatenate(scls_b, 0))
        for m_k, kp in zip(meta_b, uvc):
            kpt_map[m_k] = kp.astype(np.float32)
        crops_b.clear(); cnts_b.clear(); scls_b.clear(); meta_b.clear()

    for b0 in range(0, len(vids), 12):
        batch_v = vids[b0 : b0 + 12]
        with ThreadPoolExecutor(max_workers=6) as pool:
            blobs = list(pool.map(lambda v: _download_jaad_member(info_map[v]), batch_v))
        for vid, blob in zip(batch_v, blobs):
            mp4_tmp = scratch_dir / f"tmp_{vid}.mp4"
            mp4_tmp.write_bytes(blob)
            f_dict = vid_frame_boxes[vid]
            want_f = sorted(f_dict)
            cap = cv2.VideoCapture(str(mp4_tmp))
            ptr, f_idx = 0, 0
            while ptr < len(want_f):
                ok, bgr = cap.read()
                if not ok:
                    break
                if f_idx == want_f[ptr]:
                    items = f_dict[f_idx]
                    b_arr = np.array([bx for _, bx in items], dtype=np.float32)
                    cr, ct, sc = ext.warp_crops(bgr, b_arr)
                    crops_b.append(cr); cnts_b.append(ct); scls_b.append(sc)
                    meta_b.extend((vid, pid, f_idx) for pid, _ in items)
                    ptr += 1
                    if sum(len(x) for x in crops_b) >= 512:
                        _flush()
                f_idx += 1
            cap.release()
            mp4_tmp.unlink(missing_ok=True)
        _flush()
        print(f"[JAAD RTMPose] Processed {min(b0 + 12, len(vids))}/{len(vids)} videos ({len(kpt_map)} person-frames)", flush=True)
    return kpt_map


def _kpts5_to_autopose(k5: np.ndarray, h_ref: float) -> tuple[np.ndarray, np.ndarray]:
    """Map [5, 17, 3] COCO RTMPose keypoints to scale-invariant [5, 12] skeletal features + [2] quality."""
    u = k5[:, :, 0]
    v = k5[:, :, 1]
    c = np.clip(k5[:, :, 2], 0.0, 1.0)
    core_c = c[:, [0, 5, 6, 11, 12, 13, 14, 15, 16]].mean(axis=1)  # [5]
    gate_f = (core_c >= 0.20).astype(np.float32) * core_c.astype(np.float32)
    sh_mx, sh_my = 0.5 * (u[:, 5] + u[:, 6]), 0.5 * (v[:, 5] + v[:, 6])
    hp_mx, hp_my = 0.5 * (u[:, 11] + u[:, 12]), 0.5 * (v[:, 11] + v[:, 12])
    inv_h = 1.0 / max(float(h_ref), 25.0)
    raw11 = np.stack([
        np.abs(u[:, 5] - u[:, 6]) * inv_h,
        np.abs(u[:, 11] - u[:, 12]) * inv_h,
        (u[:, 5] - u[:, 6]) * inv_h,
        (sh_mx - hp_mx) * inv_h,
        (hp_my - sh_my) * inv_h,
        (u[:, 0] - sh_mx) * inv_h,
        (sh_my - v[:, 0]) * inv_h,
        np.abs(u[:, 15] - u[:, 16]) * inv_h,
        (u[:, 15] - u[:, 16]) * inv_h,
        (v[:, 16] - v[:, 15]) * inv_h,
        ((u[:, 15] - u[:, 13]) - (u[:, 16] - u[:, 14])) * inv_h,
    ], axis=-1).astype(np.float32)
    raw11 = np.clip(raw11, -2.5, 2.5) * gate_f[:, None]
    feat12 = np.concatenate([raw11, core_c[:, None].astype(np.float32)], axis=-1)
    torso_feet_ok = float(((c[:, [5, 6, 11, 12, 15, 16]] >= 0.25).all(axis=-1).mean()) * float(h_ref >= 50.0))
    q2 = np.array([float(core_c.mean()), torso_feet_ok], dtype=np.float32)
    return feat12, q2


def build_jaad2_strict_and_sliding_cohorts(scratch_dir: Path) -> tuple[dict, dict]:
    """Parse JAAD 2.0 XMLs, collect all required person-frames once, extract RTMPose-m, and build both cohorts."""
    blob = _fetch_url_bytes(JAAD_ZIP_URL, scratch_dir / "jaad2.zip")
    splits: dict[str, str] = {}
    parsed_tracks: list[tuple[str, str, str, str, dict[int, dict], dict[tuple[str, int], np.ndarray], dict[int, float]]] = []
    vid_frame_boxes: dict[str, dict[int, dict[str, tuple[float, float, float, float]]]] = {}

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        pfx = "JAAD-JAAD_2.0/"
        for sp in ("train", "val", "test"):
            for vid in zf.read(f"{pfx}split_ids/default/{sp}.txt").decode("utf-8").splitlines():
                if vid.strip():
                    splits[vid.strip()] = sp
        ann_files = sorted(n for n in zf.namelist() if n.startswith(f"{pfx}annotations/video_") and n.endswith(".xml"))
        for af in ann_files:
            vid = Path(af).stem
            sp = splits.get(vid, "train")
            root = ET.fromstring(zf.read(af))
            app_name = f"{pfx}annotations_appearance/{vid}_appearance.xml"
            pose_map: dict[tuple[str, int], np.ndarray] = {}
            if app_name in zf.namelist():
                for tr in ET.fromstring(zf.read(app_name)).findall(".//track"):
                    pa = tr.attrib.get("id", "")
                    for bx in tr.findall("box"):
                        pose_map[(pa, int(bx.attrib["frame"]))] = np.array([
                            float(bx.attrib.get("pose_front", "0")), float(bx.attrib.get("pose_back", "0")),
                            float(bx.attrib.get("pose_left", "0")), float(bx.attrib.get("pose_right", "0")),
                        ], dtype=np.float32)
            veh_name = f"{pfx}annotations_vehicle/{vid}_vehicle.xml"
            veh_map: dict[int, float] = {}
            if veh_name in zf.namelist():
                for vf in ET.fromstring(zf.read(veh_name)).findall(".//frame"):
                    veh_map[int(vf.attrib["id"])] = VEH_ACT_MAP.get(vf.attrib.get("action", "moving_slow"), 0.5)
            for tr_i, tr in enumerate(root.findall(".//track")):
                if tr.attrib.get("label") != "pedestrian":
                    continue
                boxes: dict[int, dict] = {}
                pid_t = ""
                for bx in tr.findall("box"):
                    if bx.attrib.get("outside", "0") == "1":
                        continue
                    fr = int(bx.attrib["frame"])
                    at = {a.attrib.get("name", ""): (a.text or "").strip() for a in bx.findall("attribute")}
                    pid_t = at.get("id", pid_t)
                    xtl, ytl, xbr, ybr = float(bx.attrib["xtl"]), float(bx.attrib["ytl"]), float(bx.attrib["xbr"]), float(bx.attrib["ybr"])
                    w, h = max(xbr - xtl, 4.0), max(ybr - ytl, 12.0)
                    boxes[fr] = {
                        "fcx": 0.5 * (xtl + xbr), "cy": 0.5 * (ytl + ybr), "fby": ybr, "w": w, "h": h,
                        "occ": at.get("occlusion", "none"), "walk": 1 if at.get("action", "standing") == "walking" else 0,
                        "look": 1.0 if at.get("look", "") == "looking" else 0.0,
                        "gest": 0.0 if at.get("hand_gesture", "__undefined__") == "__undefined__" else 1.0,
                        "nod": 1.0 if at.get("nod", "__undefined__") == "nodding" else 0.0,
                        "cross": 1.0 if at.get("cross", "") == "crossing" else 0.0,
                    }
                if len(boxes) >= 48:
                    parsed_tracks.append((vid, sp, f"{tr_i}:{pid_t}", pid_t, boxes, pose_map, veh_map))

    # First pass: register every required (vid, frame, trk_key) crop across BOTH trajectory & sliding cohorts
    traj_specs: list[tuple] = []
    slide_specs: list[tuple] = []
    ev_counter = 0

    def _reg_frames(vid: str, trk_k: str, boxes: dict[int, dict], h_r: list[int]) -> None:
        vd = vid_frame_boxes.setdefault(vid, {})
        for rf in h_r:
            b = boxes[rf]
            vd.setdefault(rf, {})[trk_k] = (b["fcx"], b["cy"], b["w"], b["h"])

    for vid, sp, trk_k, pid_t, boxes, pose_map, veh_map in parsed_tracks:
        frs = sorted(boxes)
        f_min = frs[0]
        if len(boxes) >= 75:
            starts = list(range(f_min, frs[-1] - 72, 45))
            for k_i in range(4, len(frs) - 70):
                fn_c = frs[k_i]
                if boxes[fn_c]["occ"] == "none" and any(boxes.get(fn_c - d, {}).get("occ") in ("full", "part") for d in (3, 6, 9, 12)):
                    if all(abs(fn_c - 12 - ss) >= 18 for ss in starts) and fn_c - 12 >= f_min:
                        starts.append(fn_c - 12)
            for s_raw in sorted(starts):
                req = [s_raw + 3 * k for k in range(25)]
                if all(r_f in boxes for r_f in req):
                    _reg_frames(vid, trk_k, boxes, req[:5])
                    traj_specs.append((vid, sp, trk_k, pid_t, boxes, pose_map, veh_map, f_min, s_raw, req))

        # Sliding lead-time events + pure steady negative controls
        evs: list[tuple[int, int]] = []
        for k in range(15, len(frs) - 15):
            fc = frs[k]
            pr = [fc - 3 * d for d in range(1, 6)]
            po = [fc + 3 * d for d in range(0, 4)]
            if not all(rf in boxes for rf in pr + po):
                continue
            if all(boxes[rf]["walk"] == 0 for rf in pr) and all(boxes[rf]["walk"] == 1 for rf in po) and not any(c == 0 and abs(fc - fo) < 45 for fo, c in evs):
                evs.append((fc, 0))
            if all(boxes[rf]["walk"] == 1 for rf in pr) and all(boxes[rf]["walk"] == 0 for rf in po) and not any(c == 1 and abs(fc - fo) < 45 for fo, c in evs):
                evs.append((fc, 1))
            if all(boxes[rf]["cross"] == 0 for rf in pr) and all(boxes[rf]["cross"] == 1 for rf in po) and not any(c == 2 and abs(fc - fo) < 45 for fo, c in evs):
                evs.append((fc, 2))
        for f_on, c_id in evs:
            eid = ev_counter; ev_counter += 1
            for st in TAUS_10HZ:
                t0_r = f_on + 3 * st
                hr = [t0_r - 3 * (4 - j) for j in range(5)]
                if all(rf in boxes for rf in hr):
                    _reg_frames(vid, trk_k, boxes, hr)
                    slide_specs.append((vid, sp, trk_k, pid_t, boxes, pose_map, veh_map, t0_r, hr, c_id, 0.1 * st, eid, False, f_on))
        if not evs:
            for f_neg in frs[18:-24:45]:
                wv = {boxes[rf]["walk"] for rf in frs if abs(rf - f_neg) <= 54}
                cv_s = {boxes[rf]["cross"] for rf in frs if abs(rf - f_neg) <= 54}
                hr = [f_neg - 3 * (4 - j) for j in range(5)]
                if len(wv) == 1 and len(cv_s) == 1 and all(rf in boxes for rf in hr):
                    _reg_frames(vid, trk_k, boxes, hr)
                    eid = ev_counter; ev_counter += 1
                    slide_specs.append((vid, sp, trk_k, pid_t, boxes, pose_map, veh_map, f_neg, hr, -1, 0.0, eid, True, f_neg))

    vfb_list = {v: {f: list(d.items()) for f, d in fd.items()} for v, fd in vid_frame_boxes.items()}
    kpt_map = extract_jaad_rtmpose_keypoints(vfb_list, scratch_dir)

    hist_l, fut_l, auto_l, aq_l, hum_l, veh_l = [], [], [], [], [], []
    ya_l, va_l, on_l, vid_l, sp_l = [], [], [], [], []
    m_strict_deocc, m_part_t0, m_full_t0, m_late_start, m_broad_406, m_lowdisp_strict = [], [], [], [], [], []
    z17 = np.zeros((17, 3), dtype=np.float32)

    for vid, sp, trk_k, pid_t, boxes, pose_map, veh_map, f_min, s_raw, req in traj_specs:
        h_r, f_r = req[:5], req[5:]
        h_ref = max(float(np.median([boxes[rf]["h"] for rf in h_r])), 25.0)
        pts = np.array([[(boxes[rf]["fcx"] - boxes[h_r[-1]]["fcx"]) / h_ref, (boxes[rf]["fby"] - boxes[h_r[-1]]["fby"]) / h_ref] for rf in req], dtype=np.float64)
        w_h = [boxes[rf]["walk"] for rf in h_r]
        w_f = np.array([boxes[rf]["walk"] for rf in f_r], dtype=int)
        c_f = np.array([boxes[rf]["cross"] > 0.5 for rf in f_r], dtype=bool)
        is_stand, is_walk = sum(w_h[-3:]) <= 1, sum(w_h[-3:]) >= 2
        t_st = next((i for i in range(len(w_f) - 2) if is_stand and w_f[i : i + 3].all()), -1)
        t_sp = next((i for i in range(len(w_f) - 2) if is_walk and (1 - w_f[i : i + 3]).all()), -1)
        t_cr = next((i for i in range(len(c_f) - 2) if (boxes[h_r[-1]]["cross"] < 0.5) and c_f[i : i + 3].all()), -1)
        ya = np.array([t_st >= 0, t_sp >= 0, t_cr >= 0, (t_st < 0 and t_sp < 0)], dtype=bool)
        va = np.array([is_stand, is_walk, boxes[h_r[-1]]["cross"] < 0.5, True], dtype=bool)
        on = np.array([t_st, t_sp, t_cr, -1, 0], dtype=np.int64)

        k5 = np.stack([kpt_map.get((vid, trk_k, rf), z17) for rf in h_r], axis=0)
        ap5, aq2 = _kpts5_to_autopose(k5, h_ref)
        hm5 = np.zeros((5, 8), dtype=np.float32)
        v5 = np.zeros((5, 3), dtype=np.float32)
        for ti, rf in enumerate(h_r):
            b = boxes[rf]
            po4 = pose_map.get((pid_t, rf), np.zeros(4, dtype=np.float32))
            hm5[ti, :4] = po4
            hm5[ti, 4:] = [b["look"], b["gest"], b["nod"], b["w"] / b["h"]]
            dh_dt = (boxes[rf]["h"] - boxes[h_r[max(0, ti - 1)]]["h"]) / h_ref
            v5[ti] = [veh_map.get(rf, 0.5), dh_dt * 10.0, float(b["h"] >= 60.0)]

        occ_5 = [boxes[rf]["occ"] for rf in h_r]
        had_occ_broad = any(boxes.get(rf, {}).get("occ") in ("full", "part") for rf in range(max(f_min, h_r[0] - 12), h_r[-1]))
        mid_em = (s_raw == f_min) and (f_min > 9)
        broad = bool(had_occ_broad or mid_em)
        # Strict 5-frame de-occlusion: partially/fully occluded in observed frames 0..3 (or prior 0.4s) AND recovered to 'none' at frame 4!
        strict_5f = any(oc in ("full", "part") for oc in occ_5[:4]) and (occ_5[4] == "none")
        part_t0 = broad and (occ_5[4] == "part")
        full_t0 = broad and (occ_5[4] == "full")
        late_unocc = broad and (not strict_5f) and (occ_5[4] == "none")
        disp_bh = float(np.linalg.norm(np.diff(pts[:5], axis=0), axis=-1).sum())

        hist_l.append(pts[:5]); fut_l.append(pts[5:])
        auto_l.append(ap5); aq_l.append(aq2); hum_l.append(hm5); veh_l.append(v5)
        ya_l.append(ya); va_l.append(va); on_l.append(on); vid_l.append(vid); sp_l.append(sp)
        m_strict_deocc.append(strict_5f); m_part_t0.append(part_t0); m_full_t0.append(full_t0)
        m_late_start.append(late_unocc); m_broad_406.append(broad)
        m_lowdisp_strict.append(bool(strict_5f and disp_bh < 0.25))

    # Materialize sliding lead-time cohort with BOTH AutoPose (RTMPose-17) and HumanAttr (orientation+gaze)
    sk_l, sa_l, sq_a_l, sh_l, sq_h_l, sv_l, sy_l, stau_l, seid_l, secls_l, sneg_l, sdeocc_b_l, sdeocc_s_l, svid_l, ssp_l = ([] for _ in range(15))
    for vid, sp, trk_k, pid_t, boxes, pose_map, veh_map, t0_r, hr, c_id, tau_v, eid, is_neg, f_on in slide_specs:
        href = max(float(np.median([boxes[rf]["h"] for rf in hr])), 25.0)
        xy = np.array([[(boxes[rf]["fcx"] - boxes[t0_r]["fcx"]) / href, (boxes[rf]["fby"] - boxes[t0_r]["fby"]) / href] for rf in hr], dtype=np.float32)
        vel = np.vstack((np.zeros((1, 2), dtype=np.float32), (xy[1:] - xy[:-1]) / 0.1))
        kin5 = np.concatenate((xy, vel, np.linalg.norm(vel, axis=-1, keepdims=True)), axis=-1)
        k5 = np.stack([kpt_map.get((vid, trk_k, rf), z17) for rf in hr], axis=0)
        ap5, aq2 = _kpts5_to_autopose(k5, href)
        hm5 = np.zeros((5, 8), dtype=np.float32)
        for j, rf in enumerate(hr):
            b = boxes[rf]
            p4 = pose_map.get((pid_t, rf), np.zeros(4, dtype=np.float32))
            hm5[j] = [p4[0], p4[1], p4[2], p4[3], b["look"], b["gest"], b["nod"], b["w"] / max(b["h"], 12.0)]
        oc_w = float(np.mean([0.25 if boxes[rf]["occ"] == "full" else (0.65 if boxes[rf]["occ"] == "part" else 1.0) for rf in hr]))
        qh2 = np.array([min(href / 80.0, 1.5), oc_w], dtype=np.float32)
        vs = np.array([veh_map.get(rf, 0.5) for rf in hr], dtype=np.float32)
        vh3 = np.array([vs[-1], vs[-1] - vs[0], float(vs.mean())], dtype=np.float32)
        yv = np.zeros(3, dtype=np.float32)
        if not is_neg:
            yv[c_id] = 1.0
        was_occ_b = any(boxes.get(t0_r - 3 * d, {}).get("occ") in ("full", "part") for d in range(1, 16))
        was_occ_strict = any(boxes.get(f_on - 3 * d, {}).get("occ") in ("full", "part") for d in range(2, 16)) and (boxes.get(f_on, {}).get("occ") == "none")
        sk_l.append(kin5); sa_l.append(ap5); sq_a_l.append(aq2); sh_l.append(hm5); sq_h_l.append(qh2)
        sv_l.append(vh3); sy_l.append(yv); stau_l.append(tau_v); seid_l.append(eid); secls_l.append(c_id)
        sneg_l.append(is_neg); sdeocc_b_l.append(was_occ_b); sdeocc_s_l.append(was_occ_strict); svid_l.append(vid); ssp_l.append(sp)

    traj_ds = {
        "hist": np.asarray(hist_l, dtype=np.float64), "truth": np.asarray(fut_l, dtype=np.float64),
        "auto_pose5": np.asarray(auto_l, dtype=np.float32), "auto_q": np.asarray(aq_l, dtype=np.float32),
        "human_attr5": np.asarray(hum_l, dtype=np.float32), "veh5": np.asarray(veh_l, dtype=np.float32),
        "y_act": np.asarray(ya_l, dtype=bool), "v_act": np.asarray(va_l, dtype=bool), "onset": np.asarray(on_l, dtype=np.int64),
        "strict_5f_deocc": np.asarray(m_strict_deocc, dtype=bool), "ongoing_part_t0": np.asarray(m_part_t0, dtype=bool),
        "ongoing_full_t0": np.asarray(m_full_t0, dtype=bool), "late_unocc_start": np.asarray(m_late_start, dtype=bool),
        "broad_occ_406": np.asarray(m_broad_406, dtype=bool), "strict_deocc_lowdisp_lt0p25_bh": np.asarray(m_lowdisp_strict, dtype=bool),
        "vid": np.asarray(vid_l), "split": np.asarray(sp_l), "n_rtmpose_person_frames": len(kpt_map),
    }
    slide_ds = {
        "kin": np.asarray(sk_l, dtype=np.float32), "auto_pose": np.asarray(sa_l, dtype=np.float32), "auto_q": np.asarray(sq_a_l, dtype=np.float32),
        "human_attr": np.asarray(sh_l, dtype=np.float32), "human_q": np.asarray(sq_h_l, dtype=np.float32), "veh": np.asarray(sv_l, dtype=np.float32),
        "y": np.asarray(sy_l, dtype=np.float32), "tau_s": np.asarray(stau_l, dtype=np.float64), "event_id": np.asarray(seid_l, dtype=np.int64),
        "event_cls": np.asarray(secls_l, dtype=np.int64), "is_neg": np.asarray(sneg_l, dtype=bool),
        "is_deocc_broad": np.asarray(sdeocc_b_l, dtype=bool), "is_deocc_strict_recovered": np.asarray(sdeocc_s_l, dtype=bool),
        "vid": np.asarray(svid_l), "split": np.asarray(ssp_l),
    }
    return traj_ds, slide_ds


class JAADModalityFactorialNet(nn.Module):
    """5-frame causal bbox-motion + monotonic-gated AutoPose (RTMPose-17) / HumanAttr + Vehicle net."""

    def __init__(self, use_auto_pose: bool, use_human_attr: bool, use_veh: bool) -> None:
        super().__init__()
        self.use_auto_pose = use_auto_pose
        self.use_human_attr = use_human_attr
        self.use_veh = use_veh
        self.mot_gru = nn.GRU(4, 48, batch_first=True)
        self.auto_gru = nn.GRU(12, 24, batch_first=True)
        self.dauto_mlp = nn.Sequential(nn.Linear(12, 24), nn.SiLU(), nn.Linear(24, 24))
        self.q_w_raw = nn.Parameter(torch.tensor([[0.8, 0.8]]))
        self.q_b = nn.Parameter(torch.tensor([-0.2]))
        self.hum_gru = nn.GRU(8, 24, batch_first=True)
        self.dhum_mlp = nn.Sequential(nn.Linear(8, 24), nn.SiLU(), nn.Linear(24, 24))
        self.veh_gru = nn.GRU(3, 16, batch_first=True)
        d_in = 48 + (24 if use_auto_pose else 0) + (24 if use_human_attr else 0) + (16 if use_veh else 0)
        self.act_head = nn.Sequential(nn.Linear(d_in, 48), nn.SiLU(), nn.Linear(48, 4))
        self.traj_res = nn.Linear(d_in, 4 * 20 * 2)
        self.mode_lg = nn.Linear(d_in, 4)
        nn.init.normal_(self.traj_res.weight, std=0.01)
        nn.init.zeros_(self.traj_res.bias)

    def forward(self, hist: torch.Tensor, auto_p: torch.Tensor, auto_q: torch.Tensor, hum_p: torch.Tensor, veh5: torch.Tensor) -> dict[str, torch.Tensor]:
        vel = torch.cat((torch.zeros_like(hist[:, :1]), (hist[:, 1:] - hist[:, :-1]) / 0.1), dim=1)
        _, hm = self.mot_gru(torch.cat((hist, vel), dim=-1).float())
        feats = [hm[0]]
        if self.use_auto_pose:
            _, ha = self.auto_gru(auto_p)
            gq = torch.sigmoid(F.linear(auto_q, F.softplus(self.q_w_raw), self.q_b))  # [B, 1] monotonic quality gate
            feats.append(gq * (ha[0] + self.dauto_mlp(auto_p[:, -1] - auto_p[:, 0])))
        if self.use_human_attr:
            _, hh = self.hum_gru(hum_p)
            feats.append(hh[0] + self.dhum_mlp(hum_p[:, -1] - hum_p[:, 0]))
        if self.use_veh:
            _, hv = self.veh_gru(veh5)
            feats.append(hv[0])
        ctx = torch.cat(feats, dim=-1)
        t_s = torch.arange(1, 21, dtype=hist.dtype, device=hist.device)[None, :, None] * 0.1
        v_last = vel[:, -1:]
        cv = v_last * t_s
        stop_anc = v_last * (0.45 * (1.0 - torch.exp(-t_s / 0.45)))
        res = self.traj_res(ctx).view(-1, 4, 20, 2).double()
        p4 = torch.stack((cv, cv + res[:, 1], cv + res[:, 2], stop_anc + 0.3 * res[:, 3]), dim=1)
        return {"paths": p4, "logits": self.mode_lg(ctx), "act_logits": self.act_head(ctx)}


def _jaad_post_onset_err(pred: np.ndarray, truth: np.ndarray, onset: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(pred - truth, axis=-1)
    out = np.zeros((len(pred), 4), dtype=np.float64)
    out[:, 0], out[:, 1] = d.mean(axis=1), d[:, -1]
    for i in range(len(pred)):
        val_t = [int(x) for x in onset[i, :3] if x >= 0]
        t0 = min(val_t) if val_t else 0
        out[i, 2] = float(d[i, t0 : min(20, max(t0 + 1, t0 + 5))].mean())
        out[i, 3] = float(d[i, t0 : min(20, max(t0 + 1, t0 + 10))].mean())
    return out


def eval_jaad2_trajectory_factorial_5seed(jd: dict, seeds: list[int], device: torch.device) -> dict:
    sp = jd["split"]
    tr, va, te = np.flatnonzero(sp == "train"), np.flatnonzero(sp == "val"), np.flatnonzero(sp == "test")
    ht = torch.from_numpy(jd["hist"]).to(device)
    tt = torch.from_numpy(jd["truth"]).to(device)
    apt = torch.from_numpy(jd["auto_pose5"]).to(device)
    aqt = torch.from_numpy(jd["auto_q"]).to(device)
    hmt = torch.from_numpy(jd["human_attr5"]).to(device)
    vt = torch.from_numpy(jd["veh5"]).to(device)
    yat = torch.from_numpy(jd["y_act"]).to(device)
    vat = torch.from_numpy(jd["v_act"]).to(device)

    arms = (
        ("J_M0_bbox_motion_only", False, False, False),
        ("J_AutoPose_rtmpose17", True, False, False),
        ("J_AutoPose_veh", True, False, True),
        ("J_HumanAttr_orient_gaze", False, True, False),
        ("J_M2_bbox_veh", False, False, True),
        ("J_Full_human_veh", False, True, True),
        ("J_Combined_auto_human_veh", True, True, True),
    )
    t_s = np.arange(1, 21, dtype=np.float64)[None, :, None] * 0.1
    cv_te = ((jd["hist"][te, -1:] - jd["hist"][te, -2:-1]) / 0.1) * t_s
    err_cv = _jaad_post_onset_err(cv_te, jd["truth"][te], jd["onset"][te])
    err_s = {nm: np.zeros((len(seeds), len(te), 4), dtype=np.float64) for nm, *_ in arms}
    ap_probs = {nm: np.zeros((len(seeds), len(te), 4), dtype=np.float64) for nm, *_ in arms}

    pw_c = ((vat[tr].float().sum(0) - yat[tr].float().sum(0)) / yat[tr].float().sum(0).clamp_min(1.0)).clamp(1.0, 12.0)
    tr_t, va_t, te_t = torch.from_numpy(tr).to(device), torch.from_numpy(va).to(device), torch.from_numpy(te).to(device)

    for si, seed in enumerate(seeds):
        for nm, ua, uh, uv in arms:
            torch.manual_seed(seed + 17)
            net = JAADModalityFactorialNet(ua, uh, uv).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=2e-4)
            gen = torch.Generator().manual_seed(seed + 101)
            best_st, best_vl = None, 1e9
            for _ in range(28):
                net.train()
                for b in torch.randperm(len(tr_t), generator=gen).split(64):
                    ib = tr_t[b.to(device)]
                    opt.zero_grad(set_to_none=True)
                    o = net(ht[ib], apt[ib], aqt[ib], hmt[ib], vt[ib])
                    d_k = torch.linalg.vector_norm(o["paths"] - tt[ib][:, None], dim=-1).mean(dim=-1)
                    s_w = (-d_k / 0.15).softmax(dim=1).float()
                    l_tr = d_k.min(dim=1).values.mean() + 0.25 * (-(s_w * o["logits"].log_softmax(dim=1)).sum(dim=1)).mean().double()
                    bce = F.binary_cross_entropy_with_logits(o["act_logits"], yat[ib].float(), pos_weight=pw_c, reduction="none")
                    l_ac = ((bce * vat[ib].float()).sum(1) / vat[ib].float().sum(1).clamp_min(1.0)).mean().double()
                    (l_tr + 0.30 * l_ac).backward()
                    opt.step()
                net.eval()
                with torch.inference_mode():
                    ov = net(ht[va_t], apt[va_t], aqt[va_t], hmt[va_t], vt[va_t])
                    ix_v = ov["logits"].argmax(dim=1)
                    vl = float(torch.linalg.vector_norm(ov["paths"][torch.arange(len(va_t), device=device), ix_v] - tt[va_t], dim=-1).mean())
                    if vl < best_vl:
                        best_vl = vl
                        best_st = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if best_st is not None:
                net.load_state_dict(best_st)
            net.eval()
            with torch.inference_mode():
                ot = net(ht[te_t], apt[te_t], aqt[te_t], hmt[te_t], vt[te_t])
                pw = ot["paths"].cpu().numpy()
                ix = ot["logits"].argmax(dim=1).cpu().numpy()
                err_s[nm][si] = _jaad_post_onset_err(pw[np.arange(len(te)), ix], jd["truth"][te], jd["onset"][te])
                ap_probs[nm][si] = torch.sigmoid(ot["act_logits"]).cpu().numpy()

    em = {nm: err_s[nm].mean(axis=0) for nm, *_ in arms}
    v_te = jd["vid"][te]
    act_te = jd["y_act"][te, :3].any(axis=1)
    strata = {
        "all_locked_test_953": np.ones(len(te), dtype=bool),
        "strict_5frame_deoccluded_168": jd["strict_5f_deocc"][te],
        "strict_5f_deocc_active_maneuvers": jd["strict_5f_deocc"][te] & act_te,
        "strict_5f_deocc_lowdisp_lt0p25_bbox_h": jd["strict_deocc_lowdisp_lt0p25_bh"][te],
        "ongoing_part_occ_at_t0_114": jd["ongoing_part_t0"][te],
        "ongoing_full_occ_at_t0_74": jd["ongoing_full_t0"][te],
        "late_unoccluded_track_start_50": jd["late_unocc_start"][te],
        "broad_occ_or_late_union_406": jd["broad_occ_406"][te],
    }

    def _row4(a4: np.ndarray, m: np.ndarray) -> dict[str, float]:
        v = a4[m].mean(axis=0)
        return {
            "ade_2p0s_bbox_h": float(v[0]), "fde_2p0s_bbox_h": float(v[1]),
            "ade_post_onset_0p5s_bbox_h": float(v[2]), "ade_post_onset_1p0s_bbox_h": float(v[3]),
        }

    def _vci(d2: np.ndarray, m: np.ndarray) -> dict:
        if int(m.sum()) < 4 or len(set(v_te[m].tolist())) < 2:
            return {"n": int(m.sum())}
        return _loc_bootstrap_ci(d2[m], v_te[m])

    st_out = {}
    for sk, sm in strata.items():
        if int(sm.sum()) == 0:
            continue
        md = {"Last2_CV": _row4(err_cv, sm)}
        for nm, *_ in arms:
            md[nm] = _row4(em[nm], sm)
        st_out[sk] = {
            "n_windows": int(sm.sum()),
            "n_videos": int(len(set(v_te[sm].tolist()))),
            "unit": "bbox_heights",
            "horizon_s": 2.0,
            "models": md,
            "paired_video_ci": {
                "autopose_minus_bbox_only_2s": _vci(em["J_AutoPose_rtmpose17"][:, 0:2] - em["J_M0_bbox_motion_only"][:, 0:2], sm),
                "autopose_minus_bbox_only_post_onset": _vci(em["J_AutoPose_rtmpose17"][:, 2:4] - em["J_M0_bbox_motion_only"][:, 2:4], sm),
                "autopose_veh_minus_bbox_veh_post_onset": _vci(em["J_AutoPose_veh"][:, 2:4] - em["J_M2_bbox_veh"][:, 2:4], sm),
                "human_attr_minus_bbox_only_post_onset": _vci(em["J_HumanAttr_orient_gaze"][:, 2:4] - em["J_M0_bbox_motion_only"][:, 2:4], sm),
                "human_veh_minus_bbox_veh_post_onset": _vci(em["J_Full_human_veh"][:, 2:4] - em["J_M2_bbox_veh"][:, 2:4], sm),
                "combined_minus_bbox_veh_post_onset": _vci(em["J_Combined_auto_human_veh"][:, 2:4] - em["J_M2_bbox_veh"][:, 2:4], sm),
                "autopose_veh_minus_CV_2s": _vci(em["J_AutoPose_veh"][:, 0:2] - err_cv[:, 0:2], sm),
            },
        }

    # Action AP on strict_5frame_deoccluded_168 and broad_occ_or_late_union_406
    ap_summary = {}
    for sub_nm, sub_m in (("strict_5frame_deoccluded_168", jd["strict_5f_deocc"][te]), ("broad_occ_or_late_union_406", jd["broad_occ_406"][te]), ("all_locked_test_953", np.ones(len(te), dtype=bool))):
        ap_summary[sub_nm] = {}
        for nm, *_ in arms:
            pm = ap_probs[nm].mean(axis=0)
            cls_ap = {}
            for ci, cn in enumerate(("Start", "Stop", "Cross")):
                vm = sub_m & jd["v_act"][te, ci]
                yt = jd["y_act"][te, ci][vm]
                cls_ap[cn] = {"pos": int(yt.sum()), "ap": float(_binary_average_precision(yt.astype(np.float64), pm[vm, ci])) if yt.sum() > 0 else 0.0}
            ap_summary[sub_nm][nm] = cls_ap

    return {
        "n_total_windows": int(len(sp)),
        "split_counts": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "n_rtmpose_person_frames_extracted": int(jd["n_rtmpose_person_frames"]),
        "unit": "bbox_heights",
        "horizon_s": 2.0,
        "test_strata": st_out,
        "maneuver_ap_by_stratum": ap_summary,
    }


def _fit_and_predict_lead_arms(
    kin: np.ndarray,
    arms_spec: list[tuple[str, np.ndarray, np.ndarray, bool, bool]],
    veh: np.ndarray,
    y: np.ndarray,
    tau_s: np.ndarray,
    is_neg: np.ndarray,
    split: np.ndarray,
    seeds: list[int],
    device: torch.device,
    epochs: int = 26,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    tr = np.flatnonzero(split == "train")
    va = np.flatnonzero(split == "val")
    te = np.flatnonzero(split == "test")
    kt = torch.from_numpy(kin).to(device)
    vt = torch.from_numpy(veh).to(device)
    yt = torch.from_numpy(y).to(device)
    dt_tgt = torch.from_numpy(np.clip(-tau_s, 0.0, 2.0).astype(np.float32)).to(device)
    nt = torch.from_numpy(is_neg).to(device)
    tr_t, va_t, te_t = torch.from_numpy(tr).to(device), torch.from_numpy(va).to(device), torch.from_numpy(te).to(device)
    pw = ((len(tr) - yt[tr_t].sum(0)) / yt[tr_t].sum(0).clamp_min(1.0)).clamp(1.0, 12.0)

    probs_te = {nm: np.zeros((len(seeds), len(te), 3), dtype=np.float64) for nm, *_ in arms_spec}
    probs_va = {nm: np.zeros((len(seeds), len(va), 3), dtype=np.float64) for nm, *_ in arms_spec}
    for si, seed in enumerate(seeds):
        for nm, p_arr, q_arr, up, un in arms_spec:
            pt = torch.from_numpy(p_arr).to(device)
            qt = torch.from_numpy(q_arr).to(device)
            torch.manual_seed(seed + 31)
            net = CausalAPAPrecursorLeadNet(kin.shape[-1], p_arr.shape[-1], veh.shape[-1], use_pose=up, use_nbr=un).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=2e-4)
            gen = torch.Generator().manual_seed(seed + 211)
            best_sd, best_vl = None, 1e9
            for _ in range(epochs):
                net.train()
                for b in torch.randperm(len(tr_t), generator=gen).split(128):
                    ib = tr_t[b.to(device)]
                    opt.zero_grad(set_to_none=True)
                    lg, dl = net(kt[ib], pt[ib], qt[ib], vt[ib])
                    l_cls = F.binary_cross_entropy_with_logits(lg, yt[ib], pos_weight=pw)
                    pos_m = ~nt[ib]
                    l_dl = F.smooth_l1_loss(dl[pos_m], dt_tgt[ib][pos_m]) if pos_m.any() else lg.new_tensor(0.0)
                    (l_cls + 0.15 * l_dl).backward()
                    opt.step()
                net.eval()
                with torch.inference_mode():
                    lg_v, _ = net(kt[va_t], pt[va_t], qt[va_t], vt[va_t])
                    vl = float(F.binary_cross_entropy_with_logits(lg_v, yt[va_t]).item())
                    if vl < best_vl:
                        best_vl = vl
                        best_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if best_sd is not None:
                net.load_state_dict(best_sd)
            net.eval()
            with torch.inference_mode():
                probs_va[nm][si] = torch.sigmoid(net(kt[va_t], pt[va_t], qt[va_t], vt[va_t])[0]).cpu().numpy()
                probs_te[nm][si] = torch.sigmoid(net(kt[te_t], pt[te_t], qt[te_t], vt[te_t])[0]).cpu().numpy()
    return {k: v.mean(0) for k, v in probs_va.items()}, {k: v.mean(0) for k, v in probs_te.items()}


def _score_lead_operating_points(
    p_va: dict[str, np.ndarray],
    p_te: dict[str, np.ndarray],
    ds: dict[str, np.ndarray],
    subsets: dict[str, np.ndarray],
    pairs: list[tuple[str, str, str]],
) -> dict:
    va = np.flatnonzero(ds["split"] == "val")
    te = np.flatnonzero(ds["split"] == "test")
    neg_va, neg_te = ds["is_neg"][va], ds["is_neg"][te]
    tau_te, eid_te, cls_te = ds["tau_s"][te], ds["event_id"][te], ds["event_cls"][te]

    def _clean(d: dict) -> dict:
        return {c: {k: v for k, v in cv.items() if k not in ("pre_hit_lead_ms_by_eid", "event_ids_ordered")} for c, cv in d.items()}

    out: dict[str, dict] = {"iso_test_fpr": {}, "val_calibrated_fpr_0p05": {}}
    for fpr_t in (0.025, 0.05, 0.10):
        fk = f"fpr_{fpr_t:.3f}".replace(".", "p")
        out["iso_test_fpr"][fk] = {}
        for sub_nm, sub_m in subsets.items():
            eval_m = neg_te | sub_m
            raw_m = {nm: evaluate_iso_test_fpr_and_unconflated_leadtime(
                p_te[nm][eval_m], tau_te[eval_m], eid_te[eval_m], cls_te[eval_m], neg_te[eval_m], target_test_fpr=fpr_t,
            ) for nm in p_te}
            out["iso_test_fpr"][fk][sub_nm] = {
                "models": {nm: _clean(raw_m[nm]) for nm in raw_m},
                "apple_to_apple_pairs": {tag: paired_iso_fpr_comparison(raw_m[mb], raw_m[ma]) for tag, mb, ma in pairs},
            }

    for sub_nm, sub_m in subsets.items():
        eval_m = neg_te | sub_m
        raw_v = {}
        for nm in p_te:
            thr_v = calibrate_matched_fpr_thresholds(p_va[nm][neg_va], target_fpr=0.05)
            raw_v[nm] = evaluate_iso_test_fpr_and_unconflated_leadtime(
                p_te[nm][eval_m], tau_te[eval_m], eid_te[eval_m], cls_te[eval_m], neg_te[eval_m], explicit_thr=thr_v,
            )
        out["val_calibrated_fpr_0p05"][sub_nm] = {
            "models": {nm: _clean(raw_v[nm]) for nm in raw_v},
            "apple_to_apple_pairs": {tag: paired_iso_fpr_comparison(raw_v[mb], raw_v[ma]) for tag, mb, ma in pairs},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch-dir", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    traj_ds, slide_ds = build_jaad2_strict_and_sliding_cohorts(args.scratch_dir)
    jaad_traj = eval_jaad2_trajectory_factorial_5seed(traj_ds, list(SEEDS), device)

    j_arms = [
        ("Lead_M0_kin_only", slide_ds["auto_pose"], slide_ds["auto_q"], False, False),
        ("Lead_AutoPose_rtmpose17", slide_ds["auto_pose"], slide_ds["auto_q"], True, False),
        ("Lead_AutoPose_veh", slide_ds["auto_pose"], slide_ds["auto_q"], True, True),
        ("Lead_HumanAttr_orient_gaze", slide_ds["human_attr"], slide_ds["human_q"], True, False),
        ("Lead_M2_kin_veh", slide_ds["auto_pose"], slide_ds["auto_q"], False, True),
        ("Lead_Full_human_veh", slide_ds["human_attr"], slide_ds["human_q"], True, True),
    ]
    p_va_j, p_te_j = _fit_and_predict_lead_arms(
        slide_ds["kin"], j_arms, slide_ds["veh"], slide_ds["y"], slide_ds["tau_s"], slide_ds["is_neg"], slide_ds["split"], list(SEEDS), device,
    )
    te_j = np.flatnonzero(slide_ds["split"] == "test")
    j_subs = {
        "locked_test_all_events": ~slide_ds["is_neg"][te_j],
        "locked_test_post_deocclusion_broad": (~slide_ds["is_neg"][te_j]) & slide_ds["is_deocc_broad"][te_j],
        "locked_test_post_deocclusion_strict_recovered": (~slide_ds["is_neg"][te_j]) & slide_ds["is_deocc_strict_recovered"][te_j],
    }
    j_pairs = [
        ("autopose_vs_kin_only", "Lead_AutoPose_rtmpose17", "Lead_M0_kin_only"),
        ("human_attr_vs_kin_only", "Lead_HumanAttr_orient_gaze", "Lead_M0_kin_only"),
        ("autopose_veh_vs_veh_only", "Lead_AutoPose_veh", "Lead_M2_kin_veh"),
        ("human_veh_vs_veh_only", "Lead_Full_human_veh", "Lead_M2_kin_veh"),
        ("human_attr_vs_autopose", "Lead_HumanAttr_orient_gaze", "Lead_AutoPose_rtmpose17"),
    ]
    jaad_lead = _score_lead_operating_points(p_va_j, p_te_j, slide_ds, j_subs, j_pairs)
    jaad_lead["n_sliding_windows"] = int(len(slide_ds["split"]))

    # Also evaluate exact iso-Test-FPR unconflated lead-time on JTA-228 native 3D skeleton
    st_zip = args.scratch_dir / "st_releases.zip"
    if not st_zip.is_file():
        _download_verified_zip(ST_RELEASE_URL, st_zip)
    jta_ds = extract_jta_sliding_leadtime_events(st_zip)
    t_arms = [
        ("Lead_M0_kin_only", jta_ds["pose"], jta_ds["q"], False, False),
        ("Lead_M1_3dpose", jta_ds["pose"], jta_ds["q"], True, False),
        ("Lead_M2_kin_nbr", jta_ds["pose"], jta_ds["q"], False, True),
        ("Lead_Full_3dpose_nbr", jta_ds["pose"], jta_ds["q"], True, True),
    ]
    p_va_t, p_te_t = _fit_and_predict_lead_arms(
        jta_ds["kin"], t_arms, jta_ds["veh"], jta_ds["y"], jta_ds["tau_s"], jta_ds["is_neg"], jta_ds["split"], list(SEEDS), device, epochs=22,
    )
    te_t = np.flatnonzero(jta_ds["split"] == "test")
    t_subs = {"locked_test_all_events": ~jta_ds["is_neg"][te_t]}
    t_pairs = [
        ("pose3d_vs_kin_only", "Lead_M1_3dpose", "Lead_M0_kin_only"),
        ("full3d_vs_nbr_only", "Lead_Full_3dpose_nbr", "Lead_M2_kin_nbr"),
    ]
    jta_lead = _score_lead_operating_points(p_va_t, p_te_t, jta_ds, t_subs, t_pairs)
    jta_lead["n_sliding_windows"] = int(len(jta_ds["split"]))

    summary = {
        "experiment": "wave23_jaad_autopose_and_isofpr_leadtime",
        "jaad2_trajectory_autopose_vs_human_5seed": jaad_traj,
        "jaad2_sliding_isofpr_leadtime_5seed": jaad_lead,
        "jta228_sliding_isofpr_leadtime_5seed": jta_lead,
    }
    raw = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = (json.dumps({"summary.json": hashlib.sha256(raw).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw)
        zf.writestr("sha256.json", sh)
    print("SAVED:", args.zip_path, args.zip_path.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
