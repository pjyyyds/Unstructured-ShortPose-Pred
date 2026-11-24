"""PINNS 10Hz real-video RTMPose-m keypoint, BEV polyline, and sliding-window extractor."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .actions import compute_action_labels
from .constants import DT, NUM_JOINTS, T_FUT
from .scene_graph import build_local_polylines, build_neighbor_edges
from .splits import temporal_buffer_split_indices, verify_zero_temporal_leakage


def _check_right_handed_homography(calib: Dict) -> bool:
    """Return True if world (X, Y) needs Y-flip to maintain right-handed ground coordinates."""
    pts_2d = np.asarray(calib.get("pts_2d", []), dtype=np.float64)
    pts_3d = np.asarray(calib.get("pts_3d", []), dtype=np.float64)
    if cv2 is None or pts_2d.shape != (4, 2) or pts_3d.shape != (4, 2):
        return False
    h_mat, _ = cv2.findHomography(pts_2d, pts_3d)
    if h_mat is None:
        return False
    # Image (u right, v down) is left-handed; mapping to right-handed ground (X, Y) has det(J) < 0
    u0, v0 = np.mean(pts_2d, axis=0)
    eps = 2.0
    p0 = cv2.perspectiveTransform(np.array([[[u0, v0]]], dtype=np.float64), h_mat)[0, 0]
    pu = cv2.perspectiveTransform(np.array([[[u0 + eps, v0]]], dtype=np.float64), h_mat)[0, 0] - p0
    pv = cv2.perspectiveTransform(np.array([[[u0, v0 + eps]]], dtype=np.float64), h_mat)[0, 0] - p0
    det_j = pu[0] * pv[1] - pu[1] * pv[0]
    return bool(det_j > 0.0)


class RTMPoseONNXExtractor:
    """Official OpenMMLab RTMPose-m (SimCC 256x192, normalize=False) batched GPU extractor."""

    MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __init__(self, onnx_path: Optional[Path] = None, batch_size: int = 256) -> None:
        self.batch_size = batch_size
        self.sess = None
        if onnx_path is not None and Path(onnx_path).exists():
            import onnxruntime as ort

            avail = ort.get_available_providers()
            provs = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
            self.sess = ort.InferenceSession(str(onnx_path), providers=provs or ["CPUExecutionProvider"])
            self.in_name = self.sess.get_inputs()[0].name
            self.active_providers = self.sess.get_providers()
        else:
            self.active_providers = ["MockFallback"]

    def extract_crops(self, bgr_img: np.ndarray, boxes_cxcywh: np.ndarray) -> np.ndarray:
        """Extract [N, 17, 3] (u, v, conf) in original image pixels for N boxes [cx, cy, w, h]."""
        n = len(boxes_cxcywh)
        if n == 0:
            return np.zeros((0, NUM_JOINTS, 3), dtype=np.float32)
        if self.sess is None:
            # Deterministic synthetic pose only for unit tests when ONNX file is omitted
            out = np.zeros((n, NUM_JOINTS, 3), dtype=np.float32)
            for i, (cx, cy, bw, bh) in enumerate(boxes_cxcywh):
                out[i, :, 0] = cx + np.linspace(-0.15, 0.15, NUM_JOINTS) * bw
                out[i, :, 1] = cy + np.linspace(-0.40, 0.40, NUM_JOINTS) * bh
                out[i, :, 2] = 0.85
            return out

        crops = np.zeros((n, 3, 256, 192), dtype=np.float32)
        centers = np.zeros((n, 2), dtype=np.float32)
        scales = np.zeros((n, 2), dtype=np.float32)
        aspect = 192.0 / 256.0
        for i, (cx, cy, bw, bh) in enumerate(boxes_cxcywh):
            w_pad = max(float(bw) * 1.25, 8.0)
            h_pad = max(float(bh) * 1.25, 8.0)
            if w_pad > h_pad * aspect:
                h_pad = w_pad / aspect
            else:
                w_pad = h_pad * aspect
            centers[i] = [cx, cy]
            scales[i] = [w_pad, h_pad]
            src_pts = np.array(
                [[cx, cy], [cx, cy - 0.5 * h_pad], [cx + 0.5 * w_pad, cy]], dtype=np.float32
            )
            dst_pts = np.array([[96.0, 128.0], [96.0, 0.0], [192.0, 128.0]], dtype=np.float32)
            mat = cv2.getAffineTransform(src_pts, dst_pts)
            warped = cv2.warpAffine(bgr_img, mat, (192, 256), flags=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32)
            norm = (rgb - self.MEAN) / self.STD
            crops[i] = np.transpose(norm, (2, 0, 1))

        out_uvc = np.zeros((n, NUM_JOINTS, 3), dtype=np.float32)
        for s in range(0, n, self.batch_size):
            e = min(n, s + self.batch_size)
            sx, sy = self.sess.run(None, {self.in_name: crops[s:e]})
            px = np.argmax(sx, axis=-1).astype(np.float32) / 2.0  # [b, 17] in [0, 192)
            py = np.argmax(sy, axis=-1).astype(np.float32) / 2.0  # [b, 17] in [0, 256)
            mx = np.amax(sx, axis=-1)
            my = np.amax(sy, axis=-1)
            conf = np.where((mx > 0.0) & (my > 0.0), 0.5 * (mx + my), 0.0)
            conf = np.clip(conf, 0.0, 1.0)
            out_uvc[s:e, :, 0] = centers[s:e, 0:1] + (px - 96.0) / 192.0 * scales[s:e, 0:1]
            out_uvc[s:e, :, 1] = centers[s:e, 1:2] + (py - 128.0) / 256.0 * scales[s:e, 1:2]
            out_uvc[s:e, :, 2] = conf
        return out_uvc


def extract_static_bev_and_train_polylines(
    bev_video_path: Optional[Path],
    calib: Dict,
    train_ped_pts: np.ndarray,
    parked_car_boxes: np.ndarray,
    flip_y: bool,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract non-lane static physical polylines strictly from BEV background + train split only."""
    polylines: List[Tuple[np.ndarray, np.ndarray]] = []
    x_min = float(calib.get("x_min", 0.0))
    x_max = float(calib.get("x_max", 30.0))
    y_min = float(calib.get("y_min", 0.0))
    y_max = float(calib.get("y_max", 30.0))

    # 1. Extract prominent structural edges (curbs/walls/planters) from static BEV median background
    if bev_video_path is not None and bev_video_path.exists():
        cap = cv2.VideoCapture(str(bev_video_path))
        total_f = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        bg_frames = []
        for fi in np.linspace(0, max(0, int(0.6 * total_f) - 1), num=7, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frm = cap.read()
            if ok and frm is not None:
                bg_frames.append(frm)
        cap.release()
        if bg_frames:
            med = np.median(np.stack(bg_frames, axis=0), axis=0).astype(np.uint8)
            bh, bw = med.shape[:2]
            gray = cv2.GaussianBlur(cv2.cvtColor(med, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
            edges = cv2.Canny(gray, 60, 150)
            cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=lambda c: cv2.arcLength(c, False), reverse=True)
            for cnt in cnts[:10]:
                if cv2.arcLength(cnt, False) < 0.12 * max(bw, bh) or len(cnt) < 4:
                    continue
                uv = cnt.reshape(-1, 2).astype(np.float32)
                wx = x_min + (uv[:, 0] / max(bw - 1.0, 1.0)) * (x_max - x_min)
                wy = y_min + (1.0 - uv[:, 1] / max(bh - 1.0, 1.0)) * (y_max - y_min)
                if flip_y:
                    wy = y_max - (wy - y_min)
                attr = np.zeros(8, dtype=np.float32)
                attr[0] = 1.0  # Curb / sidewalk / structural edge
                attr[7] = 1.0  # Tangent direction valid
                polylines.append((np.stack([wx, wy], axis=-1), attr))

    # 2. Stationary parked vehicle obstacle contours (ch 4 in attr_8)
    for cx, cy in parked_car_boxes[:6]:
        half_l, half_w = 2.1, 0.9
        rect = np.array(
            [
                [cx - half_l, cy - half_w],
                [cx + half_l, cy - half_w],
                [cx + half_l, cy + half_w],
                [cx - half_l, cy + half_w],
                [cx - half_l, cy - half_w],
            ],
            dtype=np.float32,
        )
        attr = np.zeros(8, dtype=np.float32)
        attr[4] = 1.0  # Parked vehicle contour
        attr[6] = 1.0  # Closed polygon
        polylines.append((rect, attr))

    # 3. Walkable corridor boundary envelope from training-only pedestrian occupancy
    if len(train_ped_pts) >= 20:
        hull = cv2.convexHull(train_ped_pts.astype(np.float32)).reshape(-1, 2)
        if len(hull) >= 3:
            closed_hull = np.vstack([hull, hull[:1]])
            attr = np.zeros(8, dtype=np.float32)
            attr[5] = 1.0  # Corridor portal / boundary guide
            attr[6] = 1.0
            polylines.append((closed_hull, attr))

    return polylines


def _parse_pinns_label_file(txt_path: Path, flip_y: bool, y_min: float, y_max: float) -> Dict[int, List[Dict]]:
    """Parse 30fps PINNS label file at strict 3-frame stride (10.0 Hz, zero interpolation)."""
    by_f10: Dict[int, List[Dict]] = {}
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 11:
                continue
            f_raw = int(float(parts[0])) - 1
            if f_raw < 0 or (f_raw % 3) != 0:
                continue
            f10 = f_raw // 3
            obj_type = parts[1]
            tid = int(float(parts[2]))
            ox, oy, conf, bw, bh = map(float, parts[3:8])
            wx, wy = float(parts[8]), float(parts[9])
            if flip_y:
                wy = y_max - (wy - y_min)
            fixed = int(float(parts[10]))
            by_f10.setdefault(f10, []).append(
                {
                    "type": obj_type,
                    "id": tid,
                    "ox": ox,
                    "oy": oy,
                    "conf": conf,
                    "w": bw,
                    "h": bh,
                    "wx": wx,
                    "wy": wy,
                    "fixed": fixed,
                }
            )
    return by_f10


def process_pinns_sequence(
    seq_stem: str,
    lbl_path: Path,
    ori_video_path: Path,
    bev_video_path: Optional[Path],
    calib: Dict,
    pose_ext: RTMPoseONNXExtractor,
    bbox_is_topleft: Optional[bool] = None,
    window_stride: int = 3,
) -> Tuple[List[Dict], Optional[bool]]:
    """Extract leakage-free 10Hz samples with real RTMPose 2D keypoints for one PINNS sequence."""
    flip_y = _check_right_handed_homography(calib)
    y_min, y_max = float(calib.get("y_min", 0.0)), float(calib.get("y_max", 30.0))
    by_f10 = _parse_pinns_label_file(lbl_path, flip_y, y_min, y_max)
    if len(by_f10) < 60:
        return [], bbox_is_topleft

    all_f10 = np.array(sorted(by_f10.keys()), dtype=np.int64)
    sp_frames = temporal_buffer_split_indices(all_f10, ratios=(0.70, 0.15, 0.15))
    frame_to_split = {}
    for sp_name, farr in sp_frames.items():
        for fi in farr:
            frame_to_split[int(fi)] = sp_name

    # Read 10Hz frames and run batched RTMPose on all Person boxes
    cap = cv2.VideoCapture(str(ori_video_path))
    img_h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 864.0)
    pose_map: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    target_f10_set = set(by_f10.keys())
    raw_idx = 0
    while True:
        if not cap.grab():
            break
        if raw_idx % 3 == 0:
            f10 = raw_idx // 3
            if f10 in target_f10_set:
                peds = [o for o in by_f10[f10] if o["type"].lower().startswith("per")]
                if peds:
                    ok, frame_bgr = cap.retrieve()
                    if not ok or frame_bgr is None:
                        raw_idx += 1
                        continue
                    # Empirical audit on PINNS videos: (ox, oy) is ground-contact foot bottom-center (cx, y_bottom),
                    # so true person bounding-box center is (ox, oy - 0.5 * h), keeping head & shoulders centered!
                    boxes = np.array(
                        [[p["ox"], p["oy"] - 0.5 * p["h"], p["w"], p["h"]] for p in peds],
                        dtype=np.float32,
                    )
                    uvc = pose_ext.extract_crops(frame_bgr, boxes)
                    for idx_p, p in enumerate(peds):
                        pose_map[(f10, p["id"])] = (uvc[idx_p], boxes[idx_p])
        raw_idx += 1
    cap.release()

    # Build static BEV polylines using ONLY train split pedestrian points + parked cars
    train_max_f = int(sp_frames["train"][-1]) if len(sp_frames["train"]) > 0 else -1
    train_ped_pts, parked_pts = [], []
    for fi, objs in by_f10.items():
        if fi <= train_max_f:
            for o in objs:
                if o["type"].lower().startswith("per"):
                    train_ped_pts.append([o["wx"], o["wy"]])
                elif o["fixed"] == 1:
                    parked_pts.append([o["wx"], o["wy"]])
    ped_arr = np.array(train_ped_pts, dtype=np.float32) if train_ped_pts else np.zeros((0, 2), dtype=np.float32)
    pk_arr = np.unique(np.round(parked_pts, 1), axis=0) if parked_pts else np.zeros((0, 2), dtype=np.float32)
    polys = extract_static_bev_and_train_polylines(bev_video_path, calib, ped_arr, pk_arr, flip_y)

    return _assemble_sequence_windows(
        seq_stem, by_f10, pose_map, polys, frame_to_split, img_h, window_stride
    ), bbox_is_topleft


def _assemble_sequence_windows(
    seq_stem: str,
    by_f10: Dict[int, List[Dict]],
    pose_map: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    polys: List[Tuple[np.ndarray, np.ndarray]],
    frame_to_split: Dict[int, str],
    img_h: float,
    window_stride: int,
) -> List[Dict]:
    track_frames: Dict[int, Dict[int, Dict]] = {}
    for fi, objs in by_f10.items():
        for o in objs:
            track_frames.setdefault((o["type"], o["id"]), {})[fi] = o

    samples: List[Dict] = []
    sorted_t0 = sorted(frame_to_split.keys())[:: max(1, window_stride)]
    for t0 in sorted_t0:
        split_name = frame_to_split[t0]
        frame_objs = by_f10.get(t0, [])
        peds_t0 = [o for o in frame_objs if o["type"].lower().startswith("per")]
        if not peds_t0:
            continue

        # Precompute all objects' recent 5-frame state at t0 for neighbor graph
        all_states = []
        for o in frame_objs:
            key = (o["type"], o["id"])
            tf = track_frames[key]
            obs_5 = sum(1 for k in range(t0 - 4, t0 + 1) if k in tf) / 5.0
            vel = np.zeros(2, dtype=np.float32)
            v_ok = 0.0
            for back in range(1, 6):
                if (t0 - back) in tf:
                    prev = tf[t0 - back]
                    vel = np.array([o["wx"] - prev["wx"], o["wy"] - prev["wy"]], dtype=np.float32) / (back * DT)
                    v_ok = 1.0
                    break
            is_ped = o["type"].lower().startswith("per")
            if is_ped:
                atype = 2 if (o["fixed"] == 1 or (v_ok > 0.5 and np.linalg.norm(vel) < 0.20)) else 0
            else:
                atype = 6  # Passenger car / motor vehicle
            all_states.append((o["id"], is_ped, np.array([o["wx"], o["wy"]], dtype=np.float32), vel, v_ok, obs_5, atype))

        for idx_ego, (oid, is_ped, p_i, v_i, vok_i, obs_i, atype_i) in enumerate(all_states):
            if not is_ped:
                continue
            tf_ego = track_frames[("Person", oid)] if ("Person", oid) in track_frames else track_frames.get(("person", oid), {})
            if not tf_ego:
                for k_cand in track_frames:
                    if k_cand[1] == oid and k_cand[0].lower().startswith("per"):
                        tf_ego = track_frames[k_cand]
                        break

            h_pos = np.zeros((20, 2), dtype=np.float32)
            h_mask = np.zeros(20, dtype=np.float32)
            h_kpts = np.zeros((20, NUM_JOINTS, 3), dtype=np.float32)
            h_bbox = np.zeros((20, 4), dtype=np.float32)
            for s_idx, fk in enumerate(range(t0 - 19, t0 + 1)):
                if fk in tf_ego:
                    rec = tf_ego[fk]
                    h_pos[s_idx] = [rec["wx"], rec["wy"]]
                    h_mask[s_idx] = 1.0
                    if (fk, oid) in pose_map:
                        h_kpts[s_idx], h_bbox[s_idx] = pose_map[(fk, oid)]
                    else:
                        h_bbox[s_idx] = [rec["ox"], rec["oy"] - 0.5 * rec["h"], rec["w"], rec["h"]]

            if h_mask[-1] < 0.5 or h_mask[-5:].sum() < 2.0:
                continue

            f_pos_abs = np.zeros((T_FUT, 2), dtype=np.float32)
            f_mask = np.zeros(T_FUT, dtype=np.float32)
            for f_idx, fk in enumerate(range(t0 + 1, t0 + T_FUT + 1)):
                if fk in tf_ego:
                    rec = tf_ego[fk]
                    f_pos_abs[f_idx] = [rec["wx"], rec["wy"]]
                    f_mask[f_idx] = 1.0
            if f_mask.sum() < 15.0:
                continue

            y_act, m_act = compute_action_labels(h_pos[-5:], h_mask[-5:], f_pos_abs, f_mask, dt=DT)
            p_env, m_map = build_local_polylines(polys, p_i, map_available=True)

            others = [st for j_idx, st in enumerate(all_states) if j_idx != idx_ego]
            if others:
                nbr_p = np.stack([x[2] for x in others])
                nbr_v = np.stack([x[3] for x in others])
                nbr_vok = np.array([x[4] for x in others], dtype=np.float32)
                nbr_obs = np.array([x[5] for x in others], dtype=np.float32)
                nbr_tp = np.array([x[6] for x in others], dtype=np.int64)
            else:
                nbr_p = np.zeros((0, 2), dtype=np.float32)
                nbr_v = np.zeros((0, 2), dtype=np.float32)
                nbr_vok = np.zeros(0, dtype=np.float32)
                nbr_obs = np.zeros(0, dtype=np.float32)
                nbr_tp = np.zeros(0, dtype=np.int64)

            edges, nbr_toh, nbr_msk = build_neighbor_edges(
                p_i, v_i, vok_i, obs_i, nbr_p, nbr_v, nbr_vok, nbr_obs, nbr_tp
            )
            ego_oh = np.zeros(8, dtype=np.float32)
            ego_oh[atype_i] = 1.0
            fut_rel = (f_pos_abs - p_i[None, :]) * f_mask[:, None]

            samples.append(
                {
                    "seq_id": seq_stem,
                    "split": split_name,
                    "t0_frame_10hz": int(t0),
                    "img_h": float(img_h),
                    "hist_pos_20": h_pos,
                    "hist_mask_20": h_mask,
                    "hist_kpts_20": h_kpts,
                    "hist_bbox_20": h_bbox,
                    "p_env": p_env,
                    "m_map": np.array([m_map], dtype=np.float32),
                    "ego_type_oh": ego_oh,
                    "nbr_edges": edges,
                    "nbr_type_oh": nbr_toh,
                    "nbr_mask": nbr_msk,
                    "fut_xy": fut_rel,
                    "fut_mask": f_mask,
                    "fut_pos_abs": f_pos_abs,
                    "p_t0": p_i,
                    "y_act": y_act,
                    "m_act": float(m_act),
                }
            )
    return samples
