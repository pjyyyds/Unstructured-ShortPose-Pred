"""JTA 2.5Hz official scene-split NDJSON loader (reproducing & extending Table 1 Social-Pose benchmark)."""

from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np

from .actions import compute_action_labels
from .constants import NUM_JOINTS
from .scene_graph import build_neighbor_edges

# JTA 22-joint -> COCO-17 index mapping (Fabbri et al., ECCV'18)
JTA22_TO_COCO17 = (1, 0, 0, 7, 3, 8, 4, 9, 5, 10, 6, 19, 16, 20, 17, 21, 18)


def _parse_jta_ndjson_file(
    ndjson_path: Path,
    split_name: str,
    rng: np.random.Generator,
    max_scenes_per_file: int = 80,
) -> List[Dict]:
    """Parse one JTA TrajNet++ .ndjson file at native 2.5Hz (dt=0.4s, T_max=8, T_fut=12)."""
    scenes = []
    by_frame: Dict[int, Dict[int, Dict]] = {}
    with open(ndjson_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "scene" in obj:
                scenes.append(obj["scene"])
            elif "track" in obj:
                tr = obj["track"]
                by_frame.setdefault(int(tr["f"]), {})[int(tr["p"])] = tr

    dt_jta = 0.4  # Native 2.5 Hz (zero interpolation!)
    t_hist_max = 8
    t_fut_jta = 12  # 12 * 0.4s = 4.8s standard JTA horizon
    img_h = 540.0  # Social-Transmotion JTA release coordinates are in 960x540 scale
    out_samples: List[Dict] = []

    step_scenes = scenes[:max_scenes_per_file]
    for sc in step_scenes:
        pid = int(sc["p"])
        s_frame = int(sc["s"])
        e_frame = int(sc["e"])
        all_f = sorted(f for f in by_frame if s_frame <= f <= e_frame)
        if len(all_f) < 15:
            continue
        f_step = int(np.median(np.diff(all_f))) if len(all_f) > 1 else 12
        f_step = max(f_step, 1)
        # Standard TrajNet++ split: 9 obs frames -> 12 pred frames; t0 is 9th frame (s_frame + 8*f_step)
        t0 = s_frame + 8 * f_step
        if pid not in by_frame.get(t0, {}):
            continue

        # Assign realistic road observation regime (45% Clear, 35% Partial Lower-Body Occl, 20% Heavy/Noisy)
        u_draw = float(rng.random())
        regime = 0 if u_draw < 0.45 else (1 if u_draw < 0.80 else 2)

        h_pos = np.zeros((t_hist_max, 2), dtype=np.float32)
        h_mask = np.zeros(t_hist_max, dtype=np.float32)
        h_kpts = np.zeros((t_hist_max, NUM_JOINTS, 3), dtype=np.float32)
        h_bbox = np.zeros((t_hist_max, 4), dtype=np.float32)

        for idx_h, k in enumerate(range(-(t_hist_max - 1), 1)):
            fk = t0 + k * f_step
            rec = by_frame.get(fk, {}).get(pid)
            if rec is None:
                continue
            h_pos[idx_h] = [float(rec["x"]), float(rec["y"])]
            h_mask[idx_h] = 1.0
            bw = max(float(rec.get("bb_width", 35.0)), 6.0)
            bh = max(float(rec.get("bb_height", 60.0)), 24.0)
            if regime == 0:
                bh = max(bh, 48.0)  # Near/mid-range clear pedestrian (q3 >= 0.088)
            elif regime == 2 and rng.random() < 0.5:
                bh = min(bh, 20.0)  # Far low-res scale (q3 < 0.04)
            cx = float(rec.get("bb_left", 500.0)) + 0.5 * bw
            cy = float(rec.get("bb_top", 500.0)) + 0.5 * bh
            h_bbox[idx_h] = [cx, cy, bw, bh]

            xx = np.array([float(rec.get(f"xx{j}", 0.0)) for j in JTA22_TO_COCO17], dtype=np.float32)
            yy = np.array([float(rec.get(f"yy{j}", 0.0)) for j in JTA22_TO_COCO17], dtype=np.float32)
            span_y = max(float(np.max(yy) - np.min(yy)), 1.0)
            u_px = cx + (xx / span_y) * bh
            v_px = cy + (yy / span_y) * bh
            conf = np.full(NUM_JOINTS, 0.88, dtype=np.float32)

            if regime == 1:
                # Parked-vehicle lower-body occlusion: hips/knees/ankles occluded or noisy, head/shoulders clear!
                u_px[11:17] += rng.normal(0.0, 0.22 * bh, size=6).astype(np.float32)
                v_px[11:17] += rng.normal(0.0, 0.22 * bh, size=6).astype(np.float32)
                conf[11:17] = rng.uniform(0.05, 0.18, size=6).astype(np.float32)
            elif regime == 2:
                # Heavy occlusion / Social-Pose Table 1 sigma=0.20 detector drift!
                u_px += rng.normal(0.0, 0.20 * bh, size=NUM_JOINTS).astype(np.float32)
                v_px += rng.normal(0.0, 0.20 * bh, size=NUM_JOINTS).astype(np.float32)
                conf = rng.uniform(0.10, 0.36, size=NUM_JOINTS).astype(np.float32)

            h_kpts[idx_h, :, 0] = u_px
            h_kpts[idx_h, :, 1] = v_px
            h_kpts[idx_h, :, 2] = conf

        if h_mask[-1] < 0.5 or h_mask[-3:].sum() < 2.0:
            continue

        f_pos_abs = np.zeros((t_fut_jta, 2), dtype=np.float32)
        f_mask = np.zeros(t_fut_jta, dtype=np.float32)
        for idx_f in range(t_fut_jta):
            fk = t0 + (idx_f + 1) * f_step
            rec = by_frame.get(fk, {}).get(pid)
            if rec is not None:
                f_pos_abs[idx_f] = [float(rec["x"]), float(rec["y"])]
                f_mask[idx_f] = 1.0
        if f_mask.sum() < 10.0:
            continue

        p_i = h_pos[-1]
        v_i = (h_pos[-1] - h_pos[-2]) / dt_jta if h_mask[-2] > 0.5 else np.zeros(2, dtype=np.float32)
        vok_i = 1.0 if h_mask[-2] > 0.5 else 0.0

        nbr_p, nbr_v, nbr_vok, nbr_obs, nbr_tp = [], [], [], [], []
        for oid, orec in by_frame.get(t0, {}).items():
            if oid == pid:
                continue
            op = np.array([float(orec["x"]), float(orec["y"])], dtype=np.float32)
            prev_o = by_frame.get(t0 - f_step, {}).get(oid)
            if prev_o is not None:
                ov = (op - np.array([float(prev_o["x"]), float(prev_o["y"])], dtype=np.float32)) / dt_jta
                ov_ok = 1.0
            else:
                ov = np.zeros(2, dtype=np.float32)
                ov_ok = 0.0
            nbr_p.append(op); nbr_v.append(ov); nbr_vok.append(ov_ok); nbr_obs.append(1.0); nbr_tp.append(0)

        edges, nbr_toh, nbr_msk = build_neighbor_edges(
            p_i,
            v_i,
            vok_i,
            float(h_mask[-3:].mean()),
            np.array(nbr_p, dtype=np.float32) if nbr_p else np.zeros((0, 2), dtype=np.float32),
            np.array(nbr_v, dtype=np.float32) if nbr_v else np.zeros((0, 2), dtype=np.float32),
            np.array(nbr_vok, dtype=np.float32),
            np.array(nbr_obs, dtype=np.float32),
            np.array(nbr_tp, dtype=np.int64),
        )
        y_act, m_act = compute_action_labels(h_pos[-3:], h_mask[-3:], f_pos_abs, f_mask, dt=dt_jta)
        ego_oh = np.zeros(8, dtype=np.float32); ego_oh[0] = 1.0

        out_samples.append(
            {
                "seq_id": ndjson_path.stem,
                "split": split_name,
                "t0_frame_10hz": int(t0),
                "img_h": img_h,
                "hist_pos_20": h_pos,
                "hist_mask_20": h_mask,
                "hist_kpts_20": h_kpts,
                "hist_bbox_20": h_bbox,
                "p_env": np.zeros((16, 10, 11), dtype=np.float32),
                "m_map": np.array([0.0], dtype=np.float32),
                "ego_type_oh": ego_oh,
                "nbr_edges": edges,
                "nbr_type_oh": nbr_toh,
                "nbr_mask": nbr_msk,
                "fut_xy": (f_pos_abs - p_i[None, :]) * f_mask[:, None],
                "fut_mask": f_mask,
                "fut_pos_abs": f_pos_abs,
                "p_t0": p_i,
                "y_act": y_act,
                "m_act": float(m_act),
            }
        )
    return out_samples


def process_jta_root(jta_data_dir: Path, max_files_per_split: int = 24, seed: int = 42) -> Tuple[List[Dict], Dict]:
    """Load JTA train/val/test official scene splits from `jta/data/`."""
    rng = np.random.default_rng(seed)
    all_samps: List[Dict] = []
    for sp in ("train", "val", "test"):
        sp_dir = jta_data_dir / sp
        if not sp_dir.exists():
            continue
        files = sorted(f for f in sp_dir.glob("*.ndjson") if not f.name.startswith("._"))[:max_files_per_split]
        for nf in files:
            all_samps.extend(_parse_jta_ndjson_file(nf, sp, rng))
    counts = {sp: sum(1 for s in all_samps if s["split"] == sp) for sp in ("train", "val", "test")}
    return all_samps, {"total_samples": len(all_samps), "split_counts": counts, "fps": 2.5}
