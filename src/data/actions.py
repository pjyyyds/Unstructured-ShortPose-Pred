"""Table 7 observable 5-class dynamic action event labeler with m_act validity mask."""

from typing import Tuple
import numpy as np

from .constants import DT


def _max_consecutive_zeros(mask_bool: np.ndarray) -> int:
    max_gap, cur = 0, 0
    for val in mask_bool:
        cur = 0 if val else cur + 1
        if cur > max_gap:
            max_gap = cur
    return max_gap


def _first_sustained_run(cond: np.ndarray, min_len: int = 3) -> int:
    """Return earliest start index of >= min_len consecutive True values, else -1."""
    run = 0
    for i, flag in enumerate(cond):
        run = run + 1 if flag else 0
        if run >= min_len:
            return i - min_len + 1
    return -1


def _estimate_causal_velocities(
    pos_xy: np.ndarray,
    mask_bool: np.ndarray,
    dt: float = DT,
    max_gap_frames: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute causal 3-frame smoothed velocities (svx, svy, speed) without future leakage."""
    n = pos_xy.shape[0]
    raw_vx = np.zeros(n, dtype=np.float64)
    raw_vy = np.zeros(n, dtype=np.float64)
    vel_valid = np.zeros(n, dtype=bool)

    last_idx = -1
    for i in range(n):
        if mask_bool[i]:
            if last_idx >= 0 and (i - last_idx) <= max_gap_frames:
                gap_t = (i - last_idx) * dt
                raw_vx[i] = (pos_xy[i, 0] - pos_xy[last_idx, 0]) / gap_t
                raw_vy[i] = (pos_xy[i, 1] - pos_xy[last_idx, 1]) / gap_t
                vel_valid[i] = True
            last_idx = i

    svx = np.zeros(n, dtype=np.float64)
    svy = np.zeros(n, dtype=np.float64)
    out_valid = np.zeros(n, dtype=bool)
    for i in range(n):
        s = max(0, i - 2)  # W_theta = 3 frames (0.3s) causal window
        w_valid = vel_valid[s : i + 1]
        if np.any(w_valid):
            svx[i] = float(np.mean(raw_vx[s : i + 1][w_valid]))
            svy[i] = float(np.mean(raw_vy[s : i + 1][w_valid]))
            out_valid[i] = True
    speed = np.hypot(svx, svy)
    return svx, svy, np.where(out_valid, speed, np.nan)


def compute_action_labels(
    hist_pos: np.ndarray,
    hist_mask: np.ndarray,
    fut_pos: np.ndarray,
    fut_mask: np.ndarray,
    dt: float = DT,
    speed_scale: float = 1.0,
    angle_scale: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Compute Table 7 5-class observable action labels y_act in {0,1}^5 and m_act in {0,1}."""
    h_mask = np.asarray(hist_mask, dtype=np.float32) > 0.5
    f_mask = np.asarray(fut_mask, dtype=np.float32) > 0.5
    y_act = np.zeros(5, dtype=np.float32)

    pre_win = h_mask[-3:] if len(h_mask) >= 3 else h_mask
    rho_fut = float(np.mean(f_mask)) if len(f_mask) > 0 else 0.0
    if int(np.sum(pre_win)) < 2 or rho_fut < 0.80 - 1e-6 or _max_consecutive_zeros(f_mask) > 3:
        return y_act, 0.0

    all_pos = np.vstack([hist_pos, fut_pos]).astype(np.float64)
    all_mask = np.concatenate([h_mask, f_mask])
    t0_idx = len(hist_pos) - 1
    svx, svy, spd = _estimate_causal_velocities(all_pos, all_mask, dt=dt)

    pre_slice = spd[max(0, t0_idx - 2) : t0_idx + 1]
    valid_pre = pre_slice[~np.isnan(pre_slice)]
    if len(valid_pre) == 0:
        return y_act, 0.0
    v_pre = float(np.mean(valid_pre))

    v_low = 0.20 * speed_scale
    v_high = 0.60 * speed_scale
    v_turn = 0.50 * speed_scale
    ang_thr = np.deg2rad(30.0 * angle_scale)

    fut_spd = spd[t0_idx + 1 :]
    t_start = _first_sustained_run((~np.isnan(fut_spd)) & (fut_spd >= v_high), min_len=3) if v_pre <= v_low else -1
    t_stop = _first_sustained_run((~np.isnan(fut_spd)) & (fut_spd <= v_low), min_len=3) if v_pre >= v_high else -1
    if t_start >= 0 and (t_stop < 0 or t_start <= t_stop):
        y_act[0] = 1.0
    elif t_stop >= 0:
        y_act[1] = 1.0

    # Contiguous moving segments starting no earlier than t0_idx - 2
    s_anchor = max(0, t0_idx - 2)
    moving = (~np.isnan(spd[s_anchor:])) & (spd[s_anchor:] >= v_turn)
    best_left, best_right = 10**9, 10**9
    idx = 0
    while idx < len(moving):
        if not moving[idx]:
            idx += 1
            continue
        end = idx
        while end < len(moving) and moving[end]:
            end += 1
        if end - idx >= 3:
            seg_svx = svx[s_anchor + idx : s_anchor + end]
            seg_svy = svy[s_anchor + idx : s_anchor + end]
            seg_theta = np.unwrap(np.arctan2(seg_svy, seg_svx))
            dtheta = seg_theta - seg_theta[0]
            fut_offset = max(0, (t0_idx + 1) - (s_anchor + idx))
            if fut_offset < len(dtheta):
                d_fut = dtheta[fut_offset:]
                tl = _first_sustained_run(d_fut >= ang_thr, min_len=3)
                tr = _first_sustained_run(d_fut <= -ang_thr, min_len=3)
                if tl >= 0:
                    best_left = min(best_left, s_anchor + idx + fut_offset + tl)
                if tr >= 0:
                    best_right = min(best_right, s_anchor + idx + fut_offset + tr)
        idx = end

    if best_left < 10**9 or best_right < 10**9:
        if best_left <= best_right:
            y_act[2] = 1.0
        else:
            y_act[3] = 1.0

    if float(y_act[:4].sum()) == 0.0:
        y_act[4] = 1.0
    return y_act, 1.0
