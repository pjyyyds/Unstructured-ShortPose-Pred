"""Exact feature builders for Eq. (3) motion, Eq. (4) pose, and Eq. (5) quality vector."""

from typing import Tuple
import numpy as np

from .constants import (
    DT,
    HEAD_SHOULDER_JOINTS,
    LEFT_HIP_IDX,
    LOWER_LIMB_GAIT_JOINTS,
    MAX_VEL_GAP_SEC,
    NUM_JOINTS,
    RIGHT_HIP_IDX,
    TAU_CONF,
    UPPER_TORSO_ARMS_JOINTS,
)


def build_motion_features(
    pos_xy: np.ndarray,
    pos_mask: np.ndarray,
    dt: float = DT,
    max_vel_gap_sec: float = MAX_VEL_GAP_SEC,
) -> np.ndarray:
    """Build Eq. (3) local-translated motion tensor X_m in R^{T x 9}.

    Args:
        pos_xy: [T, 2] world ground coordinates in meters; last frame T-1 is t_0.
        pos_mask: [T] binary observation mask in {0, 1}; pos_mask[-1] should be 1.
        dt: sampling step in seconds (default 0.1s at 10Hz).
        max_vel_gap_sec: clear finite-difference velocity if gap > 0.5s.

    Returns:
        X_m: [T, 9] float32 array with zero NaNs on missing frames:
             [dx_t0, dy_t0, vx, vy, speed, m_pos, m_vel, dt_last, rho_prefix].
    """
    T = pos_xy.shape[0]
    x_m = np.zeros((T, 9), dtype=np.float32)
    mask_bool = np.asarray(pos_mask, dtype=np.float32) > 0.5

    # Anchor p_{t_0}: use last valid frame (always T-1 for valid triggers)
    valid_indices = np.where(mask_bool)[0]
    p_t0 = pos_xy[valid_indices[-1]] if len(valid_indices) > 0 else np.zeros(2, dtype=np.float32)

    last_valid_idx = -1
    valid_count = 0
    for t in range(T):
        if mask_bool[t]:
            valid_count += 1
            rel_xy = pos_xy[t] - p_t0
            x_m[t, 0] = float(rel_xy[0])
            x_m[t, 1] = float(rel_xy[1])
            x_m[t, 5] = 1.0

            if last_valid_idx >= 0:
                gap_sec = (t - last_valid_idx) * dt
                x_m[t, 7] = float(min(gap_sec, 5.0))
                if gap_sec <= max_vel_gap_sec + 1e-6:
                    vel = (pos_xy[t] - pos_xy[last_valid_idx]) / gap_sec
                    x_m[t, 2] = float(vel[0])
                    x_m[t, 3] = float(vel[1])
                    x_m[t, 4] = float(np.linalg.norm(vel))
                    x_m[t, 6] = 1.0
            else:
                x_m[t, 7] = 0.0
            last_valid_idx = t
        else:
            gap_sec = (t - last_valid_idx) * dt if last_valid_idx >= 0 else (t + 1) * dt
            x_m[t, 7] = float(min(gap_sec, 5.0))

        x_m[t, 8] = float(valid_count) / float(t + 1)

    return np.nan_to_num(x_m, nan=0.0, posinf=0.0, neginf=0.0)


def assign_quality_bucket(q_vec: np.ndarray) -> int:
    """Assign exact Section 5.3 quality bucket: 0=Clear, 1=Partial, 2=Heavy."""
    q1, q2, q3 = float(q_vec[0]), float(q_vec[1]), float(q_vec[2])
    if q1 >= 0.85 and q2 >= 0.60 and q3 >= 0.08:
        return 0  # Clear
    if q1 < 0.50 or q2 < 0.40 or q3 < 0.04:
        return 2  # Heavy Occlusion / Low-Quality
    return 1  # Partial Occlusion


def _group_quality_stats(m_tj: np.ndarray, c_tj: np.ndarray, idxs: Tuple[int, ...]) -> Tuple[float, float]:
    sub_m = m_tj[:, idxs]
    sub_c = c_tj[:, idxs]
    q1_g = float(sub_m.mean()) if sub_m.size > 0 else 0.0
    denom = float(sub_m.sum()) + 1e-6
    q2_g = float((sub_m * sub_c).sum()) / denom
    return q1_g, q2_g


def build_pose_and_quality_features(
    kpts_uvc: np.ndarray,
    bbox_xywh: np.ndarray,
    pos_mask: np.ndarray,
    img_h: float = 864.0,
    tau_c: float = TAU_CONF,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build Eq. (4) X_p in R^{T x 17 x 6}, Eq. (5) q in [0,1]^4, part-q in [0,1]^6, and bucket.

    Args:
        kpts_uvc: [T, 17, 3] pixel (u, v, conf) in image coordinates.
        bbox_xywh: [T, 4] person bounding box [center_u, center_v, w, h] in pixels.
        pos_mask: [T] binary frame observation mask in {0, 1}.
        img_h: source image height in pixels.
        tau_c: keypoint confidence threshold (0.20).
    """
    T = kpts_uvc.shape[0]
    x_p = np.zeros((T, NUM_JOINTS, 6), dtype=np.float32)
    mask_frame = (np.asarray(pos_mask, dtype=np.float32) > 0.5).astype(np.float32)
    safe_img_h = max(float(img_h), 1.0)

    valid_bbox_heights = []
    for t in range(T):
        if mask_frame[t] < 0.5:
            continue
        h_box = float(bbox_xywh[t, 3])
        if h_box <= 1.0:
            continue
        valid_bbox_heights.append(min(h_box / safe_img_h, 1.0))

        c_t = np.clip(np.nan_to_num(kpts_uvc[t, :, 2], nan=0.0), 0.0, 1.0)
        m_t = (c_t >= tau_c).astype(np.float32)
        if m_t[LEFT_HIP_IDX] > 0.5 and m_t[RIGHT_HIP_IDX] > 0.5:
            u_root = 0.5 * float(kpts_uvc[t, LEFT_HIP_IDX, 0] + kpts_uvc[t, RIGHT_HIP_IDX, 0])
            v_root = 0.5 * float(kpts_uvc[t, LEFT_HIP_IDX, 1] + kpts_uvc[t, RIGHT_HIP_IDX, 1])
            r_hip = 1.0
        else:
            u_root = float(bbox_xywh[t, 0])
            v_root = float(bbox_xywh[t, 1])
            r_hip = 0.0

        du = np.clip((np.nan_to_num(kpts_uvc[t, :, 0]) - u_root) / h_box, -3.0, 3.0) * m_t
        dv = np.clip((np.nan_to_num(kpts_uvc[t, :, 1]) - v_root) / h_box, -3.0, 3.0) * m_t
        log_scale = float(np.log(max(h_box / safe_img_h, 1e-4)))

        x_p[t, :, 0] = du
        x_p[t, :, 1] = dv
        x_p[t, :, 2] = c_t * m_t
        x_p[t, :, 3] = m_t
        x_p[t, :, 4] = r_hip * (1.0 if m_t.sum() > 0 else 0.0)
        x_p[t, :, 5] = log_scale

    m_tj = x_p[:, :, 3]
    if float(m_tj.sum()) == 0.0:
        x_p.fill(0.0)
    c_tj = np.clip(np.nan_to_num(kpts_uvc[:, :, 2], nan=0.0), 0.0, 1.0) * mask_frame[:, None]
    q1 = float(m_tj.sum()) / float(NUM_JOINTS * T)
    q2 = float((m_tj * c_tj).sum()) / (float(m_tj.sum()) + 1e-6)
    q3 = float(np.mean(valid_bbox_heights)) if valid_bbox_heights else 0.0
    q4 = float(mask_frame.mean())
    q_vec = np.clip(np.array([q1, q2, q3, q4], dtype=np.float32), 0.0, 1.0)

    q_hs = _group_quality_stats(m_tj, c_tj, HEAD_SHOULDER_JOINTS)
    q_tr = _group_quality_stats(m_tj, c_tj, UPPER_TORSO_ARMS_JOINTS)
    q_lg = _group_quality_stats(m_tj, c_tj, LOWER_LIMB_GAIT_JOINTS)
    q_part = np.clip(np.array([*q_hs, *q_tr, *q_lg], dtype=np.float32), 0.0, 1.0)

    return x_p, q_vec, q_part, assign_quality_bucket(q_vec)
