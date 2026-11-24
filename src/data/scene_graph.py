"""Section 4.3 local polyline builder P_env in R^{16 x 10 x 11} and directed edges e_{j->i} in R^9."""

from typing import List, Tuple
import numpy as np


def resample_polyline_uniform(pts_xy: np.ndarray, num_pts: int = 10) -> np.ndarray:
    """Resample an ordered 2D polyline uniformly by arc length to `num_pts` points."""
    pts = np.asarray(pts_xy, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((num_pts, 2), dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts.astype(np.float32), num_pts, axis=0)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < 1e-6:
        return np.repeat(pts[:1].astype(np.float32), num_pts, axis=0)
    targets = np.linspace(0.0, total, num_pts)
    out_x = np.interp(targets, cum, pts[:, 0])
    out_y = np.interp(targets, cum, pts[:, 1])
    return np.stack([out_x, out_y], axis=-1).astype(np.float32)


def build_local_polylines(
    polylines: List[Tuple[np.ndarray, np.ndarray]],
    p_t0: np.ndarray,
    map_available: bool,
    radius: float = 20.0,
    max_polys: int = 16,
    pts_per_poly: int = 10,
) -> Tuple[np.ndarray, float]:
    """Build P_env in R^{16 x 10 x 11} and map state m_map in {0.0, 1.0}.

    Args:
        polylines: list of (pts_xy [L, 2], attr_8 [8]) in world ground frame.
        p_t0: [2] target ground position at t_0.
        map_available: False if scene lacks map annotation (m_map=0); True if annotated (m_map=1,
                       even when 20m neighborhood is open space with zero polylines!).
    """
    p_env = np.zeros((max_polys, pts_per_poly, 11), dtype=np.float32)
    if not map_available:
        return p_env, 0.0

    origin = np.asarray(p_t0, dtype=np.float32)
    scored = []
    for pts_xy, attr_8 in polylines:
        pts = np.asarray(pts_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        dists = np.linalg.norm(pts - origin[None, :], axis=-1)
        min_d = float(np.min(dists))
        if min_d <= radius:
            scored.append((min_d, pts, np.asarray(attr_8, dtype=np.float32)))

    scored.sort(key=lambda item: item[0])
    for k, (_, pts, attr_8) in enumerate(scored[:max_polys]):
        res = resample_polyline_uniform(pts, num_pts=pts_per_poly)
        p_env[k, :, 0:2] = res - origin[None, :]
        p_env[k, :, 2:10] = np.clip(attr_8[:8], 0.0, 1.0)[None, :]
        p_env[k, :, 10] = 1.0  # m_pt = 1
    return p_env, 1.0


def build_neighbor_edges(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    ego_vel_valid: float,
    ego_obs_ratio: float,
    nbr_pos: np.ndarray,
    nbr_vel: np.ndarray,
    nbr_vel_valid: np.ndarray,
    nbr_obs_ratio: np.ndarray,
    nbr_types: np.ndarray,
    radius: float = 10.0,
    max_nbrs: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Eq. (7) directed edge features e_{j->i} in R^{M x 9}, nbr type one-hot, and valid mask.

    Returns:
        edges: [max_nbrs, 9] float32:
               [dx_ij, dy_ij, dvx_ij, dvy_ij, d_ij, v_rad, m_spd_ij, rho_j, rho_i]
        nbr_type_oh: [max_nbrs, 8] float32 one-hot type vector for each neighbor
        nbr_mask: [max_nbrs] float32 in {0.0, 1.0}
    """
    edges = np.zeros((max_nbrs, 9), dtype=np.float32)
    nbr_type_oh = np.zeros((max_nbrs, 8), dtype=np.float32)
    nbr_mask = np.zeros((max_nbrs,), dtype=np.float32)

    if nbr_pos is None or len(nbr_pos) == 0:
        return edges, nbr_type_oh, nbr_mask

    p_i = np.asarray(ego_pos, dtype=np.float64)
    v_i = np.asarray(ego_vel, dtype=np.float64)
    dp_all = np.asarray(nbr_pos, dtype=np.float64) - p_i[None, :]
    d_all = np.hypot(dp_all[:, 0], dp_all[:, 1])

    cand = np.where((d_all <= radius) & (d_all > 1e-5))[0]
    if len(cand) == 0:
        return edges, nbr_type_oh, nbr_mask

    order = cand[np.argsort(d_all[cand])[:max_nbrs]]
    for slot, j in enumerate(order):
        dp = dp_all[j]
        dist = float(d_all[j])
        m_spd = 1.0 if (ego_vel_valid > 0.5 and float(nbr_vel_valid[j]) > 0.5) else 0.0
        if m_spd > 0.5:
            dv = np.asarray(nbr_vel[j], dtype=np.float64) - v_i
            v_rad = float(-np.dot(dp, dv) / (dist + 1e-6))
        else:
            dv = np.zeros(2, dtype=np.float64)
            v_rad = 0.0

        edges[slot, 0] = float(dp[0])
        edges[slot, 1] = float(dp[1])
        edges[slot, 2] = float(dv[0])
        edges[slot, 3] = float(dv[1])
        edges[slot, 4] = dist
        edges[slot, 5] = v_rad
        edges[slot, 6] = m_spd
        edges[slot, 7] = float(np.clip(nbr_obs_ratio[j], 0.0, 1.0))
        edges[slot, 8] = float(np.clip(ego_obs_ratio, 0.0, 1.0))

        t_idx = int(np.clip(nbr_types[j], 0, 7))
        nbr_type_oh[slot, t_idx] = 1.0
        nbr_mask[slot] = 1.0

    return edges, nbr_type_oh, nbr_mask
