"""Camera-homography vehicle projection and raw-video occlusion diagnostics for PINNS."""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

VEH_DIMS_M: dict[str, tuple[float, float, float]] = {
    "car": (2.15, 0.92, 1.55),
    "bus": (4.50, 1.25, 2.95),
    "tru": (4.20, 1.22, 2.75),
    "van": (2.35, 0.98, 1.85),
    "mot": (0.98, 0.42, 1.35),
    "bic": (0.90, 0.35, 1.30),
    "cyc": (0.90, 0.35, 1.30),
}


def project_homog(xy: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Project [N, 2] points through a 3x3 homography matrix."""
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack((pts, np.ones(len(pts), dtype=np.float64))) @ mat.T
    den = np.where(np.abs(hom[:, 2:3]) < 1e-9, 1e-9, hom[:, 2:3])
    return hom[:, :2] / den


def fit_verified_ground_to_camera_homography(
    calib: dict,
    ped_check_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, float | None, float]:
    """Fit H_gnd_to_img and verify whether label (wx, wy) uses y_sum - wy conjugation.

    ped_check_rows: optional [M, 6] array of [ox, oy, w, h, wx, wy] in camera & world units.
    Returns (h_gnd_to_img, y_conj_sum_or_None, median_ped_foot_reproj_px).
    """
    pts_2d = np.asarray(calib["pts_2d"], dtype=np.float64)
    pts_3d = np.asarray(calib["pts_3d"], dtype=np.float64)
    if cv2 is not None:
        h_i2g, _ = cv2.findHomography(pts_2d, pts_3d, method=0)
    else:
        a_rows = []
        for (u, v), (x, y) in zip(pts_2d, pts_3d):
            a_rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, u * x, v * x, x])
            a_rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, u * y, v * y, y])
        _, _, vt = np.linalg.svd(np.asarray(a_rows, dtype=np.float64))
        h_i2g = vt[-1].reshape(3, 3)
    h_g2i = np.linalg.inv(h_i2g)
    y_sum = float(calib.get("y_min", 0.0)) + float(calib.get("y_max", 30.0))
    if ped_check_rows is None or len(ped_check_rows) == 0:
        return h_g2i, y_sum, 0.0

    cam_foot = ped_check_rows[:, :2]
    w_raw = ped_check_rows[:, 4:6].copy()
    w_conj = w_raw.copy()
    w_conj[:, 1] = y_sum - w_conj[:, 1]
    err_raw = float(np.median(np.linalg.norm(project_homog(w_raw, h_g2i) - cam_foot, axis=1)))
    err_conj = float(np.median(np.linalg.norm(project_homog(w_conj, h_g2i) - cam_foot, axis=1)))
    if err_conj <= err_raw:
        return h_g2i, y_sum, err_conj
    return h_g2i, None, err_raw


def project_vehicle_3d_to_camera_box(
    wx: float,
    wy: float,
    obj_type: str,
    h_g2i: np.ndarray,
    y_conj_sum: float | None,
    heading_xy: tuple[float, float] | None = None,
) -> tuple[tuple[float, float, float, float], np.ndarray]:
    """Project vehicle ground footprint (wx, wy) in metres to camera [x1, y1, x2, y2] + 4 footprint pts."""
    pfx = obj_type.lower()[:3]
    hl, hw, h_m = VEH_DIMS_M.get(pfx, (2.15, 0.92, 1.55))
    if heading_xy is not None and np.hypot(heading_xy[0], heading_xy[1]) > 1e-3:
        ux, uy = heading_xy[0] / np.hypot(*heading_xy), heading_xy[1] / np.hypot(*heading_xy)
    else:
        ux, uy = 1.0, 0.0
    vx, vy = -uy, ux
    corners_w = np.array([
        [wx + hl * ux + hw * vx, wy + hl * uy + hw * vy],
        [wx + hl * ux - hw * vx, wy + hl * uy - hw * vy],
        [wx - hl * ux - hw * vx, wy - hl * uy - hw * vy],
        [wx - hl * ux + hw * vx, wy - hl * uy + hw * vy],
    ], dtype=np.float64)
    c_proj = corners_w.copy()
    if y_conj_sum is not None:
        c_proj[:, 1] = y_conj_sum - c_proj[:, 1]
    uv_foot = project_homog(c_proj, h_g2i)
    # Estimate local pixel-per-metre scale from ground Jacobian around vehicle center
    c0 = np.array([[wx, y_conj_sum - wy if y_conj_sum is not None else wy],
                   [wx + 1.0, y_conj_sum - wy if y_conj_sum is not None else wy],
                   [wx, (y_conj_sum - (wy + 1.0)) if y_conj_sum is not None else (wy + 1.0)]], dtype=np.float64)
    uv0 = project_homog(c0, h_g2i)
    px_per_m = float(np.clip(0.55 * (np.linalg.norm(uv0[1] - uv0[0]) + np.linalg.norm(uv0[2] - uv0[0])), 8.0, 160.0))
    x1, x2 = float(uv_foot[:, 0].min()), float(uv_foot[:, 0].max())
    y2 = float(uv_foot[:, 1].max())
    y1 = float(uv_foot[:, 1].min()) - h_m * px_per_m
    return (x1, y1, x2, y2), uv_foot


def camera_box_occludes_pedestrian(
    occ_xyxy: tuple[float, float, float, float],
    ped_xyxy: tuple[float, float, float, float],
    min_h_overlap_px: float = -8.0,
    min_vert_cover_frac: float = 0.25,
) -> bool:
    """Perspective camera occlusion check: occluder ground contact closer/equal in v & 2D body overlap."""
    ox1, oy1, ox2, oy2 = occ_xyxy
    px1, py1, px2, py2 = ped_xyxy
    pbh = max(py2 - py1, 8.0)
    # Depth ordering in perspective camera: larger v (bottom y2) means closer to camera on ground plane
    if oy2 < py2 - 0.14 * pbh:
        return False
    h_ov = min(ox2, px2) - max(ox1, px1)
    if h_ov < min_h_overlap_px:
        return False
    v_ov = max(0.0, min(oy2, py2) - max(oy1, py1))
    return (v_ov / pbh) >= min_vert_cover_frac
