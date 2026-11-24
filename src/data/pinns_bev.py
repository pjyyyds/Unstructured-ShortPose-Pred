"""PINNS BEV pixel conventions for published metric tracks and archived contour experiments."""
import numpy as np

SOURCE_LABEL = "source_label"
LEGACY_RASTER = "legacy_raster"
CONVENTIONS = (SOURCE_LABEL, LEGACY_RASTER)


def bev_to_track_xy(uv, width, height, calibration, flip_y, convention=SOURCE_LABEL):
    """Map BEV pixels to the same metre axes as parsed source tracks.

    Source Car annotations encode increasing raw world X/Y with increasing u/v.
    The model's handedness transformation, when requested, applies afterwards.
    The legacy convention reproduces the original image-edge feature matrices.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"Unknown PINNS BEV convention: {convention}")
    if width < 2 or height < 2:
        raise ValueError("Expected a BEV raster at least two pixels across each axis")
    u, v = np.asarray(uv, dtype=np.float32).T
    xmin, xmax = calibration["x_min"], calibration["x_max"]
    ymin, ymax = calibration["y_min"], calibration["y_max"]
    if convention == SOURCE_LABEL:
        if (width, height) != (calibration["bev_width"], calibration["bev_height"]):
            raise ValueError("Decoded BEV size differs from the official calibrated raster")
        x = xmin + u / width * (xmax - xmin)
        y = ymin + v / height * (ymax - ymin)
    else:
        x = xmin + u / (width - 1) * (xmax - xmin)
        y = ymin + (1 - v / (height - 1)) * (ymax - ymin)
    if flip_y:
        y = ymax - (y - ymin)
    return np.stack((x, y), axis=-1)


def require_cache_convention(diag, expected):
    """Only an explicit legacy request may read an earlier, unversioned cache."""
    if expected not in CONVENTIONS or diag["pinns"].get("bev_convention", LEGACY_RASTER) != expected:
        raise ValueError(f"PINNS cache does not match BEV convention {expected}")
