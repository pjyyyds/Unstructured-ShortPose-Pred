"""Expose the previous JRDB study's frozen helpers when running directly in this checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reviews.jrdb_development.native_pose_residual.run_jrdb_pose_residual import (
    features, infer, residual_fit,
)

__all__ = ["features", "infer", "residual_fit"]
