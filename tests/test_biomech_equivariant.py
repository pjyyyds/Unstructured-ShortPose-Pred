"""Unit tests for O(2)-equivariant biomechanical pose forecaster (M1_BioEq)."""
import unittest
import numpy as np

from src.models.biomech_equivariant import (
    biomech_features,
    fit_biomech_equivariant,
    predict_biomech_equivariant,
)

COCO_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class TestBiomechEquivariant(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(2026)
        n = 60
        self.sites = np.array(["s0"] * 20 + ["s1"] * 20 + ["s2"] * 20)
        t_hist = np.arange(-4, 1, dtype=np.float64)[None, :, None] * 0.1
        vel = rng.uniform(-0.8, 0.8, size=(n, 1, 2))
        acc = rng.normal(0.0, 0.3, size=(n, 1, 2))
        self.hist = vel * t_hist + 0.5 * acc * (t_hist ** 2) + rng.normal(0.0, 0.005, size=(n, 5, 2))
        t_fut = np.arange(1, 31, dtype=np.float64)[None, :, None] * 0.1
        self.truth = self.hist[:, -1:, :] + vel * t_fut + 0.2 * acc * (t_fut ** 2)
        self.bbox = np.tile(np.array([200.0, 300.0, 60.0, 160.0], dtype=np.float32), (n, 5, 1))
        self.kpts = np.zeros((n, 5, 17, 3), dtype=np.float32)
        self.kpts[..., 2] = 0.85
        for j in range(17):
            self.kpts[:, :, j, 0] = 200.0 + rng.normal(0.0, 12.0, size=(n, 5))
            self.kpts[:, :, j, 1] = 240.0 + j * 6.0 + rng.normal(0.0, 4.0, size=(n, 5))

    def test_so2_rotation_and_camera_azimuth_invariance(self) -> None:
        kp64 = self.kpts.astype(np.float64)
        bx64 = self.bbox.astype(np.float64)
        model = fit_biomech_equivariant(self.hist, self.truth, self.sites, kp64, bx64)
        model["g_bio"] = 0.20
        base = predict_biomech_equivariant(model, self.hist, kp64, bx64)
        theta = 1.13
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        kpts_shifted = kp64.copy()
        kpts_shifted[..., 0] += 350.0
        out_rot = predict_biomech_equivariant(model, self.hist @ rot.T, kpts_shifted, bx64)
        np.testing.assert_allclose(out_rot["primary"], base["primary"] @ rot.T, rtol=0, atol=1e-13)
        out_f32 = predict_biomech_equivariant(model, self.hist @ rot.T, kpts_shifted.astype(np.float32), self.bbox)
        np.testing.assert_allclose(out_f32["primary"], base["primary"] @ rot.T, rtol=0, atol=5e-7)

    def test_o2_mirror_reflection_and_left_right_swap(self) -> None:
        refl = np.array([[1.0, 0.0], [0.0, -1.0]])
        kp64 = self.kpts.astype(np.float64)
        bx64 = self.bbox.astype(np.float64)
        model = fit_biomech_equivariant(self.hist, self.truth, self.sites, kp64, bx64)
        model["g_bio"] = 0.20
        base = predict_biomech_equivariant(model, self.hist, kp64, bx64)
        kp_mirror = kp64[:, :, COCO_SWAP, :].copy()
        kp_mirror[..., 0] = 640.0 - kp_mirror[..., 0]
        out_ref = predict_biomech_equivariant(model, self.hist @ refl.T, kp_mirror, bx64)
        np.testing.assert_allclose(out_ref["primary"], base["primary"] @ refl.T, rtol=0, atol=1e-13)
        np.testing.assert_allclose(out_ref["modes"][:, 3], base["modes"][:, 4] @ refl.T, rtol=0, atol=1e-13)

    def test_temporal_order_sensitivity_and_missing_pose_gate(self) -> None:
        f_fwd, q_ok = biomech_features(self.kpts, self.bbox)
        f_rev, _ = biomech_features(self.kpts[:, ::-1], self.bbox[:, ::-1])
        np.testing.assert_allclose(f_rev[:, 5:10], -f_fwd[:, 5:10], rtol=0, atol=1e-12)
        self.assertTrue(np.all(q_ok > 0.5))
        kp_zero = np.zeros_like(self.kpts)
        _, q_zero = biomech_features(kp_zero, self.bbox)
        np.testing.assert_array_equal(q_zero, 0.0)


if __name__ == "__main__":
    unittest.main()
