from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from scripts import build_pinns_causal_maps as causal
from scripts import cloud_experiment_runner as cloud
from src.data.pinns_bev import LEGACY_RASTER, SOURCE_LABEL, bev_to_track_xy, require_cache_convention
from src.data.pinns_extractor import extract_static_bev_and_train_polylines


class PinnsBEVTest(unittest.TestCase):
    def setUp(self):
        self.cal = {"x_min": -2., "x_max": 18., "y_min": 3., "y_max": 13.,
                    "bev_width": 100, "bev_height": 40}

    def test_published_coordinates_and_hand_reflection(self):
        uv = [[0, 0], [25, 10], [99, 39]]
        expected = np.array([[-2, 3], [3, 5.5], [17.8, 12.75]])
        for flipped in (False, True):
            want = expected.copy()
            if flipped:
                want[:, 1] = 16. - want[:, 1]
            np.testing.assert_allclose(bev_to_track_xy(uv, 100, 40, self.cal, flipped), want,
                                       rtol=0., atol=2e-6)
        legacy = bev_to_track_xy(uv, 100, 40, self.cal, False, LEGACY_RASTER)
        np.testing.assert_allclose(legacy[[0, -1]], [[-2, 13], [18, 3]])
        with self.assertRaisesRegex(ValueError, "raster"):
            bev_to_track_xy(uv, 80, 40, self.cal, False)
        with self.assertRaisesRegex(ValueError, "convention"):
            bev_to_track_xy(uv, 100, 40, self.cal, False, "unknown")

    def test_both_contour_entry_points_and_metric_occupancy(self):
        with TemporaryDirectory() as d:
            video = Path(d) / "scene.avi"
            out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (100,40))
            self.assertTrue(out.isOpened())
            try:
                frame = np.zeros((40,100,3), np.uint8)
                cv2.rectangle(frame, (10, 8), (70, 31), (255,255,255), thickness=2)
                for _ in range(10):
                    out.write(frame)
            finally:
                out.release()
            people = np.tile([[-1,0],[1,0],[0,2]], (8,1)).astype("f4")
            cars = np.array([[3.,4.]],dtype="f4")
            previous = None
            for options in ({}, {"bev_convention":LEGACY_RASTER}):
                learned = extract_static_bev_and_train_polylines(video, self.cal, people, cars, False, **options)
                edges = causal.edges_from_frames(video, [0], self.cal, False, **options)
                self.assertTrue(edges)
                self.assertEqual(len(learned),len(edges)+2)
                for left,right in zip(learned,edges):
                    np.testing.assert_array_equal(left[0], right[0])
                if previous is not None:
                    self.assertFalse(np.allclose(learned[0][0],previous[0][0]))
                    for i in (-1,-2):
                        np.testing.assert_array_equal(learned[i][0], previous[i][0])
                previous = learned

    def test_only_explicit_legacy_accepts_old_cache(self):
        legacy = {"pinns": {"total_samples": 20}}
        corrected = {"pinns": {"bev_convention": SOURCE_LABEL}}
        require_cache_convention(legacy, LEGACY_RASTER)
        require_cache_convention(corrected, SOURCE_LABEL)
        with self.assertRaises(ValueError):
            require_cache_convention(legacy, SOURCE_LABEL)
        with self.assertRaises(ValueError):
            require_cache_convention(corrected, LEGACY_RASTER)
        with TemporaryDirectory() as d:
            root = Path(d)
            # Simulate an old cache mounted under its real historical name.
            pkg = {"pinns": [], "jta": [], "diag": legacy}
            torch.save(pkg, root/"extracted_samples_v1.pt")
            with patch.object(cloud, "_download_verified_zip", side_effect=RuntimeError("fresh extraction")):
                with self.assertRaisesRegex(RuntimeError, "fresh extraction"):
                    cloud.stage_and_extract_all_datasets(root, root)
                self.assertEqual(cloud.stage_and_extract_all_datasets(root,root,bev_convention=LEGACY_RASTER)[2], legacy)
                torch.save(pkg, root/"extracted_samples_source_label.pt")
                with self.assertRaisesRegex(RuntimeError, "fresh extraction"):
                    cloud.stage_and_extract_all_datasets(root, root)
                torch.save(pkg, root/"extracted_samples_source_label_with_person_id.pt")
                with self.assertRaisesRegex(ValueError, "convention"):
                    cloud.stage_and_extract_all_datasets(root, root)
            with self.assertRaisesRegex(ValueError, "convention"):
                cloud.execute_suites_from_samples([], [], legacy, root)


if __name__ == "__main__":
    unittest.main()
