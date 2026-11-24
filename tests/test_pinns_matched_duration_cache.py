import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from scripts import run_pinns_matched_durations as run


def fixture():
    # Rows one and zero both contribute crops at frame 7, only row zero is retained in the strict cohort.
    boxes = np.ones((2, 5, 4), np.float32)
    parent = {
        "clip": np.array(["clip", "clip"]), "person_id": np.array([7, 9]), "t0": np.array([11, 10]),
        "hist_bbox": boxes, "hist_kpts": np.full((2, 5, 17, 3), .2, np.float32),
        "pose_inferred": np.ones((2, 5), bool), "img_h": np.full(2, 100.), "ego_type": np.zeros(2, dtype=int),
    }
    keys = [("clip", 7, 11)]
    xy = np.zeros((1, 50, 2))
    ref = run.Reference(keys, {"test": np.array([0])}, xy, np.ones((1, 20, 4), np.float32),
                        parent, np.array([0]), {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}, "data")
    arrays = {name: parent[name][:1].copy() for name in run.IDENTITY_FIELDS}
    arrays.update(xy=xy.copy(), hist_bbox=ref.boxes.copy(), hist_kpts=np.full((1, 20, 17, 3), .2, np.float32),
                  pose_inferred=np.ones((1, 20), bool))
    meta = {"schema": 2, "pose_parent_sha256": run.PARENT_SHA, "onnx_sha256": run.MODEL_SHA, "data_sha256": "data",
            "keys_sha256": run.json_digest(keys), "providers": ref.parent_meta["providers"], "frame_mapping": run.CLOCK,
            "collector_sha256": run.file_hash(Path(run.__file__)), "prefix_sha256": run.prefix_digest(arrays),
            "zero_offset_controls": {"xy_atol_px": run.XY_ATOL_PX, "confidence_atol": run.CONF_ATOL,
                                     "allowed_confidence_0_2_crossings": 0, "passed": True,
                                     "clips": {"clip": {"raw_frame": 21, "crops": 2, "xy_max_abs_px": 0.,
                                                        "confidence_max_abs": 0., "confidence_0_2_crossings": 0}}}}
    arrays["metadata_json"] = np.array(json.dumps(meta))
    return ref, arrays


class MatchedCacheTests(unittest.TestCase):
    def test_largest_original_batch_includes_non_selected_windows_in_original_order(self):
        ref, arrays = fixture()
        frame, idx = run.control_batch(ref.parent, "clip")
        self.assertEqual(frame, 21)
        np.testing.assert_array_equal(idx, [[0, 1], [0, 1]])
        run.validate_cache(arrays, ref)

    def test_real_frame_control_rejects_threshold_crossing_even_with_tiny_numeric_difference(self):
        ref, _ = fixture()
        cv = MagicMock()
        cv.get.side_effect = lambda flag: 30 if flag == run.cv2.CAP_PROP_FPS else 100
        cv.retrieve.return_value = True, np.zeros((100, 50, 3), np.uint8)
        ext = MagicMock()
        ext.extract_crops.return_value = ref.parent["hist_kpts"][:, 0].copy()
        with patch.object(run.cv2, "VideoCapture", return_value=cv):
            self.assertEqual(run.zero_control("video", "clip", ref.parent, ext)["crops"], 2)
            self.assertEqual(cv.grab.call_count, 22)
            ext.extract_crops.return_value[0, 0, 2] -= 0.00001
            with self.assertRaisesRegex(ValueError, "Failed original-frame control"):
                run.zero_control("video", "clip", ref.parent, ext)

    def test_cached_arrays_are_rechecked_against_label_and_frozen_parent(self):
        mutations = [("xy", (0, 0, 0)), ("xy", (0, 49, 0)), ("ego_type", 0), ("img_h", 0),
                     ("person_id", 0), ("t0", 0), ("hist_bbox", (0, 0, 0)), ("hist_bbox", (0, 19, 0)),
                     ("hist_kpts", (0, 19, 0, 0)), ("hist_kpts", (0, 2, 0, 0)), ("pose_inferred", (0, 17))]
        for name, index in mutations:
            with self.subTest(name=name,index=index):
                ref, arrays = fixture()
                arrays[name][index] = 0 if name == "pose_inferred" else arrays[name][index] + 1
                with self.assertRaises(ValueError):
                    run.validate_cache(arrays, ref)
        for mutate in (lambda m: m["providers"].reverse(), lambda m: m["zero_offset_controls"]["clips"].clear(),
                       lambda m: m["zero_offset_controls"]["clips"]["clip"].update(crops=1)):
            ref, arrays = fixture()
            meta = json.loads(arrays["metadata_json"].item())
            mutate(meta)
            arrays["metadata_json"] = np.array(json.dumps(meta))
            with self.assertRaises(ValueError):
                run.validate_cache(arrays, ref)

    def test_undecoded_valid_prefix_crop_is_rejected_even_with_matching_fingerprint(self):
        ref, arrays = fixture()
        arrays["pose_inferred"][0, 3] = False
        arrays["hist_kpts"][0, 3] = 0
        meta = json.loads(arrays["metadata_json"].item())
        meta["prefix_sha256"] = run.prefix_digest(arrays)
        arrays["metadata_json"] = np.array(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "inference mask"):
            run.validate_cache(arrays, ref)

    def test_cache_hit_and_training_entry_verify_before_use(self):
        ref, arrays = fixture()
        arrays["xy"][0, 20, 1] = 1
        with tempfile.TemporaryDirectory() as tmp:
            location = Path(tmp) / "observation.npz"
            np.savez(location, **arrays)
            with patch.object(run, "load_reference", return_value=ref), patch.object(run, "fit_predict") as fit:
                with self.assertRaisesRegex(ValueError, "historical and 30 future"):
                    run.collect(None, None, None, location)
                with self.assertRaisesRegex(ValueError, "historical and 30 future"):
                    run.fit_and_score(None, location, Path(tmp), False, None)
                fit.assert_not_called()

    def test_provider_order_rejected_before_prefix_extraction(self):
        ref, _ = fixture()
        ext = MagicMock(active_providers=list(reversed(ref.parent_meta["providers"])))
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, "load_reference", return_value=ref), \
                patch.object(run, "file_hash", return_value=run.MODEL_SHA), \
                patch.object(run, "RTMPoseONNXExtractor", return_value=ext), patch.object(run, "build_clip") as extract:
            with self.assertRaisesRegex(RuntimeError, "Frozen providers"):
                run.collect(None, None, None, Path(tmp) / "new.npz")
            extract.assert_not_called()
