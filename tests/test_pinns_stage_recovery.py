"""Synthetic archive recovery and original-video checks for PINNS extraction."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import pickle
import unittest
from unittest.mock import Mock, patch
from zipfile import BadZipFile, ZipFile

import cv2
import numpy as np
import torch

from scripts import cloud_experiment_runner as cloud, run_stage7_extract, run_stage89_experiments
from src.data.pinns_extractor import VIDEO_DECODE_PROTOCOL, process_pinns_sequence


class StageRecoveryTest(unittest.TestCase):
    def test_archive_repairs_partial_and_bad_crc_members(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "input.zip"
            target = root / "unpacked"
            members = {"labels/a.txt": b"label", "videos/a.mp4": b"footage", "videos/calib.json": b"metadata"}
            with ZipFile(archive, "w") as output:
                for name, body in members.items():
                    output.writestr(name, body)
            cloud._unpack_verified_zip(archive, target)
            intact = target / "labels/a.txt"
            stamp = intact.stat().st_mtime_ns
            video = target / "videos/a.mp4"
            for broken in (b"x", b"badfile"):
                video.write_bytes(broken)
                (target / "videos/calib.json").unlink()
                cloud._unpack_verified_zip(archive, target)
                self.assertEqual({name: (target / name).read_bytes() for name in members}, members)
                self.assertEqual(intact.stat().st_mtime_ns, stamp)

            # A failed member read must preserve the last complete target.
            video.write_bytes(b"old")
            archive.write_bytes(archive.read_bytes().replace(b"footage", b"garbage", 1))
            with self.assertRaises(BadZipFile):
                cloud._unpack_verified_zip(archive, target)
            self.assertEqual(video.read_bytes(), b"old")


class ExtractionCacheTest(unittest.TestCase):
    def test_cloud_rebuilds_unverified_source_label_and_reuses_strict_cache(self):
        diag = {"pinns": {"bev_convention": "source_label"}}
        bundle = {"diag": diag, "pinns": ["sample"], "jta": []}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "extracted_samples_source_label_with_person_id.pt"
            torch.save(bundle, path)
            with patch.object(cloud, "stage_pinns_data", side_effect=RuntimeError("fresh source extraction")) as stage:
                with self.assertRaisesRegex(RuntimeError, "fresh source extraction"):
                    cloud.stage_and_extract_all_datasets(root, root)
                with self.assertRaisesRegex(ValueError, "decode check"):
                    cloud.execute_suites_from_samples([], [], diag, root)
                diag["pinns"]["video_decode_protocol"] = VIDEO_DECODE_PROTOCOL
                torch.save(bundle, path)
                self.assertEqual(cloud.stage_and_extract_all_datasets(root, root), (["sample"], [], diag))
                stage.assert_called_once()

    def test_local_stage_rebuild_and_training_resume_check(self):
        diag = {"pinns": {"bev_convention": "source_label"}}
        bundle = {"diag": diag, "pinns": [], "jta": []}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data_cache").mkdir()
            cache = root / "data_cache/extracted_pinns_jta_source_label.pkl"
            cache.write_bytes(pickle.dumps(bundle))
            old_diag = json.dumps(diag)
            with (patch.object(run_stage7_extract, "__file__", str(root / "scripts/stage7.py")),
                  patch.object(run_stage89_experiments, "__file__", str(root / "scripts/stage89.py")),
                  patch.object(run_stage7_extract, "_download_verified_zip", side_effect=RuntimeError("fetch raw")) as fetch):
                with self.assertRaisesRegex(RuntimeError, "fetch raw"):
                    run_stage7_extract.main()
                with self.assertRaisesRegex(ValueError, "Run Stage 7"):
                    run_stage89_experiments.main()
                diag["pinns"]["video_decode_protocol"] = VIDEO_DECODE_PROTOCOL
                cache.write_bytes(pickle.dumps(bundle))
                run_stage7_extract.main()
                fetch.assert_called_once()
                (root / "results/source_label/dataset_diagnostics.json").write_text(old_diag)
                with self.assertRaisesRegex(ValueError, "Existing results predate"):
                    run_stage89_experiments.main()


class PoseProbe:
    def __init__(self):
        self.frames = []

    def warp_crops(self, image, boxes):
        self.frames.append(float(image.mean()))
        return np.zeros((len(boxes), 1)), boxes[:, :2], boxes[:, 2:]

    def infer_warped_crops(self, crops, centers, scales):
        return np.ones((len(crops), 17, 3), dtype=np.float32)


class VideoReadTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.video = Path(tmp.name) / "clip.avi"
        self.labels = Path(tmp.name) / "clip.txt"
        self.labels.write_text("".join(f"{3*i+1} Person 7 20 25 1 8 16 {i*.1} 4 -1\n" for i in range(40)))
        self.probe = PoseProbe()

    def write_video(self, count):
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (48, 32))
        self.assertTrue(writer.isOpened())
        try:
            for i in range(count):
                writer.write(np.full((32, 48, 3), i, dtype=np.uint8))
        finally:
            writer.release()

    def extract(self):
        return process_pinns_sequence("audit_clip", self.labels, self.video, None, {}, self.probe,
                                      window_stride=4, forced_split="train", override_polys=[])

    def test_complete_file_uses_exact_zero_based_three_frame_stride(self):
        self.write_video(120)
        samples, _ = self.extract()
        np.testing.assert_allclose(self.probe.frames, np.arange(0, 118, 3), atol=1, rtol=0)
        self.assertTrue(samples)
        self.assertTrue(all(row["person_id"] == 7 and row["hist_kpts_20"][-1, :, 2].all() for row in samples))

    def test_short_file_identifies_first_unreachable_target(self):
        self.write_video(96)
        with self.assertRaisesRegex(ValueError, "audit_clip.*raw frame 96.*length 96"):
            self.extract()

    def test_grab_failure_and_unreadable_pose_target_release_capture(self):
        self.write_video(120)
        for operation in ("grab", "retrieve"):
            with self.subTest(operation=operation):
                capture = cv2.VideoCapture(str(self.video))
                wrapped = Mock(wraps=capture)
                if operation == "grab":
                    wrapped.grab.side_effect = lambda: capture.get(cv2.CAP_PROP_POS_FRAMES) != 60 and capture.grab()
                else:
                    wrapped.retrieve.side_effect = lambda: ((False, None) if capture.get(cv2.CAP_PROP_POS_FRAMES) == 61
                                                            else capture.retrieve())
                with patch.object(cv2, "VideoCapture", return_value=wrapped):
                    with self.assertRaisesRegex(ValueError, "audit_clip.*raw frame 60"):
                        self.extract()
                wrapped.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
