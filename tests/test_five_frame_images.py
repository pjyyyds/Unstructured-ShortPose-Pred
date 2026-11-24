import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from scripts.predict_five_frame_images import camera_frames, measured_observations, read_image


class FiveFrameImages(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.json"
        self.frames = [{"image": f"frame {i}.png", "timestamp_s": 1.1 + i / 10,
                        "bbox_center_xywh_px": [10.0 + i, 15.0, 8.0, 10.0]} for i in range(5)]
        self.spec = {"image_size_px": [60, 40], "frames": self.frames,
                     "image_to_ground_homography": [[0, -2, 10], [3, 0, 1], [.01, .02, 1]]}

    def load(self):
        self.input.write_text(json.dumps(self.spec), encoding="utf-8")
        return camera_frames(self.input)

    def test_projective_mapping_is_from_box_bottom_and_preserves_supplied_axes(self):
        paths, _, _, xy, size = self.load()
        x = np.arange(10., 15.)
        expected = np.stack((np.full(5, -30.), 3*x+1), axis=1) / (1.4 + .01*x[:, None])
        np.testing.assert_allclose(xy, expected, rtol=1e-14)
        self.assertEqual(paths[0], self.root.resolve() / "frame 0.png")
        self.spec["image_to_ground_homography"] = (-9.e-120 * np.array(self.spec["image_to_ground_homography"])).tolist()
        np.testing.assert_allclose(self.load()[3], expected, rtol=1e-14)
        self.assertEqual(size, (60, 40))

    def test_rejects_horizon_wrong_time_dimensions_and_duplicate_images(self):
        for modify, message in (
            (lambda: self.spec.update(image_to_ground_homography=[[1,0,0],[0,1,0],[0,1,-20]]), "projection"),
            (lambda: self.spec.update(image_to_ground_homography=[[1,0,0],[1,0,0],[0,0,1]]), "nonsingular"),
            (lambda: self.frames[2].update(timestamp_s=1.9), "spaced"),
            (lambda: self.frames[4].update(image="frame 0.png"), "distinct image files"),
            (lambda: self.frames[1].update(bbox_center_xywh_px=[0,0,0,1]), "positive"),
            (lambda: self.frames[0].update(bbox_center_xywh_px=[500,500,3,3]), "outside"),
        ):
            with self.subTest(message=message):
                current=json.dumps(self.spec)
                modify()
                with self.assertRaisesRegex(ValueError, message):
                    self.load()
                self.spec=json.loads(current)
                self.frames=self.spec["frames"]

    def test_real_file_decode_and_onnx_digest_are_checked(self):
        paths, boxes, times, xy, size = self.load()
        pixels = np.zeros((40, 60, 3), np.uint8)
        pixels[5,7] = [19, 88, 201]
        self.assertTrue(cv2.imwrite(str(paths[0]), pixels))
        np.testing.assert_array_equal(read_image(paths[0], size, 0), pixels)
        with self.assertRaisesRegex(ValueError,"decodes to"):
            read_image(paths[0], (40,60), 0)
        paths[1].write_bytes(b"not a valid image")
        with self.assertRaisesRegex(ValueError,"decode"):
            read_image(paths[1], size, 1)
        with self.assertRaisesRegex(ValueError,"does not exist"):
            read_image(paths[2], size, 2)
        wrong=self.root/"wrong.onnx"
        wrong.write_bytes(b"not a registered model")
        with self.assertRaisesRegex(ValueError,"SHA-256"):
            measured_observations(paths, boxes, times, xy, size, wrong)


if __name__ == "__main__":
    unittest.main()
