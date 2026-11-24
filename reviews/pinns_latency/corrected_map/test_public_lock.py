"""Verify the public source derivation without the private data or a GPU."""
import ast
import hashlib
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constants(path):
    fields = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Constant):
                fields[node.targets[0].id] = node.value.value
    return fields


class LockTests(unittest.TestCase):
    def setUp(self):
        self.lock = json.loads((HERE / "SOURCE_LOCK.json").read_bytes())
        self.plan = json.loads((HERE / "sync_protocol.json").read_bytes())

    def test_raw_source_and_input_lock(self):
        hashes = self.lock["files_sha256"]
        self.assertEqual(self.plan, self.lock["input_provenance"])
        for key in ("measure_latency.py", "paired_pose_compute.py"):
            self.assertEqual(fingerprint(HERE / key), hashes[key])
        self.assertEqual(fingerprint(HERE / "sync_protocol.json"), hashes["sync/protocol.json"])
        self.assertEqual(fingerprint(HERE / "prepare_corrected_sync.py"), self.plan["map_correction"]["preparation_sha256"])
        self.assertEqual(fingerprint(ROOT / "src/models/pose_inference.py"), hashes["pose_inference.py"])
        for file, value in self.plan["files_sha256"].items():
            self.assertEqual(hashes["sync/" + file], value)
        self.assertEqual(len(self.plan["map_correction"]["untouched_fields_bytewise_per_duration"]), 12)

    def test_runner_derivation(self):
        orig = HERE.parent / "measure_latency.py"
        new = HERE / "measure_latency.py"
        old = constants(orig)
        patched = orig.read_text().replace(old["PROTOCOL_HASH"], fingerprint(HERE / "sync_protocol.json"))
        self.assertEqual(new.read_text(), patched)
        paired = constants(HERE / "paired_pose_compute.py")
        self.assertEqual(paired["BUNDLE_SHA"], self.lock["files_sha256"][paired["BUNDLE"]])
        self.assertEqual(paired["RUNNER_SHA"], fingerprint(new))
        self.assertEqual(paired["TRANSFORM_SHA"], self.lock["files_sha256"]["pose_inference.py"])
        self.assertEqual(old["PROTOCOL_HASH"], self.plan["parent"]["protocol_sha256"])
        self.assertEqual(old["CHECKPOINT_HASH"], self.plan["checkpoint_sha256"])

    def test_recorded_coverage(self):
        cases = self.plan["map_correction"]["scene_comparison"]
        self.assertEqual([r["targets"] for r in cases], [12, 12])
        self.assertEqual([r["source_label_slots"] for r in cases], [102, 102])
        self.assertEqual([r["old_slots"] for r in cases], [66, 108])
        self.assertEqual([r["source_bgr_frame_sha256"] for r in cases], [
            s["map_frame_pixels_sha256"] for s in json.loads((HERE.parent / "verified_fixture.json").read_bytes())["scenes"]])


if __name__ == "__main__":
    unittest.main()
