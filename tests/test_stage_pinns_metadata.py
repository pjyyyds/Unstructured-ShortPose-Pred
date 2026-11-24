"""Exercise metadata-only staging using an in-memory synthetic official-shape ZIP."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from scripts import stage_pinns_metadata as stage


class TestMetadataStage(unittest.TestCase):
    def test_sparse_ranges_cached_repair_and_offline_local_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files, video_payloads = {}, {}
            for i in range(47):
                condition, clip = f"site_condition_{i % 22}", i // 22
                files[f"labels/{condition}_{clip}.txt"] = f"label {i}".encode()
                files[f"videos/{condition}/calib.json"] = f"calibration {condition}".encode()
                video_payloads[f"videos/{condition}/videos/ori_{clip}.mp4"] = b"private video payload " * 10
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
                for name, data in {**video_payloads, **files}.items():
                    z.writestr(name, data)
            archive = buf.getvalue()
            zip_path = root / "official.zip"
            zip_path.write_bytes(archive)
            with zipfile.ZipFile(buf) as z:
                manifest = ("\n".join(z.namelist()) + "\n").encode()
                forbidden = []
                for name in video_payloads:
                    ent = z.getinfo(name)
                    start = ent.header_offset + 30 + len(name)
                    forbidden.append((start, start + ent.compress_size))
            requests = []
            def sparse(start, stop):
                self.assertFalse(any(start < end and stop > begin for begin, end in forbidden))
                requests.append((start, stop))
                return archive[start:stop]
            h = hashlib.sha256()
            for name in sorted(files):
                h.update(name.encode() + b"\0")
                h.update(hashlib.sha256(files[name]).digest())
            (root / "results").mkdir()
            (root / "results/pinns_heldout_replay.json").write_text(json.dumps({"label_and_calibration_sha256": h.hexdigest()}))
            with patch.object(stage, "ROOT", root), patch.object(stage, "ZIP_BYTES", len(archive)), patch.object(
                    stage, "MEMBER_LIST_SHA", hashlib.sha256(manifest).hexdigest()), patch.object(
                    stage, "range_bytes", side_effect=sparse):
                remote = stage.stage_metadata(root / "remote")
                self.assertEqual(len(list(p for p in remote.rglob("*") if p.is_file())), 70)
                self.assertEqual((remote / "archive_members.txt").read_bytes(), manifest)
                self.assertGreater(len(requests), 69)
                requests.clear()
                self.assertEqual(stage.stage_metadata(root / "ignored", metadata_root=remote), remote)
                self.assertEqual(stage.stage_metadata(remote), remote)
                local = stage.stage_metadata(root / "local", archive=zip_path)
                self.assertEqual(requests, [])
                label = next((local / "labels").glob("*.txt"))
                label.write_bytes(b"X" * len(label.read_bytes()))
                stage.stage_metadata(local, archive=zip_path)
                self.assertEqual(label.read_bytes(), files[label.relative_to(local).as_posix()])

    def test_http_full_download_rejected_without_read(self):
        response = MagicMock(status=200, headers={})
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=response), patch("time.sleep"):
            with self.assertRaisesRegex(OSError, "Range unsupported"):
                stage.range_bytes(10, 20)
        response.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
