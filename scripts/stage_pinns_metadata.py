"""Stage verified PINNS labels and calibration without downloading any video or model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import tempfile
import time
import urllib.request
import zipfile
import zlib

from scripts.run_heldout_cloud import PINNS_URL, ROOT


ZIP_BYTES = 973657335
MEMBER_LIST_SHA = "dfe5a075b1a5aecb9bbf0ff9c0b9cb3a9304f2e5f9f31f07e7b7bcdb36856896"
LABEL = re.compile(r"labels/([A-Za-z0-9_]+)_([0-9]+)\.txt")


def range_bytes(start, stop):
    """Read exactly [start, stop) from the pinned ZIP; reject HTTP 200 before reading its body."""
    if not 0 <= start < stop <= ZIP_BYTES:
        raise ValueError("Invalid PINNS byte range")
    expected = f"bytes {start}-{stop-1}/{ZIP_BYTES}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(PINNS_URL, headers={
                "Range": f"bytes={start}-{stop-1}", "Accept-Encoding": "identity", "User-Agent": "ShortPose-metadata/1"})
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise OSError(f"Range unsupported or response does not match {expected}")
                data = response.read(stop - start + 1)
            if len(data) != stop - start:
                raise OSError("Incomplete or oversized PINNS byte range")
            return data
        except (OSError, EOFError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


class RemoteDirectory(io.RawIOBase):
    """Seekable input for zipfile's small central-directory lookups."""
    def __init__(self):
        super().__init__()
        self.pos = 0

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = (0, self.pos, ZIP_BYTES)[whence] + offset
        if self.pos < 0:
            raise OSError("Negative ZIP position")
        return self.pos

    def read(self, n=-1):
        end = ZIP_BYTES if n < 0 else min(ZIP_BYTES, self.pos + n)
        if end <= self.pos:
            return b""
        # This reader is used only by ZipFile's directory parser. A corrupt
        # central-directory size must not turn a metadata job into a full ZIP download.
        if end - self.pos > 1024 * 1024:
            raise ValueError("Unexpectedly large PINNS central-directory request")
        data = range_bytes(self.pos, end)
        self.pos = end
        return data


def remote_member(entry):
    pos = entry.header_offset
    header = range_bytes(pos, pos + 30)
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"Invalid local ZIP header for {entry.filename}")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    offset = name_len + extra_len
    blob = range_bytes(pos + 30, pos + 30 + offset + entry.compress_size)
    if blob[:name_len] != entry.filename.encode("ascii") or struct.unpack_from("<H", header, 8)[0] != entry.compress_type:
        raise ValueError(f"Member name or compression method mismatch for {entry.filename}")
    payload = blob[offset:]
    if entry.compress_type == zipfile.ZIP_DEFLATED:
        result = zlib.decompress(payload, -15)
    elif entry.compress_type == zipfile.ZIP_STORED:
        result = payload
    else:
        raise ValueError(f"Unexpected compression for {entry.filename}")
    if len(result) != entry.file_size or zlib.crc32(result) != entry.CRC:
        raise ValueError(f"ZIP CRC or size mismatch for {entry.filename}")
    return result


def layout(names):
    labels, calibrations, videos = [], set(), set()
    for name in names:
        match = LABEL.fullmatch(name)
        if match:
            condition, number = match.groups()
            labels.append(name)
            calibrations.add(f"videos/{condition}/calib.json")
            videos.add(f"videos/{condition}/videos/ori_{number}.mp4")
    if len(labels) != 47 or len(set(labels)) != 47 or len(calibrations) != 22:
        raise ValueError("Expected exactly 47 unique PINNS clip labels and 22 calibrations")
    return sorted([*labels, *calibrations]), videos


def metadata_identity(root, expected):
    files, videos = layout(p.relative_to(root).as_posix() for p in (root / "labels").glob("*.txt"))
    h = hashlib.sha256()
    for name in files:
        h.update(name.encode() + b"\0")
        h.update(hashlib.sha256((root / name).read_bytes()).digest())
    if h.hexdigest() != expected:
        raise ValueError("Label/calibration content differs from the accepted PINNS replay")
    return files, videos


def complete(root, videos):
    member_path = root / "archive_members.txt"
    if member_path.is_file():
        return hashlib.sha256(member_path.read_bytes()).hexdigest() == MEMBER_LIST_SHA
    return all((root / name).is_file() for name in videos)


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(data)
            stream.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def stage_metadata(output_dir, *, metadata_root=None, archive=None):
    """Return a verified data root suitable for benchmark_heldout_sites.load_site_windows."""
    output = Path(output_dir)
    expected = json.loads((ROOT / "results/pinns_heldout_replay.json").read_text())["label_and_calibration_sha256"]
    mounted = Path(metadata_root) if metadata_root is not None else None
    if mounted is not None:
        _, videos = metadata_identity(mounted, expected)
        if complete(mounted, videos):
            return mounted
    # A cleaned, verified cache survives offline reruns and interrupted staging.
    try:
        _, videos = metadata_identity(output, expected)
        if complete(output, videos):
            return output
    except (ValueError, FileNotFoundError):
        pass
    source = Path(archive) if archive is not None else RemoteDirectory()
    if archive is not None and source.stat().st_size != ZIP_BYTES:
        raise ValueError("Local PINNS archive has an unexpected byte length")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as z:
        names = z.namelist()
        manifest = ("\n".join(names) + "\n").encode()
        if hashlib.sha256(manifest).hexdigest() != MEMBER_LIST_SHA:
            raise ValueError("Archive member list differs from the published PINNS ZIP")
        files, videos = layout(names)
        if not (set(files) | videos) <= set(names):
            raise ValueError("Missing calibration or original-video member")

        def stage_one(name):
            target, info = output / name, z.getinfo(name)
            if target.is_file() and target.stat().st_size == info.file_size and zlib.crc32(target.read_bytes()) == info.CRC:
                return
            if mounted is not None:
                raw = (mounted / name).read_bytes()
            elif archive is not None:
                raw = z.read(info)  # zipfile verifies this member's decompressed CRC.
            else:
                raw = remote_member(info)
            if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
                raise ValueError(f"Invalid metadata member {name}")
            atomic_bytes(target, raw)

        with ThreadPoolExecutor(max_workers=5) as workers:
            list(workers.map(stage_one, files))
    metadata_identity(output, expected)
    atomic_bytes(output / "archive_members.txt", manifest)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/pinns_metadata"))
    parser.add_argument("--metadata-root", type=Path, help="Existing mounted labels/ and videos/<condition>/calib.json")
    parser.add_argument("--archive", type=Path, help="Existing full official PINNS ZIP; only requested members are read")
    args = parser.parse_args()
    print(stage_metadata(args.output_dir, metadata_root=args.metadata_root, archive=args.archive).resolve())
