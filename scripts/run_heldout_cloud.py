"""On a CUDA Kaggle/Colab runtime, run: python -m scripts.run_heldout_cloud.

Use --scratch-dir for a persistent download and extraction cache, and --output-dir
for the final artifacts. The notebook must allow outbound Internet connections.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DRIVE_ID = "1Pa-9WoWIM7wNOhEnWaq1_MUs37hqE8Q0"
PINNS_URL = f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&confirm=t"
MODEL_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
             "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip")
MODEL_SHA = "5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c"
SITES = ("america_crossroad", "italy_park", "japan_crossroad", "thailand_street")
BLOCK = 4 * 1024 * 1024


def pip_install(package):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", package], check=True)


def download_zip(url, archive, expected_bytes, drive_id=None):
    def validate(path):
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"Expected {expected_bytes} bytes, found {path.stat().st_size}")
        with zipfile.ZipFile(path) as source:
            bad = source.testzip()
            if bad:
                raise ValueError(f"Failed ZIP CRC for {bad}")

    if archive.exists():
        try:
            validate(archive)
            return
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as error:
            print(f"Replacing invalid cached {archive.name}: {error}", flush=True)
    temp = archive.with_suffix(".zip.part")
    for attempt in range(4):
        temp.unlink(missing_ok=True)
        try:
            print(f"Downloading {archive.name}, attempt {attempt + 1}/4", flush=True)
            if drive_id and attempt % 2:
                if importlib.util.find_spec("gdown") is None:
                    pip_install("gdown")
                subprocess.run([sys.executable, "-m", "gdown", f"https://drive.google.com/uc?id={drive_id}",
                                "-O", str(temp)], check=True)
            else:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as stream, temp.open("wb") as out:
                    header = stream.read(4)
                    if header != b"PK\x03\x04":
                        raise ValueError("Received HTML or another non-ZIP response")
                    out.write(header)
                    shutil.copyfileobj(stream, out, length=BLOCK)
            validate(temp)
            temp.replace(archive)
            return
        except Exception as error:
            temp.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"Unable to obtain verified {archive.name}") from error
            print(f"Retrying {archive.name}: {error}", flush=True)
            time.sleep(2 ** attempt)


def extract_member(archive, name, target):
    entry = archive.getinfo(name)
    if target.is_file() and target.stat().st_size == entry.file_size:
        crc = 0
        with target.open("rb") as existing:
            for data in iter(lambda: existing.read(BLOCK), b""):
                crc = zlib.crc32(data, crc)
        if crc == entry.CRC:
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    with archive.open(entry) as source, temp.open("wb") as out:
        shutil.copyfileobj(source, out, length=BLOCK)
    temp.replace(target)


def stage(scratch):
    scratch.mkdir(parents=True, exist_ok=True)
    pinns, model_zip = scratch / "pinns.zip", scratch / "rtmpose_m.zip"
    download_zip(PINNS_URL, pinns, 973657335, DRIVE_ID)
    download_zip(MODEL_URL, model_zip, 50799818)
    dataset = scratch / "pinns_original"
    with zipfile.ZipFile(pinns) as archive:
        selected = set()
        labels = [re.fullmatch(r"labels/([A-Za-z0-9_]+)_([0-9]+)\.txt", s) for s in archive.namelist()]
        for entry in filter(None, labels):
            folder, index = entry.groups()
            selected.update((entry[0], f"videos/{folder}/videos/ori_{index}.mp4", f"videos/{folder}/calib.json"))
        if (sum(k.endswith(".mp4") for k in selected) != 47
                or sum(k.endswith(".json") for k in selected) != 22):
            raise ValueError("Expected 47 labelled original videos and 22 condition calibrations")
        for name in sorted(selected):
            extract_member(archive, name, dataset / name)
    model = scratch / "end2end.onnx"
    with zipfile.ZipFile(model_zip) as archive:
        names = [k for k in archive.namelist() if PurePosixPath(k).name == "end2end.onnx"
                 and "__MACOSX" not in PurePosixPath(k).parts]
        if len(names) != 1:
            raise ValueError("Expected one end2end.onnx in the official model archive")
        extract_member(archive, names[0], model)
    with model.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    if digest.hexdigest() != MODEL_SHA:
        raise ValueError("RTMPose-m ONNX failed its published experiment SHA-256 check")
    return dataset, model


def run_logged(module, args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "scripts." + module, *map(str, args)]
    with log.open("a", encoding="utf-8") as out:
        out.write(f"\n--- {module} {time.strftime('%Y-%m-%d %H:%M:%S %Z')} ---\n")
        with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace", bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
                out.write(line)
                out.flush()
            if proc.wait():
                raise subprocess.CalledProcessError(proc.returncode, cmd)


def run(scratch, output):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime for the four training folds")
    if scratch == output or scratch.is_relative_to(output):
        raise ValueError("Place the raw-data scratch directory outside the result directory")
    if importlib.util.find_spec("cv2") is None:
        pip_install("opencv-python-headless")
    if importlib.util.find_spec("onnxruntime") is None:
        pip_install("onnxruntime-gpu")
    dataset, model = stage(scratch)
    output.mkdir(parents=True, exist_ok=True)
    cached_pose = scratch / "pose.npz"
    run_logged("cache_heldout_pose", [dataset, model, cached_pose], output / "logs/pose.log")
    pose = output / "pose.npz"
    temp = pose.with_suffix(".npz.part")
    shutil.copyfile(cached_pose, temp)
    temp.replace(pose)
    folds = output / "folds"
    train_args = [dataset, pose, "--output-dir", folds]
    for site in SITES:
        run_logged("train_pinns_heldout_sites", [*train_args, "--site", site], output / "logs" / f"{site}.log")
    run_logged("train_pinns_heldout_sites", [*train_args, "--summarize-only"], output / "logs/summary.log")
    report = json.loads((folds / "report.json").read_text())
    if report["total_windows"] != 1482 or set(report["sites"]) != set(SITES):
        raise ValueError("Final report does not cover the registered 1,482 windows and four sites")

    files = [pose, folds / "report.json", output / "logs/pose.log", output / "logs/summary.log"]
    for site in SITES:
        files += [output / "logs" / f"{site}.log", *(folds / site / name for name in ("M0.pt", "M1.pt", "fold.json"))]
    bundle = output / "shortpose_heldout.zip"
    temp = bundle.with_suffix(".zip.part")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in files:
            archive.write(path, path.relative_to(output))
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip() is not None:
            raise ValueError("Result bundle failed CRC verification")
    temp.replace(bundle)
    print(f"Verified result bundle: {bundle} ({bundle.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/tmp/shortpose_heldout_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/shortpose_heldout"))
    args = parser.parse_args()
    run(args.scratch_dir.resolve(), args.output_dir.resolve())
