"""Stage 7: Download & extract real PINNS (45 videos + RTMPose-m ONNX) and JTA into permanent cache."""

from pathlib import Path
import json
import pickle
import shutil
import time

from src.data.constants import DT
from src.data.dataset_builder import materialize_window_split, process_pinns_root
from src.data.jta_builder import process_jta_root
from scripts.cloud_experiment_runner import (
    PINNS_DIRECT_URL,
    PINNS_GDRIVE_ID,
    RTMPOSE_ONNX_URL,
    ST_RELEASE_URL,
    _download_verified_zip,
)
import numpy as np
import zipfile


def main() -> None:
    proj_root = Path(__file__).resolve().parents[1]
    cache_dir = proj_root / "data_cache"
    raw_tmp = cache_dir / "raw_tmp"
    res_dir = proj_root / "results"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_tmp.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    pkl_out = cache_dir / "extracted_pinns_jta.pkl"
    if pkl_out.exists():
        print("Cache already exists:", pkl_out)
        return

    t0 = time.perf_counter()
    rtm_zip = raw_tmp / "rtmpose_m.zip"
    rtm_dir = raw_tmp / "rtmpose_m"
    print("[1/4] Downloading RTMPose-m ONNX...", flush=True)
    _download_verified_zip(RTMPOSE_ONNX_URL, rtm_zip)
    if not rtm_dir.exists():
        with zipfile.ZipFile(rtm_zip, "r") as zf:
            zf.extractall(rtm_dir)
    onnx_path = next(p for p in rtm_dir.rglob("end2end.onnx") if "__macosx" not in str(p).lower())
    print(f"  RTMPose ONNX ready ({time.perf_counter()-t0:.1f}s): {onnx_path.name}", flush=True)

    pinns_zip = raw_tmp / "pinns.zip"
    pinns_dir = raw_tmp / "pinns_unpacked"
    print("[2/4] Downloading PINNS (973 MB)...", flush=True)
    _download_verified_zip(PINNS_DIRECT_URL, pinns_zip, gdrive_id=PINNS_GDRIVE_ID)
    if not (pinns_dir / "labels").exists():
        with zipfile.ZipFile(pinns_zip, "r") as zf:
            zf.extractall(pinns_dir)
    cand_roots = [pinns_dir] + [
        p.parent for p in pinns_dir.rglob("labels") if p.is_dir() and "__macosx" not in str(p).lower()
    ]
    pinns_root = next(p for p in cand_roots if (p / "labels").is_dir() and (p / "videos").is_dir())
    print("[3/4] Running RTMPose-m + BEV polyline + anti-leakage extraction across all 45 PINNS videos...", flush=True)
    pinns_samples, pinns_audit = process_pinns_root(pinns_root, onnx_path=onnx_path, window_stride=6)
    print(f"  PINNS done ({time.perf_counter()-t0:.1f}s): {pinns_audit}", flush=True)

    st_zip = raw_tmp / "st_releases.zip"
    st_dir = raw_tmp / "st_unpacked"
    print("[4/4] Downloading & parsing Social-Transmotion JTA 2.5Hz splits...", flush=True)
    _download_verified_zip(ST_RELEASE_URL, st_zip)
    if not (st_dir / "jta").exists():
        with zipfile.ZipFile(st_zip, "r") as zf:
            zf.extractall(st_dir)
    jta_root = next(
        p for p in st_dir.rglob("data") if p.is_dir() and "jta" in str(p).lower() and "__macosx" not in str(p).lower()
    )
    jta_samples, jta_audit = process_jta_root(jta_root, max_files_per_split=24, seed=42)

    t5_test = materialize_window_split(pinns_samples, "test", t_obs=5, dt=DT)
    bk = t5_test["bucket"].numpy()
    ya = t5_test["y_act"].numpy()
    ma = t5_test["m_act"].numpy() > 0.5
    diag = {
        "pinns": pinns_audit,
        "jta": jta_audit,
        "pinns_test_buckets": {"Clear": int(np.sum(bk == 0)), "Partial": int(np.sum(bk == 1)), "Heavy": int(np.sum(bk == 2))},
        "pinns_test_valid_actions": int(np.sum(ma)),
        "pinns_test_action_pos_counts": ya[ma].sum(axis=0).astype(int).tolist() if np.any(ma) else [0] * 5,
    }
    with open(pkl_out, "wb") as pf:
        pickle.dump({"pinns": pinns_samples, "jta": jta_samples, "diag": diag}, pf, protocol=pickle.HIGHEST_PROTOCOL)
    (res_dir / "02_datasets_extracted.json").write_text(json.dumps(diag, indent=2))
    shutil.rmtree(raw_tmp, ignore_errors=True)
    print(f"STAGE 7 COMPLETE in {time.perf_counter()-t0:.1f}s! Saved {pkl_out} ({pkl_out.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
