"""Cloud GPU orchestrator for PINNS + JTA extraction, 5-seed ablations, and Holm-Bonferroni stats."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import shutil
import subprocess
import time
import urllib.request
import zipfile
import numpy as np
import torch

from src.data.constants import DT, FIXED_SEEDS, T_FUT
from src.data.dataset_builder import materialize_window_split, process_pinns_root
from src.data.jta_builder import process_jta_root
from src.engine.trainer import (
    aggregate_multi_seed_runs,
    benchmark_inference_latency,
    train_and_eval_seed,
)
from src.metrics.metrics_and_stats import holm_bonferroni_adjust, paired_hypothesis_test

PINNS_GDRIVE_ID = "1Pa-9WoWIM7wNOhEnWaq1_MUs37hqE8Q0"
PINNS_DIRECT_URL = f"https://drive.usercontent.google.com/download?id={PINNS_GDRIVE_ID}&confirm=t"
ST_RELEASE_URL = "https://github.com/vita-epfl/social-transmotion/releases/download/ckpt_data/releases.zip"
RTMPOSE_ONNX_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)


def _download_verified_zip(url: str, dst_zip: Path, gdrive_id: Optional[str] = None) -> None:
    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    if dst_zip.exists() and dst_zip.stat().st_size > 1024:
        with open(dst_zip, "rb") as f:
            if f.read(4) == b"PK\x03\x04":
                return

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp, open(dst_zip, "wb") as out:
            shutil.copyfileobj(resp, out, length=4 * 1024 * 1024)
    except Exception as exc:
        print(f"[WARN] Direct urllib download failed ({exc}), trying gdown fallback...")

    valid = False
    if dst_zip.exists() and dst_zip.stat().st_size > 1024:
        with open(dst_zip, "rb") as f:
            valid = f.read(4) == b"PK\x03\x04"

    if not valid and gdrive_id is not None:
        subprocess.run(["gdown", "--fuzzy", gdrive_id, "-O", str(dst_zip)], check=True)
        with open(dst_zip, "rb") as f:
            if f.read(4) != b"PK\x03\x04":
                raise RuntimeError(f"Downloaded file {dst_zip} is not a valid ZIP archive!")


def stage_and_extract_all_datasets(
    scratch_dir: Path = Path("/tmp/shortpose_raw"),
    cache_dir: Optional[Path] = None,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Download RTMPose-m ONNX, PINNS, and Social-Transmotion JTA into `/tmp`, extract, and cache."""
    if cache_dir is None:
        cache_dir = Path("/kaggle/working/cache") if Path("/kaggle/working").exists() else (scratch_dir / "cache")
    cache_pt = cache_dir / "extracted_samples_v1.pt"
    for cand in [cache_pt, *Path("/kaggle/input").rglob("extracted_samples_v1.pt")] if Path("/kaggle/input").exists() else [cache_pt]:
        if cand.exists() and cand.stat().st_size > 1024:
            print(f"[Stage 7] Loading pre-extracted cache from {cand}...", flush=True)
            pkg = torch.load(cand, map_location="cpu", weights_only=False)
            return pkg["pinns"], pkg["jta"], pkg["diag"]

    scratch_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # 1. RTMPose-m ONNX
    rtm_zip = scratch_dir / "rtmpose_m.zip"
    rtm_dir = scratch_dir / "rtmpose_m"
    _download_verified_zip(RTMPOSE_ONNX_URL, rtm_zip)
    if not rtm_dir.exists():
        with zipfile.ZipFile(rtm_zip, "r") as zf:
            zf.extractall(rtm_dir)
    onnx_files = [p for p in rtm_dir.rglob("end2end.onnx") if "__macosx" not in str(p).lower()]
    onnx_path = onnx_files[0] if onnx_files else None
    print(f"[Stage 7] RTMPose ONNX ready: {onnx_path}", flush=True)

    # 2. PINNS real unstructured roadside dataset
    pinns_zip = scratch_dir / "pinns.zip"
    pinns_dir = scratch_dir / "pinns_unpacked"
    _download_verified_zip(PINNS_DIRECT_URL, pinns_zip, gdrive_id=PINNS_GDRIVE_ID)
    if not (pinns_dir / "labels").exists():
        with zipfile.ZipFile(pinns_zip, "r") as zf:
            zf.extractall(pinns_dir)
    cand_roots = [pinns_dir] + [
        p.parent for p in pinns_dir.rglob("labels") if p.is_dir() and "__macosx" not in str(p).lower()
    ]
    pinns_root = next(p for p in cand_roots if (p / "labels").is_dir() and (p / "videos").is_dir())
    pinns_samples, pinns_audit = process_pinns_root(pinns_root, onnx_path=onnx_path, window_stride=4)
    print(f"[Stage 7] PINNS extracted in {time.perf_counter()-t0:.1f}s: {pinns_audit}", flush=True)

    # 3. Social-Transmotion JTA 2.5Hz benchmark
    st_zip = scratch_dir / "st_releases.zip"
    st_dir = scratch_dir / "st_unpacked"
    _download_verified_zip(ST_RELEASE_URL, st_zip)
    if not (st_dir / "jta").exists():
        with zipfile.ZipFile(st_zip, "r") as zf:
            zf.extractall(st_dir)
    jta_data_roots = [
        p for p in st_dir.rglob("data") if p.is_dir() and "jta" in str(p).lower() and "__macosx" not in str(p).lower()
    ]
    jta_samples, jta_audit = process_jta_root(jta_data_roots[0], max_files_per_split=24, seed=42)
    print(f"[Stage 7] JTA extracted: {jta_audit}", flush=True)

    t5_test = materialize_window_split(pinns_samples, "test", t_obs=5, dt=DT)
    bk = t5_test["bucket"].numpy()
    ya = t5_test["y_act"].numpy()
    ma = t5_test["m_act"].numpy() > 0.5
    diag = {
        "pinns": pinns_audit,
        "jta": jta_audit,
        "pinns_test_buckets": {
            "Clear": int(np.sum(bk == 0)),
            "Partial": int(np.sum(bk == 1)),
            "Heavy": int(np.sum(bk == 2)),
        },
        "pinns_test_valid_actions": int(np.sum(ma)),
        "pinns_test_action_pos_counts": ya[ma].sum(axis=0).astype(int).tolist() if np.any(ma) else [0] * 5,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"pinns": pinns_samples, "jta": jta_samples, "diag": diag}, cache_pt)
    print(f"[Stage 7] Cached extracted datasets to {cache_pt} ({cache_pt.stat().st_size/1e6:.1f} MB)", flush=True)
    return pinns_samples, jta_samples, diag


def _json_clean_agg(agg: Dict) -> Dict:
    return {k: v for k, v in agg.items() if not isinstance(v, np.ndarray)}


def run_model_grid(
    tasks: List[Tuple[str, str, int, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, float]],
    seeds: Tuple[int, ...] = FIXED_SEEDS,
    epochs: int = 28,
    batch_size: int = 512,
    ckpt_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray]]:
    """Run (tag, model_name, t_obs, train_d, val_d, test_d, fut_steps, dt) across all 5 seeds on available GPUs."""
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 0:
        devices = [f"cuda:{i}" for i in range(n_gpus)]
    elif torch.backends.mps.is_available():
        devices = ["mps"]
    else:
        devices = ["cpu"]
    max_workers = max(1, len(devices))

    raw_jobs = []
    job_idx = 0
    for tag, m_name, t_obs, tr_d, va_d, te_d, f_steps, dt_val in tasks:
        for sd in seeds:
            dev_str = devices[job_idx % len(devices)]
            raw_jobs.append((tag, m_name, sd, tr_d, va_d, te_d, dev_str, f_steps, dt_val))
            job_idx += 1

    by_tag: Dict[str, List[Dict]] = {}
    t_grid0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_map = {
            pool.submit(
                train_and_eval_seed,
                m_name,
                sd,
                tr_d,
                va_d,
                te_d,
                dev_str,
                epochs,
                batch_size,
                2.0e-3,
                f_steps,
                dt_val,
                True,
            ): (tag, m_name, sd)
            for (tag, m_name, sd, tr_d, va_d, te_d, dev_str, f_steps, dt_val) in raw_jobs
        }
        done_cnt = 0
        for fut in as_completed(fut_map):
            tag, m_name, sd = fut_map[fut]
            res = fut.result()
            by_tag.setdefault(tag, []).append(res)
            if ckpt_dir is not None and sd == 42 and res.get("best_state_cpu") is not None:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                if tag in ("PINNS_T5_M0", "PINNS_T5_M1", "PINNS_T5_M2", "PINNS_T5_M3", "PINNS_T5_Ours_Full"):
                    torch.save(res["best_state_cpu"], ckpt_dir / f"{tag}_seed42.pt")
            res["best_state_cpu"] = None  # Free RAM immediately
            done_cnt += 1
            if done_cnt % 10 == 0 or done_cnt == len(raw_jobs):
                print(
                    f"  [Grid] {done_cnt}/{len(raw_jobs)} runs done ({time.perf_counter()-t_grid0:.1f}s) "
                    f"last={tag}/s{sd} ADE1={res['test']['slices']['Overall']['ADE_1']:.4f}",
                    flush=True,
                )

    summaries: Dict[str, Dict] = {}
    paired_arrays: Dict[str, np.ndarray] = {}
    for tag, run_list in sorted(by_tag.items()):
        run_list.sort(key=lambda r: int(r["seed"]))
        agg = aggregate_multi_seed_runs(run_list)
        summaries[tag] = _json_clean_agg(agg)
        paired_arrays[f"{tag}__ade1"] = agg["paired_sample_ade1"]
        paired_arrays[f"{tag}__fde1"] = agg["paired_sample_fde1"]
    return summaries, paired_arrays


def build_statistical_tables(
    paired: Dict[str, np.ndarray],
    pinns_t5_test: Dict[str, torch.Tensor],
    jta_t2_test: Dict[str, torch.Tensor],
) -> Dict[str, List[Dict]]:
    """Run Section 5.3 Shapiro-Wilk + Paired-t/Wilcoxon + Holm-Bonferroni across all research families."""
    bk_p = pinns_t5_test["bucket"].numpy()
    trans_p = (pinns_t5_test["m_act"].numpy() > 0.5) & (pinns_t5_test["y_act"].numpy()[:, :4].sum(axis=-1) > 0.5)
    bk_j = jta_t2_test["bucket"].numpy()

    def _pair(name: str, key_a: str, key_b: str, mask: Optional[np.ndarray] = None) -> Optional[Dict]:
        if f"{key_a}__ade1" not in paired or f"{key_b}__ade1" not in paired:
            return None
        ea = paired[f"{key_a}__ade1"]
        eb = paired[f"{key_b}__ade1"]
        if mask is not None:
            if int(np.sum(mask)) < 5:
                return None
            ea, eb = ea[mask], eb[mask]
        res = paired_hypothesis_test(ea, eb)
        res["comparison"] = name
        return res

    fam1_specs = [
        ("PINNS T=5 Overall: M1a vs M0", "PINNS_T5_M1a", "PINNS_T5_M0", None),
        ("PINNS T=5 Heavy: M1a vs M0", "PINNS_T5_M1a", "PINNS_T5_M0", bk_p == 2),
        ("PINNS T=5 Overall: M1 vs M0", "PINNS_T5_M1", "PINNS_T5_M0", None),
        ("PINNS T=5 Clear: M1 vs M0", "PINNS_T5_M1", "PINNS_T5_M0", bk_p == 0),
        ("PINNS T=5 Partial: M1 vs M0", "PINNS_T5_M1", "PINNS_T5_M0", bk_p == 1),
        ("PINNS T=5 Heavy: M1 vs M0", "PINNS_T5_M1", "PINNS_T5_M0", bk_p == 2),
        ("PINNS T=5 Overall: M1 vs M1a", "PINNS_T5_M1", "PINNS_T5_M1a", None),
        ("PINNS T=5 Overall: M1 vs M1b (q-vector)", "PINNS_T5_M1", "PINNS_T5_M1b", None),
        ("PINNS T=5 Overall: M1p(BPD) vs M1", "PINNS_T5_M1p", "PINNS_T5_M1", None),
        ("PINNS T=5 Partial: M1p(BPD) vs M1", "PINNS_T5_M1p", "PINNS_T5_M1", bk_p == 1),
        ("PINNS T=10 Overall: M1 vs M0", "PINNS_T10_M1", "PINNS_T10_M0", None),
        ("PINNS T=20 Overall: M1 vs M0", "PINNS_T20_M1", "PINNS_T20_M0", None),
    ]
    fam2_specs = [
        ("PINNS T=5: M2a(+Map) vs M1", "PINNS_T5_M2a", "PINNS_T5_M1", None),
        ("PINNS T=5: M2b(+GAT) vs M1", "PINNS_T5_M2b", "PINNS_T5_M1", None),
        ("PINNS T=5: M2b vs M2b_norad (v_rad)", "PINNS_T5_M2b", "PINNS_T5_M2b_norad", None),
        ("PINNS T=5: M2(Map+GAT) vs M2a", "PINNS_T5_M2", "PINNS_T5_M2a", None),
        ("PINNS T=5: M2(Map+GAT) vs M2b", "PINNS_T5_M2", "PINNS_T5_M2b", None),
        ("PINNS T=5: M2(Map+GAT) vs M1", "PINNS_T5_M2", "PINNS_T5_M1", None),
    ]
    fam3_specs = [
        ("PINNS T=5: M3a(AuxOnly) vs M2", "PINNS_T5_M3a", "PINNS_T5_M2", None),
        ("PINNS T=5: M3b(Cond-E2E) vs M2", "PINNS_T5_M3b", "PINNS_T5_M2", None),
        ("PINNS T=5: M3(StopGrad) vs M3b(E2E)", "PINNS_T5_M3", "PINNS_T5_M3b", None),
        ("PINNS T=5: M3(StopGrad) vs M2", "PINNS_T5_M3", "PINNS_T5_M2", None),
        ("PINNS T=5 Transition: M3 vs M2", "PINNS_T5_M3", "PINNS_T5_M2", trans_p),
        ("PINNS T=5: M3p(ACMC) vs M3", "PINNS_T5_M3p", "PINNS_T5_M3", None),
        ("PINNS T=5 Transition: M3p(ACMC) vs M3", "PINNS_T5_M3p", "PINNS_T5_M3", trans_p),
        ("PINNS T=5: Ours_Full vs M0", "PINNS_T5_Ours_Full", "PINNS_T5_M0", None),
        ("PINNS T=5: Ours_Full vs M3", "PINNS_T5_Ours_Full", "PINNS_T5_M3", None),
        ("PINNS T=5: Ours_Full vs Ext1_MCENET", "PINNS_T5_Ours_Full", "PINNS_T5_Ext1_MCENET", None),
        ("PINNS T=5: Ours_Full vs Ext2_HST", "PINNS_T5_Ours_Full", "PINNS_T5_Ext2_HST", None),
        ("PINNS T=5: Ours_Full vs Ext3_SocialTransmotion", "PINNS_T5_Ours_Full", "PINNS_T5_Ext3_SocialTransmotion", None),
    ]
    fam_jta_specs = [
        ("JTA 0.4s Overall: M1a vs M0", "JTA_T2_M1a", "JTA_T2_M0", None),
        ("JTA 0.4s Heavy(sigma=0.2): M1a vs M0", "JTA_T2_M1a", "JTA_T2_M0", bk_j == 2),
        ("JTA 0.4s Heavy(sigma=0.2): M1 vs M1a", "JTA_T2_M1", "JTA_T2_M1a", bk_j == 2),
        ("JTA 0.4s Partial(LowerOccl): M1p vs M1", "JTA_T2_M1p", "JTA_T2_M1", bk_j == 1),
        ("JTA 0.4s Overall: Ours_Full vs M0", "JTA_T2_Ours_Full", "JTA_T2_M0", None),
        ("JTA 0.4s Overall: Ours_Full vs Ext3_SocialTransmotion", "JTA_T2_Ours_Full", "JTA_T2_Ext3_SocialTransmotion", None),
    ]

    out = {}
    for fam_name, sp_list in (
        ("Research1_PoseQualityGate", fam1_specs),
        ("Research2_MapAndHeteroGraph", fam2_specs),
        ("Research3_ActionAndBaselines", fam3_specs),
        ("CrossDataset_JTA", fam_jta_specs),
    ):
        valid_rows = [r for spec in sp_list if (r := _pair(*spec)) is not None]
        out[fam_name] = holm_bonferroni_adjust(valid_rows, alpha=0.05)
    return out


def execute_suites_from_samples(
    pinns_samples: List[Dict],
    jta_samples: List[Dict],
    diag: Dict,
    out_dir: Path = Path("/kaggle/working"),
    epochs_pinns: int = 32,
    epochs_jta: int = 28,
) -> Path:
    """Materialize splits, execute Suites A-E across 5 seeds, compute stats, and zip results."""
    res_dir = out_dir / "results"
    ckpt_dir = out_dir / "checkpoints"
    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "dataset_diagnostics.json").write_text(json.dumps(diag, indent=2))

    # Materialize PINNS T in {5, 10, 20} and JTA T=2 (0.4s) & T=3 (0.8s)
    p_splits = {
        t: {sp: materialize_window_split(pinns_samples, sp, t_obs=t, dt=DT) for sp in ("train", "val", "test")}
        for t in (5, 10, 20)
    }
    j_splits = {
        t: {sp: materialize_window_split(jta_samples, sp, t_obs=t, dt=0.4) for sp in ("train", "val", "test")}
        for t in (2, 3)
    }

    all_summaries: Dict[str, Dict] = {}
    all_paired: Dict[str, np.ndarray] = {}

    # Suite AB: Research 1 (T=5,10,20 x M0..M1p) + Research 2 (T=5 x M2a, M2b_norad, M2b, M2)
    tasks_ab = []
    for t in (5, 10, 20):
        for m in ("M0", "M1a", "M1b", "M1", "M1p"):
            tasks_ab.append((f"PINNS_T{t}_{m}", m, t, p_splits[t]["train"], p_splits[t]["val"], p_splits[t]["test"], T_FUT, DT))
    for m in ("M2a", "M2b_norad", "M2b", "M2"):
        tasks_ab.append((f"PINNS_T5_{m}", m, 5, p_splits[5]["train"], p_splits[5]["val"], p_splits[5]["test"], T_FUT, DT))

    print(f"=== Running Suite AB ({len(tasks_ab)} configs x 5 seeds) ===")
    s_ab, p_ab = run_model_grid(tasks_ab, FIXED_SEEDS, epochs=epochs_pinns, batch_size=512, ckpt_dir=ckpt_dir)
    all_summaries.update(s_ab); all_paired.update(p_ab)
    (res_dir / "suite_AB.json").write_text(json.dumps(s_ab, indent=2))

    # Suite CD: Research 3 + Ours_Full + External Baselines (MCENET, HST, Social-Transmotion)
    tasks_cd = []
    for m in ("M3a", "M3b", "M3", "M3p", "Ours_Full"):
        tasks_cd.append((f"PINNS_T5_{m}", m, 5, p_splits[5]["train"], p_splits[5]["val"], p_splits[5]["test"], T_FUT, DT))
    for t in (10, 20):
        for m in ("M3", "Ours_Full"):
            tasks_cd.append((f"PINNS_T{t}_{m}", m, t, p_splits[t]["train"], p_splits[t]["val"], p_splits[t]["test"], T_FUT, DT))
    for t in (5, 10, 20):
        for m in ("Ext1_MCENET", "Ext2_HST", "Ext3_SocialTransmotion"):
            tasks_cd.append((f"PINNS_T{t}_{m}", m, t, p_splits[t]["train"], p_splits[t]["val"], p_splits[t]["test"], T_FUT, DT))

    print(f"=== Running Suite CD ({len(tasks_cd)} configs x 5 seeds) ===")
    s_cd, p_cd = run_model_grid(tasks_cd, FIXED_SEEDS, epochs=epochs_pinns, batch_size=512, ckpt_dir=ckpt_dir)
    all_summaries.update(s_cd); all_paired.update(p_cd)
    (res_dir / "suite_CD.json").write_text(json.dumps(s_cd, indent=2))

    # Suite E: JTA Cross-Dataset Validation (5 seeds) + Action Threshold Sensitivity (+-25%, +-50%)
    tasks_jta = []
    for m in ("M0", "M1a", "M1b", "M1", "M1p", "M3", "Ours_Full", "Ext2_HST", "Ext3_SocialTransmotion"):
        tasks_jta.append((f"JTA_T2_{m}", m, 2, j_splits[2]["train"], j_splits[2]["val"], j_splits[2]["test"], 12, 0.4))
    print(f"=== Running Suite E1: JTA Cross-Dataset ({len(tasks_jta)} configs x 5 seeds) ===")
    s_jta, p_jta = run_model_grid(tasks_jta, FIXED_SEEDS, epochs=epochs_jta, batch_size=512)
    all_summaries.update(s_jta); all_paired.update(p_jta)

    sens_tasks = []
    for scale in (0.50, 0.75, 1.25, 1.50):
        tag_s = f"SENS_{int(round(scale*100))}"
        tr_s = materialize_window_split(pinns_samples, "train", t_obs=5, dt=DT, speed_scale=scale, angle_scale=scale)
        va_s = materialize_window_split(pinns_samples, "val", t_obs=5, dt=DT, speed_scale=scale, angle_scale=scale)
        te_s = materialize_window_split(pinns_samples, "test", t_obs=5, dt=DT, speed_scale=scale, angle_scale=scale)
        sens_tasks.append((f"{tag_s}_Ours_Full", "Ours_Full", 5, tr_s, va_s, te_s, T_FUT, DT))
    print("=== Running Suite E2: Action Threshold Sensitivity (+-25%, +-50%) ===")
    s_sens, _ = run_model_grid(sens_tasks, FIXED_SEEDS[:3], epochs=epochs_pinns, batch_size=512)
    all_summaries.update(s_sens)

    # Hardware Latency & VRAM profiling across T in {5, 10, 20}
    latency = {
        f"T={t}": benchmark_inference_latency("Ours_Full", device_str="cuda:0", seq_t=t)
        for t in (5, 10, 20)
    }
    stat_tables = build_statistical_tables(all_paired, p_splits[5]["test"], j_splits[2]["test"])

    master = {
        "diagnostics": diag,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "latency_cuda": latency,
        "summaries": all_summaries,
        "statistical_tests": stat_tables,
    }
    (res_dir / "master_experiment_report.json").write_text(json.dumps(master, indent=2))
    np.savez_compressed(res_dir / "paired_sample_errors.npz", **all_paired)

    bundle_zip = out_dir / "unstructured_shortpose_v1_results.zip"
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in (res_dir, ckpt_dir):
            for fp in sorted(folder.rglob("*")):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(out_dir))
    print(f"=== ALL SUITES COMPLETED! Bundle saved: {bundle_zip} ({bundle_zip.stat().st_size/1e6:.2f} MB) ===")
    return bundle_zip
