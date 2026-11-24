"""Stage 8-9: Resumable incremental multi-seed experiment runner with auto-git checkpointing."""

from pathlib import Path
import json
import pickle
import subprocess
import time
import numpy as np
import torch

from scripts.cloud_experiment_runner import build_statistical_tables, run_model_grid
from src.data.constants import DT, FIXED_SEEDS, T_FUT
from src.data.dataset_builder import materialize_window_split
from src.engine.trainer import benchmark_inference_latency


def _git_checkpoint(proj_root: Path, msg: str) -> None:
    try:
        subprocess.run(["git", "-C", str(proj_root), "add", "results/", "CONTEXT_MEMORY.md"], check=False)
        subprocess.run(["git", "-C", str(proj_root), "commit", "-m", msg], check=False)
        subprocess.run(["git", "-C", str(proj_root), "push", "-u", "origin", "main"], timeout=15, check=False)
    except Exception:
        pass


def main() -> None:
    proj_root = Path(__file__).resolve().parents[1]
    pkl_path = proj_root / "data_cache" / "extracted_pinns_jta.pkl"
    res_dir = proj_root / "results"
    ckpt_dir = proj_root / "checkpoints"
    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "rb") as pf:
        bundle = pickle.load(pf)
    pinns_samples = bundle["pinns"]
    jta_samples = bundle["jta"]
    diag = bundle["diag"]
    print(f"Loaded cache: PINNS={len(pinns_samples)} samples, JTA={len(jta_samples)} samples", flush=True)

    p_splits = {
        t: {sp: materialize_window_split(pinns_samples, sp, t_obs=t, dt=DT) for sp in ("train", "val", "test")}
        for t in (5, 10, 20)
    }
    j_splits = {
        2: {sp: materialize_window_split(jta_samples, sp, t_obs=2, dt=0.4) for sp in ("train", "val", "test")}
    }

    all_summaries = {}
    all_paired = {}
    paired_npz = res_dir / "paired_sample_errors.npz"
    if paired_npz.exists():
        all_paired.update(dict(np.load(paired_npz)))

    ep_p, ep_j = 22, 20

    # 1. Suite AB
    ab_file = res_dir / "suite_AB.json"
    if ab_file.exists():
        all_summaries.update(json.loads(ab_file.read_text()))
        print("[Resume] Loaded existing suite_AB.json", flush=True)
    else:
        tasks_ab = [
            (f"PINNS_T{t}_{m}", m, t, p_splits[t]["train"], p_splits[t]["val"], p_splits[t]["test"], T_FUT, DT)
            for t in (5, 10, 20)
            for m in ("M0", "M1a", "M1b", "M1", "M1p")
        ] + [
            (f"PINNS_T5_{m}", m, 5, p_splits[5]["train"], p_splits[5]["val"], p_splits[5]["test"], T_FUT, DT)
            for m in ("M2a", "M2b_norad", "M2b", "M2")
        ]
        print(f"=== Running Suite AB ({len(tasks_ab)} configs x 5 seeds) ===", flush=True)
        s_ab, p_ab = run_model_grid(tasks_ab, FIXED_SEEDS, epochs=ep_p, batch_size=512, ckpt_dir=ckpt_dir)
        all_summaries.update(s_ab); all_paired.update(p_ab)
        ab_file.write_text(json.dumps(s_ab, indent=2))
        np.savez_compressed(paired_npz, **all_paired)
        _git_checkpoint(proj_root, "exp(suite-AB): complete Research 1 & 2 5-seed results on PINNS")

    # 2. Suite CD
    cd_file = res_dir / "suite_CD.json"
    if cd_file.exists():
        all_summaries.update(json.loads(cd_file.read_text()))
        print("[Resume] Loaded existing suite_CD.json", flush=True)
    else:
        tasks_cd = (
            [(f"PINNS_T5_{m}", m, 5, p_splits[5]["train"], p_splits[5]["val"], p_splits[5]["test"], T_FUT, DT) for m in ("M3a", "M3b", "M3", "M3p", "Ours_Full")]
            + [(f"PINNS_T{t}_{m}", m, t, p_splits[t]["train"], p_splits[t]["val"], p_splits[t]["test"], T_FUT, DT) for t in (10, 20) for m in ("M3", "Ours_Full")]
            + [(f"PINNS_T{t}_{m}", m, t, p_splits[t]["train"], p_splits[t]["val"], p_splits[t]["test"], T_FUT, DT) for t in (5, 10, 20) for m in ("Ext1_MCENET", "Ext2_HST", "Ext3_SocialTransmotion")]
        )
        print(f"=== Running Suite CD ({len(tasks_cd)} configs x 5 seeds) ===", flush=True)
        s_cd, p_cd = run_model_grid(tasks_cd, FIXED_SEEDS, epochs=ep_p, batch_size=512, ckpt_dir=ckpt_dir)
        all_summaries.update(s_cd); all_paired.update(p_cd)
        cd_file.write_text(json.dumps(s_cd, indent=2))
        np.savez_compressed(paired_npz, **all_paired)
        _git_checkpoint(proj_root, "exp(suite-CD): complete Research 3 & SOTA baselines 5-seed results on PINNS")

    # 3. Suite E (JTA + Sensitivity + Latency + Statistical Tables)
    tasks_jta = [
        (f"JTA_T2_{m}", m, 2, j_splits[2]["train"], j_splits[2]["val"], j_splits[2]["test"], 12, 0.4)
        for m in ("M0", "M1a", "M1b", "M1", "M1p", "M3", "Ours_Full", "Ext2_HST", "Ext3_SocialTransmotion")
    ]
    print(f"=== Running Suite E1: JTA Cross-Dataset ({len(tasks_jta)} configs x 5 seeds) ===", flush=True)
    s_jta, p_jta = run_model_grid(tasks_jta, FIXED_SEEDS, epochs=ep_j, batch_size=512)
    all_summaries.update(s_jta); all_paired.update(p_jta)

    sens_tasks = []
    for scale in (0.50, 0.75, 1.25, 1.50):
        tag_s = f"SENS_{int(round(scale*100))}"
        tr_s = materialize_window_split(pinns_samples, "train", t_obs=5, dt=DT, speed_scale=scale, angle_scale=scale)
        va_s = materialize_window_split(pinns_samples, "val", t_obs=5, dt=DT, speed_scale=scale, angle_scale=scale)
        te_s = materialize_window_split(pinns_samples, "test", t_obs=5, dt=DT, speed_scale=scale, angle_scale=scale)
        sens_tasks.append((f"{tag_s}_Ours_Full", "Ours_Full", 5, tr_s, va_s, te_s, T_FUT, DT))
    print("=== Running Suite E2: Action Threshold Sensitivity ===", flush=True)
    s_sens, _ = run_model_grid(sens_tasks, FIXED_SEEDS[:3], epochs=ep_p, batch_size=512)
    all_summaries.update(s_sens)

    latency = {f"T={t}": benchmark_inference_latency("Ours_Full", seq_t=t) for t in (5, 10, 20)}
    stat_tables = build_statistical_tables(all_paired, p_splits[5]["test"], j_splits[2]["test"])
    hw_name = "Apple M5 GPU (MPS)" if torch.backends.mps.is_available() else "CPU"
    master = {
        "diagnostics": diag,
        "gpu_name": hw_name,
        "gpu_count": 1,
        "latency_cuda": latency,
        "summaries": all_summaries,
        "statistical_tests": stat_tables,
    }
    (res_dir / "master_experiment_report.json").write_text(json.dumps(master, indent=2))
    np.savez_compressed(paired_npz, **all_paired)
    _git_checkpoint(proj_root, "exp(master): complete JTA cross-dataset, sensitivity, latency, and Holm-Bonferroni tables")
    print("ALL STAGE 8-9 SUITES FINISHED!", flush=True)


if __name__ == "__main__":
    main()
