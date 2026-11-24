"""Stage 8-9: Resumable incremental multi-seed experiment runner with local result checkpoints."""

from pathlib import Path
import json
import pickle
import time
import numpy as np
import torch

from scripts.cloud_experiment_runner import build_statistical_tables, cache_has_verified_video, run_model_grid
from src.data.pinns_bev import CONVENTIONS, LEGACY_RASTER, SOURCE_LABEL
from src.data.constants import DT, FIXED_SEEDS, T_FUT
from src.data.dataset_builder import materialize_window_split
from src.engine.trainer import benchmark_inference_latency


def main(*, bev_convention=SOURCE_LABEL) -> None:
    proj_root = Path(__file__).resolve().parents[1]
    filename = "extracted_pinns_jta.pkl" if bev_convention == LEGACY_RASTER else f"extracted_pinns_jta_{bev_convention}.pkl"
    pkl_path = proj_root / "data_cache" / filename
    res_dir = proj_root / "results"
    ckpt_dir = proj_root / "checkpoints"
    if bev_convention != LEGACY_RASTER:
        res_dir, ckpt_dir = res_dir / bev_convention, ckpt_dir / bev_convention
    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "rb") as pf:
        bundle = pickle.load(pf)
    if not cache_has_verified_video(bundle["diag"], bev_convention):
        raise ValueError("Run Stage 7 to rebuild the source_label cache with the current original-video decode check.")
    if bev_convention == SOURCE_LABEL:
        marker = res_dir / "dataset_diagnostics.json"
        if marker.exists():
            if not cache_has_verified_video(json.loads(marker.read_text()), bev_convention):
                raise ValueError("Existing results predate the original-video decode check. Use a fresh run directory.")
        else:
            resumable = [*res_dir.glob("suite_*.json"), *res_dir.glob("*.npz"), *ckpt_dir.glob("*.pt")]
            if resumable:
                raise ValueError("Missing BEV convention for existing results. Use a clean source_label run directory.")
            marker.write_text(json.dumps(bundle["diag"], indent=2))

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
    print("ALL STAGE 8-9 SUITES FINISHED!", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bev-convention", choices=CONVENTIONS, default=SOURCE_LABEL)
    main(bev_convention=parser.parse_args().bev_convention)
