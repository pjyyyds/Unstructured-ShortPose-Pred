"""Clip-clustered (21 clips) & person-clustered paired bootstrap CIs for Table 1/2 Ours_O2_Full (497 windows)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.cloud_experiment_runner import FIXED_SEEDS, stage_pinns_data
from scripts.run_pinns_matched_o2_multiduration import SITES
from scripts.run_table1_apple_to_apple_o2_full import (
    build_pinns_split_tensors,
    eval_table1_seed,
    fit_table1_seed,
)
from src.data.pinns_bev import SOURCE_LABEL


def cluster_paired_ci(diff_1d: np.ndarray, clusters: np.ndarray, sites: np.ndarray, n_boot: int = 20000, seed: int = 2026) -> dict:
    uniq = sorted(set(clusters.tolist()))
    rows_c = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    pt_win = float(diff_1d.mean())
    pt_site = float(np.mean([diff_1d[sites == s].mean() for s in SITES]))
    d_win = np.empty(n_boot, dtype=np.float64)
    d_cl = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        picked = [uniq[j] for j in rng.integers(0, len(uniq), size=len(uniq))]
        pooled = np.concatenate([rows_c[c] for c in picked])
        d_win[b] = diff_1d[pooled].mean()
        d_cl[b] = np.mean([diff_1d[rows_c[c]].mean() for c in picked])
    lw, hw = np.percentile(d_win, [2.5, 97.5])
    lc, hc = np.percentile(d_cl, [2.5, 97.5])
    return {
        "n_clusters": len(uniq),
        "window_weighted": {"mean_ade_m": pt_win, "ci95_ade_m": [float(lw), float(hw)], "strictly_negative": bool(hw < 0.0)},
        "cluster_equal": {"mean_ade_m": float(np.mean([diff_1d[rows_c[c]].mean() for c in uniq])), "ci95_ade_m": [float(lc), float(hc)], "strictly_negative": bool(hc < 0.0)},
        "four_site_equal_mean_ade_m": pt_site,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paired_errors_npz", type=Path)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/w13_table1_cluster"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(FIXED_SEEDS))
    args = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    samples, audit = stage_pinns_data(args.scratch, bev_convention=SOURCE_LABEL)
    te_samples = [s for s in samples if s["split"] == "test"]
    clips = np.array([str(s["seq_id"]) for s in te_samples], dtype=str)
    persons = np.array([f"{s['seq_id']}:{s.get('person_id', i)}" for i, s in enumerate(te_samples)], dtype=str)
    sites = np.array(["_".join(str(s["seq_id"]).split("_")[:2]) for s in te_samples], dtype=str)

    with np.load(args.paired_errors_npz, allow_pickle=False) as p_ar:
        paired_ref = {k: p_ar[k] for k in p_ar.files}

    refs = ("M0", "M1", "M1p", "M2", "M3", "Ours_Full", "Ext1_MCENET", "Ext2_HST", "Ext3_SocialTransmotion")
    out_durs = {}
    for t_obs in (5, 10, 20):
        tr = build_pinns_split_tensors(samples, "train", t_obs)
        va = build_pinns_split_tensors(samples, "val", t_obs)
        te = build_pinns_split_tensors(samples, "test", t_obs)
        s_top1, s_bayes = [], []
        for seed in args.seeds:
            net, bsc, gsc = fit_table1_seed(seed, tr, va, dev)
            sm, _ = eval_table1_seed(net, te, bsc, gsc, dev)
            s_top1.append(sm["ADE_1"]); s_bayes.append(sm["Bayes_ADE"])
        m_top1, m_bayes = np.mean(s_top1, axis=0), np.mean(s_bayes, axis=0)
        comp = {}
        for r_name in refs:
            rk = f"PINNS_T{t_obs}_{r_name}__ade1"
            if rk in paired_ref:
                comp[r_name] = {
                    "top1_clip_cluster": cluster_paired_ci(m_top1 - paired_ref[rk], clips, sites),
                    "top1_person_cluster": cluster_paired_ci(m_top1 - paired_ref[rk], persons, sites),
                    "bayes_clip_cluster": cluster_paired_ci(m_bayes - paired_ref[rk], clips, sites),
                    "bayes_person_cluster": cluster_paired_ci(m_bayes - paired_ref[rk], persons, sites),
                }
        out_durs[f"T{t_obs}"] = {
            "four_site_equal_ours_o2_full": {
                "top1_ade_m": float(np.mean([m_top1[sites == s].mean() for s in SITES])),
                "bayes_ade_m": float(np.mean([m_bayes[sites == s].mean() for s in SITES])),
            },
            "clustered_comparisons": comp,
        }
    report = {"study": "Table 1/2 497-window whole-clip & whole-person clustered bootstrap CIs for Ours_O2_Full", "n_clips": int(len(set(clips.tolist()))), "n_person_tracks": int(len(set(persons.tolist()))), "seeds": list(args.seeds), "durations": out_durs, "pinns_audit": audit}
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    mb = {"report.json": (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()}
    mb["sha256.json"] = (json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in mb.items()}, indent=2) + "\n").encode()
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in mb.items():
            zf.writestr(k, v)


if __name__ == "__main__":
    main()
