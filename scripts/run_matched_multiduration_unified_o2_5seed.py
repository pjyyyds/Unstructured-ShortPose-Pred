"""Five-seed matched multi-duration (T=5, 10, 20) trajectory & 5-action scaling of unified Ours_O2_Full (623 windows)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.benchmark_heldout_sites import load_site_windows
from scripts.run_o2_hetero_graph_pinns_and_dut import load_pinns_graph_cohort
from scripts.run_pinns_matched_o2_multiduration import DURATIONS, SEEDS, SITES, load_matched_623, paired_site_clip_ci
from scripts.run_unified_o2_full_gpu_benchmark import eval_unified_rows, fit_unified_fold, select_unified_scales
from src.metrics.metrics_and_stats import _binary_average_precision


def load_matched_623_unified(matched_npz: Path, meta_dir: Path, pose_npz: Path, joint_zip: Path) -> dict:
    m623 = load_matched_623(matched_npz, pose_npz)
    gr = load_pinns_graph_cohort(meta_dir, pose_npz)
    groups, _, _ = load_site_windows(meta_dir)
    g_keys = [(str(r.clip), int(r.person_id), int(r.t0)) for s in sorted(groups) for r in groups[s]]
    g_map = {k: i for i, k in enumerate(g_keys)}

    with np.load(matched_npz, allow_pickle=False) as m_ar:
        m_raw = {k: m_ar[k] for k in m_ar.files}
    sel = sorted(
        np.flatnonzero(m_raw["strict_disjoint_20"]).tolist(),
        key=lambda i: (str(m_raw["clip"][i]), int(m_raw["person_id"][i]), int(m_raw["t0"][i])),
    )
    m_keys = [(str(m_raw["clip"][i]), int(m_raw["person_id"][i]), int(m_raw["t0"][i])) for i in sel]
    g_idx = np.array([g_map[k] for k in m_keys], dtype=int)

    act_map = {}
    with zipfile.ZipFile(joint_zip) as zf:
        for s in SITES:
            with np.load(io.BytesIO(zf.read(f"{s}/targets.npz")), allow_pickle=False) as t_ar:
                for c, pid, t0, ya, va in zip(t_ar["clip"], t_ar["person_id"], t_ar["t0"], t_ar["y_act"], t_ar["valid_by_class"]):
                    act_map[(str(c), int(pid), int(t0))] = (ya.astype(bool), va.astype(bool))
    ya_arr = np.stack([act_map[k][0] for k in m_keys], axis=0)
    va_arr = np.stack([act_map[k][1] for k in m_keys], axis=0)
    step_v = np.linalg.norm(np.diff(m623["hist20"][:, -5:].numpy(), axis=1), axis=-1) / 0.1
    vh = (0.5 * (step_v[:, 1:] + step_v[:, :-1])).mean(axis=1)
    return {
        "site": m623["site"],
        "clip": m623["clip"],
        "hist20": m623["hist20"],
        "truth": m623["truth30"],
        "bio_seq": m623["bio_seq"],
        "q_bio": m623["q_bio"],
        "edges": gr["edges"][g_idx],
        "is_car": gr["is_car"][g_idx],
        "nbr_mask": gr["nbr_mask"][g_idx],
        "y_act": torch.from_numpy(ya_arr),
        "valid_act": torch.from_numpy(va_arr),
        "v_hist_mean": vh,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("matched_npz", type=Path)
    ap.add_argument("pinns_metadata", type=Path)
    ap.add_argument("pose_npz", type=Path)
    ap.add_argument("joint_zip", type=Path)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    torch.set_num_threads(4)
    dev = torch.device("cpu")
    base_data = load_matched_623_unified(args.matched_npz, args.pinns_metadata, args.pose_npz, args.joint_zip)
    n = len(base_data["site"])
    site_arr, clip_arr = base_data["site"], base_data["clip"]
    yt_all = base_data["y_act"].numpy().astype(bool)
    vm_all = base_data["valid_act"].numpy().astype(bool)

    dur_reports = {}
    t5_bayes_ref = None
    for dur in DURATIONS:
        data = {**base_data, "history": base_data["hist20"][:, -dur:]}
        t_seeds, a_seeds = [], []
        for seed in args.seeds:
            t_cur = np.zeros((n, 5), dtype=np.float64)
            a_cur = np.zeros((n, 5), dtype=np.float64)
            for s in SITES:
                tr_ids = np.flatnonzero(site_arr != s)
                te_ids = np.flatnonzero(site_arr == s)
                net = fit_unified_fold(seed, data, tr_ids, dev)
                bsc, gsc = select_unified_scales(seed, data, tr_ids, dev)
                t_cur[te_ids], a_cur[te_ids] = eval_unified_rows(net, data, te_ids, bsc, gsc, True, dev)
            t_seeds.append(t_cur); a_seeds.append(a_cur)
            print(f"Done matched unified Ours_O2_Full T={dur} seed={seed}", flush=True)

        t_avg = np.mean(t_seeds, axis=0)
        a_avg = np.mean(a_seeds, axis=0)
        v = np.mean([t_avg[site_arr == s].mean(0) for s in SITES], axis=0)
        traj_eq = {"ade_m": float(v[0]), "fde_m": float(v[1]), "bayes_ade_m": float(v[2]), "bayes_fde_m": float(v[3]), "minfde6_m": float(v[4])}
        if dur == 5:
            t5_bayes_ref = t_avg[:, 2:4].copy()

        s_aps, s_f1s, s_brs = [], [], []
        for s in SITES:
            sm = site_arr == s; aps, f1s, brs = [], [], []
            for c_i in range(5):
                vm = sm & vm_all[:, c_i]; yt = yt_all[vm, c_i]; ps = a_avg[vm, c_i]
                if yt.sum() > 0:
                    aps.append(float(_binary_average_precision(yt.astype(np.float64), ps)))
                pr = ps >= 0.5; tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
                f1s.append((2.0 * tp) / max(2 * tp + fp + fn, 1))
                brs.append(float(((ps - yt.astype(np.float64)) ** 2).mean()))
            s_aps.append(float(np.mean(aps))); s_f1s.append(float(np.mean(f1s))); s_brs.append(float(np.mean(brs)))

        entry = {
            "traj_site_equal": traj_eq,
            "action_eligible_site_equal": {"macro_ap": float(np.mean(s_aps)), "macro_f1": float(np.mean(s_f1s)), "mean_brier": float(np.mean(s_brs))},
        }
        if dur > 5 and t5_bayes_ref is not None:
            entry["paired_bayes_minus_T5"] = paired_site_clip_ci(t_avg[:, 2:4] - t5_bayes_ref, site_arr, clip_arr)
        dur_reports[str(dur)] = entry

    report = {"windows": n, "seeds": list(args.seeds), "durations": dur_reports}
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    members = {"report.json": (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()}
    members["sha256.json"] = (json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in members.items()}, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in members.items():
            zf.writestr(k, v)
    print("Saved", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
