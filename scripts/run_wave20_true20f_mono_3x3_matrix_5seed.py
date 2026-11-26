"""Wave 20 GPU1: True 20-frame 623-window 3x3 (T=5,10,20 x Clear,Partial,Heavy) matrix with M1_mono_q & Spearman rho(g, q1)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from scipy.stats import spearmanr
import torch

from scripts.ppt_mainline_o2_models import PPTMainlineO2Forecaster
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_matched_o2_multiduration import DURATIONS, SEEDS, SITES
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import _copy_matching
from scripts.run_ppt_rc1_3x3_duration_quality_matrix import eval_rc1_dur, load_matched_623_rc1_20frame
from scripts.run_wave18_conflict_gated_interaction_5seed import safe_site_clip_ci
from scripts.run_wave18_t4_latency_and_monotonic_gate import MonotonicQualityM1Forecaster
from src.models.bioeq_o2_joint import o2_joint_loss

W20_DUR_ARMS = ("M0", "M1a", "M1b", "M1_unconstrained", "M1_mono_q")


def fit_true20f_mono_fold(seed: int, data: dict, dur: int, tr_ids: np.ndarray, device: torch.device) -> dict[str, torch.nn.Module]:
    w_tr = make_site_weights(data["site"], tr_ids).to(device)
    truth_tr = data["truth"][tr_ids].to(device)
    tr_t = torch.from_numpy(tr_ids)
    dd = data["per_dur"][dur]

    def _fwd(net: torch.nn.Module, ids: torch.Tensor) -> dict:
        return net(
            data["hist20"][ids, -dur:].to(device), dd["bio_even"][ids].to(device), dd["pose_vec_world"][ids].to(device),
            dd["signed_knee"][ids].to(device), dd["quality"][ids].to(device), dd["m_pose"][ids].to(device),
        )

    def _train(net: torch.nn.Module, prefs: tuple[str, ...], ep: int, lr: float, wd: float, soff: int) -> torch.nn.Module:
        for nm, p in net.named_parameters():
            p.requires_grad = nm.startswith(prefs) or nm in prefs
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(seed + soff)
        net.train()
        for _ in range(ep):
            for ch in torch.randperm(len(tr_ids), generator=gen).split(64):
                loss = o2_joint_loss(_fwd(net, tr_t[ch]), truth_tr[ch], w_tr[ch], act_weight=0.0)
                if loss is not None:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return net.eval()

    torch.manual_seed(seed)
    m0 = _train(PPTMainlineO2Forecaster("M0").to(device), ("motion.", "ctx_proj.", "cv_low_bonus", "central_", "pair_"), 42, 2e-3, 1e-4, 0)
    models: dict[str, torch.nn.Module] = {"M0": m0}
    for i, (key, arm) in enumerate((("M1a", "M1a"), ("M1b", "M1b"), ("M1_unconstrained", "M1"))):
        torch.manual_seed(seed)
        net = PPTMainlineO2Forecaster(arm).to(device); _copy_matching(net, m0)
        models[key] = _train(net, ("pose_", "gate_", "central_", "pair_", "cv_low_bonus"), 26, 1.2e-3, 3e-4, 10 + i)
    torch.manual_seed(seed)
    mm = MonotonicQualityM1Forecaster().to(device); _copy_matching(mm, m0)
    models["M1_mono_q"] = _train(mm, ("pose_", "mono_", "central_", "pair_", "cv_low_bonus"), 26, 1.2e-3, 3e-4, 15)
    return models


def run_true20f_mono_matrix(data: dict, seeds: list[int], device: torch.device) -> dict:
    n, site_arr, clip_arr = len(data["site"]), data["site"], data["clip"]
    qb5 = data["per_dur"][5]["q_buckets"]
    h5 = data["hist20"][:, -5:].numpy()
    cv_p = (h5[:, -1:] - h5[:, -2:-1]) * np.arange(1, 31, dtype=np.float64)[None, :, None]
    d_cv = np.linalg.norm(cv_p - data["truth"].numpy(), axis=-1)
    cv_m = np.column_stack((d_cv.mean(1), d_cv[:, -1], d_cv.mean(1), d_cv[:, -1], d_cv.mean(1), d_cv[:, -1], np.zeros(n)))

    cols = ("ade1_m", "fde1_m", "min_ade6_m", "min_fde6_m", "bayes_ade_m", "bayes_fde_m", "mean_gate")
    contrasts = (
        ("M1_mono_q_minus_M0", "M1_mono_q", "M0"),
        ("M1_mono_q_minus_M1b", "M1_mono_q", "M1b"),
        ("M1_mono_q_minus_M1_unconstrained", "M1_mono_q", "M1_unconstrained"),
        ("M1_unconstrained_minus_M1b", "M1_unconstrained", "M1b"),
        ("M1_mono_q_minus_CV", "M1_mono_q", "CV"),
    )

    by_dur = {}
    for dur in DURATIONS:
        acc = {a: np.zeros((len(seeds), n, 7), dtype=np.float64) for a in W20_DUR_ARMS}
        for si, seed in enumerate(seeds):
            for s in sorted(set(site_arr.tolist())):
                tr_ids, te_ids = np.flatnonzero(site_arr != s), np.flatnonzero(site_arr == s)
                mods = fit_true20f_mono_fold(seed, data, dur, tr_ids, device)
                for a in W20_DUR_ARMS:
                    acc[a][si, te_ids] = eval_rc1_dur(mods[a], data, dur, te_ids, device)  # type: ignore[arg-type]
        mm = {"CV": cv_m, **{a: acc[a].mean(0) for a in W20_DUR_ARMS}}
        dd = data["per_dur"][dur]
        q_np = dd["quality"].numpy()
        mp_np = dd["m_pose"].numpy() > 0.5
        gate_audit = {}
        for a in ("M1_unconstrained", "M1_mono_q"):
            g_v = mm[a][mp_np, 6]
            gate_audit[a] = {
                "spearman_gate_vs_q1": float(spearmanr(g_v, q_np[mp_np, 0]).statistic) if mp_np.sum() >= 5 and np.std(q_np[mp_np, 0]) > 1e-8 else 0.0,
                "spearman_gate_vs_q2": float(spearmanr(g_v, q_np[mp_np, 1]).statistic) if mp_np.sum() >= 5 and np.std(q_np[mp_np, 1]) > 1e-8 else 0.0,
            }
        qbd = dd["q_buckets"]
        slices = {
            "Overall_623": np.ones(n, dtype=bool),
            "Clear": qbd == 0, "Partial": qbd == 1, "Heavy": qbd == 2,
            "Clear_T5fixed": qb5 == 0, "Partial_T5fixed": qb5 == 1, "Heavy_T5fixed": qb5 == 2,
        }
        sl_out = {}
        for sn, sm in slices.items():
            if int(sm.sum()) == 0:
                continue
            sub_s, sub_c = site_arr[sm], clip_arr[sm]
            p_sites = sorted(set(sub_s.tolist()))
            md = {}
            for m, arr in mm.items():
                sub = arr[sm]
                bys = {s: {c: float(sub[sub_s == s, j].mean()) for j, c in enumerate(cols)} for s in p_sites}
                md[m] = {"site_equal": {c: float(np.mean([v[c] for v in bys.values()])) for c in cols},
                         "pooled": {c: float(sub[:, j].mean()) for j, c in enumerate(cols)}}
            sl_out[sn] = {
                "n_windows": int(sm.sum()),
                "models": md,
                "paired_top1_clip_ci": {cn: safe_site_clip_ci(mm[ma][sm, :2] - mm[mb][sm, :2], sub_s, sub_c) for cn, ma, mb in contrasts},
            }
        by_dur[f"T{dur}"] = {"gate_audit": gate_audit, "slices": sl_out}
        print(f"[W20-GPU1] Finished true-20f monotonic 3x3 duration T={dur}", flush=True)
    return by_dur


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matched-npz", type=Path, required=True)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--matched-obs-npz", type=Path, required=True)
    ap.add_argument("--scratch-dir", type=Path, default=Path("/tmp/w20_true20f_scratch"))
    ap.add_argument("--zip-path", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    if not args.matched_obs_npz.exists():
        from scripts.run_heldout_cloud import pip_install, stage
        from scripts.run_pinns_matched_durations import collect
        probe = subprocess.run([sys.executable, "-c", "import onnxruntime as ort; raise SystemExit('CUDAExecutionProvider' not in ort.get_available_providers())"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if probe.returncode:
            pip_install("onnxruntime-gpu")
        ds_dir, onnx_p = stage(args.scratch_dir)
        collect(ds_dir, onnx_p, args.pose_npz, args.matched_obs_npz)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_matched_623_rc1_20frame(args.matched_npz, args.meta_dir, args.pose_npz, args.matched_obs_npz)
    by_dur = run_true20f_mono_matrix(data, list(args.seeds), dev)
    summary = {"experiment": "wave20_true20f_mono_3x3_matrix_5seed", "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
               "n_windows": len(data["site"]), "seeds": list(args.seeds), "coordinate_diagnostics_20f": data["coord_diag_20f"], "durations": by_dur}
    raw = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = (json.dumps({"summary.json": hashlib.sha256(raw).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw); zf.writestr("sha256.json", sh)
    print("Saved:", args.zip_path, flush=True)


if __name__ == "__main__":
    main()
