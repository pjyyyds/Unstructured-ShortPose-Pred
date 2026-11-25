"""Five-seed unified Ours_O2_Full (20,739 params) progressive LOSO ablation & Tesla T4 latency benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from scripts.run_o2_hetero_graph_pinns_and_dut import load_pinns_graph_cohort
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_matched_o2_multiduration import SITES, paired_site_clip_ci
from scripts.run_pinns_o2_joint_actions_5seed import ACTIONS, SEEDS, load_joint_cohort
from src.metrics.metrics_and_stats import _binary_average_precision
from src.models.bioeq_o2_joint import o2_joint_loss
from src.models.o2_unified_full import O2UnifiedFullForecaster


def load_unified_cohort(meta_dir: Path, pose_npz: Path, joint_zip: Path) -> dict:
    import io
    from scripts.benchmark_heldout_sites import load_site_windows
    gr = load_pinns_graph_cohort(meta_dir, pose_npz)
    groups, _, _ = load_site_windows(meta_dir)
    g_keys = [(str(r.clip), int(r.person_id), int(r.t0)) for s in sorted(groups) for r in groups[s]]
    act_map = {}
    with zipfile.ZipFile(joint_zip) as zf:
        for s in SITES:
            with np.load(io.BytesIO(zf.read(f"{s}/targets.npz")), allow_pickle=False) as t_ar:
                for c, pid, t0, ya, va in zip(t_ar["clip"], t_ar["person_id"], t_ar["t0"], t_ar["y_act"], t_ar["valid_by_class"]):
                    act_map[(str(c), int(pid), int(t0))] = (ya.astype(bool), va.astype(bool))
    ya_arr = np.stack([act_map[k][0] for k in g_keys], axis=0)
    va_arr = np.stack([act_map[k][1] for k in g_keys], axis=0)
    step_v = np.linalg.norm(np.diff(gr["history"].numpy(), axis=1), axis=-1) / 0.1
    vh = (0.5 * (step_v[:, 1:] + step_v[:, :-1])).mean(axis=1)
    return {**gr, "y_act": torch.from_numpy(ya_arr), "valid_act": torch.from_numpy(va_arr), "v_hist_mean": vh}


def fit_unified_fold(seed: int, data: dict, tr_ids: np.ndarray, device: torch.device) -> O2UnifiedFullForecaster:
    torch.manual_seed(seed)
    net = O2UnifiedFullForecaster().to(device)
    w_tr = make_site_weights(data["site"], tr_ids).to(device)
    hist, truth = data["history"][tr_ids].to(device), data["truth"][tr_ids].to(device)
    bseq, qbio = data["bio_seq"][tr_ids].to(device), data["q_bio"][tr_ids].to(device)
    edg, car, msk = data["edges"][tr_ids].to(device), data["is_car"][tr_ids].to(device), data["nbr_mask"][tr_ids].to(device)
    ya, va = data["y_act"][tr_ids].to(device), data["valid_act"][tr_ids].to(device)
    pos = (ya & va).sum(0).float().clamp_min(1.0)
    neg = ((~ya) & va).sum(0).float().clamp_min(1.0)
    pw = (neg / pos).clamp(1.0, 12.0).to(device)
    gen = torch.Generator().manual_seed(seed)

    # Stage 1: O(2) motion + parity action head + stop-grad bridge
    s1_names = ("motion.", "ctx_proj.", "act_even.", "act_turn.", "bridge_even.", "bridge_odd.", "central_", "pair_")
    for n, p in net.named_parameters():
        p.requires_grad = n.startswith(s1_names)
    opt1 = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-4)
    for _ in range(45):
        for c_cpu in torch.randperm(len(tr_ids), generator=gen).split(64):
            ch = c_cpu.to(device)
            out = net(hist[ch], bio_scale=0.0, graph_scale=0.0, use_action_bridge=True)
            loss = o2_joint_loss(out, truth[ch], w_tr[ch], ya[ch], va[ch], pw, act_weight=0.25)
            if loss is not None:
                opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()

    # Stage 2: Decoupled regime-gated biomechanical pose residual
    for n, p in net.named_parameters():
        p.requires_grad = n.startswith("bio_") or n == "gate_logit"
    opt2 = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1.5e-3, weight_decay=2e-4)
    for _ in range(25):
        for c_cpu in torch.randperm(len(tr_ids), generator=gen).split(64):
            ch = c_cpu.to(device)
            out = net(hist[ch], bseq[ch], qbio[ch], bio_scale=1.0, graph_scale=0.0, use_action_bridge=True)
            loss = o2_joint_loss(out, truth[ch], w_tr[ch], ya[ch], va[ch], pw, act_weight=0.15, reg_only=False)
            if loss is not None:
                opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step()

    # Stage 3: O(2) null-sink heterogeneous neighbor graph
    for n, p in net.named_parameters():
        p.requires_grad = n.startswith(("edge_", "graph_proj")) or n == "null_logit"
    opt3 = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1.5e-3, weight_decay=3e-4)
    for _ in range(28):
        for c_cpu in torch.randperm(len(tr_ids), generator=gen).split(64):
            ch = c_cpu.to(device)
            out = net(hist[ch], bseq[ch], qbio[ch], edg[ch], car[ch], msk[ch], bio_scale=0.5, graph_scale=1.0, use_action_bridge=True)
            loss = o2_joint_loss(out, truth[ch], w_tr[ch], act_weight=0.0)
            if loss is not None:
                opt3.zero_grad(set_to_none=True); loss.backward(); opt3.step()
    return net.eval()


def select_unified_scales(seed: int, data: dict, tr_ids: np.ndarray, device: torch.device) -> tuple[float, float]:
    src = sorted(set(data["site"][tr_ids].tolist()))
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    b_sc = np.zeros(len(grid))
    g_sc = np.zeros(len(grid))
    nets_v = []
    for vs in src:
        fi = tr_ids[data["site"][tr_ids] != vs]
        vi = tr_ids[data["site"][tr_ids] == vs]
        nv = fit_unified_fold(seed, data, fi, device)
        nets_v.append((nv, vi))
        with torch.inference_mode():
            hv, tv = data["history"][vi].to(device), data["truth"][vi].to(device)
            bv, qv = data["bio_seq"][vi].to(device), data["q_bio"][vi].to(device)
            for i, bs in enumerate(grid):
                oo = nv(hv, bv, qv, bio_scale=bs, graph_scale=0.0, use_action_bridge=True)
                pr = oo["paths"][torch.arange(len(vi), device=device), oo["logits"].argmax(1)]
                b_sc[i] += float(torch.linalg.vector_norm(pr - tv, dim=-1).mean())
    best_b = float(grid[int(b_sc.argmin())])
    for nv, vi in nets_v:
        with torch.inference_mode():
            hv, tv = data["history"][vi].to(device), data["truth"][vi].to(device)
            bv, qv = data["bio_seq"][vi].to(device), data["q_bio"][vi].to(device)
            ev, cv, mv = data["edges"][vi].to(device), data["is_car"][vi].to(device), data["nbr_mask"][vi].to(device)
            for i, gs in enumerate(grid):
                oo = nv(hv, bv, qv, ev, cv, mv, bio_scale=best_b, graph_scale=gs, use_action_bridge=True)
                pr = oo["paths"][torch.arange(len(vi), device=device), oo["logits"].argmax(1)]
                g_sc[i] += float(torch.linalg.vector_norm(pr - tv, dim=-1).mean())
    return best_b, float(grid[int(g_sc.argmin())])


def benchmark_gpu_latency_and_flops(device: torch.device) -> dict:
    net = O2UnifiedFullForecaster().to(device).eval()
    params = sum(p.numel() for p in net.parameters())
    cells = {}
    for t_obs, n_ped in ((5, 1), (5, 12), (10, 12), (20, 12)):
        torch.manual_seed(42)
        h = (torch.randn(n_ped, t_obs, 2, dtype=torch.float64, device=device) * 0.4)
        h[:, -1] = 0.0
        b = torch.randn(n_ped, t_obs, 11, device=device)
        q = torch.full((n_ped,), 0.8, device=device)
        e = torch.randn(n_ped, 12, 9, device=device)
        c = torch.zeros(n_ped, 12, device=device)
        m = torch.ones(n_ped, 12, device=device)
        with FlopCounterMode(display=False) as fc:
            _ = net(h, b, q, e, c, m, bio_scale=0.5, graph_scale=0.5, use_action_bridge=True)
        flops = int(fc.get_total_flops())
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            for _ in range(25):
                with torch.inference_mode():
                    _ = net(h, b, q, e, c, m, bio_scale=0.5, graph_scale=0.5, use_action_bridge=True)
            torch.cuda.synchronize(device)
            times = []
            for _ in range(200):
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    _ = net(h, b, q, e, c, m, bio_scale=0.5, graph_scale=0.5, use_action_bridge=True)
                torch.cuda.synchronize(device)
                times.append((time.perf_counter() - t0) * 1000.0)
            arr = np.array(times, dtype=np.float64)
            mem_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
        else:
            arr = np.zeros(10)
            mem_mib = 0.0
        cells[f"T{t_obs}_N{n_ped}"] = {
            "flops": flops,
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "peak_cuda_mem_mib": mem_mib,
        }
    return {"trainable_params": int(params), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "cells": cells}


@torch.inference_mode()
def eval_unified_rows(net: O2UnifiedFullForecaster, data: dict, ids: np.ndarray, bsc: float, gsc: float, bridge: bool, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    out = net(
        data["history"][ids].to(device),
        data["bio_seq"][ids].to(device),
        data["q_bio"][ids].to(device),
        data["edges"][ids].to(device),
        data["is_car"][ids].to(device),
        data["nbr_mask"][ids].to(device),
        bio_scale=bsc,
        graph_scale=gsc,
        use_action_bridge=bridge,
    )
    p = out["paths"].cpu()
    q = out["logits"].softmax(1).cpu()
    truth = data["truth"][ids]
    dst = torch.linalg.vector_norm(p - truth[:, None], dim=-1)
    top = p[torch.arange(len(ids)), q.argmax(1)]
    bay = torch.einsum("bk,bktw->btw", q.double(), p)
    dt = torch.linalg.vector_norm(top - truth, dim=-1)
    db = torch.linalg.vector_norm(bay - truth, dim=-1)
    mf6 = dst[:, :, -1].amin(dim=1)
    pa = torch.sigmoid(out["act_logits"]).cpu().numpy().astype(np.float64)
    vh = data["v_hist_mean"][ids]
    pa[:, 0] = np.where(vh <= 0.20, pa[:, 0], 0.0)
    pa[:, 1] = np.where(vh >= 0.60, pa[:, 1], 0.0)
    return torch.stack((dt.mean(1), dt[:, -1], db.mean(1), db[:, -1], mf6), dim=1).numpy(), pa


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pinns_metadata", type=Path)
    ap.add_argument("pose_npz", type=Path)
    ap.add_argument("joint_zip", type=Path)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = load_unified_cohort(args.pinns_metadata, args.pose_npz, args.joint_zip)
    n = len(data["site"])
    site_arr, clip_arr = data["site"], data["clip"]
    yt_all = data["y_act"].numpy().astype(bool)
    vm_all = data["valid_act"].numpy().astype(bool)

    arms = ("U0_O2_Motion", "U1_O2_BioEq", "U2_O2_BioEq_Graph", "Ours_O2_Full")
    traj_acc = {a: [] for a in arms}
    act_acc = []
    for seed in args.seeds:
        cur = {a: np.zeros((n, 5), dtype=np.float64) for a in arms}
        pa_cur = np.zeros((n, 5), dtype=np.float64)
        for s in SITES:
            tr_ids = np.flatnonzero(site_arr != s)
            te_ids = np.flatnonzero(site_arr == s)
            net = fit_unified_fold(seed, data, tr_ids, device)
            bsc, gsc = select_unified_scales(seed, data, tr_ids, device)
            cur["U0_O2_Motion"][te_ids], _ = eval_unified_rows(net, data, te_ids, 0.0, 0.0, False, device)
            cur["U1_O2_BioEq"][te_ids], _ = eval_unified_rows(net, data, te_ids, bsc, 0.0, False, device)
            cur["U2_O2_BioEq_Graph"][te_ids], _ = eval_unified_rows(net, data, te_ids, bsc, gsc, False, device)
            cur["Ours_O2_Full"][te_ids], pa_cur[te_ids] = eval_unified_rows(net, data, te_ids, bsc, gsc, True, device)
        for a in arms:
            traj_acc[a].append(cur[a])
        act_acc.append(pa_cur)
        print(f"Done unified Ours_O2_Full seed={seed}", flush=True)

    t_avg = {a: np.mean(traj_acc[a], axis=0) for a in arms}
    pa_avg = np.mean(act_acc, axis=0)
    def teq(arr: np.ndarray) -> dict:
        v = np.mean([arr[site_arr == s].mean(0) for s in SITES], axis=0)
        return {"ade_m": float(v[0]), "fde_m": float(v[1]), "bayes_ade_m": float(v[2]), "bayes_fde_m": float(v[3]), "minfde6_m": float(v[4])}

    s_aps, s_f1s, s_brs = [], [], []
    for s in SITES:
        sm = site_arr == s; aps, f1s, brs = [], [], []
        for c_i in range(5):
            vm = sm & vm_all[:, c_i]; yt = yt_all[vm, c_i]; ps = pa_avg[vm, c_i]
            if yt.sum() > 0:
                aps.append(float(_binary_average_precision(yt.astype(np.float64), ps)))
            pr = ps >= 0.5; tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
            f1s.append((2.0 * tp) / max(2 * tp + fp + fn, 1))
            brs.append(float(((ps - yt.astype(np.float64)) ** 2).mean()))
        s_aps.append(float(np.mean(aps))); s_f1s.append(float(np.mean(f1s))); s_brs.append(float(np.mean(brs)))

    hw_bench = benchmark_gpu_latency_and_flops(device)
    report = {
        "windows": n,
        "seeds": list(args.seeds),
        "hardware_and_complexity": hw_bench,
        "progressive_ablation_5seed_site_equal": {a: teq(t_avg[a]) for a in arms},
        "ours_o2_full_action_eligible_site_equal": {
            "macro_ap": float(np.mean(s_aps)),
            "macro_f1": float(np.mean(s_f1s)),
            "mean_brier": float(np.mean(s_brs)),
        },
        "paired_contrasts": {
            "full_minus_motion_top1": paired_site_clip_ci(t_avg["Ours_O2_Full"][:, :2] - t_avg["U0_O2_Motion"][:, :2], site_arr, clip_arr),
            "full_minus_motion_bayes": paired_site_clip_ci(t_avg["Ours_O2_Full"][:, 2:4] - t_avg["U0_O2_Motion"][:, 2:4], site_arr, clip_arr),
            "full_minus_bioeq_bayes": paired_site_clip_ci(t_avg["Ours_O2_Full"][:, 2:4] - t_avg["U1_O2_BioEq"][:, 2:4], site_arr, clip_arr),
        },
    }
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    members = {"report.json": (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()}
    members["sha256.json"] = (json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in members.items()}, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in members.items():
            zf.writestr(k, v)
    print("Saved", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
