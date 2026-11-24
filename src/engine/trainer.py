"""Deterministic GPU-resident training, bucket-balanced sampling, and evaluation engine."""

from typing import Dict, Optional, Tuple
import copy
import random
import threading
import time
import numpy as np
import torch
import torch.nn as nn

_MODEL_INIT_LOCK = threading.Lock()

from ..data.constants import DT, T_FUT
from ..losses.losses import JointPredictionLoss
from ..metrics.metrics_and_stats import (
    compute_action_metrics,
    compute_sample_trajectory_metrics,
    summarize_bucket_slices,
)
from ..models.baselines import HSTBaseline, MCENETBaseline, SocialTransmotionBaseline
from ..models.predictor import MODEL_SPECS, UnstructuredShortPosePredictor


def set_deterministic_seed(seed: int, dev: Optional[torch.device] = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if dev is not None and dev.type == "cuda" and dev.index is not None:
            torch.cuda.set_device(dev)
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)


def build_any_model(name: str, fut_steps: int = T_FUT, dt: float = DT) -> Tuple[nn.Module, bool]:
    """Instantiate model variant or external baseline, returning (model, uses_action_loss)."""
    if name in MODEL_SPECS:
        mod = UnstructuredShortPosePredictor(variant=name, fut_steps=fut_steps, dt=dt)
        return mod, (MODEL_SPECS[name][4] != "none")
    if name == "Ext1_MCENET":
        return MCENETBaseline(fut_steps=fut_steps, dt=dt), False
    if name == "Ext2_HST":
        return HSTBaseline(fut_steps=fut_steps, dt=dt), False
    if name == "Ext3_SocialTransmotion":
        return SocialTransmotionBaseline(fut_steps=fut_steps, dt=dt), False
    raise ValueError(f"Unknown model name: {name}")


def _compute_train_action_pos_weights(y_act: torch.Tensor, m_act: torch.Tensor) -> torch.Tensor:
    valid = m_act > 0.5
    if int(valid.sum().item()) < 5:
        return torch.ones(5, dtype=torch.float32)
    yv = y_act[valid]
    pos = yv.sum(dim=0) + 1.0
    neg = (len(yv) - yv.sum(dim=0)) + 1.0
    # Dampened inverse class ratio in [1.0, 6.0] prevents unstable gradient spikes on rare transitions
    return torch.clamp(torch.sqrt(neg / pos), min=1.0, max=6.0).float()


def _compute_bucket_sampling_probs(buckets: torch.Tensor) -> torch.Tensor:
    b_np = buckets.cpu().numpy()
    counts = np.bincount(b_np, minlength=3).astype(np.float64)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    w = inv_sqrt[b_np]
    w = w / np.sum(w)
    return torch.from_numpy(w.astype(np.float32))


def _slice_gpu_batch(gpu_data: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: v.index_select(0, idx) for k, v in gpu_data.items() if k != "bucket"}


@torch.no_grad()
def evaluate_split_tensors(
    model: nn.Module,
    split_data: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = 512,
    compute_viol: bool = True,
) -> Dict[str, object]:
    """Evaluate a trained model on `split_data`, returning slice summaries and per-sample ADE_1."""
    model.eval()
    n = split_data["x_m"].shape[0]
    trajs_list, probs_list, act_p_list, gates_list = [], [], [], []
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        sub = {
            k: v[s:e].to(device, non_blocking=True)
            for k, v in split_data.items()
            if k not in ("bucket", "fut_xy", "fut_mask", "y_act", "m_act")
        }
        out = model(sub)
        trajs_list.append(out["trajs"].cpu().numpy())
        probs_list.append(out["mode_probs"].cpu().numpy())
        act_p_list.append(out["act_probs"].cpu().numpy())
        if "gate_weights" in out:
            gates_list.append(out["gate_weights"].cpu().numpy())

    pred_trajs = np.concatenate(trajs_list, axis=0)
    mode_probs = np.concatenate(probs_list, axis=0)
    act_probs = np.concatenate(act_p_list, axis=0)
    fut_xy = split_data["fut_xy"].cpu().numpy()
    fut_mask = split_data["fut_mask"].cpu().numpy()
    p_env = split_data["p_env"].cpu().numpy() if compute_viol else None

    samp_m = compute_sample_trajectory_metrics(pred_trajs, mode_probs, fut_xy, fut_mask, p_env=p_env)
    buckets = split_data["bucket"].cpu().numpy()
    y_act = split_data["y_act"].cpu().numpy()
    m_act = split_data["m_act"].cpu().numpy()

    slices = summarize_bucket_slices(samp_m, buckets, y_act=y_act, m_act=m_act)
    act_m = compute_action_metrics(y_act, act_probs, m_act)
    gate_stats = {}
    if gates_list:
        g_all = np.concatenate(gates_list, axis=0)
        for b_id, b_name in ((0, "Clear"), (1, "Partial"), (2, "Heavy")):
            bm = buckets == b_id
            if np.any(bm):
                gate_stats[b_name] = np.mean(g_all[bm], axis=0).tolist()

    return {
        "slices": slices,
        "action": act_m,
        "gate_by_bucket": gate_stats,
        "sample_ade1": samp_m["ADE_1"],
        "sample_fde1": samp_m["FDE_1"],
        "sample_minade6": samp_m["minADE_6"],
    }


def train_and_eval_seed(
    model_name: str,
    seed: int,
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    device_str: str = "cuda:0",
    epochs: int = 32,
    batch_size: int = 256,
    lr: float = 1.5e-3,
    fut_steps: int = T_FUT,
    dt: float = DT,
    balanced_buckets: bool = True,
) -> Dict[str, object]:
    """Train one (model, seed) end-to-end with val ADE_1 checkpointing and test evaluation."""
    dev = torch.device(device_str if (torch.cuda.is_available() or device_str in ("cpu", "mps")) else "cpu")
    with _MODEL_INIT_LOCK:
        set_deterministic_seed(seed, dev=dev)
        model, uses_act_loss = build_any_model(model_name, fut_steps=fut_steps, dt=dt)
    model = model.to(dev)

    pos_w = _compute_train_action_pos_weights(train_data["y_act"], train_data["m_act"])
    criterion = JointPredictionLoss(use_action_loss=uses_act_loss, act_pos_weights=pos_w).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * 0.05)

    gpu_train = {k: v.to(dev) for k, v in train_data.items()}
    n_train = train_data["x_m"].shape[0]
    samp_probs = _compute_bucket_sampling_probs(train_data["bucket"]).to(dev) if balanced_buckets else None
    gen = torch.Generator(device="cpu").manual_seed(seed + 7)

    best_val_ade1 = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    t_start = time.perf_counter()

    for ep in range(epochs):
        model.train()
        if samp_probs is not None:
            perm = torch.multinomial(samp_probs.cpu(), num_samples=n_train, replacement=True, generator=gen).to(dev)
        else:
            perm = torch.randperm(n_train, generator=gen).to(dev)

        for s in range(0, n_train, batch_size):
            idx = perm[s : min(n_train, s + batch_size)]
            if len(idx) < 2:
                continue
            b = _slice_gpu_batch(gpu_train, idx)
            opt.zero_grad(set_to_none=True)
            out = model(b)
            l_dict = criterion(out, b["fut_xy"], b["fut_mask"], b["y_act"], b["m_act"])
            l_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        if (ep + 1) % 4 == 0 or ep == epochs - 1:
            val_res = evaluate_split_tensors(model, val_data, dev, compute_viol=False)
            v_ade1 = float(val_res["slices"]["Overall"]["ADE_1"])
            if v_ade1 < best_val_ade1:
                best_val_ade1 = v_ade1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    test_res = evaluate_split_tensors(model, test_data, dev, compute_viol=True)
    elapsed = time.perf_counter() - t_start
    return {
        "model": model_name,
        "seed": seed,
        "params": int(sum(p.numel() for p in model.parameters())),
        "best_val_ade1": best_val_ade1,
        "train_sec": round(elapsed, 2),
        "test": test_res,
        "best_state_cpu": best_state if seed == 42 else None,
    }


def aggregate_multi_seed_runs(seed_runs: list) -> Dict[str, object]:
    """Aggregate 5-seed runs into mean/std summary tables and sample-averaged paired ADE_1 vector."""
    assert len(seed_runs) > 0
    slice_names = list(seed_runs[0]["test"]["slices"].keys())
    summary_slices: Dict[str, Dict[str, float]] = {}

    for sl in slice_names:
        cnt = float(seed_runs[0]["test"]["slices"][sl].get("count", 0.0))
        if cnt == 0.0:
            summary_slices[sl] = {"count": 0.0}
            continue
        metric_keys = [k for k in seed_runs[0]["test"]["slices"][sl].keys() if k != "count"]
        sl_dict: Dict[str, float] = {"count": cnt}
        for mk in metric_keys:
            vals = np.array([float(r["test"]["slices"][sl][mk]) for r in seed_runs], dtype=np.float64)
            sl_dict[f"{mk}_mean"] = float(np.mean(vals))
            sl_dict[f"{mk}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            sl_dict[f"{mk}_seeds"] = [round(float(x), 6) for x in vals]
        summary_slices[sl] = sl_dict

    act_f1 = np.array([float(r["test"]["action"]["macro_f1"]) for r in seed_runs], dtype=np.float64)
    act_map = np.array([float(r["test"]["action"]["mAP"]) for r in seed_runs], dtype=np.float64)
    paired_sample_ade1 = np.mean(np.stack([r["test"]["sample_ade1"] for r in seed_runs], axis=0), axis=0)
    paired_sample_fde1 = np.mean(np.stack([r["test"]["sample_fde1"] for r in seed_runs], axis=0), axis=0)

    gate_agg = {}
    for b_name in ("Clear", "Partial", "Heavy"):
        g_vals = [r["test"]["gate_by_bucket"][b_name] for r in seed_runs if b_name in r["test"]["gate_by_bucket"]]
        if g_vals:
            gate_agg[b_name] = np.mean(np.array(g_vals, dtype=np.float64), axis=0).tolist()

    return {
        "model": seed_runs[0]["model"],
        "params": seed_runs[0]["params"],
        "num_seeds": len(seed_runs),
        "mean_train_sec": float(np.mean([r["train_sec"] for r in seed_runs])),
        "slices": summary_slices,
        "action_macro_f1_mean": float(np.nanmean(act_f1)),
        "action_macro_f1_std": float(np.nanstd(act_f1, ddof=1)) if len(act_f1) > 1 else 0.0,
        "action_mAP_mean": float(np.nanmean(act_map)),
        "action_mAP_std": float(np.nanstd(act_map, ddof=1)) if len(act_map) > 1 else 0.0,
        "gate_by_bucket": gate_agg,
        "paired_sample_ade1": paired_sample_ade1,
        "paired_sample_fde1": paired_sample_fde1,
    }


@torch.no_grad()
def benchmark_inference_latency(
    model_name: str = "Ours_Full",
    device_str: str = "cuda:0",
    seq_t: int = 5,
    warmup: int = 25,
    iters: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Measure real single-scene inference latency (ms) and peak VRAM across N in {1, 4, 8, 12}."""
    if torch.cuda.is_available():
        dev = torch.device(device_str)
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    mod, _ = build_any_model(model_name)
    mod = mod.to(dev).eval()
    results: Dict[str, Dict[str, float]] = {}

    for n_agents in (1, 4, 8, 12):
        b = {
            "x_m": torch.ones((n_agents, seq_t, 9), device=dev),
            "x_p": torch.ones((n_agents, seq_t, 17, 6), device=dev),
            "q_vec": torch.full((n_agents, 4), 0.8, device=dev),
            "q_part": torch.full((n_agents, 6), 0.8, device=dev),
            "p_env": torch.ones((n_agents, 16, 10, 11), device=dev),
            "m_map": torch.ones((n_agents, 1), device=dev),
            "ego_type_oh": torch.nn.functional.one_hot(torch.zeros(n_agents, dtype=torch.long, device=dev), 8).float(),
            "nbr_edges": torch.ones((n_agents, 12, 9), device=dev),
            "nbr_type_oh": torch.nn.functional.one_hot(torch.ones(n_agents, 12, dtype=torch.long, device=dev), 8).float(),
            "nbr_mask": torch.ones((n_agents, 12), device=dev),
        }
        for _ in range(warmup):
            _ = mod(b)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        elif dev.type == "mps":
            torch.mps.synchronize()

        times_ms = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = mod(b)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            elif dev.type == "mps":
                torch.mps.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.array(times_ms, dtype=np.float64)
        vram_mb = (torch.cuda.max_memory_allocated(dev) / (1024.0 ** 2)) if dev.type == "cuda" else 0.0
        results[f"N={n_agents}"] = {
            "mean_ms": round(float(np.mean(arr)), 3),
            "p50_ms": round(float(np.median(arr)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "peak_vram_mb": round(float(vram_mb), 2),
        }
    return results
