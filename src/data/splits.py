"""Section 5.1 3-level anti-leakage splitter and zero-overlap physical time auditor."""

from typing import Dict, List, Sequence, Tuple
import numpy as np

from .constants import DT, TEMPORAL_ISOLATION_SEC


def temporal_buffer_split_indices(
    frame_indices_10hz: np.ndarray,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    buffer_sec: float = TEMPORAL_ISOLATION_SEC,
    max_hist_sec: float = 1.9,
    fut_sec: float = 3.0,
    dt: float = DT,
) -> Dict[str, np.ndarray]:
    """Split t_0 trigger frame indices along time with >= 6.0s physical isolation bands.

    Guarantees that for any split A < split B in chronological order:
        max(t_0^A + fut_sec) < min(t_0^B - max_hist_sec)
    """
    frames = np.sort(np.unique(np.asarray(frame_indices_10hz, dtype=np.int64)))
    if len(frames) == 0:
        return {"train": frames, "val": frames, "test": frames}

    t_min, t_max = int(frames[0]), int(frames[-1])
    # Buffer in 10Hz steps: max(buffer_sec, max_hist_sec + fut_sec + 1.1s) >= 60 steps (6.0s)
    buf_steps = int(np.ceil(max(buffer_sec, max_hist_sec + fut_sec + 1.1) / dt))
    min_fut_steps = int(np.ceil(0.5 * fut_sec / dt))

    usable_hi = max(t_max - min_fut_steps, t_min + 3)
    net_span = max(usable_hi - t_min - 2 * buf_steps, 3)
    tr_span = max(1, int(round(ratios[0] * net_span)))
    rem_ratio = ratios[1] / max(ratios[1] + ratios[2], 1e-6)
    va_span = max(1, int(round(rem_ratio * max(net_span - tr_span, 2))))

    cut1 = t_min + tr_span
    cut2 = cut1 + buf_steps + va_span

    train_f = frames[frames <= cut1]
    val_f = frames[(frames >= cut1 + buf_steps) & (frames <= cut2)]
    test_f = frames[frames >= cut2 + buf_steps]
    return {"train": train_f, "val": val_f, "test": test_f}


def assign_site_clip_splits(
    clip_ids: Sequence[int],
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    boundary_guard_steps: int = 60,
) -> Dict[int, Tuple[str, int]]:
    """Assign chronological clips of one camera site to (split, min_t0_guard_steps).

    Preserves all three splits across multi-clip sites while enforcing a 6.0s (60-step)
    guard at split-transition clips in case consecutive clips came from one raw stream.
    """
    ordered = sorted({int(c) for c in clip_ids})
    m = len(ordered)
    if m < 3:
        return {}
    n_test = max(1, int(round(ratios[2] * m)))
    n_val = max(1, int(round(ratios[1] * m)))
    if n_val + n_test >= m:
        n_val, n_test = 1, 1
    n_train = m - n_val - n_test

    mapping: Dict[int, Tuple[str, int]] = {}
    for idx, cid in enumerate(ordered):
        if idx < n_train:
            mapping[cid] = ("train", 0)
        elif idx < n_train + n_val:
            guard = boundary_guard_steps if idx == n_train else 0
            mapping[cid] = ("val", guard)
        else:
            guard = boundary_guard_steps if idx == n_train + n_val else 0
            mapping[cid] = ("test", guard)
    return mapping


def verify_zero_temporal_leakage(
    samples_meta: Sequence[Dict],
    max_hist_steps: int = 19,
    fut_steps: int = 30,
) -> Tuple[bool, str]:
    """Audit that no two samples from different splits overlap in physical [t0-19, t0+30] frames."""
    by_seq: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
    for s in samples_meta:
        seq = str(s["seq_id"])
        sp = str(s["split"])
        t0 = int(s["t0_frame_10hz"])
        by_seq.setdefault(seq, {}).setdefault(sp, []).append((t0 - max_hist_steps, t0 + fut_steps))

    for seq, sp_map in by_seq.items():
        all_iv: List[Tuple[int, int, str]] = []
        for sp, iv_list in sp_map.items():
            for lo, hi in iv_list:
                all_iv.append((lo, hi, sp))
        all_iv.sort(key=lambda x: (x[0], x[1]))
        active_by_sp: Dict[str, int] = {}
        for lo, hi, sp in all_iv:
            for other_sp, other_max_hi in active_by_sp.items():
                if other_sp != sp and other_max_hi >= lo:
                    return False, f"Temporal overlap in {seq} between {other_sp}(<= {other_max_hi}) and {sp}[{lo},{hi}]"
            active_by_sp[sp] = max(active_by_sp.get(sp, -10**9), hi)
    return True, "PASS: zero physical time overlap across train/val/test"
