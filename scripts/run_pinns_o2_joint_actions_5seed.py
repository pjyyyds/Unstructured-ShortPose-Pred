"""Five-seed 4-site LOSO O(2) parity-equivariant joint action & trajectory benchmark (1,482 windows)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_matched_o2_multiduration import SITES, paired_site_clip_ci
from src.metrics.metrics_and_stats import _binary_average_precision
from src.models.bioeq_neural import biomech_sequence_torch
from src.models.bioeq_o2_joint import O2BioEqJointForecaster, o2_joint_loss

SEEDS = (42, 1024, 2025, 3407, 4096)
ARMS = ("none", "auxiliary", "end_to_end", "stop_grad")
ACTIONS = ("start", "stop", "left", "right", "maintain")


def load_joint_cohort(joint_zip: Path, pose_npz: Path) -> dict:
    with np.load(pose_npz, allow_pickle=False) as p_ar:
        p = {k: p_ar[k] for k in p_ar.files}
    p_map = {(str(c), int(pid), int(t)): i for i, (c, pid, t) in enumerate(zip(p["clip"], p["person_id"], p["t0"]))}
    rows_clip, rows_site, p_idx_list, y_act_list, v_act_list, fut_list = [], [], [], [], [], []
    with zipfile.ZipFile(joint_zip) as zf:
        for s in SITES:
            with np.load(io.BytesIO(zf.read(f"{s}/targets.npz")), allow_pickle=False) as t_ar:
                for c, pid, t0, ya, va, fx in zip(
                    t_ar["clip"], t_ar["person_id"], t_ar["t0"], t_ar["y_act"], t_ar["valid_by_class"], t_ar["fut_xy"]
                ):
                    key = (str(c), int(pid), int(t0))
                    rows_clip.append(str(c))
                    rows_site.append(s)
                    p_idx_list.append(p_map[key])
                    y_act_list.append(ya.astype(bool))
                    v_act_list.append(va.astype(bool))
                    fut_list.append(fx.astype(np.float64))
    p_idx = np.array(p_idx_list, dtype=int)
    if len(p_idx) != 1482:
        raise ValueError(f"Expected 1482 windows, found {len(p_idx)}")
    hist = p["hist_pos_relative"][p_idx].astype(np.float64)
    kpts = torch.from_numpy(p["hist_kpts"][p_idx].astype(np.float32))
    bbox = torch.from_numpy(p["hist_bbox"][p_idx].astype(np.float32))
    bio_seq, q_bio = biomech_sequence_torch(kpts, bbox)
    step_v = np.linalg.norm(np.diff(hist, axis=1), axis=-1) / 0.1
    # 3-step smoothed historical speed matching causal eligibility audit
    v_smooth = 0.5 * (step_v[:, 1:] + step_v[:, :-1])
    v_hist_mean = v_smooth.mean(axis=1)
    return {
        "clip": np.array(rows_clip, dtype=str),
        "site": np.array(rows_site, dtype=str),
        "history": torch.from_numpy(hist),
        "truth": torch.from_numpy(np.stack(fut_list, axis=0)),
        "bio_seq": bio_seq,
        "q_bio": q_bio,
        "y_act": torch.from_numpy(np.stack(y_act_list, axis=0)),
        "valid_act": torch.from_numpy(np.stack(v_act_list, axis=0)),
        "v_hist_mean": v_hist_mean,
    }


def dampened_pos_weight(y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    pos = (y & v).float().sum(dim=0)
    neg = ((~y) & v).float().sum(dim=0)
    ratio = torch.sqrt(neg / pos.clamp_min(1.0)).clamp(1.0, 5.0)
    return torch.where(pos > 0, ratio, torch.ones_like(ratio))


def action_site_macro(y_true: np.ndarray, y_prob: np.ndarray, valid_act: np.ndarray, site: np.ndarray) -> dict:
    by_site = {}
    ap_sites, f1_sites, brier_sites = [], [], []
    for s in SITES:
        m_s = site == s
        yt_s, yp_s, va_s = y_true[m_s], y_prob[m_s], valid_act[m_s]
        c_ap, c_f1, c_br = [], [], []
        per_class = {}
        for c, name in enumerate(ACTIONS):
            ok = va_s[:, c]
            yt = yt_s[ok, c].astype(np.float64)
            yp = yp_s[ok, c].astype(np.float64)
            br = float(np.mean((yp - yt) ** 2))
            c_br.append(br)
            pos = int(np.sum(yt > 0.5))
            if pos > 0:
                ap = _binary_average_precision(yt, yp)
                pred = yp >= 0.5
                tp = int(np.sum((yt > 0.5) & pred))
                fp = int(np.sum((yt <= 0.5) & pred))
                fn = int(np.sum((yt > 0.5) & (~pred)))
                f1 = float((2.0 * tp) / max(2 * tp + fp + fn, 1))
                c_ap.append(ap)
                c_f1.append(f1)
                per_class[name] = {"pos": pos, "ap": ap, "f1": f1, "tp": tp, "fp": fp, "brier": br}
            else:
                per_class[name] = {"pos": 0, "ap": None, "f1": None, "tp": 0, "fp": int(np.sum(yp >= 0.5)), "brier": br}
        by_site[s] = {
            "macro_ap": float(np.mean(c_ap)),
            "macro_f1": float(np.mean(c_f1)),
            "mean_brier": float(np.mean(c_br)),
            "classes": per_class,
        }
        ap_sites.append(by_site[s]["macro_ap"])
        f1_sites.append(by_site[s]["macro_f1"])
        brier_sites.append(by_site[s]["mean_brier"])
    return {
        "site_equal": {
            "macro_ap": float(np.mean(ap_sites)),
            "macro_f1": float(np.mean(f1_sites)),
            "mean_brier": float(np.mean(brier_sites)),
        },
        "by_site": by_site,
    }


def fit_joint_arm(seed: int, pose_mode: str, action_mode: str, data: dict, tr_ids: np.ndarray) -> O2BioEqJointForecaster:
    torch.manual_seed(seed)
    net = O2BioEqJointForecaster(pose_mode=pose_mode, action_mode=action_mode)
    hist, truth = data["history"][tr_ids], data["truth"][tr_ids]
    bs, qb = data["bio_seq"][tr_ids], data["q_bio"][tr_ids]
    ya, va = data["y_act"][tr_ids], data["valid_act"][tr_ids]
    pw = dampened_pos_weight(ya, va)
    sw = make_site_weights(data["site"], tr_ids)
    act_w = 0.0 if action_mode == "none" else 0.15
    opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=28, eta_min=1e-4)
    gen = torch.Generator().manual_seed(seed + 49)
    net.train()
    for _ in range(28):
        for b in torch.randperm(len(tr_ids), generator=gen).split(64):
            opt.zero_grad(set_to_none=True)
            out = net(hist[b], bs[b], qb[b])
            loss = o2_joint_loss(out, truth[b], sw[b], ya[b], va[b], pw, act_weight=act_w)
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
        sched.step()
    return net.eval()


@torch.inference_mode()
def eval_joint_arm(net: O2BioEqJointForecaster, data: dict, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = net(data["history"][ids], data["bio_seq"][ids], data["q_bio"][ids])
    paths = out["paths"].cpu().numpy()
    probs = out["logits"].softmax(dim=1).cpu().numpy()
    act_p = torch.sigmoid(out["act_logits"]).cpu().numpy()
    truth = data["truth"][ids].numpy()
    dst = np.linalg.norm(paths - truth[:, None], axis=3)
    ade_k, fde_k = dst.mean(axis=2), dst[:, :, -1]
    top = probs.argmax(axis=1)
    ii = np.arange(len(ids))
    bayes = np.einsum("nk,nktw->ntw", probs, paths)
    bd = np.linalg.norm(bayes - truth, axis=2)
    traj = np.column_stack((ade_k[ii, top], fde_k[ii, top], ade_k.min(1), fde_k.min(1), bd.mean(1), bd[:, -1]))
    return traj, act_p


def apply_causal_eligibility(act_p: np.ndarray, v_hist_mean: np.ndarray) -> np.ndarray:
    out = act_p.copy()
    out[:, 0] = np.where(v_hist_mean <= 0.20, out[:, 0], 0.0)
    out[:, 1] = np.where(v_hist_mean >= 0.60, out[:, 1], 0.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("joint_zip", type=Path)
    ap.add_argument("pose_npz", type=Path)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    torch.set_num_threads(4)
    data = load_joint_cohort(args.joint_zip, args.pose_npz)
    n = len(data["site"])
    y_np = data["y_act"].numpy()
    va_np = data["valid_act"].numpy()
    specs = [
        ("bio_none", "bio_pose", "none"),
        ("bio_auxiliary", "bio_pose", "auxiliary"),
        ("bio_end_to_end", "bio_pose", "end_to_end"),
        ("bio_stop_grad", "bio_pose", "stop_grad"),
        ("motion_stop_grad", "motion", "stop_grad"),
    ]
    traj_seeds = {tag: [] for tag, _, _ in specs}
    act_seeds = {tag: [] for tag, _, _ in specs}
    ckpts: dict[str, bytes] = {}

    for seed in args.seeds:
        for tag, p_mode, a_mode in specs:
            t_arr = np.zeros((n, 6), dtype=np.float64)
            a_arr = np.zeros((n, 5), dtype=np.float64)
            for s in SITES:
                tr_ids = np.flatnonzero(data["site"] != s)
                te_ids = np.flatnonzero(data["site"] == s)
                net = fit_joint_arm(seed, p_mode, a_mode, data, tr_ids)
                t_arr[te_ids], a_arr[te_ids] = eval_joint_arm(net, data, te_ids)
                if seed == 42:
                    buf = io.BytesIO()
                    torch.save({"state_dict": net.state_dict(), "pose_mode": p_mode, "action_mode": a_mode, "site": s}, buf)
                    ckpts[f"{s}/{tag}.pt"] = buf.getvalue()
            traj_seeds[tag].append(t_arr)
            act_seeds[tag].append(a_arr)
        print(f"Done seed={seed}", flush=True)

    names = ("ade_m", "fde_m", "minade6_m", "minfde6_m", "bayes_ade_m", "bayes_fde_m")
    summary = {"windows": n, "seeds": list(args.seeds), "arms": {}}
    traj_avg, brier_win = {}, {}
    for tag, _, a_mode in specs:
        tm = np.mean(traj_seeds[tag], axis=0)
        traj_avg[tag] = tm
        arm_out = {
            "traj_site_equal": dict(zip(names, np.mean([tm[data["site"] == s].mean(0) for s in SITES], axis=0).tolist())),
            "traj_by_site": {s: dict(zip(names, tm[data["site"] == s].mean(0).tolist())) for s in SITES},
        }
        if a_mode != "none":
            am = np.mean(act_seeds[tag], axis=0)
            am_elig = apply_causal_eligibility(am, data["v_hist_mean"])
            arm_out["action_raw"] = action_site_macro(y_np, am, va_np, data["site"])
            arm_out["action_eligible"] = action_site_macro(y_np, am_elig, va_np, data["site"])
            brier_win[tag] = ((am - y_np.astype(np.float64)) ** 2).mean(axis=1, keepdims=True).repeat(2, axis=1)
        summary["arms"][tag] = arm_out

    summary["paired_comparisons"] = {
        "stop_grad_minus_end_to_end_traj": paired_site_clip_ci(traj_avg["bio_stop_grad"][:, :2] - traj_avg["bio_end_to_end"][:, :2], data["site"], data["clip"]),
        "stop_grad_minus_end_to_end_brier": paired_site_clip_ci(brier_win["bio_stop_grad"] - brier_win["bio_end_to_end"], data["site"], data["clip"])["mean"]["ade_m"],
        "bio_minus_motion_stop_grad_traj": paired_site_clip_ci(traj_avg["bio_stop_grad"][:, :2] - traj_avg["motion_stop_grad"][:, :2], data["site"], data["clip"]),
        "stop_grad_minus_none_traj": paired_site_clip_ci(traj_avg["bio_stop_grad"][:, :2] - traj_avg["bio_none"][:, :2], data["site"], data["clip"]),
    }
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    members = {"report.json": (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode(), **ckpts}
    members["sha256.json"] = (json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in members.items()}, indent=2, sort_keys=True) + "\n").encode()
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in members.items():
            zf.writestr(k, v)
    print("Saved", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
