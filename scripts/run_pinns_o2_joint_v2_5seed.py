"""Two-stage decoupled O(2) BioEq-Act v2 5-seed 4-site LOSO benchmark (1,482 windows)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_matched_o2_multiduration import BIO_TRAINABLE, SITES, paired_site_clip_ci
from scripts.run_pinns_o2_joint_actions_5seed import (
    SEEDS,
    action_site_macro,
    apply_causal_eligibility,
    dampened_pos_weight,
    fit_joint_arm,
    load_joint_cohort,
)
from src.models.bioeq_o2_joint import O2BioEqJointForecaster, o2_joint_loss

STAGE2_JOINT_TRAINABLE = set(BIO_TRAINABLE) | {"act_even.weight", "act_even.bias"}


@torch.inference_mode()
def eval_joint_scaled(net: O2BioEqJointForecaster, data: dict, ids: np.ndarray, bio_scale: float, act_scale: float) -> tuple[np.ndarray, np.ndarray]:
    hist = data["history"][ids]
    bs = data["bio_seq"][ids]
    qb_traj = data["q_bio"][ids] * float(bio_scale)
    qb_act = data["q_bio"][ids] * float(act_scale)
    out_t = net(hist, bs, qb_traj)
    out_a = net(hist, bs, qb_act) if abs(act_scale - bio_scale) > 1e-9 else out_t
    paths = out_t["paths"].cpu().numpy()
    probs = out_t["logits"].softmax(dim=1).cpu().numpy()
    act_p = torch.sigmoid(out_a["act_logits"]).cpu().numpy()
    truth = data["truth"][ids].numpy()
    dst = np.linalg.norm(paths - truth[:, None], axis=3)
    ade_k, fde_k = dst.mean(axis=2), dst[:, :, -1]
    top = probs.argmax(axis=1)
    ii = np.arange(len(ids))
    bayes = np.einsum("nk,nktw->ntw", probs, paths)
    bd = np.linalg.norm(bayes - truth, axis=2)
    traj = np.column_stack((ade_k[ii, top], fde_k[ii, top], ade_k.min(1), fde_k.min(1), bd.mean(1), bd[:, -1]))
    return traj, act_p


def fit_stage2_joint(mot_net: O2BioEqJointForecaster, seed: int, data: dict, tr_ids: np.ndarray) -> tuple[O2BioEqJointForecaster, float, float]:
    def train_on(sub_ids: np.ndarray) -> O2BioEqJointForecaster:
        torch.manual_seed(seed + 211)
        bio = O2BioEqJointForecaster(pose_mode="bio_pose", action_mode="stop_grad")
        st = bio.state_dict()
        for k, v in mot_net.state_dict().items():
            if not k.startswith("bio_"):
                st[k].copy_(v)
        # Keep motion's pretrained [48] columns in act_even fixed via copy; train bio [24] columns + bio traj knots
        bio.load_state_dict(st)
        for name, param in bio.named_parameters():
            param.requires_grad_(name in STAGE2_JOINT_TRAINABLE)
        h_s, t_s = data["history"][sub_ids], data["truth"][sub_ids]
        bs_s, qb_s = data["bio_seq"][sub_ids], data["q_bio"][sub_ids]
        ya_s, va_s = data["y_act"][sub_ids], data["valid_act"][sub_ids]
        pw_s = dampened_pos_weight(ya_s, va_s)
        sw_s = make_site_weights(data["site"], sub_ids)
        mot_act_w = mot_net.act_even.weight[:, :48].detach().clone()
        mot_act_b = mot_net.act_even.bias.detach().clone()
        opt = torch.optim.AdamW([p for p in bio.parameters() if p.requires_grad], lr=0.0015, weight_decay=3e-4)
        gen = torch.Generator().manual_seed(seed + 77)
        bio.train()
        for _ in range(18):
            for b in torch.randperm(len(sub_ids), generator=gen).split(64):
                opt.zero_grad(set_to_none=True)
                out = bio(h_s[b], bs_s[b], qb_s[b])
                reg_l = o2_joint_loss(out, t_s[b], sw_s[b], reg_only=True)
                m_a = out["valid"][:, None] & va_s[b]
                bce = F.binary_cross_entropy_with_logits(out["act_logits"], ya_s[b].float(), pos_weight=pw_s, reduction="none").double()
                act_l = (bce * m_a.double()).sum() / m_a.double().sum().clamp_min(1.0)
                loss = reg_l + 0.15 * act_l
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bio.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    bio.act_even.weight[:, :48].copy_(mot_act_w)
                    bio.act_even.bias.copy_(mot_act_b)
        return bio.eval()

    tr_sites = sorted(set(data["site"][tr_ids].tolist()))
    scales = (0.0, 0.25, 0.5, 1.0)
    oof_t = {sc: [] for sc in scales}
    oof_a = {sc: [] for sc in scales}
    for vs in tr_sites:
        in_tr = tr_ids[data["site"][tr_ids] != vs]
        in_va = tr_ids[data["site"][tr_ids] == vs]
        b_in = train_on(in_tr)
        ya_v = data["y_act"][in_va].numpy().astype(np.float64)
        for sc in scales:
            t_m, a_p = eval_joint_scaled(b_in, data, in_va, sc, sc)
            oof_t[sc].append(float(t_m[:, 0].mean()))
            oof_a[sc].append(float(((a_p - ya_v) ** 2).mean()))
    best_t_sc = min(scales, key=lambda sc: np.mean(oof_t[sc]) + (1e-7 if sc > 0 else 0.0))
    best_a_sc = min(scales, key=lambda sc: np.mean(oof_a[sc]) + (1e-7 if sc > 0 else 0.0))
    return train_on(tr_ids), float(best_t_sc), float(best_a_sc)


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

    mot_t_seeds, bio_t_seeds = [], []
    mot_a_seeds, bio_a_seeds = [], []
    scales_rec, ckpts = {}, {}

    for seed in args.seeds:
        mt = np.zeros((n, 6), dtype=np.float64)
        bt = np.zeros((n, 6), dtype=np.float64)
        ma = np.zeros((n, 5), dtype=np.float64)
        ba = np.zeros((n, 5), dtype=np.float64)
        scales_rec[str(seed)] = {}
        for s in SITES:
            tr_ids = np.flatnonzero(data["site"] != s)
            te_ids = np.flatnonzero(data["site"] == s)
            mot = fit_joint_arm(seed, "motion", "stop_grad", data, tr_ids)
            bio, sc_t, sc_a = fit_stage2_joint(mot, seed, data, tr_ids)
            mt[te_ids], ma[te_ids] = eval_joint_scaled(mot, data, te_ids, 0.0, 0.0)
            bt[te_ids], ba[te_ids] = eval_joint_scaled(bio, data, te_ids, sc_t, sc_a)
            scales_rec[str(seed)][s] = {"traj_scale": sc_t, "act_scale": sc_a}
            if seed == 42:
                for tag, model in (("motion_stop_grad", mot), ("bio_v2_stop_grad", bio)):
                    buf = io.BytesIO()
                    torch.save({"state_dict": model.state_dict(), "traj_scale": sc_t, "act_scale": sc_a, "site": s}, buf)
                    ckpts[f"{s}/{tag}.pt"] = buf.getvalue()
        mot_t_seeds.append(mt)
        bio_t_seeds.append(bt)
        mot_a_seeds.append(ma)
        bio_a_seeds.append(ba)
        print(f"Done seed={seed}", flush=True)

    names = ("ade_m", "fde_m", "minade6_m", "minfde6_m", "bayes_ade_m", "bayes_fde_m")
    mt_avg, bt_avg = np.mean(mot_t_seeds, axis=0), np.mean(bio_t_seeds, axis=0)
    ma_avg, ba_avg = np.mean(mot_a_seeds, axis=0), np.mean(bio_a_seeds, axis=0)
    summary = {
        "windows": n,
        "seeds": list(args.seeds),
        "selected_scales": scales_rec,
        "motion_stop_grad": {
            "traj_site_equal": dict(zip(names, np.mean([mt_avg[data["site"] == s].mean(0) for s in SITES], axis=0).tolist())),
            "action_raw": action_site_macro(y_np, ma_avg, va_np, data["site"]),
            "action_eligible": action_site_macro(y_np, apply_causal_eligibility(ma_avg, data["v_hist_mean"]), va_np, data["site"]),
        },
        "bio_v2_stop_grad": {
            "traj_site_equal": dict(zip(names, np.mean([bt_avg[data["site"] == s].mean(0) for s in SITES], axis=0).tolist())),
            "action_raw": action_site_macro(y_np, ba_avg, va_np, data["site"]),
            "action_eligible": action_site_macro(y_np, apply_causal_eligibility(ba_avg, data["v_hist_mean"]), va_np, data["site"]),
        },
        "paired_bio_v2_minus_motion": paired_site_clip_ci(bt_avg[:, :2] - mt_avg[:, :2], data["site"], data["clip"]),
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
