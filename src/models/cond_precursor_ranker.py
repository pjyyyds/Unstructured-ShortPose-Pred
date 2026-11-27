"""Eligibility-conditioned pose x traffic-clearance maneuver ranker & 5-seed selection auditor."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_conditioned_precursor_features(
    history: torch.Tensor,
    bio_even: torch.Tensor,
    pose_vec_world: torch.Tensor,
    signed_knee: torch.Tensor,
    quality: torch.Tensor,
    m_pose: torch.Tensor,
    edges: torch.Tensor,
    is_car: torch.Tensor,
    nbr_mask: torch.Tensor,
    obs_dt: torch.Tensor,
    use_pose: bool = True,
    use_interaction: bool = True,
) -> torch.Tensor:
    """Extract O(2)-invariant 20-D precursor features for Start/Stop/Turn eligibility ranking."""
    dt = obs_dt.clamp_min(1e-4)
    vel = (history[:, 1:] - history[:, :-1]) / dt[:, :, None]
    spd = torch.linalg.vector_norm(vel, dim=-1)  # [N, 4]
    v_last = spd[:, -1]
    v_first = spd[:, 0]
    accel = (v_last - v_first) / dt.sum(dim=-1).clamp_min(0.1)
    if vel.size(1) >= 2:
        cross_z = vel[:, -2, 0] * vel[:, -1, 1] - vel[:, -2, 1] * vel[:, -1, 0]
        dot_xy = (vel[:, -2] * vel[:, -1]).sum(dim=-1)
        yaw_rate = torch.atan2(cross_z, dot_xy.clamp_min(1e-4))
    else:
        yaw_rate = torch.zeros_like(v_last)
    kin_f = torch.stack((
        v_last, v_first, spd.mean(dim=-1), accel,
        yaw_rate.abs(), yaw_rate,
    ), dim=-1).float()  # [N, 6]

    n = len(history)
    dev = history.device
    if use_pose:
        q1 = (quality[:, 0] if quality.ndim > 1 else quality).clamp(0.0, 1.0).float() * m_pose.float()
        be = bio_even.float()
        # Temporal change in biomechanics across 5 frames (e.g. ankle spread, hip/shoulder ratio, knee bend)
        dbe = (be[:, -1, :4] - be[:, 0, :4]) * q1[:, None]
        be_last = be[:, -1, :3] * q1[:, None]
        sk_chg = ((signed_knee[:, -1] - signed_knee[:, 0]).float().squeeze(-1) if signed_knee.ndim == 3 else (signed_knee[:, -1] - signed_knee[:, 0]).float()) * q1
        pw_norm = torch.linalg.vector_norm(pose_vec_world[:, -1], dim=-1).mean(dim=-1).float() * q1
        pose_f = torch.cat((dbe, be_last, sk_chg[:, None], pw_norm[:, None], q1[:, None]), dim=-1)  # [N, 10]
    else:
        pose_f = torch.zeros((n, 10), dtype=torch.float32, device=dev)

    if use_interaction and edges.size(1) > 0:
        rel_p = edges[:, :, 0:2].float()
        rel_v = edges[:, :, 2:4].float()
        dist = torch.linalg.vector_norm(rel_p, dim=-1).clamp_min(0.25)
        r_dot = (rel_p * rel_v).sum(dim=-1) / dist  # < 0 approaching, > 0 receding (gap opening!)
        valid_e = nbr_mask.float() > 0.5
        car_e = valid_e & (is_car.float() > 0.5)
        ttc_app = torch.where(valid_e & (r_dot < -0.15), dist / (-r_dot).clamp_min(0.15), torch.full_like(dist, 20.0))
        min_ttc = ttc_app.min(dim=-1).values.clamp(0.0, 15.0)
        min_car_d = torch.where(car_e, dist, torch.full_like(dist, 25.0)).min(dim=-1).values.clamp(0.0, 25.0)
        gap_clear = torch.where(car_e & (dist < 8.0) & (r_dot > 0.20), r_dot, torch.zeros_like(r_dot)).max(dim=-1).values
        nbr_cnt = valid_e.float().sum(dim=-1).clamp(0.0, 10.0) / 5.0
        int_f = torch.stack((torch.exp(-min_ttc / 3.0), torch.exp(-min_car_d / 5.0), gap_clear.clamp(0.0, 5.0), nbr_cnt), dim=-1)
    else:
        int_f = torch.zeros((n, 4), dtype=torch.float32, device=dev)
    return torch.cat((kin_f, pose_f, int_f), dim=-1)  # [N, 20]


class ParityExactCondPrecursorNet(nn.Module):
    """O(2)-parity-exact eligibility-conditioned precursor head for Start, Stop, Turn-L/R, Maintain."""

    def __init__(self) -> None:
        super().__init__()
        # Even features: indices [0,1,2,3,4, 6..12, 14,15, 16..19] (18-D O(2)-invariant)
        self.even_idx = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
        # Odd pseudoscalar features: [5 (yaw_rate), 13 (signed_knee_change)] (2-D O(2)-odd)
        self.odd_idx = [5, 13]
        self.even_trunk = nn.Sequential(nn.LayerNorm(18), nn.Linear(18, 32), nn.SiLU(), nn.Dropout(0.10))
        self.head_start = nn.Linear(32, 1)
        self.head_stop = nn.Linear(32, 1)
        self.head_maint = nn.Linear(32, 1)
        self.odd_scale = nn.Sequential(nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 2))

    def forward(self, feat20: torch.Tensor, base_act_logits: torch.Tensor) -> torch.Tensor:
        ev = self.even_trunk(feat20[:, self.even_idx])
        odd_f = feat20[:, self.odd_idx]
        # Odd turn score flips exact sign when odd_f -> -odd_f under reflection!
        turn_delta = (self.odd_scale(ev) * odd_f).sum(dim=-1, keepdim=True)
        d_logits = torch.cat((
            self.head_start(ev),
            self.head_stop(ev),
            turn_delta,
            -turn_delta,
            self.head_maint(ev),
        ), dim=-1)
        return base_act_logits + d_logits


def select_medoid_expected_ade(
    paths: np.ndarray,
    mode_logits: np.ndarray,
    cal_act_probs: np.ndarray,
    v_act: np.ndarray,
    beta: float = 0.18,
    sw_thr: np.ndarray | None = None,
) -> np.ndarray:
    """Select Physical Medoid k* = argmin_k sum_m pi_m ADE(Y_k, Y_m) among the 6 generated slots."""
    log_pi = mode_logits - mode_logits.max(1, keepdims=True)
    log_pi = log_pi - np.log(np.exp(log_pi).sum(1, keepdims=True))
    slot_p = np.column_stack((
        cal_act_probs[:, 4], cal_act_probs[:, 0], cal_act_probs[:, 1],
        cal_act_probs[:, 2], cal_act_probs[:, 3], cal_act_probs[:, 4],
    ))
    comb = log_pi + float(beta) * np.log(np.clip(slot_p, 1e-4, 1.0 - 1e-4))
    comb = comb - comb.max(1, keepdims=True)
    pi = np.exp(comb)
    pi = pi / np.maximum(pi.sum(1, keepdims=True), 1e-8)  # [N, 6]
    # Pairwise ADE matrix [N, 6, 6] across the 6 physical trajectories
    pw_ade = np.linalg.norm(paths[:, :, None, :, :] - paths[:, None, :, :, :], axis=-1).mean(axis=-1)
    exp_risk = (pw_ade * pi[:, None, :]).sum(axis=-1)  # [N, 6]
    idx = exp_risk.argmin(axis=1)
    if sw_thr is not None:
        for c_act, slot_k in ((0, 1), (1, 2), (2, 3), (3, 4)):
            if sw_thr[c_act] <= 1.0:
                fire = v_act[:, c_act].astype(bool) & (cal_act_probs[:, c_act] >= sw_thr[c_act])
                idx[fire] = slot_k
    return idx


def summarize_6candidate_5seed(per_seed_audits: list[dict[str, dict]]) -> dict[str, dict]:
    """Aggregate 6-candidate generation vs Top-1 selection across ALL 5 seeds (never Seed-0 only)."""
    classes = tuple(per_seed_audits[0].keys())
    out: dict[str, dict] = {}
    for cls in classes:
        n_true = int(per_seed_audits[0][cls]["n_true"])
        n_neg = int(per_seed_audits[0][cls]["n_neg_eligible"])
        gen = [int(d[cls]["any6_generated_hits"]) for d in per_seed_audits]
        sel = [int(d[cls]["top1_selected_hits"]) for d in per_seed_audits]
        orac = [int(d[cls]["oracle6_beats_cv_by_2cm"]) for d in per_seed_audits]
        fp1 = [int(d[cls]["top1_neg_false_positives"]) for d in per_seed_audits]
        out[cls] = {
            "n_true": n_true,
            "n_neg_eligible": n_neg,
            "any6_generated_hits_5seed_mean": float(np.mean(gen)),
            "any6_generated_hits_per_seed": gen,
            "top1_selected_hits_5seed_mean": float(np.mean(sel)),
            "top1_selected_hits_per_seed": sel,
            "oracle6_beats_cv_5seed_mean": float(np.mean(orac)),
            "top1_neg_fp_5seed_mean": float(np.mean(fp1)),
            "top1_neg_fp_per_seed": fp1,
        }
    return out
