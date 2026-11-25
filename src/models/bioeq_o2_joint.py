"""Exact O(2) parity-equivariant multi-duration & joint action-trajectory forecaster."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.models.bioeq_neural import biomech_sequence_torch, local_frenet_torch


class O2BioEqJointForecaster(nn.Module):
    """Exact O(2)-equivariant forecaster supporting T in {5,10,20}, variable clocks, and 5 actions."""

    def __init__(
        self,
        pose_mode: str = "bio_pose",
        action_mode: str = "none",
        v_ref: float = 0.75,
    ) -> None:
        super().__init__()
        if pose_mode not in ("motion", "bio_pose"):
            raise ValueError(f"Unsupported pose_mode: {pose_mode}")
        if action_mode not in ("none", "auxiliary", "end_to_end", "stop_grad"):
            raise ValueError(f"Unsupported action_mode: {action_mode}")
        self.pose_mode = pose_mode
        self.action_mode = action_mode
        self.v_ref = v_ref

        self.motion = nn.GRU(5, 48, batch_first=True)
        self.ctx_proj = nn.Sequential(nn.LayerNorm(48), nn.Linear(48, 48), nn.SiLU())

        self.bio_in = nn.Linear(11, 24)
        self.bio_gru = nn.GRU(24, 24, batch_first=True)
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

        # 5 observable actions: [Start, Stop, Turn-Left, Turn-Right, Keep]
        self.act_even = nn.Linear(48 + 24, 3)
        self.act_turn = nn.Linear(48 + 24, 1)

        # Action conditioning into even (4 dims) and odd (1 dim, bias-free) contexts
        self.act_to_even = nn.Linear(4, 48)
        self.act_to_odd = nn.Linear(1, 48, bias=False)

        # Modes 1, 2, 5 (3 central modes): even -> dx (3*3), odd (bias-free) -> dy (3*3)
        self.central_dx = nn.Linear(48, 3 * 3)
        self.central_dy = nn.Linear(48, 3 * 3, bias=False)
        self.central_scores = nn.Linear(48, 4)  # Mode 0 (CV) + Modes 1, 2, 5

        # Modes 3, 4 (mirror-twin pair): shared head applied to ctx_+ and ctx_-
        self.pair_knots = nn.Linear(48, 3 * 2)
        self.pair_score = nn.Linear(48, 1)

        # Stage-2 longitudinal biomechanical refinement on the 5 non-CV modes (3central + pair)
        self.bio_par_central = nn.Linear(24, 3 * 3)
        self.bio_par_pair = nn.Linear(24, 3)
        self.bio_score_central = nn.Linear(24, 4)
        self.bio_score_pair = nn.Linear(24, 1)
        self._reset_heads()

    def _reset_heads(self) -> None:
        for mod in (self.central_dx, self.central_dy, self.pair_knots):
            nn.init.normal_(mod.weight, mean=0.0, std=0.01)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        for mod in (self.bio_par_central, self.bio_par_pair, self.bio_score_central, self.bio_score_pair):
            nn.init.zeros_(mod.weight)
            nn.init.zeros_(mod.bias)

    def forward(
        self,
        history: torch.Tensor,
        bio_seq: torch.Tensor | None = None,
        q_bio: torch.Tensor | None = None,
        obs_dt: torch.Tensor | None = None,
        fut_times_s: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, t_obs, _ = history.shape
        origin, basis, valid = local_frenet_torch(history)
        body = (history - origin) @ basis
        if obs_dt is None:
            dt_step = history.new_full((bsz, t_obs - 1), 0.1)
        else:
            dt_step = obs_dt.to(dtype=history.dtype, device=history.device).clamp_min(1e-4)
        vel = torch.cat((torch.zeros_like(body[:, :1]), (body[:, 1:] - body[:, :-1]) / dt_step[:, :, None]), dim=1)
        spd = torch.linalg.vector_norm(vel, dim=2, keepdim=True)

        feat_pos = torch.cat((body, vel, spd), dim=2).float()
        flip_y = feat_pos.new_tensor([1.0, -1.0, 1.0, -1.0, 1.0])
        feat_neg = feat_pos * flip_y
        both = torch.cat((feat_pos, feat_neg), dim=0)
        _, h_both = self.motion(both)
        ctx_both = self.ctx_proj(h_both[0])
        ctx_pos, ctx_neg = ctx_both[:bsz], ctx_both[bsz:]
        ctx_even = 0.5 * (ctx_pos + ctx_neg)
        ctx_odd = 0.5 * (ctx_pos - ctx_neg)

        last_v = (torch.linalg.vector_norm(history[:, -1] - history[:, -2], dim=-1) / dt_step[:, -1]).float()
        if self.pose_mode == "bio_pose" and bio_seq is not None and q_bio is not None:
            regime = q_bio * torch.exp(-((last_v / self.v_ref) ** 2)) * torch.sigmoid(self.gate_logit)
            _, hb = self.bio_gru(F.silu(self.bio_in(bio_seq)))
            h_bio = hb[0]
            h_bio_gated = regime[:, None] * h_bio
        else:
            regime = last_v.new_zeros(bsz)
            h_bio = last_v.new_zeros((bsz, 24))
            h_bio_gated = h_bio

        ssk = self.act_even(torch.cat((ctx_even, h_bio_gated), dim=1))
        tl = self.act_turn(torch.cat((ctx_pos, h_bio_gated), dim=1))
        tr = self.act_turn(torch.cat((ctx_neg, h_bio_gated), dim=1))
        act_logits = torch.cat((ssk[:, :2], tl, tr, ssk[:, 2:]), dim=1)
        act_logits = torch.where(valid[:, None], act_logits, torch.zeros_like(act_logits))

        if self.action_mode in ("end_to_end", "stop_grad"):
            src_lg = act_logits.detach() if self.action_mode == "stop_grad" else act_logits
            p_act = torch.sigmoid(src_lg)
            a_ev = torch.stack((p_act[:, 0], p_act[:, 1], p_act[:, 4], 0.5 * (p_act[:, 2] + p_act[:, 3])), dim=1)
            a_od = p_act[:, 2:3] - p_act[:, 3:4]
            ctx_even = ctx_even + self.act_to_even(a_ev)
            ctx_odd = ctx_odd + self.act_to_odd(a_od)
            ctx_pos, ctx_neg = ctx_even + ctx_odd, ctx_even - ctx_odd

        c_dx = self.central_dx(ctx_even).view(bsz, 3, 3)
        c_dy = self.central_dy(ctx_odd).view(bsz, 3, 3)
        p_pos = self.pair_knots(ctx_pos).view(bsz, 3, 2)
        p_neg = self.pair_knots(ctx_neg).view(bsz, 3, 2)
        if self.pose_mode == "bio_pose" and bio_seq is not None and q_bio is not None:
            c_dx = c_dx + regime[:, None, None] * self.bio_par_central(h_bio).view(bsz, 3, 3)
            dp = regime[:, None] * self.bio_par_pair(h_bio)
            p_pos = torch.stack((p_pos[..., 0] + dp, p_pos[..., 1]), dim=-1)
            p_neg = torch.stack((p_neg[..., 0] + dp, p_neg[..., 1]), dim=-1)

        c_knots = torch.stack((c_dx, c_dy), dim=-1)
        m3 = p_pos[:, None]
        m4 = torch.stack((p_neg[..., 0], -p_neg[..., 1]), dim=-1)[:, None]
        knots5 = torch.cat((c_knots[:, :2], m3, m4, c_knots[:, 2:]), dim=1).double()
        knots5 = torch.cat((torch.zeros_like(knots5[:, :, :1]), knots5), dim=2)

        c_sc = self.central_scores(ctx_even)
        s3, s4 = self.pair_score(ctx_pos), self.pair_score(ctx_neg)
        if self.pose_mode == "bio_pose" and bio_seq is not None and q_bio is not None:
            c_sc = c_sc + regime[:, None] * self.bio_score_central(h_bio)
            ds = regime[:, None] * self.bio_score_pair(h_bio)
            s3, s4 = s3 + ds, s4 + ds
        logits = torch.cat((c_sc[:, :3], s3, s4, c_sc[:, 3:]), dim=1)
        logits = torch.where(valid[:, None], logits, torch.zeros_like(logits))

        if fut_times_s is None:
            t_sec = torch.arange(1, 31, dtype=history.dtype, device=history.device)[None, :] * 0.1
        else:
            t_sec = fut_times_s.to(dtype=history.dtype, device=history.device)
        idx = (t_sec - 1e-9).floor().long().clamp(0, 2)
        phase = (t_sec - idx.to(history.dtype))[:, None, :, None]
        b_idx = torch.arange(bsz, device=history.device)[:, None, None]
        m_idx = torch.arange(5, device=history.device)[None, :, None]
        k_left = knots5[b_idx, m_idx, idx[:, None, :]]
        k_right = knots5[b_idx, m_idx, (idx + 1)[:, None, :]]
        delta = k_left * (1.0 - phase) + k_right * phase
        world = torch.einsum("bkts,bws->bktw", delta, basis)
        v_last = (history[:, -1:] - history[:, -2:-1]) / dt_step[:, -1:, None]
        cv = origin + v_last * t_sec[:, :, None]
        paths = torch.cat((cv[:, None], cv[:, None] + world), dim=1)
        return {"paths": paths, "logits": logits, "act_logits": act_logits, "valid": valid, "regime": regime}


def o2_joint_loss(
    out: dict[str, torch.Tensor],
    target_xy: torch.Tensor,
    sample_w: torch.Tensor,
    y_act: torch.Tensor | None = None,
    valid_act: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
    act_weight: float = 0.25,
    reg_only: bool = False,
) -> torch.Tensor | None:
    valid = out["valid"]
    if not torch.any(valid):
        return None
    w_v = sample_w[valid]
    w_v = (w_v / w_v.mean().clamp_min(1e-8)).double()
    dist = torch.linalg.vector_norm(out["paths"][valid] - target_xy[valid, None], dim=-1)
    ade = dist.detach().mean(dim=2)
    winner = ade[:, 1:] == ade[:, 1:].amin(dim=1, keepdim=True)
    m_w = winner.double() / winner.sum(dim=1, keepdim=True)
    reg_i = ((torch.sqrt(dist[:, 1:].square() + 0.25) - 0.5).mean(dim=2) * m_w).sum(dim=1)
    if reg_only:
        return (reg_i * w_v).mean()
    soft = (-ade / 0.15).softmax(dim=1).to(out["logits"].dtype)
    cls_i = -(soft * out["logits"][valid].log_softmax(dim=1)).sum(dim=1).double()
    total = ((reg_i + 0.25 * cls_i) * w_v).mean()
    if y_act is not None and valid_act is not None and act_weight > 0.0:
        m_a = valid[:, None] & valid_act
        if torch.any(m_a):
            bce = F.binary_cross_entropy_with_logits(
                out["act_logits"], y_act.float(), pos_weight=pos_weight, reduction="none"
            ).double()
            w_broad = (sample_w / sample_w.mean().clamp_min(1e-8)).double()[:, None]
            act_l = (bce * w_broad * m_a.double()).sum() / m_a.double().sum().clamp_min(1.0)
            total = total + act_weight * act_l
    return total
