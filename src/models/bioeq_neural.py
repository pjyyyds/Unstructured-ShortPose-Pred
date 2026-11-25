"""PyTorch O(2)-equivariant 6-mode neural forecaster with regime-gated biomechanical pose."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

AXIS_THRESHOLD_M = 0.05
KP_CONF_MIN = 0.20


def local_frenet_torch(history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    origin = history[:, -1:]
    back = (origin - history[:, :-1]).flip(1)
    lengths = torch.linalg.vector_norm(back, dim=2)
    eligible = lengths >= (AXIS_THRESHOLD_M - 1e-12)
    valid = eligible.any(1)
    idx = eligible.long().argmax(1)
    rows = torch.arange(len(history), device=history.device)
    unit = back[rows, idx] / lengths[rows, idx, None].clamp_min(1e-12)
    left = torch.stack((-unit[:, 1], unit[:, 0]), dim=1)
    basis = torch.where(valid[:, None, None], torch.stack((unit, left), dim=2), 0.0)
    return origin, basis, valid


def biomech_sequence_torch(kpts: torch.Tensor, bbox: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (B, 5, 11) camera-azimuth & O(2)-reflection invariant 5-frame gait sequence + q_bio."""
    h = bbox[..., 3].double()
    f_ok = torch.isfinite(h) & (h > 1.0)
    kp = kpts.double()
    good = torch.isfinite(kp).all(dim=-1) & (kp[..., 2] >= KP_CONF_MIN) & f_ok[..., None]
    xy = torch.where(good[..., None], kp[..., :2], 0.0) / torch.where(f_ok, h, 1.0)[..., None, None]

    def knee_cos(hip: int, knee: int, ank: int) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = xy[:, :, hip] - xy[:, :, knee], xy[:, :, ank] - xy[:, :, knee]
        nu, nv = torch.linalg.vector_norm(u, dim=-1), torch.linalg.vector_norm(v, dim=-1)
        ok = good[:, :, hip] & good[:, :, knee] & good[:, :, ank] & (nu > 1e-6) & (nv > 1e-6)
        cos = ((u * v).sum(dim=-1) / torch.where(ok, nu * nv, 1.0)).clamp(-1.0, 1.0)
        return torch.where(ok, cos, 0.0), ok

    ank_ok = good[:, :, 15] & good[:, :, 16]
    ank_span = torch.where(ank_ok, torch.linalg.vector_norm(xy[:, :, 15] - xy[:, :, 16], dim=-1), 0.0)
    cl, okl = knee_cos(11, 13, 15)
    cr, okr = knee_cos(12, 14, 16)
    kok = okl & okr
    k_mean = torch.where(kok, 0.5 * (cl + cr), 0.0)
    k_diff = torch.where(kok, (cl - cr).abs(), 0.0)
    lok = good[:, :, 11] & good[:, :, 15] & good[:, :, 12] & good[:, :, 16]
    leg_ext = torch.where(
        lok,
        0.5 * (torch.linalg.vector_norm(xy[:, :, 11] - xy[:, :, 15], dim=-1) + torch.linalg.vector_norm(xy[:, :, 12] - xy[:, :, 16], dim=-1)),
        0.0,
    )
    tok = good[:, :, [5, 6, 11, 12]].all(dim=-1)
    tr_up = torch.where(tok, -0.5 * (xy[:, :, 5, 1] + xy[:, :, 6, 1] - xy[:, :, 11, 1] - xy[:, :, 12, 1]), 0.0)

    vals = torch.stack((ank_span, k_mean, k_diff, leg_ext, tr_up), dim=-1)
    msks = torch.stack((ank_ok, kok, kok, lok, tok), dim=-1)
    adj_ok = msks[:, 1:] & msks[:, :-1]
    step_d = torch.where(adj_ok, (vals[:, 1:] - vals[:, :-1]) / 0.1, 0.0)
    delta = torch.cat((torch.zeros_like(step_d[:, :1]), step_d), dim=1)
    frame_valid = msks[:, :, :4].float().mean(dim=-1, keepdim=True)
    lower_conf = torch.where(good[:, :, 11:], kp[:, :, 11:, 2], 0.0).mean(dim=(1, 2))
    q_bio = (msks[:, :, :4].double().mean(dim=(1, 2)) * lower_conf).clamp(0.0, 1.0)
    seq = torch.cat((vals.float(), delta.float(), frame_valid), dim=-1)
    return seq, q_bio.float()


class BioEqGroundForecaster(nn.Module):
    """SO(2)/O(2)-aware 6-mode forecaster comparing motion, raw pose, and regime-gated bio-pose."""

    def __init__(self, mode: str = "bio_pose", v_ref: float = 0.75) -> None:
        super().__init__()
        if mode not in ("motion", "raw_pose", "bio_pose"):
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.v_ref = v_ref
        self.motion = nn.GRU(5, 48, batch_first=True)
        self.context = nn.Sequential(nn.LayerNorm(48), nn.Linear(48, 48), nn.SiLU())
        self.residual = nn.Linear(48, 5 * 3 * 2)
        self.scores = nn.Linear(48, 6)
        nn.init.normal_(self.residual.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.residual.bias)

        self.raw_in = nn.Linear(17 * 6, 24)
        self.raw_gru = nn.GRU(24, 24, batch_first=True)
        self.raw_out = nn.Linear(24 + 4, 48)

        self.bio_in = nn.Linear(11, 24)
        self.bio_gru = nn.GRU(24, 24, batch_first=True)
        self.bio_par = nn.Linear(24, 5 * 3)
        self.bio_score = nn.Linear(24, 6)
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.bio_par.weight)
        nn.init.zeros_(self.bio_par.bias)
        nn.init.zeros_(self.bio_score.weight)
        nn.init.zeros_(self.bio_score.bias)

    def forward(
        self,
        history: torch.Tensor,
        x_p: torch.Tensor | None = None,
        q_vec: torch.Tensor | None = None,
        bio_seq: torch.Tensor | None = None,
        q_bio: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        origin, basis, valid = local_frenet_torch(history)
        body = (history - origin) @ basis
        step = torch.diff(body, dim=1, prepend=body[:, :1]) / 0.1
        feat = torch.cat((body, step, torch.linalg.vector_norm(step, dim=2, keepdim=True)), dim=2)
        _, hm = self.motion(feat.float())
        hidden = hm[0]

        if self.mode == "raw_pose":
            present = (x_p[:, :, :, 3] > 0.5).any(2).any(1)
            _, hp = self.raw_gru(F.silu(self.raw_in(x_p.flatten(2))))
            hidden = hidden + torch.where(present[:, None], self.raw_out(torch.cat((hp[0], q_vec), dim=1)), 0.0)

        ctx = self.context(hidden)
        logits = self.scores(ctx)
        knots = self.residual(ctx).view(len(history), 5, 3, 2)

        if self.mode == "bio_pose":
            last_v = torch.linalg.vector_norm(history[:, -1] - history[:, -2], dim=-1).float() / 0.1
            regime = q_bio * torch.exp(-((last_v / self.v_ref) ** 2)) * torch.sigmoid(self.gate_logit)
            _, hb = self.bio_gru(F.silu(self.bio_in(bio_seq)))
            h_bio = hb[0]
            d_par = regime[:, None, None] * self.bio_par(h_bio).view(len(history), 5, 3)
            knots = torch.stack((knots[..., 0] + d_par, knots[..., 1]), dim=-1)
            logits = logits + regime[:, None] * self.bio_score(h_bio)

        knots64 = knots.double()
        knots64 = torch.cat((torch.zeros_like(knots64[:, :, :1]), knots64), dim=2)
        times = torch.arange(1, 31, dtype=history.dtype, device=history.device)
        left = (torch.arange(30, device=history.device) // 10).long()
        phase = (times / 10.0 - left)[None, None, :, None]
        delta = knots64[:, :, left] * (1.0 - phase) + knots64[:, :, left + 1] * phase
        world = torch.einsum("bkts,bws->bktw", delta, basis)
        cv = origin + (history[:, -1:] - history[:, -2:-1]) * times[None, :, None]
        paths = torch.cat((cv[:, None], cv[:, None] + world), dim=1)
        logits = torch.where(valid[:, None], logits, torch.zeros_like(logits))
        return paths, logits, valid


def bioeq_loss(
    paths: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor | None:
    if not torch.any(valid):
        return None
    dist = torch.linalg.vector_norm(paths[valid] - target[valid, None], dim=-1)
    ade = dist.detach().mean(dim=2)
    winner = ade[:, 1:] == ade[:, 1:].amin(dim=1, keepdim=True)
    weight = winner.double() / winner.sum(dim=1, keepdim=True)
    reg = ((torch.sqrt(dist[:, 1:].square() + 0.25) - 0.5).mean(dim=2) * weight).sum(dim=1).mean()
    soft = (-ade / 0.15).softmax(dim=1).to(logits.dtype)
    cls = -(soft * logits[valid].log_softmax(dim=1)).sum(dim=1).mean()
    return reg + 0.25 * cls
