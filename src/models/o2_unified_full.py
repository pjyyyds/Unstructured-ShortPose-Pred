"""Unified O(2) parity-equivariant forecaster combining gated biomechanics, null-sink graph, and 5-action bridge."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.models.bioeq_neural import local_frenet_torch


class O2UnifiedFullForecaster(nn.Module):
    """Complete <22k-parameter O(2)-equivariant joint action-trajectory forecaster (`Ours_O2_Full`)."""

    def __init__(self, v_ref: float = 0.75) -> None:
        super().__init__()
        self.v_ref = v_ref
        self.motion = nn.GRU(5, 48, batch_first=True)
        self.ctx_proj = nn.Sequential(nn.LayerNorm(48), nn.Linear(48, 48), nn.SiLU())

        # Pillar 1: Regime-gated 11-D biomechanical pose branch
        self.bio_in = nn.Linear(11, 24)
        self.bio_gru = nn.GRU(24, 24, batch_first=True)
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))
        self.bio_par_central = nn.Linear(24, 3 * 3, bias=False)
        self.bio_par_pair = nn.Linear(24, 3, bias=False)
        nn.init.zeros_(self.bio_par_central.weight)
        nn.init.zeros_(self.bio_par_pair.weight)

        # Pillar 2: O(2) parity-preserving null-sink heterogeneous ped-vehicle graph
        self.edge_mlp = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 48))
        self.edge_att = nn.Linear(96, 1)
        self.null_logit = nn.Parameter(torch.tensor(1.5))
        self.graph_proj = nn.Linear(48, 48, bias=False)
        nn.init.zeros_(self.graph_proj.weight)

        # Pillar 3: Parity-decomposed 5-action head (Start, Stop, Left, Right, Maintain) + Stop-Grad bridge
        self.act_even = nn.Linear(48 + 24, 3)
        self.act_turn = nn.Linear(48, 1)
        self.bridge_even = nn.Linear(4, 48, bias=False)
        self.bridge_odd = nn.Linear(1, 48, bias=False)
        nn.init.zeros_(self.bridge_even.weight)
        nn.init.zeros_(self.bridge_odd.weight)

        # 6-mode O(2) trajectory spline knots & mode scores (Mode 0 = CV, Modes 1,2,5 central, Modes 3<->4 twin)
        self.central_dx = nn.Linear(48, 3 * 3)
        self.central_dy = nn.Linear(48, 3 * 3, bias=False)
        self.pair_knots = nn.Linear(48, 3 * 2)
        self.central_scores = nn.Linear(48, 4)
        self.pair_score = nn.Linear(48, 1)
        for m in (self.central_dx, self.pair_knots):
            nn.init.normal_(m.weight, std=0.01); nn.init.zeros_(m.bias)
        nn.init.normal_(self.central_dy.weight, std=0.01)

    def _graph_delta(self, ctx: torch.Tensor, e_feat: torch.Tensor, nbr_mask: torch.Tensor) -> torch.Tensor:
        emb = self.edge_mlp(e_feat)
        att = self.edge_att(torch.cat((ctx[:, None, :].expand_as(emb), emb), dim=-1)).squeeze(-1)
        att = att.masked_fill(nbr_mask < 0.5, -1e9)
        w = torch.softmax(torch.cat((att, self.null_logit.expand(len(ctx), 1)), dim=1), dim=1)[:, :12]
        return self.graph_proj((w[:, :, None] * emb).sum(dim=1))

    def forward(
        self,
        history: torch.Tensor,
        bio_seq: torch.Tensor | None = None,
        q_bio: torch.Tensor | None = None,
        edges: torch.Tensor | None = None,
        is_car: torch.Tensor | None = None,
        nbr_mask: torch.Tensor | None = None,
        bio_scale: float = 1.0,
        graph_scale: float = 1.0,
        use_action_bridge: bool = True,
        obs_dt: torch.Tensor | None = None,
        fut_times_s: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, t_obs, _ = history.shape
        origin, basis, valid = local_frenet_torch(history)
        body = (history - origin) @ basis
        dt_step = history.new_full((bsz, t_obs - 1), 0.1) if obs_dt is None else obs_dt.to(history.dtype).clamp_min(1e-4)
        vel = torch.cat((torch.zeros_like(body[:, :1]), (body[:, 1:] - body[:, :-1]) / dt_step[:, :, None]), dim=1)
        spd = torch.linalg.vector_norm(vel, dim=2, keepdim=True)
        fp = torch.cat((body, vel, spd), dim=2).float()
        fn = fp * fp.new_tensor([1.0, -1.0, 1.0, -1.0, 1.0])
        _, hb = self.motion(torch.cat((fp, fn), dim=0))
        cb = self.ctx_proj(hb[0])
        ctx_pos, ctx_neg = cb[:bsz], cb[bsz:]

        if graph_scale > 0.0 and edges is not None and is_car is not None and nbr_mask is not None:
            dp_l = torch.einsum("bmk,bks->bms", edges[:, :, 0:2].double(), basis).float()
            dv_l = torch.einsum("bmk,bks->bms", edges[:, :, 2:4].double(), basis).float()
            inv = torch.stack((edges[:, :, 4] / 10.0, edges[:, :, 5], edges[:, :, 6], is_car), dim=-1).float()
            ep = torch.cat((dp_l, dv_l, inv), dim=-1)
            en = torch.cat((dp_l * ep.new_tensor([1.0, -1.0]), dv_l * ep.new_tensor([1.0, -1.0]), inv), dim=-1)
            ctx_pos = ctx_pos + float(graph_scale) * self._graph_delta(ctx_pos, ep, nbr_mask)
            ctx_neg = ctx_neg + float(graph_scale) * self._graph_delta(ctx_neg, en, nbr_mask)

        ctx_even = 0.5 * (ctx_pos + ctx_neg)
        ctx_odd = 0.5 * (ctx_pos - ctx_neg)
        last_v = (torch.linalg.vector_norm(history[:, -1] - history[:, -2], dim=-1) / dt_step[:, -1]).float()
        if bio_scale > 0.0 and bio_seq is not None and q_bio is not None:
            regime = (q_bio * float(bio_scale)) * torch.exp(-((last_v / self.v_ref) ** 2)) * torch.sigmoid(self.gate_logit)
            _, hbio = self.bio_gru(F.silu(self.bio_in(bio_seq)))
            h_bio = regime[:, None] * hbio[0]
        else:
            regime, h_bio = last_v.new_zeros(bsz), last_v.new_zeros((bsz, 24))

        ev_a = self.act_even(torch.cat((ctx_even, h_bio), dim=-1))
        tl, tr = self.act_turn(ctx_pos), self.act_turn(ctx_neg)
        act_logits = torch.cat((ev_a[:, :2], tl, tr, ev_a[:, 2:3]), dim=-1)

        if use_action_bridge:
            pa = torch.sigmoid(act_logits.detach())
            ev_in = torch.stack((pa[:, 0], pa[:, 1], pa[:, 4], 0.5 * (pa[:, 2] + pa[:, 3])), dim=-1)
            od_in = 0.5 * (pa[:, 2:3] - pa[:, 3:4])
            de, do = self.bridge_even(ev_in), self.bridge_odd(od_in)
            d_even, d_odd, d_pos, d_neg = ctx_even + de, ctx_odd + do, ctx_pos + de + do, ctx_neg + de - do
        else:
            d_even, d_odd, d_pos, d_neg = ctx_even, ctx_odd, ctx_pos, ctx_neg

        c_dx = self.central_dx(d_even).view(bsz, 3, 3) + self.bio_par_central(h_bio).view(bsz, 3, 3)
        c_dy = self.central_dy(d_odd).view(bsz, 3, 3)
        p_pos = self.pair_knots(d_pos).view(bsz, 3, 2)
        p_neg = self.pair_knots(d_neg).view(bsz, 3, 2)
        dp_b = self.bio_par_pair(h_bio)
        p_pos = torch.stack((p_pos[..., 0] + dp_b, p_pos[..., 1]), dim=-1)
        p_neg = torch.stack((p_neg[..., 0] + dp_b, p_neg[..., 1]), dim=-1)

        ck = torch.stack((c_dx, c_dy), dim=-1)
        m4 = torch.stack((p_neg[..., 0], -p_neg[..., 1]), dim=-1)[:, None]
        knots5 = torch.cat((ck[:, :2], p_pos[:, None], m4, ck[:, 2:]), dim=1).double()
        knots5 = torch.cat((torch.zeros_like(knots5[:, :, :1]), knots5), dim=2)
        csc = self.central_scores(ctx_even)
        logits = torch.where(valid[:, None], torch.cat((csc[:, :3], self.pair_score(ctx_pos), self.pair_score(ctx_neg), csc[:, 3:]), dim=1), 0.0)

        t_sec = torch.arange(1, 31, dtype=history.dtype, device=history.device)[None, :] * 0.1 if fut_times_s is None else fut_times_s.to(history.dtype)
        idx = (t_sec - 1e-9).floor().long().clamp(0, 2)
        ph = (t_sec - idx.to(history.dtype))[:, None, :, None]
        bi = torch.arange(bsz, device=history.device)[:, None, None]
        mi = torch.arange(5, device=history.device)[None, :, None]
        delta = knots5[bi, mi, idx[:, None, :]] * (1.0 - ph) + knots5[bi, mi, (idx + 1)[:, None, :]] * ph
        world = torch.einsum("bkts,bws->bktw", delta, basis)
        cv = origin + ((history[:, -1:] - history[:, -2:-1]) / dt_step[:, -1:, None]) * t_sec[:, :, None]
        return {"paths": torch.cat((cv[:, None], cv[:, None] + world), dim=1), "logits": logits, "act_logits": act_logits, "valid": valid, "regime": regime}
