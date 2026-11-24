"""Table 9 External Baselines: Ext-1 (MCENET), Ext-2 (HST), and Ext-3 (Social-Transmotion)."""

from typing import Dict
import math
import torch
import torch.nn as nn

from ..data.constants import DT, K_MODES, NUM_ACTIONS, NUM_JOINTS, T_FUT
from .env_and_gatv2 import LocalPolylineEncoder


class SharedMultiModalHead(nn.Module):
    """Standard K=6 trajectory + mode probability head for fair同口径 comparison across baselines."""

    def __init__(
        self, hidden_dim: int = 128, num_modes: int = K_MODES, fut_steps: int = T_FUT, dt: float = DT
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.fut_steps = fut_steps
        self.dt = dt
        self.mode_emb = nn.Parameter(torch.randn(num_modes, 32) * 0.02)
        self.gru = nn.GRUCell(2 + 32, hidden_dim)
        self.init_proj = nn.Linear(hidden_dim + 32, hidden_dim)
        self.out_xy = nn.Linear(hidden_dim, 2)
        self.pi_mlp = nn.Sequential(nn.Linear(hidden_dim * 2 + 32, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, ctx: torch.Tensor, v0_xy: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz = ctx.shape[0]
        k = self.num_modes
        m_q = self.mode_emb.unsqueeze(0).expand(bsz, -1, -1)
        s = self.init_proj(torch.cat([ctx.unsqueeze(1).expand(bsz, k, -1), m_q], dim=-1))
        s_flat = s.reshape(bsz * k, self.hidden_dim)
        mq_flat = m_q.reshape(bsz * k, 32)

        pos = ctx.new_zeros((bsz * k, 2))
        prev_d = (v0_xy * self.dt).unsqueeze(1).expand(bsz, k, 2).reshape(bsz * k, 2)
        steps = []
        for _ in range(self.fut_steps):
            s_flat = self.gru(torch.cat([prev_d, mq_flat], dim=-1), s_flat)
            d = self.out_xy(s_flat)
            pos = pos + d
            prev_d = d
            steps.append(pos.view(bsz, k, 2))
        trajs = torch.stack(steps, dim=2)
        sf = s_flat.view(bsz, k, self.hidden_dim)
        logits = self.pi_mlp(torch.cat([ctx.unsqueeze(1).expand(bsz, k, -1), sf, m_q], dim=-1)).squeeze(-1)
        return {
            "trajs": trajs,
            "mode_logits": logits,
            "mode_probs": torch.softmax(logits, dim=-1),
            "act_logits": ctx.new_zeros((bsz, NUM_ACTIONS)),
            "act_probs": ctx.new_zeros((bsz, NUM_ACTIONS)),
        }


class MCENETBaseline(nn.Module):
    """Ext-1: MCENET (Cheng et al., ITSC'20) — Motion + Local Env + Polar Social Context + CVAE."""

    def __init__(self, hidden_dim: int = 128, fut_steps: int = T_FUT, dt: float = DT) -> None:
        super().__init__()
        self.mot_gru = nn.GRU(9, hidden_dim, batch_first=True)
        self.env_enc = LocalPolylineEncoder(pt_dim=11, ego_dim=hidden_dim, env_dim=64)
        # Polar social encoder: [r, sin(theta), cos(theta), dvx, dvy] + agent type (8) = 13
        self.polar_mlp = nn.Sequential(nn.Linear(13, 64), nn.SiLU(), nn.Linear(64, 64))
        self.cvae_fuse = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.head = SharedMultiModalHead(hidden_dim=hidden_dim, fut_steps=fut_steps, dt=dt)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        xm = torch.nan_to_num(batch["x_m"], nan=0.0)
        _, h_m = self.mot_gru(xm)
        h_m = h_m.squeeze(0)
        if "p_env" in batch and "m_map" in batch:
            h_env, _ = self.env_enc(batch["p_env"], batch["m_map"], h_m)
        else:
            h_env = h_m.new_zeros((xm.shape[0], 64))

        if "nbr_edges" in batch and "nbr_mask" in batch and "nbr_type_oh" in batch:
            e = torch.nan_to_num(batch["nbr_edges"], nan=0.0)
            nm = (batch["nbr_mask"] > 0.5).to(xm.dtype).unsqueeze(-1)
            r = e[:, :, 4:5]
            ang = torch.atan2(e[:, :, 1:2], e[:, :, 0:1] + 1e-6)
            polar = torch.cat([r, torch.sin(ang), torch.cos(ang), e[:, :, 2:4], batch["nbr_type_oh"]], dim=-1)
            p_feat = self.polar_mlp(polar * nm) * nm
            h_soc = p_feat.sum(dim=1) / (nm.sum(dim=1) + 1.0)
        else:
            h_soc = h_m.new_zeros((xm.shape[0], 64))

        ctx = self.cvae_fuse(torch.cat([h_m, h_env, h_soc], dim=-1))
        v0 = xm[:, -1, 2:4] * (xm[:, -1, 6:7] > 0.5).to(xm.dtype)
        return self.head(ctx, v0)


class HSTBaseline(nn.Module):
    """Ext-2: HST (Salzmann et al., IEEE RA-L'23) — Spatio-temporal skeletal-trajectory Transformer."""

    def __init__(self, hidden_dim: int = 128, max_len: int = 32, fut_steps: int = T_FUT, dt: float = DT) -> None:
        super().__init__()
        self.mot_tok = nn.Linear(9, hidden_dim)
        self.joint_tok = nn.Linear(6, hidden_dim)
        self.joint_id = nn.Embedding(NUM_JOINTS, hidden_dim)
        self.time_emb = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.joint_attn = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.temp_attn = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = SharedMultiModalHead(hidden_dim=hidden_dim, fut_steps=fut_steps, dt=dt)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        xm = torch.nan_to_num(batch["x_m"], nan=0.0)
        xp = torch.nan_to_num(batch["x_p"], nan=0.0)
        bsz, seq_t, _, _ = xp.shape
        jid = torch.arange(NUM_JOINTS, device=xp.device).view(1, 1, NUM_JOINTS)
        j_emb = self.joint_tok(xp) + self.joint_id(jid)  # [B, T, 17, D]
        j_flat = self.joint_attn(j_emb.reshape(bsz * seq_t, NUM_JOINTS, -1)).mean(dim=1).view(bsz, seq_t, -1)

        t_emb = self.time_emb[:, :seq_t, :]
        # Interleave motion and equal-weight skeletal tokens across time (no explicit quality gate)
        tokens = torch.cat([self.mot_tok(xm) + t_emb, j_flat + t_emb], dim=1)  # [B, 2T, D]
        ctx = self.temp_attn(tokens)[:, seq_t - 1, :]
        v0 = xm[:, -1, 2:4] * (xm[:, -1, 6:7] > 0.5).to(xm.dtype)
        return self.head(ctx, v0)


class SocialTransmotionBaseline(nn.Module):
    """Ext-3: Social-Transmotion (Saadatnejad et al., ICLR'24) — Promptable Dual-Transformer + Modality Masking."""

    def __init__(
        self, hidden_dim: int = 128, modality_drop_p: float = 0.25, fut_steps: int = T_FUT, dt: float = DT
    ) -> None:
        super().__init__()
        self.modality_drop_p = modality_drop_p
        self.traj_proj = nn.Linear(9, hidden_dim)
        self.pose_proj = nn.Linear(NUM_JOINTS * 6, hidden_dim)
        self.nbr_proj = nn.Linear(9 + 8, hidden_dim)
        self.cue_type = nn.Embedding(3, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.cross_cue_tf = nn.TransformerEncoder(layer, num_layers=2)
        self.social_tf = nn.TransformerEncoder(layer, num_layers=1)
        self.head = SharedMultiModalHead(hidden_dim=hidden_dim, fut_steps=fut_steps, dt=dt)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        xm = torch.nan_to_num(batch["x_m"], nan=0.0)
        xp = torch.nan_to_num(batch["x_p"], nan=0.0)
        bsz, seq_t, _ = xm.shape
        dev = xm.device

        t_tok = self.traj_proj(xm) + self.cue_type(torch.zeros(1, 1, dtype=torch.long, device=dev))
        p_flat = xp.reshape(bsz, seq_t, NUM_JOINTS * 6)
        if self.training and self.modality_drop_p > 0.0:
            keep = (torch.rand((bsz, 1, 1), device=dev) >= self.modality_drop_p).to(xm.dtype)
            p_flat = p_flat * keep
        p_tok = self.pose_proj(p_flat) + self.cue_type(torch.ones(1, 1, dtype=torch.long, device=dev))

        ego_seq = self.cross_cue_tf(torch.cat([t_tok, p_tok], dim=1))
        ego_tok = ego_seq[:, seq_t - 1 : seq_t, :]  # [B, 1, D]

        if "nbr_edges" in batch and "nbr_mask" in batch and "nbr_type_oh" in batch:
            nm = (batch["nbr_mask"] > 0.5).to(xm.dtype).unsqueeze(-1)
            n_in = torch.cat([torch.nan_to_num(batch["nbr_edges"], nan=0.0), batch["nbr_type_oh"]], dim=-1) * nm
            n_tok = self.nbr_proj(n_in) + self.cue_type(torch.full((1, 1), 2, dtype=torch.long, device=dev))
            soc_out = self.social_tf(torch.cat([ego_tok, n_tok * nm], dim=1))
            ctx = soc_out[:, 0, :]
        else:
            ctx = ego_tok.squeeze(1)

        v0 = xm[:, -1, 2:4] * (xm[:, -1, 6:7] > 0.5).to(xm.dtype)
        return self.head(ctx, v0)
