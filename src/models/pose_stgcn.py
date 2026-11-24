"""Table 5 Row 2: 3-layer Mask-Normalized ST-GCN pose encoder with confidence pooling."""

from typing import Dict, Tuple
import torch
import torch.nn as nn

from ..data.constants import (
    COCO17_EDGES,
    HEAD_SHOULDER_JOINTS,
    LOWER_LIMB_GAIT_JOINTS,
    NUM_JOINTS,
    UPPER_TORSO_ARMS_JOINTS,
)


def build_coco17_adjacency() -> torch.Tensor:
    """Build symmetric binary neighbor adjacency matrix A (without self-loop) in R^{17 x 17}."""
    adj = torch.zeros((NUM_JOINTS, NUM_JOINTS), dtype=torch.float32)
    for u, v in COCO17_EDGES:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


class MaskedSTGCNBlock(nn.Module):
    """Single spatial-temporal graph conv block immune to occluded-joint coordinate bleed."""

    def __init__(self, in_ch: int, out_ch: int, kernel_t: int = 3) -> None:
        super().__init__()
        self.self_lin = nn.Linear(in_ch, out_ch)
        self.nbr_lin = nn.Linear(in_ch, out_ch, bias=False)
        pad_t = kernel_t // 2
        # Depthwise-separable temporal conv preserves full temporal window at 2.8x lower FLOPs
        self.t_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(kernel_t, 1), padding=(pad_t, 0), groups=out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=(1, 1)),
        )
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.SiLU()
        self.res = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, h: torch.Tensor, m_tj: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Args: h [B, T, V, C_in], m_tj [B, T, V, 1], adj [V, V]."""
        h_masked = h * m_tj
        # Valid-neighbor normalized message: [B, T, V, V]
        valid_k = m_tj.squeeze(-1).unsqueeze(-2)  # [B, T, 1, V]
        eff_adj = adj.view(1, 1, NUM_JOINTS, NUM_JOINTS) * valid_k
        deg = eff_adj.sum(dim=-1, keepdim=True) + 1.0
        norm_adj = eff_adj / deg

        nbr_feat = torch.matmul(norm_adj, self.nbr_lin(h_masked))
        sp_out = self.act(self.norm(self.self_lin(h_masked) + nbr_feat)) * m_tj

        # Temporal conv across T for each joint V independently
        t_in = sp_out.permute(0, 3, 1, 2)  # [B, C_out, T, V]
        t_out = self.t_conv(t_in).permute(0, 2, 3, 1)  # [B, T, V, C_out]
        out = self.act(self.norm(t_out + self.res(h_masked))) * m_tj
        return out


class PoseSTGCNEncoder(nn.Module):
    """3-layer ST-GCN(6 -> 32 -> 64 -> 128) with confidence pooling and exact zero fallback."""

    def __init__(self, in_dim: int = 6, out_dim: int = 128) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.register_buffer("adj", build_coco17_adjacency(), persistent=False)
        self.block1 = MaskedSTGCNBlock(in_dim, 32, kernel_t=3)
        self.block2 = MaskedSTGCNBlock(32, 64, kernel_t=3)
        self.block3 = MaskedSTGCNBlock(64, out_dim, kernel_t=3)
        self.out_norm = nn.LayerNorm(out_dim)
        self.part_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(3)])

    @staticmethod
    def _conf_pool(
        feat: torch.Tensor,
        w_tj: torch.Tensor,
        joint_idxs: Tuple[int, ...],
        norm_layer: nn.Module,
    ) -> torch.Tensor:
        sub_f = feat[:, :, joint_idxs, :]  # [B, T, Vg, D]
        sub_w = w_tj[:, :, joint_idxs, :]  # [B, T, Vg, 1]
        w_sum_4d = sub_w.sum(dim=(1, 2), keepdim=True)  # [B, 1, 1, 1]
        has_valid = (w_sum_4d.squeeze(-1).squeeze(-1) > 1e-7).to(feat.dtype)  # [B, 1]
        norm_w = sub_w / (w_sum_4d + 1e-6)
        pooled = (sub_f * norm_w).sum(dim=(1, 2))  # [B, D]
        # Multiply has_valid AFTER LayerNorm so empty groups equal exact 0.0 bit-for-bit
        return has_valid * norm_layer(pooled)

    def forward(self, x_p: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode pose tensor x_p [B, T, 17, 6].

        Returns:
            h_p: [B, 128] global pose feature (exact 0 when q_1 == 0).
            parts: dict with 'head_sho', 'torso', 'legs' in [B, 128] for BPD-Gate (M1p).
        """
        clean_xp = torch.nan_to_num(x_p, nan=0.0, posinf=0.0, neginf=0.0)
        m_tj = (clean_xp[:, :, :, 3:4] > 0.5).to(x_p.dtype)
        c_tj = clean_xp[:, :, :, 2:3].clamp(1e-3, 1.0) * m_tj

        h = self.block1(clean_xp, m_tj, self.adj)
        h = self.block2(h, m_tj, self.adj)
        h = self.block3(h, m_tj, self.adj)

        all_joints = tuple(range(NUM_JOINTS))
        h_p = self._conf_pool(h, c_tj, all_joints, self.out_norm)
        parts = {
            "head_sho": self._conf_pool(h, c_tj, HEAD_SHOULDER_JOINTS, self.part_norms[0]),
            "torso": self._conf_pool(h, c_tj, UPPER_TORSO_ARMS_JOINTS, self.part_norms[1]),
            "legs": self._conf_pool(h, c_tj, LOWER_LIMB_GAIT_JOINTS, self.part_norms[2]),
        }
        return h_p, parts
