"""Table 5 Row 3 & Eq. (6): Explicit quality-regulated residual fusion (M0/M1a/M1b/M1/M1p)."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class QualityGatedFusion(nn.Module):
    """Unified motion-pose fusion module supporting strict single-variable ablations.

    Modes:
        - 'M0': Motion only (g = 0).
        - 'M1a': Unconditional equal-weight pose residual (g = m_pose).
        - 'M1b': Implicit feature gate without q (passes 0_4 for exact param parity with M1).
        - 'M1': Explicit quality vector q in [0, 1]^4 residual gate (Eq. 6).
        - 'M1p': Biomechanical Part-Decomposed Quality Gate (BPD-Gate, paper upgrade).
    """

    VALID_MODES = ("M0", "M1a", "M1b", "M1", "M1p")

    def __init__(self, hidden_dim: int = 128, fusion_mode: str = "M1") -> None:
        super().__init__()
        if fusion_mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode}, expected {self.VALID_MODES}")
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode
        self.out_norm = nn.LayerNorm(hidden_dim)

        if fusion_mode in ("M1b", "M1"):
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 4, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )
        elif fusion_mode == "M1p":
            self.part_gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 4 + 3, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )

    def forward(
        self,
        h_m: torch.Tensor,
        h_p: Optional[torch.Tensor] = None,
        q_vec: Optional[torch.Tensor] = None,
        q_part: Optional[torch.Tensor] = None,
        part_feats: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse h_m [B, 128] and pose features into h_mp [B, 128], returning gate g [B, 1 or 3]."""
        bsz = h_m.shape[0]
        if self.fusion_mode == "M0" or h_p is None or q_vec is None:
            g = h_m.new_zeros((bsz, 1))
            return self.out_norm(h_m), g

        m_pose = (q_vec[:, 0:1] > 0.0).to(h_m.dtype)
        safe_hp = m_pose * h_p

        if self.fusion_mode == "M1a":
            g = m_pose
            return self.out_norm(h_m + g * safe_hp), g

        if self.fusion_mode in ("M1b", "M1"):
            q_in = torch.zeros_like(q_vec) if self.fusion_mode == "M1b" else q_vec
            gate_in = torch.cat([h_m, safe_hp, q_in], dim=-1)
            g = m_pose * torch.sigmoid(self.gate_mlp(gate_in))
            return self.out_norm(h_m + g * safe_hp), g

        # M1p: Biomechanical Part-Decomposed Quality Gate (BPD-Gate)
        assert q_part is not None and part_feats is not None, "M1p requires q_part and part_feats"
        part_keys = ("head_sho", "torso", "legs")
        gates = []
        weighted_parts = []
        eye3 = torch.eye(3, device=h_m.device, dtype=h_m.dtype)
        for r_idx, key in enumerate(part_keys):
            q1_r = q_part[:, 2 * r_idx : 2 * r_idx + 1]
            q2_r = q_part[:, 2 * r_idx + 1 : 2 * r_idx + 2]
            m_r = (q1_r > 0.0).to(h_m.dtype)
            hp_r = m_r * part_feats[key]
            qr_vec = torch.cat([q1_r, q2_r, q_vec[:, 2:4]], dim=-1)
            er_vec = eye3[r_idx].unsqueeze(0).expand(bsz, -1)
            gin = torch.cat([h_m, hp_r, qr_vec, er_vec], dim=-1)
            g_r = m_r * torch.sigmoid(self.part_gate_mlp(gin))
            gates.append(g_r)
            weighted_parts.append(g_r * hp_r)

        g_stack = torch.cat(gates, dim=-1)  # [B, 3]
        denom = torch.clamp(g_stack.sum(dim=-1, keepdim=True), min=1.0)
        agg_hp = m_pose * (sum(weighted_parts) / denom)
        return self.out_norm(h_m + agg_hp), g_stack
