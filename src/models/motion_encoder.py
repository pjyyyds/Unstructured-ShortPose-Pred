"""Table 5 Row 1: Masked GRU historical motion encoder (X_m in R^{B x T x 9} -> h_m in R^{B x 128})."""

import torch
import torch.nn as nn


class MaskedMotionEncoder(nn.Module):
    """Encodes local-translated motion sequence while freezing hidden state on missing frames."""

    def __init__(self, in_dim: int = 9, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x_m: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_m: [B, T, 9] float tensor; channel 5 is m_pos in {0, 1}.

        Returns:
            h_m: [B, 128] motion hidden representation at t_0.
        """
        bsz, seq_len, _ = x_m.shape
        m_pos = (x_m[:, :, 5:6] > 0.5).to(x_m.dtype)
        # Sanitize input before MLP so missing frames never inject garbage values
        clean_x = torch.nan_to_num(x_m, nan=0.0, posinf=0.0, neginf=0.0)
        emb = self.input_mlp(clean_x)

        h = x_m.new_zeros((bsz, self.hidden_dim))
        for t in range(seq_len):
            cand_h = self.gru_cell(emb[:, t, :], h)
            mt = m_pos[:, t, :]
            h = mt * cand_h + (1.0 - mt) * h
        return h
