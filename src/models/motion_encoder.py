"""Table 5 Row 1: Masked GRU historical motion encoder (X_m in R^{B x T x 9} -> h_m in R^{B x 128})."""

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def _jit_masked_gru_scan(
    emb: torch.Tensor,
    m_pos: torch.Tensor,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: torch.Tensor,
    b_hh: torch.Tensor,
) -> torch.Tensor:
    bsz, seq_len, hidden_dim = emb.shape
    gi_all = F.linear(emb, w_ih, b_ih)  # Precompute all T input gates in ONE batched GEMM!
    h = torch.zeros((bsz, hidden_dim), dtype=emb.dtype, device=emb.device)
    for t in range(seq_len):
        gi = gi_all[:, t, :]
        gh = F.linear(h, w_hh, b_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        cand_h = (1.0 - z) * n + z * h
        mt = m_pos[:, t, :]
        h = mt * cand_h + (1.0 - mt) * h
    return h


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
        return _jit_masked_gru_scan(
            emb, m_pos, self.gru_cell.weight_ih, self.gru_cell.weight_hh, self.gru_cell.bias_ih, self.gru_cell.bias_hh
        )
