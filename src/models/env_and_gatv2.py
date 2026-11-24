"""Table 5 Rows 4-5 & Eq. (7)-(8): Local polyline encoder and 2-layer 4-head Directed GATv2."""

from typing import Optional, Tuple
import torch
import torch.nn as nn


def _safe_masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute softmax over masked entries without IEEE-754 0 * NaN gradient poisoning when all-zero."""
    mask_bool = mask > 0.5
    any_valid = mask_bool.any(dim=dim, keepdim=True)
    # Substitute all-True row when a sample has zero valid items so softmax stays finite
    safe_bool = torch.where(any_valid, mask_bool, torch.ones_like(mask_bool))
    filled = logits.masked_fill(~safe_bool, -1e9)
    probs = torch.softmax(filled, dim=dim)
    return probs * mask.to(logits.dtype)


class LocalPolylineEncoder(nn.Module):
    """Encodes up to 16 local boundary/attractor polylines P_env in R^{B x 16 x 10 x 11} -> R^{B x 64}."""

    def __init__(self, pt_dim: int = 11, ego_dim: int = 128, env_dim: int = 64) -> None:
        super().__init__()
        self.env_dim = env_dim
        self.pt_mlp = nn.Sequential(
            nn.Linear(pt_dim, env_dim),
            nn.LayerNorm(env_dim),
            nn.SiLU(),
            nn.Linear(env_dim, env_dim),
            nn.LayerNorm(env_dim),
            nn.SiLU(),
        )
        self.q_proj = nn.Linear(ego_dim, env_dim)
        self.k_proj = nn.Linear(env_dim, env_dim)
        self.v_proj = nn.Linear(env_dim, env_dim)
        self.out_norm = nn.LayerNorm(env_dim)
        self.env_drop = nn.Dropout(0.15)

    def forward(
        self,
        p_env: torch.Tensor,
        m_map: torch.Tensor,
        h_mp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return h_env [B, 64] and effective map flag eff_map [B, 1]."""
        clean_p = torch.nan_to_num(p_env, nan=0.0, posinf=0.0, neginf=0.0)
        if self.training:
            clean_p = clean_p.clone()
            clean_p[:, :, :, :2] = clean_p[:, :, :, :2] + 0.06 * torch.randn_like(clean_p[:, :, :, :2])
        m_pt = (clean_p[:, :, :, 10:11] > 0.5).to(h_mp.dtype)  # [B, 16, 10, 1]
        pt_feat = self.pt_mlp(clean_p * m_pt) * m_pt  # [B, 16, 10, 64]

        pt_cnt = m_pt.sum(dim=2)  # [B, 16, 1]
        m_poly = (pt_cnt.squeeze(-1) > 0.5).to(h_mp.dtype)  # [B, 16]
        mean_pool = pt_feat.sum(dim=2) / (pt_cnt + 1e-6)
        max_pool = pt_feat.masked_fill(m_pt < 0.5, -1e4).max(dim=2).values
        poly_feat = 0.5 * (mean_pool + max_pool) * m_poly.unsqueeze(-1)  # [B, 16, 64]

        q = self.q_proj(h_mp).unsqueeze(1)  # [B, 1, 64]
        k = self.k_proj(poly_feat)  # [B, 16, 64]
        v = self.v_proj(poly_feat)  # [B, 16, 64]
        scores = (q * k).sum(dim=-1) / (self.env_dim ** 0.5)  # [B, 16]
        attn = _safe_masked_softmax(scores, m_poly, dim=-1)  # [B, 16]

        has_poly = (m_poly.sum(dim=-1, keepdim=True) > 0.5).to(h_mp.dtype)  # [B, 1]
        eff_map = (m_map > 0.5).to(h_mp.dtype)
        agg = (attn.unsqueeze(-1) * v).sum(dim=1)  # [B, 64]
        h_env = (eff_map * has_poly) * self.env_drop(self.out_norm(agg))
        return h_env, eff_map


class DirectedGATv2Layer(nn.Module):
    """Single multi-head edge-aware Directed GATv2 layer (Eq. 8: 4 heads x 32 dim = 128)."""

    def __init__(
        self, hidden_dim: int = 128, edge_dim: int = 32, num_heads: int = 4, update_nbr: bool = True
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.update_nbr = update_nbr

        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_e = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.attn_vec = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(-1))
        # Learnable Null-Sink logit avoids forcing bystander attention to sum to 1.0 when non-conflicting
        self.sink_logit = nn.Parameter(torch.full((1, 1, num_heads), 0.5))

        self.w_v = nn.Linear(hidden_dim + edge_dim, hidden_dim)
        self.leaky = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(hidden_dim)
        if update_nbr:
            self.nbr_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )

    def forward(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        e_ji: torch.Tensor,
        nbr_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args: h_i [B, D], h_j [B, M, D], e_ji [B, M, De], nbr_mask [B, M]."""
        bsz, max_m, _ = h_j.shape
        q = self.w_q(h_i).view(bsz, 1, self.num_heads, self.head_dim)
        k = self.w_k(h_j).view(bsz, max_m, self.num_heads, self.head_dim)
        e = self.w_e(e_ji).view(bsz, max_m, self.num_heads, self.head_dim)

        act = self.leaky(q + k + e)  # [B, M, H, Dh]
        logits = (act * self.attn_vec.view(1, 1, self.num_heads, self.head_dim)).sum(dim=-1)  # [B, M, H]
        m_exp = nbr_mask.unsqueeze(-1) > 0.5
        cat_logits = torch.cat([logits.masked_fill(~m_exp, -1e9), self.sink_logit.expand(bsz, 1, self.num_heads)], dim=1)
        alpha = torch.softmax(cat_logits, dim=1)[:, :max_m, :] * nbr_mask.unsqueeze(-1).to(h_i.dtype)

        val = self.w_v(torch.cat([h_j, e_ji], dim=-1)).view(bsz, max_m, self.num_heads, self.head_dim)
        msg = (alpha.unsqueeze(-1) * val).sum(dim=1).reshape(bsz, self.hidden_dim)  # [B, D]
        has_nbr = (nbr_mask.sum(dim=-1, keepdim=True) > 0.5).to(h_i.dtype)
        h_i_next = self.norm(h_i + has_nbr * msg)
        if self.update_nbr:
            h_j_next = (h_j + self.nbr_ffn(h_j)) * nbr_mask.unsqueeze(-1).to(h_i.dtype)
        else:
            h_j_next = h_j
        return h_i_next, h_j_next, alpha.mean(dim=-1)


class HeteroInteractionGraph(nn.Module):
    """2-layer 4-head Directed GATv2 fusing z_i in R^{201} and edges e_{j->i} in R^{9}."""

    def __init__(
        self,
        node_in_dim: int = 201,
        hidden_dim: int = 128,
        edge_in_dim: int = 9,
        edge_emb_dim: int = 32,
        use_graph: bool = True,
        use_v_rad: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_graph = use_graph
        self.use_v_rad = use_v_rad
        env_flag_dim = max(1, node_in_dim - hidden_dim - 8)
        self.type_proj = nn.Linear(8, hidden_dim, bias=False)
        self.map_proj = nn.Linear(env_flag_dim, hidden_dim, bias=False)
        self.map_scale = nn.Parameter(torch.tensor(0.10))
        self.map_drop = nn.Dropout(0.15)
        self.base_norm = nn.LayerNorm(hidden_dim)

        if use_graph:
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_in_dim, edge_emb_dim),
                nn.LayerNorm(edge_emb_dim),
                nn.SiLU(),
                nn.Linear(edge_emb_dim, edge_emb_dim),
                nn.LayerNorm(edge_emb_dim),
                nn.SiLU(),
            )
            self.nbr_init_proj = nn.Sequential(
                nn.Linear(edge_in_dim + 8, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.layer1 = DirectedGATv2Layer(hidden_dim, edge_emb_dim, num_heads=4, update_nbr=True)
            self.layer2 = DirectedGATv2Layer(hidden_dim, edge_emb_dim, num_heads=4, update_nbr=False)
            self.nbr_gate = nn.Linear(hidden_dim * 2, 1)
            nn.init.constant_(self.nbr_gate.bias, -1.5)
            self.nbr_drop = nn.Dropout(0.15)
            self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_mp: torch.Tensor,
        h_env: torch.Tensor,
        ego_type_oh: torch.Tensor,
        eff_map: torch.Tensor,
        nbr_edges: Optional[torch.Tensor] = None,
        nbr_type_oh: Optional[torch.Tensor] = None,
        nbr_mask: Optional[torch.Tensor] = None,
        nbr_h_mp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return ego-preserving interaction representation h_tilde [B, 128] and attention [B, M]."""
        map_delta = self.map_drop(self.map_proj(torch.cat([h_env, eff_map], dim=-1)))
        h0 = self.base_norm(h_mp + self.type_proj(ego_type_oh) + torch.tanh(self.map_scale) * map_delta)
        if not self.use_graph or nbr_edges is None or nbr_mask is None or nbr_type_oh is None:
            return h0, None

        clean_e = torch.nan_to_num(nbr_edges, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.use_v_rad:
            clean_e = clean_e.clone()
            clean_e[:, :, 5] = 0.0  # Single-variable ablation removing v_rad!

        mask_f = (nbr_mask > 0.5).to(h_mp.dtype)
        e_emb = self.edge_mlp(clean_e * mask_f.unsqueeze(-1)) * mask_f.unsqueeze(-1)
        if nbr_h_mp is not None:
            hj = self.base_norm(nbr_h_mp + self.type_proj(nbr_type_oh)) * mask_f.unsqueeze(-1)
        else:
            hj = self.nbr_init_proj(torch.cat([clean_e, nbr_type_oh], dim=-1)) * mask_f.unsqueeze(-1)

        h1, hj1, _ = self.layer1(h0, hj, e_emb, mask_f)
        h2, _, attn2 = self.layer2(h1, hj1, e_emb, mask_f)
        delta_gat = h2 - h0
        has_nbr = (mask_f.sum(dim=-1, keepdim=True) > 0.5).to(h_mp.dtype)
        g_nbr = has_nbr * torch.sigmoid(self.nbr_gate(torch.cat([h0, delta_gat], dim=-1)))
        return self.out_norm(h0 + g_nbr * self.nbr_drop(delta_gat)), attn2
