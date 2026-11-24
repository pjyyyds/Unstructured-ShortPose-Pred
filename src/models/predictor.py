"""Full modular predictor assembling M0-M3, single-factor ablations, and BPD-Gate + ACMC upgrades."""

from typing import Dict, Tuple
import torch
import torch.nn as nn

from ..data.constants import DT, T_FUT
from .action_and_decoder import ActionConditionedDecoder
from .env_and_gatv2 import HeteroInteractionGraph, LocalPolylineEncoder
from .motion_encoder import MaskedMotionEncoder
from .pose_stgcn import PoseSTGCNEncoder
from .quality_gate import QualityGatedFusion

# (fusion_mode, use_map, use_graph, use_v_rad, action_mode)
MODEL_SPECS: Dict[str, Tuple[str, bool, bool, bool, str]] = {
    "M0": ("M0", False, False, False, "none"),
    "M1a": ("M1a", False, False, False, "none"),
    "M1b": ("M1b", False, False, False, "none"),
    "M1": ("M1", False, False, False, "none"),
    "M1p": ("M1p", False, False, False, "none"),
    "M2a": ("M1", True, False, False, "none"),
    "M2b_norad": ("M1", False, True, False, "none"),
    "M2b": ("M1", False, True, True, "none"),
    "M2": ("M1", True, True, True, "none"),
    "M3a": ("M1", True, True, True, "aux_only"),
    "M3b": ("M1", True, True, True, "e2e"),
    "M3": ("M1", True, True, True, "stop_grad"),
    "M3p": ("M1", True, True, True, "acmc"),
    "Ours_Full": ("M1p", True, True, True, "acmc"),
}


class UnstructuredShortPosePredictor(nn.Module):
    """Unified predictor strictly adhering to Eq. (1)-(10) and Table 5/9 specifications."""

    def __init__(
        self,
        variant: str = "M3",
        hidden_dim: int = 128,
        env_dim: int = 64,
        fut_steps: int = T_FUT,
        dt: float = DT,
    ) -> None:
        super().__init__()
        if variant not in MODEL_SPECS:
            raise ValueError(f"Unknown variant={variant}, available: {list(MODEL_SPECS)}")
        fusion_mode, use_map, use_graph, use_v_rad, action_mode = MODEL_SPECS[variant]
        self.variant = variant
        self.fusion_mode = fusion_mode
        self.use_map = use_map
        self.use_graph = use_graph
        self.env_dim = env_dim
        self.action_mode = action_mode

        self.motion_enc = MaskedMotionEncoder(in_dim=9, hidden_dim=hidden_dim)
        if fusion_mode != "M0":
            self.pose_enc = PoseSTGCNEncoder(in_dim=6, out_dim=hidden_dim)
        self.quality_gate = QualityGatedFusion(hidden_dim=hidden_dim, fusion_mode=fusion_mode)

        if use_map:
            self.poly_enc = LocalPolylineEncoder(pt_dim=11, ego_dim=hidden_dim, env_dim=env_dim)
        self.interaction = HeteroInteractionGraph(
            node_in_dim=hidden_dim + env_dim + 8 + 1,
            hidden_dim=hidden_dim,
            edge_in_dim=9,
            edge_emb_dim=32,
            use_graph=use_graph,
            use_v_rad=use_v_rad,
        )
        self.decoder = ActionConditionedDecoder(
            hidden_dim=hidden_dim,
            env_dim=env_dim,
            fut_steps=fut_steps,
            dt=dt,
            action_mode=action_mode,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run end-to-end prediction on a collated batch dictionary."""
        x_m = batch["x_m"]  # [B, T, 9]
        bsz = x_m.shape[0]
        h_m = self.motion_enc(x_m)

        if self.fusion_mode == "M0":
            h_mp, gate_w = self.quality_gate(h_m)
        else:
            h_p, part_feats = self.pose_enc(batch["x_p"])
            h_mp, gate_w = self.quality_gate(
                h_m, h_p=h_p, q_vec=batch["q_vec"], q_part=batch.get("q_part"), part_feats=part_feats
            )

        if self.use_map and "p_env" in batch and "m_map" in batch:
            h_env, eff_map = self.poly_enc(batch["p_env"], batch["m_map"], h_mp)
        else:
            h_env = h_mp.new_zeros((bsz, self.env_dim))
            eff_map = h_mp.new_zeros((bsz, 1))

        ego_type = batch.get("ego_type_oh")
        if ego_type is None:
            ego_type = h_mp.new_zeros((bsz, 8))
            ego_type[:, 0] = 1.0

        h_tilde, nbr_attn = self.interaction(
            h_mp=h_mp,
            h_env=h_env,
            ego_type_oh=ego_type,
            eff_map=eff_map,
            nbr_edges=batch.get("nbr_edges"),
            nbr_type_oh=batch.get("nbr_type_oh"),
            nbr_mask=batch.get("nbr_mask"),
        )
        # Observed velocity at t_0 (valid-masked in channel 2:4 and channel 6 of x_m[:, -1])
        m_vel_t0 = (x_m[:, -1, 6:7] > 0.5).to(x_m.dtype)
        v0_xy = torch.nan_to_num(x_m[:, -1, 2:4], nan=0.0) * m_vel_t0

        out = self.decoder(h_tilde, h_env, v0_xy)
        out["gate_weights"] = gate_w
        if nbr_attn is not None:
            out["nbr_attn"] = nbr_attn
        return out
