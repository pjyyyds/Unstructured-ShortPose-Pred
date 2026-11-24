"""Table 5 Row 6 & Eq. (9): 5-class action classifier and K=6 conditional autoregressive GRU decoder."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.constants import DT, K_MODES, NUM_ACTIONS, T_FUT


@torch.jit.script
def _jit_decoder_rollout(
    prev_delta: torch.Tensor,
    s_flat: torch.Tensor,
    static_gates: torch.Tensor,
    ih_w: torch.Tensor,
    ih_b: torch.Tensor,
    hh_w: torch.Tensor,
    hh_b: torch.Tensor,
    out_w: torch.Tensor,
    out_b: torch.Tensor,
    bsz: int,
    k_num: int,
    fut_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cur_pos = torch.zeros((bsz * k_num, 2), dtype=s_flat.dtype, device=s_flat.device)
    traj_steps = torch.jit.annotate(list[torch.Tensor], [])
    delta_steps = torch.jit.annotate(list[torch.Tensor], [])
    for _ in range(fut_steps):
        gi = F.linear(prev_delta, ih_w, ih_b) + static_gates
        gh = F.linear(s_flat, hh_w, hh_b)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r_gate = torch.sigmoid(i_r + h_r)
        z_gate = torch.sigmoid(i_z + h_z)
        n_cand = torch.tanh(i_n + r_gate * h_n)
        s_flat = (1.0 - z_gate) * n_cand + z_gate * s_flat
        delta = F.linear(s_flat, out_w, out_b)
        cur_pos = cur_pos + delta
        prev_delta = delta
        traj_steps.append(cur_pos.view(bsz, k_num, 2))
        delta_steps.append(delta.view(bsz, k_num, 2))
    return torch.stack(traj_steps, dim=2), torch.stack(delta_steps, dim=2), s_flat.view(bsz, k_num, -1)


def extract_trajectory_kinematics(
    trajs: torch.Tensor,
    deltas: torch.Tensor,
    v0_xy: torch.Tensor,
    dt: float = DT,
) -> torch.Tensor:
    """Compute 6D physical summary psi(Y_k) in R^{B x K x 6} for ACMC mode calibration (M3p)."""
    fut_t = deltas.shape[2]
    early_end = max(1, min(5, fut_t // 3))
    late_start = max(0, fut_t // 2)
    step_spd = torch.linalg.vector_norm(deltas, dim=-1) / dt  # [B, K, T_f]
    v_early = step_spd[:, :, :early_end].mean(dim=-1)
    v_late = step_spd[:, :, late_start:].mean(dim=-1)
    v0_norm = torch.linalg.vector_norm(v0_xy, dim=-1, keepdim=True)  # [B, 1]
    dv_acc = v_late - v0_norm

    end_vec = trajs[:, :, -1, :]  # [B, K, 2]
    end_dist = torch.linalg.vector_norm(end_vec, dim=-1)
    path_len = torch.linalg.vector_norm(deltas, dim=-1).sum(dim=-1) + 1e-4
    straight_ratio = end_dist / path_len

    # Signed lateral turn sin(dtheta) relative to initial direction (or early displacement if still)
    early_dir = trajs[:, :, min(4, fut_t - 1), :]  # [B, K, 2]
    ref_dir = torch.where(v0_norm.unsqueeze(-1) > 0.3, v0_xy.unsqueeze(1).expand_as(early_dir), early_dir)
    ref_unit = ref_dir / (torch.linalg.vector_norm(ref_dir, dim=-1, keepdim=True) + 1e-4)
    end_unit = end_vec / (end_dist.unsqueeze(-1) + 1e-4)
    sin_turn = ref_unit[:, :, 0] * end_unit[:, :, 1] - ref_unit[:, :, 1] * end_unit[:, :, 0]

    return torch.stack([v_early, v_late, dv_acc, end_dist, sin_turn, straight_ratio], dim=-1)


class ActionConditionedDecoder(nn.Module):
    """5-class action head + gradient control bridge + K=6 GRU decoder + optional ACMC (M3p).

    action_mode choices:
        - 'none': no action loss, c_act = 0 (for M0, M1*, M2*)
        - 'aux_only': supervise action head, keep c_act = 0 (M3a)
        - 'e2e': c_act = MLP(p_act) with full backprop (M3b)
        - 'stop_grad': c_act = MLP(p_act.detach()) blocking regression gradients (M3)
        - 'acmc': stop_grad + Action-Consistent Mode Calibration on K=6 rollouts (M3p)
    """

    VALID_ACTION_MODES = ("none", "aux_only", "e2e", "stop_grad", "acmc")

    def __init__(
        self,
        hidden_dim: int = 128,
        env_dim: int = 64,
        mode_dim: int = 32,
        cond_dim: int = 16,
        num_modes: int = K_MODES,
        fut_steps: int = T_FUT,
        dt: float = DT,
        action_mode: str = "stop_grad",
    ) -> None:
        super().__init__()
        if action_mode not in self.VALID_ACTION_MODES:
            raise ValueError(f"Invalid action_mode={action_mode}")
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.num_modes = num_modes
        self.fut_steps = fut_steps
        self.dt = dt
        self.action_mode = action_mode

        if action_mode != "none":
            self.act_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Dropout(0.15),
                nn.Linear(64, NUM_ACTIONS),
            )
        if action_mode in ("e2e", "stop_grad", "acmc"):
            # Smooth Tanh bottleneck without LayerNorm avoids amplifying detached calibration jitter
            self.cond_mlp = nn.Sequential(
                nn.Linear(NUM_ACTIONS, 32),
                nn.SiLU(),
                nn.Linear(32, cond_dim),
                nn.Tanh(),
            )
            self.cond_scale = nn.Parameter(torch.tensor(0.25))
        if action_mode == "acmc":
            self.acmc_norm = nn.LayerNorm(6)
            self.acmc_head = nn.Sequential(
                nn.Linear(6 + NUM_ACTIONS + 2, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
            )
            nn.init.normal_(self.acmc_head[-1].weight, std=0.02)
            nn.init.zeros_(self.acmc_head[-1].bias)

        self.dec_env_scale = nn.Parameter(torch.tensor(0.10))
        self.mode_queries = nn.Parameter(torch.empty(num_modes, mode_dim))
        nn.init.orthogonal_(self.mode_queries)
        self.mode_init_proj = nn.Linear(mode_dim, hidden_dim)

        in_step_dim = 2 + env_dim + mode_dim + cond_dim  # 2 + 64 + 32 + 16 = 114 (Eq. 9)
        self.ih_delta = nn.Linear(2, 3 * hidden_dim)
        self.ih_static = nn.Linear(in_step_dim - 2, 3 * hidden_dim)
        self.hh_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.delta_out = nn.Linear(hidden_dim, 2)
        self.pi_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + mode_dim + cond_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )


    def forward(
        self,
        h_tilde: torch.Tensor,
        h_env: torch.Tensor,
        v0_xy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Decode K=6 trajectories [B, 6, 30, 2], mode probs [B, 6], and action probs [B, 5]."""
        bsz = h_tilde.shape[0]
        k_num = self.num_modes

        if self.action_mode == "none":
            act_logits = h_tilde.new_zeros((bsz, NUM_ACTIONS))
            p_act = h_tilde.new_zeros((bsz, NUM_ACTIONS))
            c_act = h_tilde.new_zeros((bsz, self.cond_dim))
        else:
            act_logits = self.act_head(h_tilde)
            p_act = torch.sigmoid(act_logits)
            if self.action_mode == "aux_only":
                c_act = h_tilde.new_zeros((bsz, self.cond_dim))
            elif self.action_mode == "e2e":
                c_act = torch.tanh(self.cond_scale) * self.cond_mlp(2.0 * p_act - 1.0)
            else:  # 'stop_grad' or 'acmc'
                p_det = p_act.detach()
                c_act = torch.tanh(self.cond_scale) * self.cond_mlp(2.0 * p_det - 1.0)

        q_k = self.mode_queries.unsqueeze(0).expand(bsz, -1, -1)  # [B, K, 32]
        s_k = h_tilde.unsqueeze(1) + self.mode_init_proj(q_k)  # [B, K, 128]
        s_flat = s_k.reshape(bsz * k_num, self.hidden_dim)

        env_scaled = torch.tanh(self.dec_env_scale) * h_env
        env_flat = env_scaled.unsqueeze(1).expand(bsz, k_num, -1).reshape(bsz * k_num, -1)
        qk_flat = q_k.reshape(bsz * k_num, -1)
        cond_flat = c_act.unsqueeze(1).expand(bsz, k_num, -1).reshape(bsz * k_num, -1)
        static_ctx = torch.cat([env_flat, qk_flat, cond_flat], dim=-1)  # [B*K, 112]

        static_gates = self.ih_static(static_ctx)  # [B*K, 3*D]
        prev_delta = (v0_xy * self.dt).unsqueeze(1).expand(bsz, k_num, 2).reshape(bsz * k_num, 2)
        trajs, deltas, s_final = _jit_decoder_rollout(
            prev_delta,
            s_flat,
            static_gates,
            self.ih_delta.weight,
            self.ih_delta.bias,
            self.hh_proj.weight,
            self.hh_proj.bias,
            self.delta_out.weight,
            self.delta_out.bias,
            bsz,
            k_num,
            self.fut_steps,
        )

        pi_in = torch.cat(
            [
                h_tilde.unsqueeze(1).expand(bsz, k_num, -1),
                s_final - h_tilde.unsqueeze(1),
                q_k,
                c_act.unsqueeze(1).expand(bsz, k_num, -1),
            ],
            dim=-1,
        )
        mode_logits = self.pi_head(pi_in).squeeze(-1)  # [B, K]

        if self.action_mode == "acmc":
            psi = extract_trajectory_kinematics(trajs.detach(), deltas.detach(), v0_xy.detach(), dt=self.dt)
            p_rep = p_act.detach().unsqueeze(1).expand(bsz, k_num, -1)
            lon_align = psi[:, :, 2:3] * (p_rep[:, :, 0:1] - p_rep[:, :, 1:2])
            lat_align = psi[:, :, 4:5] * (p_rep[:, :, 2:3] - p_rep[:, :, 3:4])
            compat_in = torch.cat([self.acmc_norm(psi), p_rep, lon_align, lat_align], dim=-1)
            mode_logits = mode_logits + self.acmc_head(compat_in).squeeze(-1)

        mode_probs = torch.softmax(mode_logits, dim=-1)
        return {
            "trajs": trajs,
            "mode_logits": mode_logits,
            "mode_probs": mode_probs,
            "act_logits": act_logits,
            "act_probs": p_act,
        }
