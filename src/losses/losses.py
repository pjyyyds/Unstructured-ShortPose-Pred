"""Eq. (10): Shared-winner Smooth-L1 + Mode CE + m_act-masked 5-class BCE joint loss."""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class JointPredictionLoss(nn.Module):
    """Computes Eq. (10) joint loss with exact masking on missing future frames and invalid m_act."""

    def __init__(
        self,
        lambda_mode: float = 0.25,
        lambda_act: float = 0.50,
        smooth_l1_beta: float = 0.50,
        use_action_loss: bool = True,
        act_pos_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.lambda_mode = lambda_mode
        self.lambda_act = lambda_act
        self.smooth_l1_beta = smooth_l1_beta
        self.use_action_loss = use_action_loss
        if act_pos_weights is not None:
            self.register_buffer("act_pos_weights", act_pos_weights.float(), persistent=False)
        else:
            self.register_buffer("act_pos_weights", torch.ones(5, dtype=torch.float32), persistent=False)

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        fut_xy: torch.Tensor,
        fut_mask: torch.Tensor,
        y_act: Optional[torch.Tensor] = None,
        m_act: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Args:
            pred['trajs']: [B, K, T_f, 2], pred['mode_logits']: [B, K], pred['act_logits']: [B, 5]
            fut_xy: [B, T_f, 2] local-translated ground-truth future offsets (y_t - p_{t_0})
            fut_mask: [B, T_f] binary mask in {0, 1}
        """
        trajs = pred["trajs"]
        mode_logits = pred["mode_logits"]
        bsz, num_k, fut_t, _ = trajs.shape

        clean_gt = torch.nan_to_num(fut_xy, nan=0.0).unsqueeze(1).expand(bsz, num_k, fut_t, 2)
        f_mask = (fut_mask > 0.5).to(trajs.dtype)  # [B, T_f]
        valid_cnt = f_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]

        # Winner selection k* on valid future steps only (Eq. 10)
        l2_step = torch.linalg.vector_norm(trajs - clean_gt, dim=-1) * f_mask.unsqueeze(1)  # [B, K, T_f]
        ade_per_mode = l2_step.sum(dim=-1) / valid_cnt  # [B, K]
        k_star = torch.argmin(ade_per_mode.detach(), dim=-1)  # [B]

        idx_gather = k_star.view(bsz, 1, 1, 1).expand(bsz, 1, fut_t, 2)
        win_traj = torch.gather(trajs, dim=1, index=idx_gather).squeeze(1)  # [B, T_f, 2]
        gt_clean_2d = torch.nan_to_num(fut_xy, nan=0.0)

        reg_elem = F.smooth_l1_loss(win_traj, gt_clean_2d, beta=self.smooth_l1_beta, reduction="none").sum(dim=-1)
        loss_reg = ((reg_elem * f_mask).sum(dim=-1) / valid_cnt.squeeze(-1)).mean()
        loss_mode = F.cross_entropy(mode_logits, k_star)

        loss_act = trajs.new_tensor(0.0)
        if self.use_action_loss and y_act is not None and m_act is not None:
            act_mask = (m_act.view(-1) > 0.5).to(trajs.dtype)  # [B]
            if float(act_mask.sum().item()) > 0.0:
                bce = F.binary_cross_entropy_with_logits(
                    pred["act_logits"],
                    y_act.to(trajs.dtype),
                    pos_weight=self.act_pos_weights.to(trajs.device, trajs.dtype),
                    reduction="none",
                )  # [B, 5]
                per_sample_act = bce.mean(dim=-1) * act_mask
                loss_act = per_sample_act.sum() / act_mask.sum().clamp(min=1.0)

        total = loss_reg + self.lambda_mode * loss_mode + self.lambda_act * loss_act
        return {
            "loss": total,
            "loss_reg": loss_reg.detach(),
            "loss_mode": loss_mode.detach(),
            "loss_act": loss_act.detach(),
        }
