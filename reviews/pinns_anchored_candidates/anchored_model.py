"""Observed-only six-mode predictor and training-only responsibility assignment."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.data.constants import DT, T_FUT
from src.models.predictor import UnstructuredShortPosePredictor


def observed_base(history, anchored):
    """Ground xy at the five actual observations, relative to the fifth."""
    if history.dtype != torch.float64 or history.shape[1:] != (5, 2):
        raise ValueError("Expected the five original float64 positions")
    delta = history[:, -1] - history[:, -2]
    times = DT * torch.arange(1, T_FUT + 1, device=history.device, dtype=history.dtype)
    if anchored:
        path = history[:, -1:] + (delta / DT)[:, None] * times[None, :, None]
    else:
        path = history.new_zeros((len(history), T_FUT, 2))
    return path, delta


class SixCandidates(nn.Module):
    def __init__(self, variant, anchored, seed=42):
        super().__init__()
        if variant not in ("M0", "M1"):
            raise ValueError("Only individual M0 and quality-gated M1 are registered")
        self.anchored = anchored
        self.variant = variant
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            shared = UnstructuredShortPosePredictor("M0")
            if variant == "M0":
                self.network = shared
            else:
                torch.manual_seed(seed)
                self.network = UnstructuredShortPosePredictor("M1")
                common = shared.state_dict()
                own = self.network.state_dict()
                if not common.keys() <= own.keys():
                    raise ValueError("Common M0 tensor is missing from M1")
                self.network.load_state_dict({k: common.get(k, value) for k, value in own.items()})
            nn.init.zeros_(self.network.decoder.delta_out.weight)
            nn.init.zeros_(self.network.decoder.delta_out.bias)

    def forward(self, history_xy, x_m, ego_type_oh, x_p=None, q_vec=None):
        base, last_step = observed_base(history_xy, self.anchored)
        batch = {"x_m": x_m, "ego_type_oh": ego_type_oh}
        if self.variant == "M1":
            batch.update(x_p=x_p, q_vec=q_vec)
        output = self.network(batch)
        # Keep the exact observed float64 base. The GRU and its parameters remain float32.
        paths = base[:, None] + output["trajs"].double()
        return paths, output["mode_logits"], base, last_step


@torch.no_grad()
def responsibilities(ade, target, base, last_step, valid):
    """For exact best-ADE ties, choose the closest motion-relative sector among tied modes."""
    tied = ade == ade.amin(dim=1, keepdim=True)
    residuals = target - base
    size = torch.hypot(residuals[..., 0], residuals[..., 1]).masked_fill(~valid, -1.)
    row = torch.arange(len(ade), device=ade.device)
    offset = residuals[row, size.argmax(1)]  # earliest maximum on valid future
    speed = torch.hypot(last_step[:, 0], last_step[:, 1])
    moving = speed > 0
    along = last_step / torch.where(moving, speed, 1.)[:, None]
    along = torch.where(moving[:, None], along, last_step.new_tensor([1., 0.]))
    left = torch.stack((-along[:, 1], along[:, 0]), dim=1)
    relative = torch.stack(((offset * along).sum(-1), (offset * left).sum(-1)), dim=1)
    s = math.sqrt(3.) / 2.
    directions = target.new_tensor([[1., 0.], [.5, s], [-.5, s],
                                    [-1., 0.], [-.5, -s], [.5, -s]])
    alignment = (relative @ directions.T).masked_fill(~tied, -torch.inf)
    selected = alignment == alignment.amax(1, keepdim=True)
    return selected.to(ade.dtype) / selected.sum(1, keepdim=True), tied.sum(1) > 1


def training_loss(paths, logits, base, last_step, target, mask):
    valid = mask.bool()
    if not valid.any(1).all():
        raise ValueError("Training windows must have valid future supervision")
    # Invalid steps cannot influence a tie through a NaN or an unobserved large excursion.
    target = torch.where(valid[..., None], target.double(), base)
    count = valid.sum(1)
    ade = (torch.linalg.vector_norm(paths.detach() - target[:, None], dim=-1) *
           valid[:, None]).sum(2) / count[:, None]
    responsibility, tied = responsibilities(ade, target, base, last_step, valid)
    reg = F.smooth_l1_loss(paths, target[:, None].expand_as(paths), beta=.5,
                           reduction="none").sum(-1)
    per_mode = (reg * valid[:, None]).sum(2) / count[:, None]
    regression = (per_mode * responsibility).sum(1).mean()
    hard = .96 * responsibility + .04 / paths.shape[1]
    soft = torch.softmax(-ade / .15, dim=1)
    class_target = (.65 * hard + .35 * soft).to(logits.dtype)
    classification = -(class_target * logits.log_softmax(1)).sum(1).mean()
    return regression + .25 * classification, responsibility, tied
