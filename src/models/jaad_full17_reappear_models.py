"""Wave 26B: Full 17-joint Head/Torso/Arm/Ankle split RTMPose features, post-reappearance precursor net, & NP conformal."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.emergence_kalman_residual_6slot import compute_coherence_kalman_velocity


def kpts5_to_full17_split_autopose(k5: np.ndarray, h_ref: float) -> tuple[np.ndarray, np.ndarray]:
    """Extract [5, 20] co-visibility-imputed Head/Torso/Arm/Lower features + [4] split quality from COCO-17."""
    u = k5[:, :, 0].astype(np.float64)
    v = k5[:, :, 1].astype(np.float64)
    c = np.clip(k5[:, :, 2].astype(np.float64), 0.0, 1.0)
    inv_h = 1.0 / max(float(h_ref), 25.0)

    m_no = c[:, 0] >= 0.25
    m_head = m_no & ((c[:, 1:5] >= 0.20).any(axis=1))
    m_sh = (c[:, 5] >= 0.25) & (c[:, 6] >= 0.25)
    m_tor = m_sh & (c[:, 11] >= 0.25) & (c[:, 12] >= 0.25)
    m_wr = (c[:, 9] >= 0.22) & (c[:, 10] >= 0.22)
    m_el = m_sh & (c[:, 7] >= 0.22) & (c[:, 8] >= 0.22)
    m_low = (c[:, 13] >= 0.25) & (c[:, 14] >= 0.25) & (c[:, 15] >= 0.25) & (c[:, 16] >= 0.25)

    sh_mx, sh_my = 0.5 * (u[:, 5] + u[:, 6]), 0.5 * (v[:, 5] + v[:, 6])
    hp_mx, hp_my = 0.5 * (u[:, 11] + u[:, 12]), 0.5 * (v[:, 11] + v[:, 12])
    c_face = c[:, 0] + 0.5 * (c[:, 1] + c[:, 2])
    c_ears = 0.5 * (c[:, 3] + c[:, 4])
    face_front = (c_face / (0.25 + c_face + c_ears)) * m_no
    eye_span = np.abs(u[:, 1] - u[:, 2]) * inv_h * ((c[:, 1] >= 0.22) & (c[:, 2] >= 0.22))
    ear_mid = (c[:, 3] * u[:, 3] + c[:, 4] * u[:, 4]) / np.maximum(c[:, 3] + c[:, 4], 1e-4)
    head_yaw = np.clip((u[:, 0] - ear_mid) / max(0.18 * float(h_ref), 8.0), -2.0, 2.0) * m_head

    sgn_sh = (u[:, 5] - u[:, 6]) * inv_h * m_sh
    abs_sh = np.abs(u[:, 5] - u[:, 6]) * inv_h * m_sh
    sgn_hp = (u[:, 11] - u[:, 12]) * inv_h * m_tor
    twist = (sgn_sh - sgn_hp) * m_tor
    lean_x = (sh_mx - hp_mx) * inv_h * m_tor
    tor_ext = (hp_my - sh_my) * inv_h * m_tor
    no_dx = (u[:, 0] - sh_mx) * inv_h * (m_sh & m_no)
    no_dy = (sh_my - v[:, 0]) * inv_h * (m_sh & m_no)

    wr_span = np.abs(u[:, 9] - u[:, 10]) * inv_h * m_wr
    wr_min_v = np.minimum(np.where(c[:, 9] >= 0.22, v[:, 9], 1e9), np.where(c[:, 10] >= 0.22, v[:, 10], 1e9))
    wr_raise = np.where(wr_min_v < 1e8, np.maximum(0.0, (sh_my - wr_min_v) * inv_h), 0.0) * m_sh
    el_asym = ((u[:, 7] - u[:, 5]) - (u[:, 8] - u[:, 6])) * inv_h * m_el

    ank_abs = np.abs(u[:, 15] - u[:, 16]) * inv_h * m_low
    ank_sgn = (u[:, 15] - u[:, 16]) * inv_h * m_low
    foot_lift = (v[:, 16] - v[:, 15]) * inv_h * m_low
    kn_stride = ((u[:, 15] - u[:, 13]) - (u[:, 16] - u[:, 14])) * inv_h * m_low

    up14 = np.clip(np.stack([
        face_front, eye_span, head_yaw, sgn_sh, abs_sh, sgn_hp, twist,
        lean_x, tor_ext, no_dx, no_dy, wr_span, wr_raise, el_asym,
    ], axis=-1), -2.5, 2.5)
    lo4 = np.clip(np.stack([ank_abs, ank_sgn, foot_lift, kn_stride], axis=-1), -2.5, 2.5)

    # Co-visibility forward-fill of early occluded frames so temporal differences have zero truncation jump!
    vis_u = np.flatnonzero(m_sh)
    if len(vis_u) > 0:
        f_u = int(vis_u[0])
        for t in range(5):
            if not m_sh[t]:
                up14[t] = up14[f_u]
    else:
        up14[:] = 0.0
    vis_l = np.flatnonzero(m_low)
    if len(vis_l) > 0:
        f_l = int(vis_l[0])
        for t in range(5):
            if not m_low[t]:
                lo4[t] = lo4[f_l]
    else:
        lo4[:] = 0.0

    up_c = c[:, [0, 5, 6, 11, 12]].mean(axis=1, keepdims=True)
    lo_c = (c[:, [13, 14, 15, 16]].mean(axis=1) * m_low)[:, None]
    feat20 = np.concatenate([up14, lo4, up_c, lo_c], axis=-1).astype(np.float32)
    h_sc = float(np.clip((float(h_ref) - 35.0) / 45.0, 0.0, 1.0))
    q4 = np.array([
        float(m_head[-1]) * float(c[-1, [0, 1, 2, 3, 4]].mean()),
        float(m_tor[-1]) * float(m_sh.mean()),
        float(m_low[-1]) * float(len(vis_l) >= 2),
        h_sc,
    ], dtype=np.float32)
    return feat20, q4


class JAADPostReappearPrecursorNet(nn.Module):
    """Causal 5-frame precursor net with Split Upper/Lower pose gating & observable speed-regime conditioning."""

    def __init__(self, d_kin: int, d_pose: int, d_q: int, d_veh: int, use_pose: bool = True, use_veh: bool = False) -> None:
        super().__init__()
        self.use_pose = use_pose
        self.use_veh = use_veh
        self.d_pose = d_pose
        self.kin_gru = nn.GRU(d_kin, 40, batch_first=True)
        self.coh_mlp = nn.Sequential(nn.Linear(6, 20), nn.SiLU(), nn.Linear(20, 24))
        d_ctx = 40 + 24
        if use_pose:
            self.pose_gru = nn.GRU(d_pose, 32, batch_first=True)
            self.dpose_mlp = nn.Sequential(nn.Linear(2 * d_pose, 32), nn.SiLU(), nn.Linear(32, 32))
            self.q_w_raw = nn.Parameter(torch.full((1, d_q), 0.65))
            self.q_b = nn.Parameter(torch.tensor([-0.15]))
            d_ctx += 32
        if use_veh:
            self.veh_mlp = nn.Sequential(nn.Linear(d_veh, 16), nn.SiLU(), nn.Linear(16, 16))
            d_ctx += 16
        self.cls_head = nn.Sequential(nn.LayerNorm(d_ctx + 3), nn.Linear(d_ctx + 3, 48), nn.SiLU(), nn.Dropout(0.10), nn.Linear(48, 3))
        self.lead_head = nn.Sequential(nn.Linear(d_ctx + 3, 24), nn.SiLU(), nn.Linear(24, 3))

    def forward(self, kin5: torch.Tensor, pose5: torch.Tensor, q_vec: torch.Tensor, veh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, hk = self.kin_gru(kin5)
        xy = kin5[:, :, :2].double()
        kf = compute_coherence_kalman_velocity(xy, sigma_js=0.02, w_chord=0.60, gamma_neg=0.40, d_ref=0.25)
        spd_steps = kin5[:, 1:, 4]
        spd_med = spd_steps.median(dim=1).values
        spd_last = spd_steps[:, -1]
        coh6 = torch.stack((
            kf["path_len"].float(), kf["chord_len"].float(), kf["kappa_dir"].float(),
            kf["cos_last"].float(), spd_med, spd_last - spd_steps[:, 0],
        ), dim=-1)
        parts = [hk[0], self.coh_mlp(coh6)]
        if self.use_pose:
            _, hp = self.pose_gru(pose5)
            dp = torch.cat((pose5[:, -1] - pose5[:, 0], pose5[:, -1] - pose5[:, -2]), dim=-1)
            gq = torch.sigmoid(F.linear(q_vec.clamp(0.0, 1.5), F.softplus(self.q_w_raw), self.q_b))
            parts.append(gq * (hp[0] + self.dpose_mlp(dp)))
        if self.use_veh:
            parts.append(self.veh_mlp(veh))
        stand_env = torch.exp(-((spd_med / 0.45) ** 2))
        walk_env = torch.tanh((spd_med / 0.40) ** 2)
        reg3 = torch.stack((stand_env, walk_env, spd_med.clamp(0.0, 3.0)), dim=-1)
        ctx = torch.cat(parts + [reg3], dim=-1)
        return self.cls_head(ctx), F.softplus(self.lead_head(ctx))


def calibrate_conformal_np_thresholds(
    val_neg_scores: np.ndarray,
    target_fpr: float = 0.05,
) -> np.ndarray:
    """Exact finite-sample split-conformal Neyman-Pearson threshold: ceil((n + 1)*(1 - alpha))-th order statistic."""
    n, n_cls = val_neg_scores.shape
    k_ord = min(n - 1, max(0, int(math.ceil((n + 1) * (1.0 - float(target_fpr)))) - 1))
    thr = np.zeros(n_cls, dtype=np.float64)
    for c in range(n_cls):
        s_sorted = np.sort(val_neg_scores[:, c])
        thr[c] = float(s_sorted[k_ord]) + 1e-12
    return thr


class JAADCoherenceSplitTrajectoryNet(nn.Module):
    """5-frame JAAD trajectory net with Coherence-Kalman Slot-0 + zero-init Full-17 SplitPose & Vehicle residual."""

    def __init__(
        self,
        d_pose: int,
        d_q: int,
        use_pose: bool,
        use_veh: bool,
        kalman_cfg: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.use_pose = use_pose
        self.use_veh = use_veh
        self.kalman_cfg = kalman_cfg or {"sigma_js": 0.025, "w_chord": 0.65, "gamma_neg": 0.45, "d_ref": 0.25}

        self.mot_gru = nn.GRU(4, 48, batch_first=True)
        self.coh_mlp = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, 48))
        d_in = 48
        if use_pose:
            self.pose_gru = nn.GRU(d_pose, 32, batch_first=True)
            self.dpose_mlp = nn.Sequential(nn.Linear(2 * d_pose, 32), nn.SiLU(), nn.Linear(32, 32))
            self.q_w_raw = nn.Parameter(torch.full((1, d_q), 0.70))
            self.q_b = nn.Parameter(torch.tensor([-0.15]))
            d_in += 32
        if use_veh:
            self.veh_gru = nn.GRU(3, 16, batch_first=True)
            d_in += 16

        self.slot0_knots = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 32), nn.SiLU(), nn.Linear(32, 3 * 2))
        nn.init.zeros_(self.slot0_knots[-1].weight)
        nn.init.zeros_(self.slot0_knots[-1].bias)

        self.multi_res = nn.Linear(d_in, 3 * 20 * 2)
        nn.init.normal_(self.multi_res.weight, std=0.01)
        nn.init.zeros_(self.multi_res.bias)
        self.mode_lg = nn.Linear(d_in, 4)
        self.act_head = nn.Sequential(nn.Linear(d_in, 48), nn.SiLU(), nn.Linear(48, 4))

    def forward(self, hist: torch.Tensor, pose5: torch.Tensor, q_vec: torch.Tensor, veh5: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz = hist.size(0)
        vel = torch.cat((torch.zeros_like(hist[:, :1]), (hist[:, 1:] - hist[:, :-1]) / 0.1), dim=1)
        kf = compute_coherence_kalman_velocity(
            hist,
            sigma_js=self.kalman_cfg.get("sigma_js", 0.025),
            w_chord=self.kalman_cfg.get("w_chord", 0.65),
            gamma_neg=self.kalman_cfg.get("gamma_neg", 0.45),
            d_ref=self.kalman_cfg.get("d_ref", 0.25),
        )
        coh4 = torch.stack((kf["path_len"], kf["chord_len"], kf["kappa_dir"], kf["cos_last"]), dim=-1).float()
        _, hm = self.mot_gru(torch.cat((hist, vel), dim=-1).float())
        feats = [hm[0] + self.coh_mlp(coh4)]
        if self.use_pose:
            _, hp = self.pose_gru(pose5)
            dp = torch.cat((pose5[:, -1] - pose5[:, 0], pose5[:, -1] - pose5[:, -2]), dim=-1)
            gq = torch.sigmoid(F.linear(q_vec.clamp(0.0, 1.5), F.softplus(self.q_w_raw), self.q_b))
            feats.append(gq * (hp[0] + self.dpose_mlp(dp)))
        if self.use_veh:
            _, hv = self.veh_gru(veh5)
            feats.append(hv[0])
        ctx = torch.cat(feats, dim=-1)

        t_s = torch.arange(1, 21, dtype=hist.dtype, device=hist.device)[None, :] * 0.1
        kn_idx = ((t_s / 2.0) * 3.0 - 1e-9).floor().long().clamp(0, 2)
        kn_ph = ((t_s / 2.0) * 3.0 - kn_idx.to(hist.dtype))[:, :, None]
        bi = torch.arange(bsz, device=hist.device)[:, None]

        chord = kf["chord_len"]
        mob = torch.maximum(torch.tanh((chord / 0.08) ** 2), torch.full_like(chord, 0.25 if self.use_pose or self.use_veh else 0.15))
        r_cap = (0.14 + 0.55 * torch.tanh(chord / 0.35)) * mob
        k0 = r_cap[:, None, None] * torch.tanh(self.slot0_knots(ctx).double().view(bsz, 3, 2))
        res0 = torch.cat((torch.zeros((bsz, 1, 2), dtype=torch.float64, device=hist.device), k0), dim=1)
        delta0 = res0[bi, kn_idx] * (1.0 - kn_ph) + res0[bi, kn_idx + 1] * kn_ph

        base0 = kf["v_filt"][:, None, :] * t_s[:, :, None]
        slot0 = base0 + delta0
        m_res = self.multi_res(ctx.detach()).view(bsz, 3, 20, 2).double()
        stop_anc = kf["v_filt"][:, None, :] * (0.45 * (1.0 - torch.exp(-t_s[:, :, None] / 0.45)))
        p4 = torch.stack((slot0, base0 + m_res[:, 0], base0 + mob[:, None, None] * m_res[:, 1], stop_anc + 0.3 * m_res[:, 2]), dim=1)
        return {"paths": p4, "logits": self.mode_lg(ctx), "act_logits": self.act_head(ctx), "kin_path": base0}
