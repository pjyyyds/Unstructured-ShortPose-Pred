"""Wave 26: Coherence-Kalman velocity state filter & O(2) split-covisibility continuous-Slot0 + 6-slot model (EKR6)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bioeq_neural import KP_CONF_MIN


def compute_coherence_kalman_velocity(
    history: torch.Tensor,
    obs_dt: torch.Tensor | None = None,
    bbox_h: torch.Tensor | None = None,
    sigma_js: float = 0.045,
    w_chord: float = 0.65,
    gamma_neg: float = 0.45,
    jump_thr: float = 0.12,
    d_ref: float = 0.55,
) -> dict[str, torch.Tensor]:
    """O(2)-equivariant coherence-filtered & crop-jump-guarded initial velocity estimator."""
    bsz, t_obs, _ = history.shape
    h = history.double()
    dt = h.new_full((bsz, t_obs - 1), 0.1) if obs_dt is None else obs_dt.double().clamp_min(1e-4)
    diffs = h[:, 1:] - h[:, :-1]
    step_d = torch.linalg.vector_norm(diffs, dim=-1)
    path_len = step_d.sum(dim=-1)
    chord_vec = h[:, -1] - h[:, 0]
    chord_len = torch.linalg.vector_norm(chord_vec, dim=-1)
    v_last = diffs[:, -1] / dt[:, -1:]
    v_chord = chord_vec / dt.sum(dim=-1, keepdim=True).clamp_min(1e-4)

    if t_obs >= 3:
        kappa_dir = (chord_len / path_len.clamp_min(1e-5)).clamp(0.0, 1.0)
        d_last, d_prev = diffs[:, -1], diffs[:, -2]
        denom = (torch.linalg.vector_norm(d_last, dim=-1) * torch.linalg.vector_norm(d_prev, dim=-1)).clamp_min(1e-6)
        cos_last = ((d_last * d_prev).sum(dim=-1) / denom).clamp(-1.0, 1.0)
    else:
        kappa_dir = h.new_ones(bsz)
        cos_last = h.new_ones(bsz)

    jump_pen = h.new_zeros(bsz)
    if bbox_h is not None and t_obs >= 3:
        bh = bbox_h.double().clamp_min(1.0)
        rel_j = (bh[:, -1] - bh[:, -2]).abs() / torch.maximum(bh[:, -1], bh[:, -2]).clamp_min(10.0)
        jump_pen = ((rel_j - float(jump_thr)).clamp_min(0.0) / 0.18).clamp(0.0, 0.85)

    low_env = torch.exp(-((path_len / max(float(d_ref), 0.10)) ** 2))
    inc_raw = float(w_chord) * (1.0 - kappa_dir) + float(gamma_neg) * F.relu(-cos_last) + jump_pen
    w_inc = (low_env * inc_raw.clamp(0.0, 1.0))[:, None]
    s_js = 1.0 - low_env * ((float(sigma_js) ** 2) / (chord_len.square() + float(sigma_js) ** 2 + 1e-12))
    v_filt = s_js[:, None] * ((1.0 - w_inc) * v_last + w_inc * v_chord)
    return {
        "v_filt": v_filt,
        "v_last": v_last,
        "v_chord": v_chord,
        "path_len": path_len,
        "chord_len": chord_len,
        "kappa_dir": kappa_dir,
        "cos_last": cos_last,
        "low_env": low_env,
    }


def calibrate_coherence_kalman_params(
    hist_src: torch.Tensor,
    obs_dt_src: torch.Tensor,
    fut_s_src: torch.Tensor,
    truth_src: torch.Tensor,
    bbox_h_src: torch.Tensor | None = None,
    d_ref: float = 0.55,
) -> dict[str, float]:
    """Fit (sigma_js, w_chord, gamma_neg) strictly on source-fold windows (includes exact Last-2 CV = 0,0,0)."""
    with torch.inference_mode():
        p0 = hist_src[:, -1:, :].double()
        ft = fut_s_src.double()[:, :, None]
        gt = truth_src.double()
        pl = torch.linalg.vector_norm(hist_src[:, 1:] - hist_src[:, :-1], dim=-1).sum(dim=-1).double()
        w_bal = torch.where(pl < 0.50, torch.full_like(pl, 2.0), torch.ones_like(pl))

        best_cfg = {"sigma_js": 0.0, "w_chord": 0.0, "gamma_neg": 0.0, "d_ref": float(d_ref)}
        v0 = compute_coherence_kalman_velocity(hist_src, obs_dt_src, bbox_h_src, 0.0, 0.0, 0.0, d_ref=d_ref)["v_filt"]
        best_loss = float((torch.linalg.vector_norm(p0 + v0[:, None, :] * ft - gt, dim=-1).mean(dim=-1) * w_bal).mean())

        for s_js in (0.0, 0.025, 0.045, 0.070, 0.100):
            for w_ch in (0.0, 0.35, 0.65, 0.90):
                for g_ng in (0.0, 0.30, 0.60):
                    vf = compute_coherence_kalman_velocity(
                        hist_src, obs_dt_src, bbox_h_src, s_js, w_ch, g_ng, d_ref=d_ref
                    )["v_filt"]
                    err = torch.linalg.vector_norm(p0 + vf[:, None, :] * ft - gt, dim=-1).mean(dim=-1)
                    sc = float((err * w_bal).mean())
                    if sc + 1e-7 < best_loss:
                        best_loss = sc
                        best_cfg = {"sigma_js": float(s_js), "w_chord": float(w_ch), "gamma_neg": float(g_ng), "d_ref": float(d_ref)}
    return best_cfg


def compute_split_anatomical_covisibility(
    kpts: torch.Tensor | None,
    bbox: torch.Tensor | None,
    bio_even: torch.Tensor,
    pose_vec_world: torch.Tensor,
    signed_knee: torch.Tensor,
    q_vec: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute split Upper-Body vs. Lower-Body co-visibility & earliest-to-latest valid sub-span deltas."""
    bsz, t_p, _ = bio_even.shape
    dev = bio_even.device
    if kpts is None or bbox is None:
        q1 = (q_vec[:, 0] if (q_vec is not None and q_vec.ndim > 1) else (q_vec if q_vec is not None else torch.ones(bsz, device=dev))).clamp(0.0, 1.0).float()
        p_fin = (torch.linalg.vector_norm(pose_vec_world[:, -1, 0], dim=-1) > 1e-5).float()
        qu = (q1 * p_fin).clamp(0.0, 1.0)
        return {
            "q_upper": qu,
            "q_lower": qu,
            "d_bio5": (bio_even[:, -1, :5] - bio_even[:, 0, :5]).float() * qu[:, None],
            "d_fc": (pose_vec_world[:, -1, 0] - pose_vec_world[:, 0, 0]).double() * qu[:, None].double(),
            "d_fl": (pose_vec_world[:, -1, 1] - pose_vec_world[:, 0, 1]).double() * qu[:, None].double(),
            "d_knee": (signed_knee[:, -1, 0] - signed_knee[:, 0, 0]).float() * qu,
            "head_front": qu,
        }

    kp, bb = kpts.float(), bbox.float()
    f_ok = torch.isfinite(bb[..., 3]) & (bb[..., 3] > 1.0)
    good = torch.isfinite(kp).all(dim=-1) & (kp[..., 2] >= max(float(KP_CONF_MIN), 0.28)) & f_ok[..., None]
    tor_ok = good[:, :, 5] & good[:, :, 6] & good[:, :, 11] & good[:, :, 12]
    low_ok = good[:, :, 13] & good[:, :, 14] & good[:, :, 15] & good[:, :, 16]
    bh = bb[:, :, 3].clamp_min(1.0)
    bh_scale = ((bh[:, -1] - 35.0) / 30.0).clamp(0.0, 1.0)
    rel_j = (bh[:, -1] - bh[:, -2]).abs() / torch.maximum(bh[:, -1], bh[:, -2]).clamp_min(10.0)
    stable_last = (1.0 - (rel_j - 0.12).clamp_min(0.0) / 0.20).clamp(0.0, 1.0)

    # Earliest valid frame t_first in 0..t_p-2 paired with frame t_p-1 (handles mid-window emergence!)
    idx_grid = torch.arange(t_p, device=dev)[None, :]
    first_tor = torch.where(tor_ok, idx_grid, torch.full_like(idx_grid, t_p - 1)).amin(dim=1)
    first_low = torch.where(low_ok, idx_grid, torch.full_like(idx_grid, t_p - 1)).amin(dim=1)
    has_tor_pair = tor_ok[:, -1] & (first_tor < t_p - 1)
    has_low_pair = low_ok[:, -1] & (first_low < t_p - 1)
    bi = torch.arange(bsz, device=dev)
    dt_tor = ((t_p - 1 - first_tor).clamp_min(1).float() / float(max(t_p - 1, 1)))[:, None]
    dt_low = ((t_p - 1 - first_low).clamp_min(1).float() / float(max(t_p - 1, 1)))[:, None]

    q_upper = (tor_ok[:, -1].float() * (0.45 + 0.55 * tor_ok.float().mean(1)) * bh_scale).clamp(0.0, 1.0)
    q_lower = (low_ok[:, -1].float() * has_low_pair.float() * stable_last * bh_scale).clamp(0.0, 1.0)
    c_face = kp[:, -1, [0, 1, 2], 2].clamp(0.0, 1.0).mean(dim=-1)
    c_ears = kp[:, -1, [3, 4], 2].clamp(0.0, 1.0).mean(dim=-1)
    head_front = (c_face / (0.25 + c_face + c_ears)).clamp(0.0, 1.0) * f_ok[:, -1].float()

    d_fc = ((pose_vec_world[:, -1, 0] - pose_vec_world[bi, first_tor, 0]) / dt_tor.double()) * (has_tor_pair.float() * stable_last)[:, None].double()
    d_fl = ((pose_vec_world[:, -1, 1] - pose_vec_world[bi, first_tor, 1]) / dt_fl_safe(dt_tor)) * (has_tor_pair.float() * stable_last)[:, None].double()
    d_bio5 = ((bio_even[:, -1, :5] - bio_even[bi, first_low, :5]) / dt_low) * q_lower[:, None]
    d_knee = ((signed_knee[:, -1, 0] - signed_knee[bi, first_low, 0]) / dt_low[:, 0]) * q_lower
    return {
        "q_upper": q_upper,
        "q_lower": q_lower,
        "d_bio5": d_bio5.float(),
        "d_fc": d_fc,
        "d_fl": d_fl,
        "d_knee": d_knee.float(),
        "head_front": head_front,
    }


def dt_fl_safe(dt_tor: torch.Tensor) -> torch.Tensor:
    return dt_tor.double().clamp_min(0.25)


class O2EmergenceKalmanRes6SlotNet(nn.Module):
    """O(2)-equivariant predictor with Coherence-Kalman Slot-0 trust-region residual & decoupled 6-slot hypotheses."""

    def __init__(
        self,
        use_pose: bool = True,
        use_interaction: bool = True,
        kalman_cfg: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.use_pose = use_pose
        self.use_interaction = use_interaction
        self.kalman_cfg = kalman_cfg or {"sigma_js": 0.045, "w_chord": 0.65, "gamma_neg": 0.45, "d_ref": 0.55}

        self.motion = nn.GRU(5, 48, batch_first=True)
        self.coh_mlp = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, 48))
        self.ctx_proj = nn.Sequential(nn.LayerNorm(48), nn.Linear(48, 48), nn.SiLU())

        self.q_gamma_raw = nn.Parameter(torch.tensor(-0.20))
        self.q_w_raw = nn.Parameter(torch.tensor([0.55, 0.55, 0.30, 0.30]))
        self.q_bias = nn.Parameter(torch.tensor(-0.05))
        self.pose_in = nn.Linear(16, 24)
        self.pose_gru = nn.GRU(24, 24, batch_first=True)
        self.dpose_even_mlp = nn.Sequential(nn.Linear(11, 24), nn.SiLU(), nn.Linear(24, 24))
        self.dpose_odd_lin = nn.Linear(3, 24, bias=False)
        self.pose_proj = nn.Linear(24, 48, bias=False)
        nn.init.zeros_(self.pose_proj.weight)

        self.edge_mlp = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 48))
        self.edge_att = nn.Linear(96, 1)
        self.null_logit = nn.Parameter(torch.tensor(1.4))
        self.graph_proj = nn.Linear(48, 48, bias=False)
        nn.init.zeros_(self.graph_proj.weight)

        # Primary Slot-0 continuous bounded residual (3 knots in x_par, y_perp), exact zero-init!
        self.slot0_dx = nn.Linear(48, 3)
        self.slot0_dy = nn.Linear(48, 3, bias=False)
        nn.init.zeros_(self.slot0_dx.weight)
        nn.init.zeros_(self.slot0_dx.bias)
        nn.init.zeros_(self.slot0_dy.weight)
        self.pose_steer_gain = nn.Parameter(torch.tensor(0.0))

        # Decoupled counterfactual Slots 1..5
        self.v_walk_head = nn.Linear(48, 1)
        self.slot125_dx = nn.Linear(48, 3 * 3)
        self.slot125_dy = nn.Linear(48, 3 * 3, bias=False)
        self.turn_pair_knots = nn.Linear(48, 3 * 2)
        for m in (self.v_walk_head, self.slot125_dx, self.turn_pair_knots):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        nn.init.normal_(self.slot125_dy.weight, std=0.01)

        self.even_mode_logits = nn.Linear(48 + 24 + 5, 4)
        self.turn_mode_logit = nn.Linear(48 + 24 + 5, 1)
        self.act_even = nn.Linear(48 + 24 + 5, 3)
        self.act_turn = nn.Linear(48 + 24 + 5, 1)
        self.cv_hold_bias = nn.Parameter(torch.tensor(0.90))

    def _split_gates(self, q_vec: torch.Tensor, q_up: torch.Tensor, q_lo: torch.Tensor, m_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qc = q_vec.clamp(0.0, 1.0).float()
        if qc.ndim == 1:
            qc = qc[:, None].expand(-1, 4)
        elif qc.size(1) < 4:
            qc = F.pad(qc, (0, 4 - qc.size(1)), value=1.0)
        else:
            qc = qc[:, :4]
        gamma = F.softplus(self.q_gamma_raw) + 0.20
        base = m_pose.float() * ((qc[:, 0] * qc[:, 1]).clamp_min(1e-6) ** gamma) * torch.sigmoid(self.q_bias + (qc - 0.5) @ F.softplus(self.q_w_raw))
        return base * q_up.clamp(0.0, 1.0).float(), base * q_lo.clamp(0.0, 1.0).float()

    def forward(
        self,
        history: torch.Tensor,
        bio_even: torch.Tensor | None = None,
        pose_vec_world: torch.Tensor | None = None,
        signed_knee: torch.Tensor | None = None,
        q_vec: torch.Tensor | None = None,
        m_pose: torch.Tensor | None = None,
        split_covis: dict[str, torch.Tensor] | None = None,
        bbox_h: torch.Tensor | None = None,
        edges: torch.Tensor | None = None,
        is_car: torch.Tensor | None = None,
        nbr_mask: torch.Tensor | None = None,
        obs_dt: torch.Tensor | None = None,
        fut_times_s: torch.Tensor | None = None,
        slot0_res_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        from scripts.ppt_mainline_o2_models import local_frenet_with_pose

        bsz, t_obs, _ = history.shape
        use_p = self.use_pose and (m_pose is not None) and (pose_vec_world is not None) and (bio_even is not None)
        pvw_m = (pose_vec_world * m_pose[:, None, None, None].to(pose_vec_world.dtype)) if use_p else None
        origin, basis, valid = local_frenet_with_pose(history, pvw_m, use_pose_backup=use_p)
        body = (history - origin) @ basis
        dt_s = history.new_full((bsz, t_obs - 1), 0.1) if obs_dt is None else obs_dt.to(history.dtype).clamp_min(1e-4)
        vel = torch.cat((torch.zeros_like(body[:, :1]), (body[:, 1:] - body[:, :-1]) / dt_s[:, :, None]), dim=1)
        spd = torch.linalg.vector_norm(vel, dim=2, keepdim=True)

        kf = compute_coherence_kalman_velocity(
            history, dt_s, bbox_h=bbox_h,
            sigma_js=self.kalman_cfg.get("sigma_js", 0.045),
            w_chord=self.kalman_cfg.get("w_chord", 0.65),
            gamma_neg=self.kalman_cfg.get("gamma_neg", 0.45),
            d_ref=self.kalman_cfg.get("d_ref", 0.55),
        )
        v_filt_loc = torch.einsum("bk,bks->bs", kf["v_filt"], basis)
        coh_in = torch.stack((kf["path_len"], kf["chord_len"], kf["kappa_dir"], kf["cos_last"]), dim=-1).float()
        coh_emb = self.coh_mlp(coh_in)

        fp = torch.cat((body, vel, spd), dim=2).float()
        fn = fp * fp.new_tensor([1.0, -1.0, 1.0, -1.0, 1.0])
        _, hm = self.motion(torch.cat((fp, fn), dim=0))
        cb = self.ctx_proj(hm[0] + torch.cat((coh_emb, coh_emb), dim=0))
        c_pos, c_neg = cb[:bsz], cb[bsz:]

        disp_all = kf["path_len"].float()
        chord_len = kf["chord_len"]
        g_up = disp_all.new_zeros(bsz)
        g_lo = disp_all.new_zeros(bsz)
        hp_pos = disp_all.new_zeros((bsz, 24))
        hp_neg = disp_all.new_zeros((bsz, 24))
        fc_lat_odd = history.new_zeros(bsz)

        if use_p and signed_knee is not None:
            qv = q_vec if q_vec is not None else disp_all.new_zeros((bsz, 4))
            sc = split_covis or compute_split_anatomical_covisibility(None, None, bio_even, pose_vec_world, signed_knee, qv)
            g_up, g_lo = self._split_gates(qv, sc["q_upper"], sc["q_lower"], m_pose)
            pv_l = torch.einsum("btvk,bks->btvs", pose_vec_world.double(), basis).float()
            fc_p, fl_p = pv_l[:, :, 0], pv_l[:, :, 1]
            dfc_l = torch.einsum("bk,bks->bs", sc["d_fc"], basis).float()
            dfl_l = torch.einsum("bk,bks->bs", sc["d_fl"], basis).float()
            flip = fc_p.new_tensor([1.0, -1.0])
            pp = torch.cat((bio_even, fc_p, fl_p, signed_knee), dim=-1)
            pn = torch.cat((bio_even, fc_p * flip, fl_p * flip, -signed_knee), dim=-1)
            _, hpb = self.pose_gru(F.silu(self.pose_in(torch.cat((pp, pn), dim=0))))
            dp_even = torch.cat((sc["d_bio5"], bio_even[:, -1, 5:8], fc_p[:, -1, :1], dfl_l[:, :1], sc["head_front"][:, None]), dim=-1)
            dp_odd = torch.stack((fc_p[:, -1, 1], dfc_l[:, 1], sc["d_knee"]), dim=-1)
            de_v = self.dpose_even_mlp(dp_even)
            do_v = self.dpose_odd_lin(dp_odd)
            g_comb = (0.60 * g_up + 0.40 * g_lo)[:, None]
            hp_pos = hpb[0][:bsz] + de_v + do_v
            hp_neg = hpb[0][bsz:] + de_v - do_v
            c_pos = c_pos + g_comb * self.pose_proj(hp_pos)
            c_neg = c_neg + g_comb * self.pose_proj(hp_neg)
            fc_n = torch.linalg.vector_norm(pv_l[:, -1, 0].double(), dim=-1).clamp_min(1e-6)
            fc_lat_odd = g_up.double() * (pv_l[:, -1, 0, 1].double() / fc_n)

        risk_f = disp_all.new_zeros((bsz, 5))
        w_nbr_act = disp_all.new_zeros(bsz)
        if self.use_interaction and edges is not None and is_car is not None and nbr_mask is not None:
            dp_l = torch.einsum("bmk,bks->bms", edges[:, :, 0:2].double(), basis).float()
            dv_l = torch.einsum("bmk,bks->bms", edges[:, :, 2:4].double(), basis).float()
            dist_n, vrad = edges[:, :, 4].float(), edges[:, :, 5].float()
            inv = torch.stack((dist_n / 10.0, vrad, edges[:, :, 6].float(), is_car.float()), dim=-1)
            ep = torch.cat((dp_l, dv_l, inv), dim=-1)
            en = torch.cat((dp_l * ep.new_tensor([1.0, -1.0]), dv_l * ep.new_tensor([1.0, -1.0]), inv), dim=-1)
            w_app = (nbr_mask.float() * torch.sigmoid((vrad - 0.18) / 0.25) * torch.exp(-dist_n / 5.0)).amax(dim=1)
            w_gap = (nbr_mask.float() * is_car.float() * torch.sigmoid((-vrad - 0.20) / 0.30) * torch.exp(-dist_n / 4.5)).amax(dim=1)
            w_nbr_act = torch.maximum(w_app, 0.55 * w_gap)
            max_vr = (nbr_mask.float() * F.relu(vrad)).amax(dim=1)
            min_ttc_inv = (nbr_mask.float() * F.relu(vrad) / dist_n.clamp_min(0.5)).amax(dim=1)
            car_app = (nbr_mask.float() * is_car.float() * (vrad > 0.25).float()).amax(dim=1)
            risk_f = torch.stack((w_app, w_gap, max_vr / 3.0, min_ttc_inv, car_app), dim=-1)
            for s_sgn, c_stream, e_stream in ((1, c_pos, ep), (-1, c_neg, en)):
                emb = self.edge_mlp(e_stream)
                att = self.edge_att(torch.cat((c_stream[:, None, :].expand_as(emb), emb), dim=-1)).squeeze(-1)
                att = att.masked_fill(nbr_mask < 0.5, -1e9)
                w_a = torch.softmax(torch.cat((att, self.null_logit.expand(bsz, 1)), dim=1), dim=1)[:, :12]
                g_del = w_nbr_act[:, None] * self.graph_proj((w_a[:, :, None] * emb).sum(dim=1))
                if s_sgn == 1:
                    c_pos = c_pos + g_del
                else:
                    c_neg = c_neg + g_del

        c_even, c_odd = 0.5 * (c_pos + c_neg), 0.5 * (c_pos - c_neg)
        hp_even = 0.5 * (hp_pos + hp_neg)
        g_m = (0.60 * g_up + 0.40 * g_lo)[:, None]
        feat_even = torch.cat((c_even, g_m * hp_even, risk_f), dim=-1)
        feat_pos = torch.cat((c_pos, g_m * hp_pos, risk_f), dim=-1)
        feat_neg = torch.cat((c_neg, g_m * hp_neg, risk_f), dim=-1)

        ev_act = self.act_even(feat_even)
        act_logits = torch.cat((ev_act[:, :2], self.act_turn(feat_pos), self.act_turn(feat_neg), ev_act[:, 2:3]), dim=-1)
        ev_m = self.even_mode_logits(feat_even)
        low_env = kf["low_env"].float()
        m0_lg = ev_m[:, :1] + F.softplus(self.cv_hold_bias) * low_env[:, None]
        raw_lg = torch.cat((m0_lg, ev_m[:, 1:3], self.turn_mode_logit(feat_pos), self.turn_mode_logit(feat_neg), ev_m[:, 3:]), dim=1)
        logits = torch.where(valid[:, None], raw_lg, 0.0)

        t_sec = torch.arange(1, 31, dtype=history.dtype, device=history.device)[None, :] * 0.1 if fut_times_s is None else fut_times_s.to(history.dtype)
        t_max = t_sec[:, -1:].clamp_min(1.0)
        kn_idx = ((t_sec / t_max) * 3.0 - 1e-9).floor().long().clamp(0, 2)
        kn_ph = ((t_sec / t_max) * 3.0 - kn_idx.to(history.dtype))[:, None, :, None]
        bi = torch.arange(bsz, device=history.device)[:, None]

        # Trust-region continuous residual on Primary Slot 0 (zero-drift when stationary & isolated!)
        mob0 = torch.tanh((chord_len / 0.14) ** 2)
        act_env0 = torch.maximum(mob0, 0.70 * w_nbr_act.double())
        r_cap = (0.18 + 0.75 * torch.tanh(chord_len / 0.55)) * act_env0 * float(slot0_res_scale)
        k0_x = r_cap[:, None] * torch.tanh(self.slot0_dx(c_even).double())
        k0_y = r_cap[:, None] * torch.tanh(self.slot0_dy(c_odd).double() + torch.tanh(self.pose_steer_gain.double()) * fc_lat_odd[:, None])
        res0 = torch.cat((torch.zeros((bsz, 1, 2), dtype=torch.float64, device=history.device), torch.stack((k0_x, k0_y), dim=-1)), dim=1)
        kn0_ph = kn_ph.squeeze(1)
        delta0 = res0[bi, kn_idx] * (1.0 - kn0_ph) + res0[bi, kn_idx + 1] * kn0_ph
        anc0_base = v_filt_loc[:, None, :] * t_sec[:, :, None]
        slot0_loc = anc0_base + delta0

        # Decoupled counterfactual Slots 1..5 (detach trunk so rare maneuver amplitude never pulls Slot 0!)
        ce_d, co_d, cp_d, cn_d = c_even.detach(), c_odd.detach(), c_pos.detach(), c_neg.detach()
        s_dx = self.slot125_dx(ce_d).view(bsz, 3, 3)
        s_dy = self.slot125_dy(co_d).view(bsz, 3, 3)
        tp_p, tp_n = self.turn_pair_knots(cp_d).view(bsz, 3, 2), self.turn_pair_knots(cn_d).view(bsz, 3, 2)
        tp_m = torch.stack((tp_n[..., 0], -tp_n[..., 1]), dim=-1)
        res5 = torch.cat((torch.stack((s_dx[:, :2], s_dy[:, :2]), dim=-1), tp_p[:, None], tp_m[:, None], torch.stack((s_dx[:, 2:], s_dy[:, 2:]), dim=-1)), dim=1).double()
        res5 = torch.cat((torch.zeros_like(res5[:, :, :1]), res5), dim=2)
        mi = torch.arange(5, device=history.device)[None, :, None]
        delta5 = res5[bi[:, None], mi, kn_idx[:, None, :]] * (1.0 - kn_ph) + res5[bi[:, None], mi, (kn_idx + 1)[:, None, :]] * kn_ph
        mob5 = torch.tanh((chord_len / 0.25) ** 2)
        damp5 = torch.ones_like(delta5); damp5[:, 1:] = mob5[:, None, None, None]
        delta5 = delta5 * damp5

        v_walk = (0.95 + 0.35 * torch.tanh(self.v_walk_head(ce_d).squeeze(-1))).double()
        s_acc = (v_walk - v_filt_loc[:, 0]).clamp_min(0.45)[:, None] * (t_sec - 0.45 * (1.0 - torch.exp(-t_sec / 0.45)))
        anc1 = anc0_base + torch.stack((s_acc, 0.45 * fc_lat_odd.detach()[:, None] * s_acc), dim=-1)
        anc2 = v_filt_loc[:, None, :] * (0.55 * (1.0 - torch.exp(-t_sec[:, :, None] / 0.55)))
        v_st = torch.maximum(v_filt_loc[:, 0], 0.65 * mob5)[:, None]
        th_t = 0.60 * (t_sec / 1.8).clamp(0.0, 1.0)
        anc3 = torch.stack((v_st * t_sec * torch.cos(0.5 * th_t), v_st * t_sec * torch.sin(0.5 * th_t) + 0.5 * v_filt_loc[:, 1:2] * t_sec), dim=-1)
        anc4 = torch.stack((v_st * t_sec * torch.cos(0.5 * th_t), -v_st * t_sec * torch.sin(0.5 * th_t) + 0.5 * v_filt_loc[:, 1:2] * t_sec), dim=-1)
        anc5 = 0.55 * anc0_base
        paths_loc = torch.cat((slot0_loc[:, None], torch.stack((anc1, anc2, anc3, anc4, anc5), dim=1) + delta5), dim=1)
        paths_world = origin[:, None, :, :] + torch.einsum("bkts,bws->bktw", paths_loc, basis)
        return {"paths": paths_world, "logits": logits, "act_logits": act_logits, "valid": valid, "gate_upper": g_up, "gate_lower": g_lo}


def ekr6_decoupled_slot0_and_maneuver_loss(
    out: dict[str, torch.Tensor],
    truth: torch.Tensor,
    hist: torch.Tensor,
    w_sample: torch.Tensor,
    y_act: torch.Tensor,
    valid_act: torch.Tensor,
    pos_weight: torch.Tensor,
    act_weight: float = 0.30,
) -> torch.Tensor | None:
    """Supervise Primary Slot-0 directly with regime-balanced point loss + decoupled Slots 1..5 maneuver loss."""
    vm = out["valid"]
    if not torch.any(vm):
        return None
    p6 = out["paths"][vm]
    gt = truth[vm, None]
    ws = w_sample[vm].double()
    pl = torch.linalg.vector_norm(hist[vm, 1:] - hist[vm, :-1], dim=-1).sum(dim=-1).double()
    w_reg0 = torch.where(pl < 0.55, 1.8 * ws, ws)

    d = torch.linalg.vector_norm(p6 - gt, dim=-1)
    # Tighter 8cm Charbonnier transition on Primary Slot-0 for sub-decimeter <0.5m precision
    slot0_l = (torch.sqrt(d[:, 0].square() + 0.0064) - 0.08).mean(dim=-1)
    huber = (torch.sqrt(d.square() + 0.04) - 0.20).mean(dim=-1)
    ade = d.detach().mean(dim=-1)

    ya = (y_act[vm] & valid_act[vm]).bool()
    slot_reg = huber[:, 1:].amin(dim=1)
    slot_reg = torch.where(ya[:, 0], 0.65 * huber[:, 1] + 0.35 * slot_reg, slot_reg)
    slot_reg = torch.where(ya[:, 1], 0.65 * huber[:, 2] + 0.35 * slot_reg, slot_reg)
    slot_reg = torch.where(ya[:, 2], 0.65 * huber[:, 3] + 0.35 * slot_reg, slot_reg)
    slot_reg = torch.where(ya[:, 3], 0.65 * huber[:, 4] + 0.35 * slot_reg, slot_reg)
    slot_reg = torch.where(ya[:, 4] & (~ya[:, :4].any(dim=1)), 0.50 * huber[:, 5] + 0.50 * slot_reg, slot_reg)

    soft = (-ade / 0.16).softmax(dim=1).to(out["logits"].dtype)
    cls_l = -(soft * out["logits"][vm].log_softmax(dim=1)).sum(dim=1).double()
    bce = F.binary_cross_entropy_with_logits(out["act_logits"][vm], y_act[vm].float(), pos_weight=pos_weight, reduction="none")
    va_f = valid_act[vm].float()
    act_l = ((bce * va_f).sum(dim=1) / va_f.sum(dim=1).clamp_min(1.0)).double()

    l0 = (slot0_l * w_reg0).sum() / w_reg0.sum().clamp_min(1e-8)
    lm = ((0.55 * slot_reg + 0.20 * cls_l + float(act_weight) * act_l) * ws).sum() / ws.sum().clamp_min(1e-8)
    return l0 + lm


def fit_ekr6_source_oof_regime_selector(
    oof_kin_path: np.ndarray,
    oof_slot0_path: np.ndarray,
    oof_medoid_path: np.ndarray,
    oof_truth: np.ndarray,
    oof_path_len: np.ndarray,
) -> dict[str, float]:
    """Fit (alpha_low, alpha_high, disp_switch, use_medoid_high) strictly on source-only OOF predictions."""
    low_m = oof_path_len < 0.50
    hi_m = ~low_m

    best_al, best_el = 0.0, float(np.linalg.norm(oof_kin_path[low_m] - oof_truth[low_m], axis=-1).mean()) if low_m.any() else 0.0
    if low_m.any():
        for al in (0.0, 0.20, 0.40, 0.60, 0.80, 1.0):
            pr = (1.0 - al) * oof_kin_path[low_m] + al * oof_slot0_path[low_m]
            el = float(np.linalg.norm(pr - oof_truth[low_m], axis=-1).mean())
            if el + 1e-7 < best_el:
                best_el, best_al = el, float(al)

    best_ah, best_med, best_eh = 1.0, 0.0, float(np.linalg.norm(oof_slot0_path[hi_m] - oof_truth[hi_m], axis=-1).mean()) if hi_m.any() else 0.0
    if hi_m.any():
        for ah in (0.0, 0.35, 0.70, 1.0):
            for wm in (0.0, 0.50, 1.0):
                base_h = (1.0 - ah) * oof_kin_path[hi_m] + ah * oof_slot0_path[hi_m]
                pr = (1.0 - wm) * base_h + wm * oof_medoid_path[hi_m]
                eh = float(np.linalg.norm(pr - oof_truth[hi_m], axis=-1).mean())
                if eh + 1e-7 < best_eh:
                    best_eh, best_ah, best_med = eh, float(ah), float(wm)
    return {"alpha_low": best_al, "alpha_high": best_ah, "w_medoid_high": best_med, "disp_switch": 0.50}


def apply_ekr6_regime_selector(
    kin_path: np.ndarray,
    slot0_path: np.ndarray,
    medoid_path: np.ndarray,
    path_len: np.ndarray,
    cfg: dict[str, float],
) -> np.ndarray:
    low_m = (path_len < float(cfg.get("disp_switch", 0.50)))[:, None, None]
    al = float(cfg.get("alpha_low", 0.5))
    ah = float(cfg.get("alpha_high", 1.0))
    wm = float(cfg.get("w_medoid_high", 0.0))
    p_low = (1.0 - al) * kin_path + al * slot0_path
    p_hi_base = (1.0 - ah) * kin_path + ah * slot0_path
    p_hi = (1.0 - wm) * p_hi_base + wm * medoid_path
    return np.where(low_m, p_low, p_hi)
