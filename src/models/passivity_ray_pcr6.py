"""Wave 29: Passivity-Constrained Speed-Proportional Viewing-Ray O(2) Model (PCR6)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ray_kalman_and_cusum import anisotropic_ray_kalman_filter
from src.models.tm_ray_rkr6 import compute_unitless_ray_gain_matrix


class O2PassivityRay6SlotNet(nn.Module):
    """O(2)-equivariant predictor with speed-proportional residual bound & longitudinal non-reversal passivity."""

    def __init__(
        self,
        use_pose: bool = True,
        use_interaction: bool = True,
        use_ray_aniso_gain: bool = True,
        v_gate_mps: float = 0.42,
        alpha_prop: float = 0.28,
        ramp_power: float = 1.50,
        ray_cfg: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.use_pose = use_pose
        self.use_interaction = use_interaction
        self.use_ray_aniso_gain = use_ray_aniso_gain
        self.v_gate_mps = float(v_gate_mps)
        self.alpha_prop = float(alpha_prop)
        self.ramp_power = float(ramp_power)
        self.ray_cfg = ray_cfg or {
            "sigma_u_px": 0.0, "sigma_v_px": 0.0, "kappa_dh": 0.0,
            "sigma_iso_m": 0.0, "q_acc": 1.5, "tau_eb": 0.0, "anchor_p_blend": 0.0, "d_ref": 0.55,
        }

        self.motion = nn.GRU(5, 48, batch_first=True)
        self.kin_aux_mlp = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, 48))
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

        self.slot0_dx = nn.Linear(48, 3)
        self.slot0_dy = nn.Linear(48, 3, bias=False)
        nn.init.zeros_(self.slot0_dx.weight)
        nn.init.zeros_(self.slot0_dx.bias)
        nn.init.zeros_(self.slot0_dy.weight)
        self.pose_steer_gain = nn.Parameter(torch.tensor(0.0))

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
        jac_uv_world: torch.Tensor | None = None,
        bbox_h: torch.Tensor | None = None,
        ank_conf: torch.Tensor | None = None,
        edges: torch.Tensor | None = None,
        is_car: torch.Tensor | None = None,
        nbr_mask: torch.Tensor | None = None,
        obs_dt: torch.Tensor | None = None,
        fut_times_s: torch.Tensor | None = None,
        slot0_res_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        from scripts.ppt_mainline_o2_models import local_frenet_with_pose
        from src.models.emergence_kalman_residual_6slot import compute_split_anatomical_covisibility

        bsz, t_obs, _ = history.shape
        use_p = self.use_pose and (m_pose is not None) and (pose_vec_world is not None) and (bio_even is not None)
        pvw_m = (pose_vec_world * m_pose[:, None, None, None].to(pose_vec_world.dtype)) if use_p else None
        origin, basis, valid = local_frenet_with_pose(history, pvw_m, use_pose_backup=use_p)
        body = (history - origin) @ basis
        dt_s = history.new_full((bsz, t_obs - 1), 0.1) if obs_dt is None else obs_dt.to(history.dtype).clamp_min(1e-4)
        vel = torch.cat((torch.zeros_like(body[:, :1]), (body[:, 1:] - body[:, :-1]) / dt_s[:, :, None]), dim=1)
        spd = torch.linalg.vector_norm(vel, dim=2, keepdim=True)

        kf = anisotropic_ray_kalman_filter(
            history, dt_s, jac_uv_world=jac_uv_world, bbox_h=bbox_h, ank_conf=ank_conf, **self.ray_cfg
        )
        dp0_loc = torch.einsum("bk,bks->bs", kf["p_filt"] - history[:, -1].double(), basis.double())
        v_filt_loc = torch.einsum("bk,bks->bs", kf["v_filt"], basis.double())
        v_last_w = (history[:, -1] - history[:, -2]).double() / dt_s[:, -1:].double()
        v_last_spd = torch.linalg.vector_norm(v_last_w, dim=-1)
        path_len = torch.linalg.vector_norm(history[:, 1:] - history[:, :-1], dim=-1).sum(dim=-1).double()
        chord_len = torch.linalg.vector_norm(history[:, -1] - history[:, 0], dim=-1).double()

        g_ray = compute_unitless_ray_gain_matrix(jac_uv_world if self.use_ray_aniso_gain else None, basis)
        kin_aux = torch.stack((path_len, chord_len, v_last_spd, g_ray[:, 0, 0]), dim=-1).float()
        aux_emb = self.kin_aux_mlp(kin_aux)

        fp = torch.cat((body, vel, spd), dim=2).float()
        fn = fp * fp.new_tensor([1.0, -1.0, 1.0, -1.0, 1.0])
        _, hm = self.motion(torch.cat((fp, fn), dim=0))
        cb = self.ctx_proj(hm[0] + torch.cat((aux_emb, aux_emb), dim=0))
        c_pos, c_neg = cb[:bsz], cb[bsz:]

        disp_all = path_len.float()
        g_up, g_lo = disp_all.new_zeros(bsz), disp_all.new_zeros(bsz)
        hp_pos, hp_neg = disp_all.new_zeros((bsz, 24)), disp_all.new_zeros((bsz, 24))
        fc_lat_odd = history.new_zeros(bsz, dtype=torch.float64)

        if use_p and signed_knee is not None:
            qv = q_vec if q_vec is not None else disp_all.new_zeros((bsz, 4))
            sc = split_covis or compute_split_anatomical_covisibility(None, None, bio_even, pose_vec_world, signed_knee, qv)
            g_up, g_lo = self._split_gates(qv, sc["q_upper"], sc["q_lower"], m_pose)
            pv_l = torch.einsum("btvk,bks->btvs", pose_vec_world.double(), basis.double()).float()
            fc_p, fl_p = pv_l[:, :, 0], pv_l[:, :, 1]
            dfc_l = torch.einsum("bk,bks->bs", sc["d_fc"], basis.double()).float()
            dfl_l = torch.einsum("bk,bks->bs", sc["d_fl"], basis.double()).float()
            flip = fc_p.new_tensor([1.0, -1.0])
            pp = torch.cat((bio_even, fc_p, fl_p, signed_knee), dim=-1)
            pn = torch.cat((bio_even, fc_p * flip, fl_p * flip, -signed_knee), dim=-1)
            _, hpb = self.pose_gru(F.silu(self.pose_in(torch.cat((pp, pn), dim=0))))
            dp_even = torch.cat((sc["d_bio5"], bio_even[:, -1, 5:8], fc_p[:, -1, :1], dfl_l[:, :1], sc["head_front"][:, None]), dim=-1)
            dp_odd = torch.stack((fc_p[:, -1, 1], dfc_l[:, 1], sc["d_knee"]), dim=-1)
            de_v, do_v = self.dpose_even_mlp(dp_even), self.dpose_odd_lin(dp_odd)
            g_comb = (0.60 * g_up + 0.40 * g_lo)[:, None]
            hp_pos, hp_neg = hpb[0][:bsz] + de_v + do_v, hpb[0][bsz:] + de_v - do_v
            c_pos = c_pos + g_comb * self.pose_proj(hp_pos)
            c_neg = c_neg + g_comb * self.pose_proj(hp_neg)
            fc_n = torch.linalg.vector_norm(pv_l[:, -1, 0].double(), dim=-1).clamp_min(1e-6)
            fc_lat_odd = g_up.double() * (pv_l[:, -1, 0, 1].double() / fc_n)

        risk_f = disp_all.new_zeros((bsz, 5))
        w_nbr_act = disp_all.new_zeros(bsz)
        if self.use_interaction and edges is not None and is_car is not None and nbr_mask is not None:
            dp_l = torch.einsum("bmk,bks->bms", edges[:, :, 0:2].double(), basis.double()).float()
            dv_l = torch.einsum("bmk,bks->bms", edges[:, :, 2:4].double(), basis.double()).float()
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

        t_sec = torch.arange(1, 31, dtype=torch.float64, device=history.device)[None, :] * 0.1 if fut_times_s is None else fut_times_s.double()
        t_max = t_sec[:, -1:].clamp_min(1.0)
        kn_idx = ((t_sec / t_max) * 3.0 - 1e-9).floor().long().clamp(0, 2)
        kn_ph = ((t_sec / t_max) * 3.0 - kn_idx.double())[:, None, :, None]
        bi = torch.arange(bsz, device=history.device)[:, None]

        # Speed-proportional bound (ZERO constant floor!) + longitudinal non-reversal passivity
        mob_last = torch.tanh((v_last_spd / max(self.v_gate_mps, 0.05)) ** 2)
        r_cap = (self.alpha_prop * v_last_spd.clamp_max(1.85) * t_max.squeeze(-1)) * mob_last * float(slot0_res_scale)
        k0_x = torch.tanh(self.slot0_dx(c_even).double())
        k0_y = torch.tanh(self.slot0_dy(c_odd).double() + torch.tanh(self.pose_steer_gain.double()) * fc_lat_odd[:, None])
        raw_res0 = torch.stack((k0_x, k0_y), dim=-1)
        ray_res0 = torch.einsum("bij,bkj->bki", g_ray, raw_res0) * r_cap[:, None, None]
        res0 = torch.cat((torch.zeros((bsz, 1, 2), dtype=torch.float64, device=history.device), ray_res0), dim=1)
        kn0_ph = kn_ph.squeeze(1)
        ramp_t = ((t_sec / t_max).clamp(0.0, 1.0) ** self.ramp_power)[:, :, None]
        delta0 = (res0[bi, kn_idx] * (1.0 - kn0_ph) + res0[bi, kn_idx + 1] * kn0_ph) * ramp_t
        anc0_base = dp0_loc[:, None, :] + v_filt_loc[:, None, :] * t_sec[:, :, None]
        s0_raw = anc0_base + delta0
        s0_x = torch.where(anc0_base[:, :, 0] >= 0.0, s0_raw[:, :, 0].clamp_min(0.0), s0_raw[:, :, 0])
        slot0_loc = torch.stack((s0_x, s0_raw[:, :, 1]), dim=-1)

        ce_d, co_d, cp_d, cn_d = c_even.detach(), c_odd.detach(), c_pos.detach(), c_neg.detach()
        s_dx, s_dy = self.slot125_dx(ce_d).view(bsz, 3, 3), self.slot125_dy(co_d).view(bsz, 3, 3)
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
        anc2 = dp0_loc[:, None, :] + v_filt_loc[:, None, :] * (0.55 * (1.0 - torch.exp(-t_sec[:, :, None] / 0.55)))
        v_st = torch.maximum(v_filt_loc[:, 0], 0.65 * mob5)[:, None]
        th_t = 0.60 * (t_sec / 1.8).clamp(0.0, 1.0)
        anc3 = dp0_loc[:, None, :] + torch.stack((v_st * t_sec * torch.cos(0.5 * th_t), v_st * t_sec * torch.sin(0.5 * th_t) + 0.5 * v_filt_loc[:, 1:2] * t_sec), dim=-1)
        anc4 = dp0_loc[:, None, :] + torch.stack((v_st * t_sec * torch.cos(0.5 * th_t), -v_st * t_sec * torch.sin(0.5 * th_t) + 0.5 * v_filt_loc[:, 1:2] * t_sec), dim=-1)
        anc5 = dp0_loc[:, None, :] + 0.55 * (v_filt_loc[:, None, :] * t_sec[:, :, None])
        paths_loc = torch.cat((slot0_loc[:, None], torch.stack((anc1, anc2, anc3, anc4, anc5), dim=1) + delta5), dim=1)
        paths_world = origin.double()[:, None, :, :] + torch.einsum("bkts,bws->bktw", paths_loc, basis.double())
        return {"paths": paths_world, "logits": logits, "act_logits": act_logits, "valid": valid, "gate_upper": g_up, "gate_lower": g_lo}



def fit_site_balanced_pcr6_oof_selector(
    oof_kin_path: np.ndarray,
    oof_slot0_path: np.ndarray,
    oof_medoid_path: np.ndarray,
    oof_truth: np.ndarray,
    oof_path_len: np.ndarray,
    oof_grp: np.ndarray,
) -> dict[str, float]:
    """Fit (alpha_low, alpha_high, w_medoid_high) strictly on source-only OOF balancing site-equal & pooled ADE."""
    low_m = oof_path_len < 0.50
    hi_m = ~low_m
    u_g = sorted(set(oof_grp.tolist()))

    def _obj(pred_sub: np.ndarray, gt_sub: np.ndarray, g_sub: np.ndarray) -> float:
        ade = np.linalg.norm(pred_sub - gt_sub, axis=-1).mean(axis=-1)
        g_means = [float(ade[g_sub == g].mean()) for g in u_g if np.any(g_sub == g)]
        return 0.5 * float(np.mean(g_means)) + 0.5 * float(ade.mean())

    best_al = 0.0
    best_el = _obj(oof_kin_path[low_m], oof_truth[low_m], oof_grp[low_m]) if low_m.any() else 0.0
    if low_m.any():
        for al in (0.0, 0.15, 0.30, 0.50, 0.75, 1.0):
            pr = (1.0 - al) * oof_kin_path[low_m] + al * oof_slot0_path[low_m]
            el = _obj(pr, oof_truth[low_m], oof_grp[low_m])
            if el + 1e-7 < best_el:
                best_el, best_al = el, float(al)

    best_ah, best_med = 1.0, 0.0
    best_eh = _obj(oof_slot0_path[hi_m], oof_truth[hi_m], oof_grp[hi_m]) if hi_m.any() else 0.0
    if hi_m.any():
        for ah in (0.0, 0.35, 0.70, 1.0):
            for wm in (0.0, 0.50, 1.0):
                base_h = (1.0 - ah) * oof_kin_path[hi_m] + ah * oof_slot0_path[hi_m]
                pr = (1.0 - wm) * base_h + wm * oof_medoid_path[hi_m]
                eh = _obj(pr, oof_truth[hi_m], oof_grp[hi_m])
                if eh + 1e-7 < best_eh:
                    best_eh, best_ah, best_med = eh, float(ah), float(wm)
    return {"alpha_low": best_al, "alpha_high": best_ah, "w_medoid_high": best_med, "disp_switch": 0.50}
