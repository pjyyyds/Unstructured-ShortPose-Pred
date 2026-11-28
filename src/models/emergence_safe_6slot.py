"""Emergence-safe O(2) 6-slot model with co-visible delta-pose masking & stationary residual damping."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bioeq_neural import KP_CONF_MIN


def compute_pose_covisibility_features(
    kpts: torch.Tensor,
    bbox: torch.Tensor,
    bio_even: torch.Tensor,
    pose_vec_world: torch.Tensor,
    signed_knee: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute 5-frame co-visibility stability q_stab and truncation-masked temporal pose deltas."""
    kp = kpts.float()
    bb = bbox.float()
    f_ok = torch.isfinite(bb[..., 3]) & (bb[..., 3] > 1.0)
    good = torch.isfinite(kp).all(dim=-1) & (kp[..., 2] >= max(float(KP_CONF_MIN), 0.30)) & f_ok[..., None]

    torso_ok = good[:, :, 5] & good[:, :, 6] & good[:, :, 11] & good[:, :, 12]
    lower_ok = good[:, :, 13] & good[:, :, 14] & good[:, :, 15] & good[:, :, 16]
    bh = bb[:, :, 3].clamp_min(1.0)
    # Detect rapid bbox height jump during de-occlusion (e.g. half-body -> full-body crop growth)
    bh_rel_jump = (bh[:, -1] - bh[:, 0]).abs() / torch.maximum(bh[:, -1], bh[:, 0]).clamp_min(10.0)
    stable_crop = (1.0 - (bh_rel_jump - 0.12).clamp_min(0.0) / 0.25).clamp(0.0, 1.0)
    bh_scale = ((bh.median(dim=1).values - 35.0) / 35.0).clamp(0.0, 1.0)

    torso_covis = (torso_ok[:, 0] & torso_ok[:, -1]).float()
    lower_covis = (lower_ok[:, 0] & lower_ok[:, -1]).float()
    torso_ratio = torso_ok.float().mean(dim=1)
    q_stab = (torso_ratio * (0.45 + 0.55 * torso_covis) * stable_crop * bh_scale).clamp(0.0, 1.0)

    # Co-visible delta weights: zero out finite differences when frame 0 or T-1 had occluded limbs
    w_torso = (torso_covis * stable_crop)[:, None]
    w_lower = (lower_covis * stable_crop)[:, None]
    d_bio5 = (bio_even[:, -1, :5] - bio_even[:, 0, :5]) * w_lower
    d_knee = (signed_knee[:, -1, 0] - signed_knee[:, 0, 0]) * lower_covis * stable_crop
    return {
        "q_stab": q_stab,
        "w_torso": w_torso,
        "w_lower": w_lower,
        "d_bio5": d_bio5,
        "d_knee": d_knee,
        "torso_last_ok": torso_ok[:, -1].float(),
    }


class O2EmergenceSafe6SlotNet(nn.Module):
    """O(2)-equivariant 6-slot predictor preventing partial-emergence pose & stationary-residual drift."""

    def __init__(self, use_pose: bool = True, use_interaction: bool = True, low_disp_shrink: float = 0.85) -> None:
        super().__init__()
        self.use_pose = use_pose
        self.use_interaction = use_interaction
        self.low_disp_shrink = float(low_disp_shrink)

        self.motion = nn.GRU(5, 48, batch_first=True)
        self.ctx_proj = nn.Sequential(nn.LayerNorm(48), nn.Linear(48, 48), nn.SiLU())

        self.q_gamma_raw = nn.Parameter(torch.tensor(-0.20))
        self.q_w_raw = nn.Parameter(torch.tensor([0.6, 0.6, 0.3, 0.3]))
        self.q_bias = nn.Parameter(torch.tensor(-0.10))
        self.pose_in = nn.Linear(16, 24)
        self.pose_gru = nn.GRU(24, 24, batch_first=True)
        self.dpose_even_mlp = nn.Sequential(nn.Linear(10, 24), nn.SiLU(), nn.Linear(24, 24))
        self.dpose_odd_lin = nn.Linear(3, 24, bias=False)
        self.pose_proj = nn.Linear(24, 48, bias=False)
        nn.init.zeros_(self.pose_proj.weight)

        self.edge_mlp = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 48))
        self.edge_att = nn.Linear(96, 1)
        self.null_logit = nn.Parameter(torch.tensor(1.5))
        self.graph_proj = nn.Linear(48, 48, bias=False)
        nn.init.zeros_(self.graph_proj.weight)

        self.v_walk_head = nn.Linear(48, 1)
        self.slot125_dx = nn.Linear(48, 3 * 3)
        self.slot125_dy = nn.Linear(48, 3 * 3, bias=False)
        self.turn_pair_knots = nn.Linear(48, 3 * 2)
        for m in (self.v_walk_head, self.slot125_dx, self.turn_pair_knots):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        nn.init.normal_(self.slot125_dy.weight, std=0.01)

        self.even_mode_logits = nn.Linear(48 + 24 + 4, 4)
        self.turn_mode_logit = nn.Linear(48 + 24 + 4, 1)
        self.act_even = nn.Linear(48 + 24 + 4, 3)
        self.act_turn = nn.Linear(48 + 24 + 4, 1)
        self.cv_hold_bias = nn.Parameter(torch.tensor(0.85))

    def _safe_gate(self, q_vec: torch.Tensor, q_stab: torch.Tensor, m_pose: torch.Tensor) -> torch.Tensor:
        qc = q_vec.clamp(0.0, 1.0).float()
        if qc.ndim == 1:
            qc = qc[:, None].expand(-1, 4)
        elif qc.size(1) < 4:
            qc = F.pad(qc, (0, 4 - qc.size(1)), value=1.0)
        else:
            qc = qc[:, :4]
        gamma = F.softplus(self.q_gamma_raw) + 0.20
        w_pos = F.softplus(self.q_w_raw)
        env = ((qc[:, 0] * qc[:, 1]).clamp_min(1e-6)) ** gamma
        return m_pose.float() * q_stab.clamp(0.0, 1.0).float() * env * torch.sigmoid(self.q_bias + (qc - 0.5) @ w_pos)

    def forward(
        self,
        history: torch.Tensor,
        bio_even: torch.Tensor | None = None,
        pose_vec_world: torch.Tensor | None = None,
        signed_knee: torch.Tensor | None = None,
        q_vec: torch.Tensor | None = None,
        m_pose: torch.Tensor | None = None,
        covis: dict[str, torch.Tensor] | None = None,
        edges: torch.Tensor | None = None,
        is_car: torch.Tensor | None = None,
        nbr_mask: torch.Tensor | None = None,
        obs_dt: torch.Tensor | None = None,
        fut_times_s: torch.Tensor | None = None,
        conflict_scale: float = 1.0,
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
        fp = torch.cat((body, vel, spd), dim=2).float()
        fn = fp * fp.new_tensor([1.0, -1.0, 1.0, -1.0, 1.0])
        _, hm = self.motion(torch.cat((fp, fn), dim=0))
        cb = self.ctx_proj(hm[0])
        c_pos, c_neg = cb[:bsz], cb[bsz:]

        disp_all = torch.linalg.vector_norm(history[:, 1:] - history[:, :-1], dim=-1).sum(dim=-1).float()
        v_loc = vel[:, -1]
        v_last = torch.linalg.vector_norm(v_loc, dim=-1).float()
        gate = v_last.new_zeros(bsz)
        hp_pos = v_last.new_zeros((bsz, 24))
        hp_neg = v_last.new_zeros((bsz, 24))
        fc_lat_odd = history.new_zeros(bsz)

        if use_p and signed_knee is not None:
            qv = q_vec if q_vec is not None else v_last.new_zeros((bsz, 4))
            q_st = covis["q_stab"] if covis is not None else v_last.new_ones(bsz)
            w_tor = covis["w_torso"] if covis is not None else v_last.new_ones((bsz, 1))
            d_bio = covis["d_bio5"] if covis is not None else (bio_even[:, -1, :5] - bio_even[:, 0, :5])
            d_kn = covis["d_knee"] if covis is not None else (signed_knee[:, -1, 0] - signed_knee[:, 0, 0])
            gate = self._safe_gate(qv, q_st, m_pose)
            pv_l = torch.einsum("btvk,bks->btvs", pose_vec_world.double(), basis).float()
            fc_p, fl_p = pv_l[:, :, 0], pv_l[:, :, 1]
            flip = fc_p.new_tensor([1.0, -1.0])
            pp = torch.cat((bio_even, fc_p, fl_p, signed_knee), dim=-1)
            pn = torch.cat((bio_even, fc_p * flip, fl_p * flip, -signed_knee), dim=-1)
            _, hpb = self.pose_gru(F.silu(self.pose_in(torch.cat((pp, pn), dim=0))))
            d_fc = (fc_p[:, -1] - fc_p[:, 0]) * w_tor
            d_fl = (fl_p[:, -1] - fl_p[:, 0]) * w_tor
            dp_even = torch.cat((d_bio, bio_even[:, -1, 5:8], fc_p[:, -1, :1], d_fl[:, :1]), dim=-1)
            dp_odd = torch.stack((fc_p[:, -1, 1], d_fc[:, 1], d_kn), dim=-1)
            de_v, do_v = self.dpose_even_mlp(dp_even), self.dpose_odd_lin(dp_odd)
            hp_pos = hpb[0][:bsz] + de_v + do_v
            hp_neg = hpb[0][bsz:] + de_v - do_v
            c_pos = c_pos + gate[:, None] * self.pose_proj(hp_pos)
            c_neg = c_neg + gate[:, None] * self.pose_proj(hp_neg)
            fc_n = torch.linalg.vector_norm(pv_l[:, -1, 0].double(), dim=-1).clamp_min(1e-6)
            fc_lat_odd = gate.double() * (pv_l[:, -1, 0, 1].double() / fc_n) * torch.exp(-((disp_all.double() / 0.25) ** 2))

        risk_f = v_last.new_zeros((bsz, 4))
        if self.use_interaction and edges is not None and is_car is not None and nbr_mask is not None:
            dp_l = torch.einsum("bmk,bks->bms", edges[:, :, 0:2].double(), basis).float()
            dv_l = torch.einsum("bmk,bks->bms", edges[:, :, 2:4].double(), basis).float()
            dist_n, vrad = edges[:, :, 4].float(), edges[:, :, 5].float()
            inv = torch.stack((dist_n / 10.0, vrad, edges[:, :, 6].float(), is_car.float()), dim=-1)
            ep = torch.cat((dp_l, dv_l, inv), dim=-1)
            en = torch.cat((dp_l * ep.new_tensor([1.0, -1.0]), dv_l * ep.new_tensor([1.0, -1.0]), inv), dim=-1)
            w_r = (nbr_mask.float() * torch.sigmoid((vrad - 0.20) / 0.25) * torch.exp(-dist_n / 5.0)).amax(dim=1)
            max_vr = (nbr_mask.float() * F.relu(vrad)).amax(dim=1)
            min_ttc_inv = (nbr_mask.float() * F.relu(vrad) / dist_n.clamp_min(0.5)).amax(dim=1)
            car_app = (nbr_mask.float() * is_car.float() * (vrad > 0.25).float()).amax(dim=1)
            risk_f = torch.stack((w_r, max_vr / 3.0, min_ttc_inv, car_app), dim=-1)
            for s_sgn, c_stream, e_stream in ((1, c_pos, ep), (-1, c_neg, en)):
                emb = self.edge_mlp(e_stream)
                att = self.edge_att(torch.cat((c_stream[:, None, :].expand_as(emb), emb), dim=-1)).squeeze(-1)
                att = att.masked_fill(nbr_mask < 0.5, -1e9)
                w_a = torch.softmax(torch.cat((att, self.null_logit.expand(bsz, 1)), dim=1), dim=1)[:, :12]
                g_del = float(conflict_scale) * w_r[:, None] * self.graph_proj((w_a[:, :, None] * emb).sum(dim=1))
                if s_sgn == 1:
                    c_pos = c_pos + g_del
                else:
                    c_neg = c_neg + g_del

        c_even, c_odd = 0.5 * (c_pos + c_neg), 0.5 * (c_pos - c_neg)
        hp_even = 0.5 * (hp_pos + hp_neg)
        feat_even = torch.cat((c_even, gate[:, None] * hp_even, risk_f), dim=-1)
        feat_pos = torch.cat((c_pos, gate[:, None] * hp_pos, risk_f), dim=-1)
        feat_neg = torch.cat((c_neg, gate[:, None] * hp_neg, risk_f), dim=-1)

        ev_act = self.act_even(feat_even)
        act_logits = torch.cat((ev_act[:, :2], self.act_turn(feat_pos), self.act_turn(feat_neg), ev_act[:, 2:3]), dim=-1)

        ev_m = self.even_mode_logits(feat_even)
        low_env = torch.exp(-((disp_all / 0.35) ** 2))
        m0_lg = ev_m[:, :1] + F.softplus(self.cv_hold_bias) * low_env[:, None]
        raw_lg = torch.cat((m0_lg, ev_m[:, 1:3], self.turn_mode_logit(feat_pos), self.turn_mode_logit(feat_neg), ev_m[:, 3:]), dim=1)
        logits = torch.where(valid[:, None], raw_lg, 0.0)

        t_sec = torch.arange(1, 31, dtype=history.dtype, device=history.device)[None, :] * 0.1 if fut_times_s is None else fut_times_s.to(history.dtype)
        t_max = t_sec[:, -1:].clamp_min(1.0)
        kn_idx = ((t_sec / t_max) * 3.0 - 1e-9).floor().long().clamp(0, 2)
        kn_ph = ((t_sec / t_max) * 3.0 - kn_idx.to(history.dtype))[:, None, :, None]

        s_dx = self.slot125_dx(c_even).view(bsz, 3, 3)
        s_dy = self.slot125_dy(c_odd).view(bsz, 3, 3)
        tp_p = self.turn_pair_knots(c_pos).view(bsz, 3, 2)
        tp_n = self.turn_pair_knots(c_neg).view(bsz, 3, 2)
        tp_m = torch.stack((tp_n[..., 0], -tp_n[..., 1]), dim=-1)
        res5 = torch.cat((torch.stack((s_dx[:, :2], s_dy[:, :2]), dim=-1), tp_p[:, None], tp_m[:, None], torch.stack((s_dx[:, 2:], s_dy[:, 2:]), dim=-1)), dim=1).double()
        res5 = torch.cat((torch.zeros_like(res5[:, :, :1]), res5), dim=2)
        bi = torch.arange(bsz, device=history.device)[:, None, None]
        mi = torch.arange(5, device=history.device)[None, :, None]
        delta5 = res5[bi, mi, kn_idx[:, None, :]] * (1.0 - kn_ph) + res5[bi, mi, (kn_idx + 1)[:, None, :]] * kn_ph

        # Stationary-safe envelope: Slot 1 (index 0 of delta5) keeps full Start initiation; Slots 2,3,4,5 scale with mobility
        mob_env = torch.tanh((disp_all.double() / 0.30) ** 2)
        damp_mask = torch.ones_like(delta5)
        damp_mask[:, 1:, :, :] = mob_env[:, None, None, None]
        delta5 = delta5 * damp_mask

        shrink_v = 1.0 - (1.0 - self.low_disp_shrink) * low_env.double()
        anc0 = (shrink_v[:, None] * v_loc)[:, None, :] * t_sec[:, :, None]
        v_walk = (0.95 + 0.35 * torch.tanh(self.v_walk_head(c_even).squeeze(-1))).double()
        dv_acc = (v_walk - v_loc[:, 0]).clamp_min(0.45)[:, None]
        s_acc = dv_acc * (t_sec - 0.45 * (1.0 - torch.exp(-t_sec / 0.45)))
        anc1 = anc0 + torch.stack((s_acc, 0.45 * fc_lat_odd[:, None] * s_acc), dim=-1)
        anc2 = v_loc[:, None, :] * (0.55 * (1.0 - torch.exp(-t_sec[:, :, None] / 0.55)))
        v_st = torch.maximum(v_loc[:, 0], 0.65 * mob_env)[:, None]
        th_t = 0.60 * (t_sec / 1.8).clamp(0.0, 1.0)
        st_x = v_st * t_sec * torch.cos(0.5 * th_t)
        st_y = v_st * t_sec * torch.sin(0.5 * th_t)
        anc3 = torch.stack((st_x, st_y + 0.5 * v_loc[:, 1:2] * t_sec), dim=-1)
        anc4 = torch.stack((st_x, -st_y + 0.5 * v_loc[:, 1:2] * t_sec), dim=-1)
        anc5 = 0.55 * anc0
        anc_1to5 = torch.stack((anc1, anc2, anc3, anc4, anc5), dim=1) + delta5
        paths_loc = torch.cat((anc0[:, None], anc_1to5), dim=1)
        paths_world = origin[:, None, :, :] + torch.einsum("bkts,bws->bktw", paths_loc, basis)
        return {"paths": paths_world, "logits": logits, "act_logits": act_logits, "valid": valid, "gate": gate}


def calibrate_source_low_disp_shrink(
    history_np: np.ndarray,
    truth_np: np.ndarray,
    tr_ids: np.ndarray,
    dt_obs: float = 0.1,
    fut_times_s: np.ndarray | None = None,
) -> float:
    """Select low-displacement CV shrinkage strictly on source-site training windows (< 0.5 m)."""
    h_tr = history_np[tr_ids]
    y_tr = truth_np[tr_ids]
    disp = np.linalg.norm(np.diff(h_tr, axis=1), axis=-1).sum(axis=1)
    low_env = np.exp(-((disp / 0.35) ** 2))
    v_last = (h_tr[:, -1] - h_tr[:, -2]) / max(dt_obs, 1e-4)
    ts = (np.arange(1, y_tr.shape[1] + 1, dtype=np.float64) * 0.1)[None, :] if fut_times_s is None else fut_times_s[tr_ids]
    sub = disp < 0.50
    if int(sub.sum()) < 8:
        return 1.0
    best_a, best_e = 1.0, float("inf")
    for a in (0.65, 0.75, 0.85, 0.92, 1.00):
        sh = 1.0 - (1.0 - a) * low_env[sub]
        pred = h_tr[sub, -1:, :] + (sh[:, None] * v_last[sub])[:, None, :] * ts[sub, :, None]
        err = float(np.linalg.norm(pred - y_tr[sub], axis=-1).mean())
        if err < best_e:
            best_e, best_a = err, float(a)
    return best_a
