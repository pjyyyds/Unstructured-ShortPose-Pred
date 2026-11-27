"""Explicit 6-slot maneuver-hypothesis O(2) generator, onset scorer, and 6-candidate diagnostic."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.actions import _estimate_causal_velocities, _first_sustained_run, _max_consecutive_zeros


def compute_action_labels_with_onset(
    hist_pos: np.ndarray,
    hist_mask: np.ndarray,
    fut_pos: np.ndarray,
    fut_mask: np.ndarray,
    dt: float = 0.1,
    speed_scale: float = 1.0,
    angle_scale: float = 1.0,
    min_run: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_act[5], valid_cls[5], onset_step[5]) with exact future onset index in {0..Tf-1} or -1."""
    hm = np.asarray(hist_mask, dtype=np.float32) > 0.5
    fm = np.asarray(fut_mask, dtype=np.float32) > 0.5
    y_act = np.zeros(5, dtype=bool)
    v_cls = np.zeros(5, dtype=bool)
    onset = np.full(5, -1, dtype=np.int64)
    run_k = max(2, int(round(0.3 / max(dt, 0.05)))) if min_run is None else int(min_run)
    run_k = min(run_k, max(2, len(fm) // 2))

    pre_win = hm[-3:] if len(hm) >= 3 else hm
    if int(pre_win.sum()) < 2 or float(fm.mean()) < 0.80 - 1e-6 or _max_consecutive_zeros(fm) > 3:
        return y_act, v_cls, onset

    all_pos = np.vstack([hist_pos, fut_pos]).astype(np.float64)
    all_m = np.concatenate([hm, fm])
    t0_i = len(hist_pos) - 1
    svx, svy, spd = _estimate_causal_velocities(all_pos, all_m, dt=dt)
    pre_s = spd[max(0, t0_i - 2) : t0_i + 1]
    vp = pre_s[~np.isnan(pre_s)]
    if len(vp) == 0:
        return y_act, v_cls, onset
    v_pre = float(np.mean(vp))

    v_low, v_high, v_turn = 0.20 * speed_scale, 0.60 * speed_scale, 0.50 * speed_scale
    ang_thr = np.deg2rad(30.0 * angle_scale)
    v_cls[:] = True
    v_cls[0] = v_pre <= v_low
    v_cls[1] = v_pre >= v_high

    f_spd = spd[t0_i + 1 :]
    ts = _first_sustained_run((~np.isnan(f_spd)) & (f_spd >= v_high), min_len=run_k) if v_cls[0] else -1
    tx = _first_sustained_run((~np.isnan(f_spd)) & (f_spd <= v_low), min_len=run_k) if v_cls[1] else -1
    if ts >= 0 and (tx < 0 or ts <= tx):
        y_act[0], onset[0] = True, int(ts)
    elif tx >= 0:
        y_act[1], onset[1] = True, int(tx)

    s_anc = max(0, t0_i - 2)
    mov = (~np.isnan(spd[s_anc:])) & (spd[s_anc:] >= v_turn)
    bl, br = 10**9, 10**9
    idx = 0
    while idx < len(mov):
        if not mov[idx]:
            idx += 1
            continue
        end = idx
        while end < len(mov) and mov[end]:
            end += 1
        if end - idx >= run_k:
            th = np.unwrap(np.arctan2(svy[s_anc + idx : s_anc + end], svx[s_anc + idx : s_anc + end]))
            dth = th - th[0]
            fo = max(0, (t0_i + 1) - (s_anc + idx))
            if fo < len(dth):
                tl = _first_sustained_run(dth[fo:] >= ang_thr, min_len=run_k)
                tr = _first_sustained_run(dth[fo:] <= -ang_thr, min_len=run_k)
                if tl >= 0:
                    bl = min(bl, s_anc + idx + fo + tl)
                if tr >= 0:
                    br = min(br, s_anc + idx + fo + tr)
        idx = end
    if bl < 10**9 or br < 10**9:
        if bl <= br:
            y_act[2], onset[2] = True, int(max(0, bl - (t0_i + 1)))
        else:
            y_act[3], onset[3] = True, int(max(0, br - (t0_i + 1)))
    if not y_act[:4].any():
        y_act[4], onset[4] = True, 0
    return y_act, v_cls, onset


def horizon_and_post_onset_errors(
    pred_top1: np.ndarray,
    truth_np: np.ndarray,
    onset_np: np.ndarray,
    dt: float = 0.1,
) -> np.ndarray:
    """Return [N, 6]: [ade_full, fde_full, ade_trig_0p5s, ade_trig_1p0s, ade_post_onset_0p5s, ade_post_onset_1p0s]."""
    d = np.linalg.norm(pred_top1 - truth_np, axis=-1)
    n, tf = d.shape
    k05 = min(tf, max(1, int(round(0.5 / max(dt, 0.05)))))
    k10 = min(tf, max(2, int(round(1.0 / max(dt, 0.05)))))
    post05 = d[:, :k05].mean(axis=1).copy()
    post10 = d[:, :k10].mean(axis=1).copy()
    for i in range(n):
        act_ons = [int(onset_np[i, c]) for c in range(4) if int(onset_np[i, c]) >= 0]
        if act_ons:
            s0 = min(max(0, min(act_ons)), tf - 1)
            post05[i] = float(d[i, s0 : min(tf, s0 + k05)].mean())
            post10[i] = float(d[i, s0 : min(tf, s0 + k10)].mean())
    return np.column_stack((d.mean(1), d[:, -1], d[:, :k05].mean(1), d[:, :k10].mean(1), post05, post10))


def audit_6candidate_generation_vs_selection(
    hist_np: np.ndarray,
    paths_k_rel: np.ndarray,
    top1_idx: np.ndarray,
    truth_rel: np.ndarray,
    cv_rel: np.ndarray,
    y_act: np.ndarray,
    v_act: np.ndarray,
    mask: np.ndarray,
    dt: float = 0.1,
) -> dict[str, dict]:
    """Audit per-maneuver 'not generated' vs 'not selected' across K=6 candidates on `mask`."""
    sub = np.flatnonzero(mask)
    names = ("Start", "Stop", "Turn_Left", "Turn_Right", "Maintain")
    hm = np.ones(hist_np.shape[1], dtype=np.float32)
    fm = np.ones(paths_k_rel.shape[2], dtype=np.float32)
    cand_y = np.zeros((len(sub), paths_k_rel.shape[1], 5), dtype=bool)
    for ii, i in enumerate(sub):
        hp = hist_np[i] - hist_np[i, -1:]
        for k in range(paths_k_rel.shape[1]):
            yk, _, _ = compute_action_labels_with_onset(hp, hm, paths_k_rel[i, k], fm, dt=dt)
            cand_y[ii, k] = yk

    d_k = np.linalg.norm(paths_k_rel[sub] - truth_rel[sub, None], axis=-1).mean(axis=-1)
    d_cv = np.linalg.norm(cv_rel[sub] - truth_rel[sub], axis=-1).mean(axis=-1)
    sel_y = cand_y[np.arange(len(sub)), top1_idx[sub]]
    out: dict[str, dict] = {}
    for c, nm in enumerate(names):
        elig = v_act[sub, c].astype(bool)
        yt = y_act[sub, c].astype(bool) & elig
        yn = (~y_act[sub, c].astype(bool)) & elig
        any6 = cand_y[:, :, c].any(axis=1)
        s1 = sel_y[:, c]
        n_pos, n_neg = int(yt.sum()), int(yn.sum())
        gen_hit = int((yt & any6).sum())
        sel_hit = int((yt & s1).sum())
        out[nm] = {
            "n_true": n_pos,
            "n_neg_eligible": n_neg,
            "any6_generated_hits": gen_hit,
            "not_generated_misses": int(n_pos - gen_hit),
            "generated_but_not_selected": int(gen_hit - sel_hit),
            "top1_selected_hits": sel_hit,
            "oracle6_beats_cv_by_2cm": int((yt & (d_k.min(axis=1) + 0.02 < d_cv)).sum()),
            "any6_neg_false_candidates": int((yn & any6).sum()),
            "top1_neg_false_positives": int((yn & s1).sum()),
        }
    return out


def score_site_isolated_actions(
    y_true: np.ndarray,
    v_mask: np.ndarray,
    probs_seed: np.ndarray,
    pred_bin_seed: np.ndarray,
    grp_arr: np.ndarray,
) -> dict:
    """Score 5-class actions where EVERY test group & seed used ONLY its own source-only thresholds."""
    from src.metrics.metrics_and_stats import _binary_average_precision

    pm = probs_seed.mean(axis=0)
    bm = pred_bin_seed.mean(axis=0) >= 0.5
    names = ("Start", "Stop", "Turn_Left", "Turn_Right", "Maintain")
    pooled = {}
    for c, nm in enumerate(names):
        vm = v_mask[:, c].astype(bool)
        yt, ps, pr = y_true[vm, c].astype(bool), pm[vm, c], bm[vm, c]
        tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
        pooled[nm] = {
            "pos": int(yt.sum()),
            "ap": float(_binary_average_precision(yt.astype(np.float64), ps)) if yt.sum() > 0 else 0.0,
            "f1": float((2 * tp) / max(2 * tp + fp + fn, 1)),
            "tp": tp,
            "fp": fp,
        }
    g_aps, g_f1s = [], []
    for g in sorted(set(grp_arr.tolist())):
        gm = grp_arr == g
        aps, f1s = [], []
        for c in range(5):
            vm = gm & v_mask[:, c].astype(bool)
            if not vm.any():
                continue
            yt, ps, pr = y_true[vm, c].astype(bool), pm[vm, c], bm[vm, c]
            if yt.sum() > 0:
                aps.append(float(_binary_average_precision(yt.astype(np.float64), ps)))
            tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
            f1s.append((2.0 * tp) / max(2 * tp + fp + fn, 1))
        if aps:
            g_aps.append(float(np.mean(aps)))
        if f1s:
            g_f1s.append(float(np.mean(f1s)))
    return {"group_equal_macro_ap": float(np.mean(g_aps)), "group_equal_macro_f1": float(np.mean(g_f1s)), "pooled_classes": pooled}


class O2Maneuver6SlotPrecursorNet(nn.Module):
    """O(2)-equivariant 6-slot maneuver hypothesis generator + 5-frame temporal pose-precursor ranker."""

    def __init__(self, use_pose: bool = True, use_interaction: bool = True) -> None:
        super().__init__()
        self.use_pose = use_pose
        self.use_interaction = use_interaction

        self.motion = nn.GRU(5, 48, batch_first=True)
        self.ctx_proj = nn.Sequential(nn.LayerNorm(48), nn.Linear(48, 48), nn.SiLU())

        # Monotonic quality gate + 5-frame pose & temporal delta-pose encoder
        self.q_gamma_raw = nn.Parameter(torch.tensor(-0.35))
        self.q_w_raw = nn.Parameter(torch.tensor([0.5, 0.5, 0.2, 0.2]))
        self.q_bias = nn.Parameter(torch.tensor(0.20))
        self.pose_in = nn.Linear(16, 24)
        self.pose_gru = nn.GRU(24, 24, batch_first=True)
        self.dpose_even_mlp = nn.Sequential(nn.Linear(10, 24), nn.SiLU(), nn.Linear(24, 24))
        self.dpose_odd_lin = nn.Linear(3, 24, bias=False)
        self.pose_proj = nn.Linear(24, 48, bias=False)
        nn.init.zeros_(self.pose_proj.weight)

        # Causal conflict-gated neighbor encoder
        self.edge_mlp = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 48))
        self.edge_att = nn.Linear(96, 1)
        self.null_logit = nn.Parameter(torch.tensor(1.5))
        self.graph_proj = nn.Linear(48, 48, bias=False)
        nn.init.zeros_(self.graph_proj.weight)

        # Un-squashed semantic residuals for Slots 1 (Start), 2 (Stop), 5 (Yield) + Slots 3,4 (Turn pair)
        self.v_walk_head = nn.Linear(48, 1)
        self.slot125_dx = nn.Linear(48, 3 * 3)
        self.slot125_dy = nn.Linear(48, 3 * 3, bias=False)
        self.turn_pair_knots = nn.Linear(48, 3 * 2)
        for m in (self.v_walk_head, self.slot125_dx, self.turn_pair_knots):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        nn.init.normal_(self.slot125_dy.weight, std=0.01)

        # Mode logits & 5-class observable maneuver precursor head
        self.even_mode_logits = nn.Linear(48 + 24 + 4, 4)  # Slots 0, 1, 2, 5
        self.turn_mode_logit = nn.Linear(48 + 24 + 4, 1)   # Slots 3 (+), 4 (-)
        self.act_even = nn.Linear(48 + 24 + 4, 3)          # Start, Stop, Maintain
        self.act_turn = nn.Linear(48 + 24 + 4, 1)          # Turn-Left (+), Turn-Right (-)
        self.cv_hold_bias = nn.Parameter(torch.tensor(0.30))

    def _mono_gate(self, q_vec: torch.Tensor, m_pose: torch.Tensor) -> torch.Tensor:
        qc = q_vec.clamp(0.0, 1.0).float()
        if qc.ndim == 1:
            qc = qc[:, None].expand(-1, 4)
        elif qc.size(1) < 4:
            qc = F.pad(qc, (0, 4 - qc.size(1)), value=1.0)
        else:
            qc = qc[:, :4]
        gamma = F.softplus(self.q_gamma_raw) + 0.15
        w_pos = F.softplus(self.q_w_raw)
        env = ((qc[:, 0] * qc[:, 1]).clamp_min(1e-6)) ** gamma
        return m_pose.float() * env * torch.sigmoid(self.q_bias + (qc - 0.5) @ w_pos)

    def forward(
        self,
        history: torch.Tensor,
        bio_even: torch.Tensor | None = None,
        pose_vec_world: torch.Tensor | None = None,
        signed_knee: torch.Tensor | None = None,
        q_vec: torch.Tensor | None = None,
        m_pose: torch.Tensor | None = None,
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
        v_loc = vel[:, -1]  # [B, 2] in local Frenet frame (double)
        v_last = torch.linalg.vector_norm(v_loc, dim=-1).float()
        gate = v_last.new_zeros(bsz)
        hp_pos = v_last.new_zeros((bsz, 24))
        hp_neg = v_last.new_zeros((bsz, 24))
        fc_lat_odd = history.new_zeros(bsz)

        if use_p and signed_knee is not None:
            qv = q_vec if q_vec is not None else v_last.new_zeros((bsz, 4))
            gate = self._mono_gate(qv, m_pose)
            pv_l = torch.einsum("btvk,bks->btvs", pose_vec_world.double(), basis).float()
            fc_p, fl_p = pv_l[:, :, 0], pv_l[:, :, 1]
            flip = fc_p.new_tensor([1.0, -1.0])
            pp = torch.cat((bio_even, fc_p, fl_p, signed_knee), dim=-1)
            pn = torch.cat((bio_even, fc_p * flip, fl_p * flip, -signed_knee), dim=-1)
            _, hpb = self.pose_gru(F.silu(self.pose_in(torch.cat((pp, pn), dim=0))))
            # Explicit 5-frame temporal pose-change precursors (ankle/leg rate, lean rate, yaw lead)
            d_fc = fc_p[:, -1] - fc_p[:, 0]
            d_fl = fl_p[:, -1] - fl_p[:, 0]
            dp_even = torch.cat((bio_even[:, -1, :5] - bio_even[:, 0, :5], bio_even[:, -1, 5:8], fc_p[:, -1, :1], d_fl[:, :1]), dim=-1)
            dp_odd = torch.stack((fc_p[:, -1, 1], d_fc[:, 1], signed_knee[:, -1, 0] - signed_knee[:, 0, 0]), dim=-1)
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
        m0_lg = ev_m[:, :1] + self.cv_hold_bias * torch.exp(-((disp_all / 0.25) ** 2))[:, None]
        raw_lg = torch.cat((m0_lg, ev_m[:, 1:3], self.turn_mode_logit(feat_pos), self.turn_mode_logit(feat_neg), ev_m[:, 3:]), dim=1)
        logits = torch.where(valid[:, None], raw_lg, 0.0)

        t_sec = torch.arange(1, 31, dtype=history.dtype, device=history.device)[None, :] * 0.1 if fut_times_s is None else fut_times_s.to(history.dtype)
        t_max = t_sec[:, -1:].clamp_min(1.0)
        kn_idx = ((t_sec / t_max) * 3.0 - 1e-9).floor().long().clamp(0, 2)
        kn_ph = ((t_sec / t_max) * 3.0 - kn_idx.to(history.dtype))[:, None, :, None]

        # Un-squashed learned residuals for Slots 1 (Start), 2 (Stop), 5 (Yield) and Slots 3,4 (Turn)
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

        # Explicit physical anchors in local Frenet frame: 1=Start, 2=Stop, 3=Left, 4=Right, 5=Yield
        anc0 = v_loc[:, None, :] * t_sec[:, :, None]
        v_walk = (0.95 + 0.35 * torch.tanh(self.v_walk_head(c_even).squeeze(-1))).double()
        dv_acc = (v_walk - v_loc[:, 0]).clamp_min(0.45)[:, None]
        s_acc = dv_acc * (t_sec - 0.45 * (1.0 - torch.exp(-t_sec / 0.45)))
        anc1 = anc0 + torch.stack((s_acc, 0.45 * fc_lat_odd[:, None] * s_acc), dim=-1)
        anc2 = v_loc[:, None, :] * (0.55 * (1.0 - torch.exp(-t_sec[:, :, None] / 0.55)))
        v_st = v_loc[:, 0].clamp_min(0.65)[:, None]
        th_t = (0.60 * (t_sec / 1.8).clamp(0.0, 1.0))
        st_x = v_st * t_sec * torch.cos(0.5 * th_t)
        st_y = v_st * t_sec * torch.sin(0.5 * th_t)
        anc3 = torch.stack((st_x, st_y + 0.5 * v_loc[:, 1:2] * t_sec), dim=-1)
        anc4 = torch.stack((st_x, -st_y + 0.5 * v_loc[:, 1:2] * t_sec), dim=-1)
        anc5 = 0.55 * anc0
        anc_1to5 = torch.stack((anc1, anc2, anc3, anc4, anc5), dim=1) + delta5
        paths_loc = torch.cat((anc0[:, None], anc_1to5), dim=1)
        paths_world = origin[:, None, :, :] + torch.einsum("bkts,bws->bktw", paths_loc, basis)
        return {"paths": paths_world, "logits": logits, "act_logits": act_logits, "valid": valid, "gate": gate}


def slot_semantic_maneuver_loss(
    out: dict[str, torch.Tensor],
    truth: torch.Tensor,
    w_sample: torch.Tensor,
    y_act: torch.Tensor,
    valid_act: torch.Tensor,
    pos_weight: torch.Tensor,
    act_weight: float = 0.35,
) -> torch.Tensor | None:
    """Supervise dedicated maneuver slots (1:Start, 2:Stop, 3:Left, 4:Right, 0/5:Maintain) + mode CE + BCE."""
    vm = out["valid"]
    if not torch.any(vm):
        return None
    p6 = out["paths"][vm]
    gt = truth[vm, None]
    ws = w_sample[vm].double()
    d = torch.linalg.vector_norm(p6 - gt, dim=-1)
    huber = (torch.sqrt(d.square() + 0.04) - 0.20).mean(dim=-1)  # [B, 6]
    ade = d.detach().mean(dim=-1)

    ya = (y_act[vm] & valid_act[vm]).bool()
    # Dedicated semantic slot supervision prevents rare-maneuver mode collapse
    slot_reg = huber[:, 1:].amin(dim=1)
    slot_reg = torch.where(ya[:, 0], 0.65 * huber[:, 1] + 0.35 * slot_reg, slot_reg)
    slot_reg = torch.where(ya[:, 1], 0.65 * huber[:, 2] + 0.35 * slot_reg, slot_reg)
    slot_reg = torch.where(ya[:, 2], 0.65 * huber[:, 3] + 0.35 * slot_reg, slot_reg)
    slot_reg = torch.where(ya[:, 3], 0.65 * huber[:, 4] + 0.35 * slot_reg, slot_reg)
    slot_reg = torch.where(ya[:, 4] & (~ya[:, :4].any(dim=1)), 0.50 * huber[:, 5] + 0.50 * slot_reg, slot_reg)

    soft = (-ade / 0.18).softmax(dim=1).to(out["logits"].dtype)
    cls_l = -(soft * out["logits"][vm].log_softmax(dim=1)).sum(dim=1).double()
    bce = F.binary_cross_entropy_with_logits(out["act_logits"][vm], y_act[vm].float(), pos_weight=pos_weight, reduction="none")
    va_f = valid_act[vm].float()
    act_l = ((bce * va_f).sum(dim=1) / va_f.sum(dim=1).clamp_min(1.0)).double()
    return ((slot_reg + 0.25 * cls_l + float(act_weight) * act_l) * ws).sum() / ws.sum().clamp_min(1e-8)


def fit_strict_source_platt_and_slot_selector(
    oof_mode_logits: np.ndarray,
    oof_act_logits: np.ndarray,
    oof_paths: np.ndarray,
    oof_truth: np.ndarray,
    oof_y_act: np.ndarray,
    oof_v_act: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Fit source-only Platt action calibration + precision-constrained slot-switching thresholds."""
    from scripts.run_wave18_action_calibration_and_rerank_5seed import fit_platt_and_thresholds

    a_pl, b_pl, thr_f1 = fit_platt_and_thresholds(oof_act_logits, oof_y_act, oof_v_act)
    p_cal = 1.0 / (1.0 + np.exp(-np.clip(oof_act_logits * a_pl[None, :] + b_pl[None, :], -30.0, 30.0)))

    # Calibrate conservative slot-switch probability gates [Start->1, Stop->2, Left->3, Right->4]
    # strictly on source OOF trajectories so switching only occurs when net ADE drops!
    d_all = np.linalg.norm(oof_paths - oof_truth[:, None], axis=-1).mean(axis=-1)  # [N, 6]
    base_idx = oof_mode_logits.argmax(axis=1)
    d_base = d_all[np.arange(len(d_all)), base_idx].copy()

    best_beta = 0.0
    best_err = float(d_base.mean())
    log_pi = oof_mode_logits - oof_mode_logits.max(1, keepdims=True)
    log_pi = log_pi - np.log(np.exp(log_pi).sum(1, keepdims=True))
    slot_p = np.column_stack((p_cal[:, 4], p_cal[:, 0], p_cal[:, 1], p_cal[:, 2], p_cal[:, 3], p_cal[:, 4]))
    log_sp = np.log(np.clip(slot_p, 1e-4, 1.0 - 1e-4))
    for bt in (0.0, 0.05, 0.12, 0.22, 0.35):
        idx_b = (log_pi + bt * log_sp).argmax(axis=1)
        eb = float(d_all[np.arange(len(d_all)), idx_b].mean())
        if eb + (1e-7 if bt > 0 else 0.0) < best_err:
            best_err, best_beta = eb, float(bt)

    cur_idx = (log_pi + best_beta * log_sp).argmax(axis=1)
    cur_err = d_all[np.arange(len(d_all)), cur_idx].copy()
    sw_thr = np.full(4, 1.01, dtype=np.float64)  # default > 1.0 = never force-override unless OOF net gain > 0
    for c_act, slot_k in ((0, 1), (1, 2), (2, 3), (3, 4)):
        elig = oof_v_act[:, c_act].astype(bool)
        best_gain = 0.0
        for th in np.linspace(0.32, 0.86, 28):
            fire = elig & (p_cal[:, c_act] >= th)
            if int(fire.sum()) < 3:
                continue
            gain = float((cur_err[fire] - d_all[fire, slot_k]).sum()) / max(len(d_all), 1)
            if gain > best_gain + 1e-6:
                best_gain, sw_thr[c_act] = gain, float(th)
        if sw_thr[c_act] <= 1.0:
            fire = elig & (p_cal[:, c_act] >= sw_thr[c_act])
            cur_idx[fire] = slot_k
            cur_err[fire] = d_all[fire, slot_k]
    return {"a_pl": a_pl, "b_pl": b_pl, "thr_f1": thr_f1, "beta": best_beta, "sw_thr": sw_thr}


def apply_6slot_selector(
    mode_logits: np.ndarray,
    act_logits: np.ndarray,
    v_act: np.ndarray,
    cal: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (selected_top1_idx[N], calibrated_act_probs[N,5], site_isolated_act_preds[N,5])."""
    a_pl, b_pl = np.asarray(cal["a_pl"]), np.asarray(cal["b_pl"])
    thr_f1, sw_thr = np.asarray(cal["thr_f1"]), np.asarray(cal["sw_thr"])
    p_cal = 1.0 / (1.0 + np.exp(-np.clip(act_logits * a_pl[None, :] + b_pl[None, :], -30.0, 30.0)))
    pred_bin = (p_cal >= thr_f1[None, :]) & v_act.astype(bool)
    log_pi = mode_logits - mode_logits.max(1, keepdims=True)
    log_pi = log_pi - np.log(np.exp(log_pi).sum(1, keepdims=True))
    slot_p = np.column_stack((p_cal[:, 4], p_cal[:, 0], p_cal[:, 1], p_cal[:, 2], p_cal[:, 3], p_cal[:, 4]))
    idx = (log_pi + float(cal["beta"]) * np.log(np.clip(slot_p, 1e-4, 1.0 - 1e-4))).argmax(axis=1)
    for c_act, slot_k in ((0, 1), (1, 2), (2, 3), (3, 4)):
        if sw_thr[c_act] <= 1.0:
            fire = v_act[:, c_act].astype(bool) & (p_cal[:, c_act] >= sw_thr[c_act])
            idx[fire] = slot_k
    return idx, p_cal, pred_bin
