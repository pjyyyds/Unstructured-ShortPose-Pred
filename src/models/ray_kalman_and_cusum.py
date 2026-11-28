"""Wave 27: Projective-Jacobian Anisotropic Ray-Kalman Filter (PJ-IAKF) & Causal Post-Reappearance CUSUM."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_homography_jacobian_world(
    bbox: torch.Tensor,
    h_mats: torch.Tensor,
    y_sums: torch.Tensor,
) -> torch.Tensor:
    """Exact 2x2 ground-per-pixel Jacobian [j_u, j_v] (shape B, T, 2, 2) at bbox footpoint (cx, cy + 0.5*h).

    Each row j_u = d(X,Y)/du and j_v = d(X,Y)/dv is a contravariant 2D world vector (m/px), so under any
    world orthogonal transform Q in O(2), jac_uv_world -> jac_uv_world @ Q.T and R_t -> Q R_t Q.T exactly.
    """
    bb = bbox.double()
    hm = h_mats.double()
    _ = y_sums  # y = y_sum - raw_y flips dy sign without depending on y_sum value
    u = bb[..., 0]
    v = bb[..., 1] + 0.5 * bb[..., 3]
    ones = torch.ones_like(u)
    hom = torch.stack((u, v, ones), dim=-1)  # (B, T, 3)
    raw = torch.einsum("btk,bjk->btj", hom, hm)  # (B, T, 3)
    rx, ry, w = raw[..., 0], raw[..., 1], raw[..., 2]
    ok = (w.abs() > 1e-6) & torch.isfinite(bb[..., 3]) & (bb[..., 3] > 1.0)
    w2 = torch.where(ok, w * w, torch.ones_like(w))

    dx_du = (hm[:, None, 0, 0] * w - rx * hm[:, None, 2, 0]) / w2
    dx_dv = (hm[:, None, 0, 1] * w - rx * hm[:, None, 2, 1]) / w2
    dy_du = -(hm[:, None, 1, 0] * w - ry * hm[:, None, 2, 0]) / w2
    dy_dv = -(hm[:, None, 1, 1] * w - ry * hm[:, None, 2, 1]) / w2

    j_u = torch.stack((dx_du, dy_du), dim=-1)
    j_v = torch.stack((dx_dv, dy_dv), dim=-1)
    jac = torch.stack((j_u, j_v), dim=2)  # (B, T, 2, 2): [:, :, 0] = j_u, [:, :, 1] = j_v
    # Clamp extreme horizon singularities safely to <= 0.35 m/px
    nrm = torch.linalg.vector_norm(jac, dim=-1, keepdim=True).clamp_min(1e-8)
    jac = jac * torch.clamp(0.35 / nrm, max=1.0)
    return torch.where(ok[:, :, None, None], jac, 0.0)


def _inv2x2(mat: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Exact batched 2x2 symmetric positive-definite inverse in float64."""
    a = mat[..., 0, 0] + eps
    b = mat[..., 0, 1]
    c = mat[..., 1, 0]
    d = mat[..., 1, 1] + eps
    det = (a * d - b * c).clamp_min(eps * eps)
    inv_det = 1.0 / det
    r0 = torch.stack((d * inv_det, -b * inv_det), dim=-1)
    r1 = torch.stack((-c * inv_det, a * inv_det), dim=-1)
    return torch.stack((r0, r1), dim=-2)


def anisotropic_ray_kalman_filter(
    history: torch.Tensor,
    obs_dt: torch.Tensor | None = None,
    jac_uv_world: torch.Tensor | None = None,
    bbox_h: torch.Tensor | None = None,
    ank_conf: torch.Tensor | None = None,
    sigma_u_px: float = 1.5,
    sigma_v_px: float = 2.8,
    kappa_dh: float = 0.35,
    sigma_iso_m: float = 0.012,
    q_acc: float = 1.4,
    tau_eb: float = 0.22,
    anchor_p_blend: float = 0.5,
    d_ref: float = 0.55,
) -> dict[str, torch.Tensor]:
    """O(2)-equivariant 4-state [p, v] Innovation-Adaptive Anisotropic Ray-Kalman Filter (PJ-IAKF)."""
    bsz, t_obs, _ = history.shape
    h = history.double()
    dev = h.device
    dt = h.new_full((bsz, t_obs - 1), 0.1) if obs_dt is None else obs_dt.double().clamp_min(1e-4)
    v_last = (h[:, -1] - h[:, -2]) / dt[:, -1:]
    eye2 = torch.eye(2, dtype=torch.float64, device=dev)[None].expand(bsz, 2, 2)
    path_len = torch.linalg.vector_norm(h[:, 1:] - h[:, :-1], dim=-1).sum(dim=-1)
    low_env = torch.exp(-((path_len / max(float(d_ref), 0.10)) ** 2))

    if max(abs(sigma_u_px), abs(sigma_v_px), abs(sigma_iso_m)) <= 1e-7:
        z2 = h.new_zeros((bsz, 2, 2))
        return {"p_filt": h[:, -1], "v_filt": v_last, "v_last": v_last, "P_vv": z2, "P_pp": z2, "low_env": low_env}

    jac = jac_uv_world.double() if jac_uv_world is not None else None
    bh = bbox_h.double().clamp_min(1.0) if bbox_h is not None else None
    ac = ank_conf.double().clamp(0.0, 1.0) if ank_conf is not None else None

    def _meas_cov(t: int) -> torch.Tensor:
        r_iso = (float(sigma_iso_m) ** 2 + 1e-8) * eye2
        if jac is None:
            return r_iso
        dh = (bh[:, t] - bh[:, max(t - 1, 0)]).abs() if (bh is not None and t > 0) else h.new_zeros(bsz)
        am = (1.0 - ac[:, t]) if ac is not None else h.new_zeros(bsz)
        su2 = h.new_full((bsz,), float(sigma_u_px) ** 2)
        sv = float(sigma_v_px) + float(kappa_dh) * dh + 0.5 * float(sigma_v_px) * am
        sv2 = sv.square()
        ju, jv = jac[:, t, 0], jac[:, t, 1]
        return r_iso + su2[:, None, None] * (ju[:, :, None] * ju[:, None, :]) + sv2[:, None, None] * (jv[:, :, None] * jv[:, None, :])

    r0, r1 = _meas_cov(0), _meas_cov(1)
    dt0 = dt[:, 0]
    p_est = h[:, 1]
    v_est = (h[:, 1] - h[:, 0]) / dt0[:, None]
    p_pp = r1
    p_pv = r1 / dt0[:, None, None]
    p_vv = (r1 + r0) / (dt0.square()[:, None, None]) + (0.15 * float(q_acc) ** 2) * eye2

    for t in range(2, t_obs):
        d = dt[:, t - 1]
        d1, d2 = d[:, None], d[:, None, None]
        p_pred = p_est + d1 * v_est
        rt = _meas_cov(t)
        nu = h[:, t] - p_pred
        # Base innovation check for maneuver-adaptive process noise (NIS gate)
        qa2_base = float(q_acc) ** 2
        s_base = p_pp + d2 * (p_pv + p_pv.transpose(-1, -2)) + (d2 * d2) * p_vv + (0.25 * (d2 ** 4) * qa2_base) * eye2 + rt
        nis = (nu[:, None, :] @ _inv2x2(s_base) @ nu[:, :, None]).squeeze(-1).squeeze(-1).clamp_min(0.0)
        boost = (1.0 + 0.65 * F.relu(nis - 3.5)).clamp(1.0, 12.0)[:, None, None]
        qa2 = qa2_base * boost

        p_pp_m = p_pp + d2 * (p_pv + p_pv.transpose(-1, -2)) + (d2 * d2) * p_vv + 0.25 * (d2 ** 4) * qa2 * eye2
        p_pv_m = p_pv + d2 * p_vv + 0.5 * (d2 ** 3) * qa2 * eye2
        p_vv_m = p_vv + (d2 * d2) * qa2 * eye2

        s_inv = _inv2x2(p_pp_m + rt)
        kp = p_pp_m @ s_inv
        kv = p_pv_m.transpose(-1, -2) @ s_inv
        p_est = p_pred + (kp @ nu[:, :, None]).squeeze(-1)
        v_est = v_est + (kv @ nu[:, :, None]).squeeze(-1)
        ik = eye2 - kp
        p_pp = 0.5 * (ik @ p_pp_m + (ik @ p_pp_m).transpose(-1, -2))
        p_pv = ik @ p_pv_m
        p_vv = 0.5 * ((p_vv_m - kv @ p_pv_m) + (p_vv_m - kv @ p_pv_m).transpose(-1, -2))

    if tau_eb > 1e-5:
        sh_mat = low_env[:, None, None] * (p_vv @ _inv2x2(p_vv + (float(tau_eb) ** 2) * eye2))
        v_filt = v_est - (sh_mat @ v_est[:, :, None]).squeeze(-1)
    else:
        v_filt = v_est
    ab = float(np.clip(anchor_p_blend, 0.0, 1.0))
    p_filt = (1.0 - ab) * h[:, -1] + ab * p_est
    return {"p_filt": p_filt, "v_filt": v_filt, "v_last": v_last, "P_vv": p_vv, "P_pp": p_pp, "low_env": low_env}


def calibrate_ray_kalman_params(
    hist_src: torch.Tensor,
    obs_dt_src: torch.Tensor,
    fut_s_src: torch.Tensor,
    truth_src: torch.Tensor,
    src_grp: np.ndarray | None = None,
    jac_src: torch.Tensor | None = None,
    bbox_h_src: torch.Tensor | None = None,
    ank_conf_src: torch.Tensor | None = None,
    d_ref: float = 0.55,
) -> dict[str, float]:
    """Select PJ-IAKF physical parameters strictly on source folds (includes exact Last-2 CV baseline)."""
    with torch.inference_mode():
        ft = fut_s_src.double()[:, :, None]
        gt = truth_src.double()
        pl = torch.linalg.vector_norm(hist_src[:, 1:] - hist_src[:, :-1], dim=-1).sum(dim=-1).double()
        low_m = pl < 0.50

        def _score(cfg: dict[str, float]) -> float:
            out = anisotropic_ray_kalman_filter(
                hist_src, obs_dt_src, jac_src, bbox_h_src, ank_conf_src, **cfg
            )
            pred = out["p_filt"][:, None, :] + out["v_filt"][:, None, :] * ft
            ade = torch.linalg.vector_norm(pred - gt, dim=-1).mean(dim=-1)
            w = torch.where(low_m, torch.full_like(ade, 2.5), torch.ones_like(ade))
            if src_grp is not None and len(set(src_grp.tolist())) >= 2:
                vals = []
                for g in sorted(set(src_grp.tolist())):
                    mg = torch.from_numpy(src_grp == g).to(ade.device)
                    if mg.any():
                        vals.append(float((ade[mg] * w[mg]).sum() / w[mg].sum().clamp_min(1.0)))
                return 0.5 * float(np.mean(vals)) + 0.5 * float((ade * w).sum() / w.sum().clamp_min(1.0))
            return float((ade * w).sum() / w.sum().clamp_min(1.0))

        best_cfg = {
            "sigma_u_px": 0.0, "sigma_v_px": 0.0, "kappa_dh": 0.0,
            "sigma_iso_m": 0.0, "q_acc": 1.5, "tau_eb": 0.0, "anchor_p_blend": 0.0, "d_ref": float(d_ref),
        }
        best_sc = _score(best_cfg)

        if jac_src is not None:
            cand_grid = [
                (su, sv, kdh, 0.008, qa, teb, apb)
                for (su, sv, kdh) in ((1.0, 2.0, 0.25), (1.5, 3.0, 0.35), (2.2, 4.2, 0.45))
                for qa in (1.0, 1.8, 3.0)
                for teb in (0.0, 0.18, 0.30)
                for apb in (0.0, 0.45, 0.80)
            ]
        else:
            cand_grid = [
                (0.0, 0.0, 0.0, s_iso, qa, teb, apb)
                for s_iso in (0.012, 0.025, 0.045)
                for qa in (0.9, 1.6, 2.8)
                for teb in (0.0, 0.16, 0.28)
                for apb in (0.0, 0.45)
            ]

        for su, sv, kdh, s_iso, qa, teb, apb in cand_grid:
            cfg = {
                "sigma_u_px": float(su), "sigma_v_px": float(sv), "kappa_dh": float(kdh),
                "sigma_iso_m": float(s_iso), "q_acc": float(qa), "tau_eb": float(teb),
                "anchor_p_blend": float(apb), "d_ref": float(d_ref),
            }
            sc = _score(cfg)
            if sc + 1e-7 < best_sc:
                best_sc = sc
                best_cfg = cfg
    return best_cfg


def causal_post_reappear_cusum_scores(
    z_seq: np.ndarray,
    occ_seq: np.ndarray | None = None,
    lam: float = 0.65,
    drift_nu: float = 0.15,
    w_cusum: float = 0.55,
    gate_heavy_occ: bool = True,
) -> np.ndarray:
    """Strictly causal post-reappearance leaky CUSUM / SPRT score sequence (applied identically to all tracks)."""
    z = np.asarray(z_seq, dtype=np.float64)
    t_len = z.shape[0]
    out = np.empty_like(z)
    acc = np.zeros(z.shape[1:], dtype=np.float64)
    for t in range(t_len):
        heavy = bool(occ_seq[t] >= 2) if occ_seq is not None else False
        if heavy and gate_heavy_occ:
            acc = 0.25 * acc
            out[t] = -1e4
        else:
            acc = np.maximum(0.0, float(lam) * acc + (z[t] - float(drift_nu)))
            out[t] = (1.0 - float(w_cusum)) * z[t] + float(w_cusum) * acc
    return out



class O2RayKalmanRes6SlotNet(nn.Module):
    """O(2)-equivariant predictor with PJ-IAKF Ray-Kalman state & 2x2 Covariance-Guided Slot-0 fusion (RKR6)."""

    def __init__(
        self,
        use_pose: bool = True,
        use_interaction: bool = True,
        ray_cfg: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.use_pose = use_pose
        self.use_interaction = use_interaction
        self.ray_cfg = ray_cfg or {
            "sigma_u_px": 1.5, "sigma_v_px": 3.0, "kappa_dh": 0.35,
            "sigma_iso_m": 0.012, "q_acc": 1.8, "tau_eb": 0.20, "anchor_p_blend": 0.45, "d_ref": 0.55,
        }
        self.motion = nn.GRU(5, 48, batch_first=True)
        self.cov_mlp = nn.Sequential(nn.Linear(4, 24), nn.SiLU(), nn.Linear(24, 48))
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

        # Covariance-guided Slot-0 residual knots (exact zero-init!)
        self.slot0_dx = nn.Linear(48, 3)
        self.slot0_dy = nn.Linear(48, 3, bias=False)
        nn.init.zeros_(self.slot0_dx.weight)
        nn.init.zeros_(self.slot0_dx.bias)
        nn.init.zeros_(self.slot0_dy.weight)
        self.pose_steer_gain = nn.Parameter(torch.tensor(0.0))
        self.tau_prior_raw = nn.Parameter(torch.tensor(-0.8))

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
        dp0_loc = torch.einsum("bk,bks->bs", kf["p_filt"] - history[:, -1].double(), basis)
        v_filt_loc = torch.einsum("bk,bks->bs", kf["v_filt"], basis)
        p_vv_loc = torch.einsum("bki,bkl,blj->bij", basis, kf["P_vv"], basis)
        path_len = torch.linalg.vector_norm(history[:, 1:] - history[:, :-1], dim=-1).sum(dim=-1).double()
        chord_len = torch.linalg.vector_norm(history[:, -1] - history[:, 0], dim=-1).double()
        cov_in = torch.stack((
            path_len, chord_len,
            p_vv_loc[:, 0, 0].clamp_min(0.0).sqrt(),
            p_vv_loc[:, 1, 1].clamp_min(0.0).sqrt(),
        ), dim=-1).float()
        cov_emb = self.cov_mlp(cov_in)

        fp = torch.cat((body, vel, spd), dim=2).float()
        fn = fp * fp.new_tensor([1.0, -1.0, 1.0, -1.0, 1.0])
        _, hm = self.motion(torch.cat((fp, fn), dim=0))
        cb = self.ctx_proj(hm[0] + torch.cat((cov_emb, cov_emb), dim=0))
        c_pos, c_neg = cb[:bsz], cb[bsz:]

        disp_all = path_len.float()
        g_up, g_lo = disp_all.new_zeros(bsz), disp_all.new_zeros(bsz)
        hp_pos, hp_neg = disp_all.new_zeros((bsz, 24)), disp_all.new_zeros((bsz, 24))
        fc_lat_odd = history.new_zeros(bsz, dtype=torch.float64)

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

        t_sec = torch.arange(1, 31, dtype=torch.float64, device=history.device)[None, :] * 0.1 if fut_times_s is None else fut_times_s.double()
        t_max = t_sec[:, -1:].clamp_min(1.0)
        kn_idx = ((t_sec / t_max) * 3.0 - 1e-9).floor().long().clamp(0, 2)
        kn_ph = ((t_sec / t_max) * 3.0 - kn_idx.double())[:, None, :, None]
        bi = torch.arange(bsz, device=history.device)[:, None]

        # 2x2 Anisotropic Covariance-Guided Slot-0 residual (routes pose/conflict along uncertain camera ray!)
        eye2 = torch.eye(2, dtype=torch.float64, device=history.device)[None].expand(bsz, 2, 2)
        tau_p2 = (0.12 + F.softplus(self.tau_prior_raw.double())).square()
        p_eff = p_vv_loc + 0.015 * eye2
        g_cov = p_eff @ _inv2x2(p_eff + tau_p2 * eye2)
        mob0 = torch.tanh((chord_len / 0.14) ** 2)
        act_env0 = torch.maximum(mob0, 0.70 * w_nbr_act.double())
        r_cap = (0.20 + 0.80 * torch.tanh(chord_len / 0.55)) * act_env0 * float(slot0_res_scale)
        k0_x = torch.tanh(self.slot0_dx(c_even).double())
        k0_y = torch.tanh(self.slot0_dy(c_odd).double() + torch.tanh(self.pose_steer_gain.double()) * fc_lat_odd[:, None])
        raw_res0 = torch.stack((k0_x, k0_y), dim=-1)  # (B, 3, 2)
        cov_res0 = torch.einsum("bij,bkj->bki", g_cov, raw_res0) * r_cap[:, None, None]
        res0 = torch.cat((torch.zeros((bsz, 1, 2), dtype=torch.float64, device=history.device), cov_res0), dim=1)
        kn0_ph = kn_ph.squeeze(1)
        delta0 = res0[bi, kn_idx] * (1.0 - kn0_ph) + res0[bi, kn_idx + 1] * kn0_ph
        anc0_base = dp0_loc[:, None, :] + v_filt_loc[:, None, :] * t_sec[:, :, None]
        slot0_loc = anc0_base + delta0

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
