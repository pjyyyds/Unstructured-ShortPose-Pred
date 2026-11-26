"""PPT Slide 7-13 O(2)-equivariant M0-M3 model family, ground-pose features, and crossing metric."""
from __future__ import annotations

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.models.bioeq_neural import AXIS_THRESHOLD_M, KP_CONF_MIN, biomech_sequence_torch

PPT_ARMS = (
    "M0", "M1a", "M1b", "M1",
    "M2a", "M2b_no_vrad", "M2b", "M2",
    "M3a", "M3b", "M3",
)


def fit_pinns_ground_homography(calib: dict) -> tuple[np.ndarray, float]:
    """Fit verified 3x3 image-to-ground homography H and y_sum = y_min + y_max from PINNS calib.json."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required to fit 3x3 ground homography")
    pts_2d = np.asarray(calib.get("pts_2d", []), dtype=np.float64)
    pts_3d = np.asarray(calib.get("pts_3d", []), dtype=np.float64)
    if pts_2d.shape != (4, 2) or pts_3d.shape != (4, 2):
        raise ValueError(f"Expected 4x2 pts_2d/pts_3d in calib, got {pts_2d.shape} and {pts_3d.shape}")
    h_mat, _ = cv2.findHomography(pts_2d, pts_3d, method=0)
    if h_mat is None or h_mat.shape != (3, 3) or not np.isfinite(h_mat).all() or abs(float(np.linalg.det(h_mat))) <= 1e-12:
        raise ValueError("Degenerate or singular 3x3 ground homography fitted from calibration points")
    return h_mat.astype(np.float64), float(calib["y_min"]) + float(calib["y_max"])


def _map_uv_torch(uv: torch.Tensor, h_mat: torch.Tensor, y_sum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones((*uv.shape[:-1], 1), dtype=uv.dtype, device=uv.device)
    hom = torch.cat((uv, ones), dim=-1)
    raw = torch.einsum("...k,bjk->...j", hom, h_mat)
    ok = raw[..., 2].abs() > 1e-6
    xy = raw[..., :2] / torch.where(ok[..., None], raw[..., 2:3], 1.0)
    y_ref = y_sum.view(len(h_mat), *([1] * (xy.ndim - 2)))
    xy = torch.stack((xy[..., 0], y_ref - xy[..., 1]), dim=-1)
    return torch.where(ok[..., None], xy, 0.0), ok


def build_ground_pose_o2_inputs(
    history: torch.Tensor,
    kpts: torch.Tensor,
    bbox: torch.Tensor,
    h_mats: torch.Tensor | None = None,
    y_sums: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute 11-D even gait invariants, O(2) ground facing/lean vectors, signed knee diff, & m_pose."""
    bio_even, q_bio = biomech_sequence_torch(kpts, bbox)
    bsz, t_obs, _, _ = kpts.shape
    kp = kpts.double()
    bb = bbox.double()
    f_ok = torch.isfinite(bb[..., 3]) & (bb[..., 3] > 1.0)
    good = torch.isfinite(kp).all(dim=-1) & (kp[..., 2] >= KP_CONF_MIN) & f_ok[..., None]
    m_pose = good.any(dim=-1).any(dim=-1).float()

    # Odd-parity signed left-minus-right knee cosine asymmetry (flips sign under COCO L<->R swap)
    h_box = torch.where(f_ok, bb[..., 3], 1.0)[..., None, None]
    xy_n = torch.where(good[..., None], kp[..., :2], 0.0) / h_box

    def _kcos(hip: int, knee: int, ank: int) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = xy_n[:, :, hip] - xy_n[:, :, knee], xy_n[:, :, ank] - xy_n[:, :, knee]
        nu, nv = torch.linalg.vector_norm(u, dim=-1), torch.linalg.vector_norm(v, dim=-1)
        ok = good[:, :, hip] & good[:, :, knee] & good[:, :, ank] & (nu > 1e-6) & (nv > 1e-6)
        c = ((u * v).sum(dim=-1) / torch.where(ok, nu * nv, 1.0)).clamp(-1.0, 1.0)
        return torch.where(ok, c, 0.0), ok

    cl, okl = _kcos(11, 13, 15)
    cr, okr = _kcos(12, 14, 16)
    signed_knee = torch.where(okl & okr, cl - cr, 0.0).float().unsqueeze(-1)
    f_coronal = history.new_zeros((bsz, t_obs, 2))
    f_lean = history.new_zeros((bsz, t_obs, 2))
    if h_mats is not None and y_sums is not None:
        hm64, ys64 = h_mats.double(), y_sums.double()
        if hm64.ndim != 3 or hm64.shape[1:] != (3, 3) or not bool((torch.linalg.det(hm64).abs() > 1e-12).all()):
            raise ValueError("build_ground_pose_o2_inputs requires non-singular (B, 3, 3) homography matrices")
        v_bbox_foot = bb[:, :, 1] + 0.5 * bb[:, :, 3]
        ank_ok = good[:, :, 15] & good[:, :, 16] & f_ok
        v_foot = torch.where(ank_ok, 0.5 * (kp[:, :, 15, 1] + kp[:, :, 16, 1]), v_bbox_foot)
        uv_foot = torch.stack((bb[:, :, 0], v_foot), dim=-1)
        p0, ok0 = _map_uv_torch(uv_foot, hm64, ys64)
        pu, oku = _map_uv_torch(uv_foot + uv_foot.new_tensor([2.0, 0.0]), hm64, ys64)
        pv, okv = _map_uv_torch(uv_foot + uv_foot.new_tensor([0.0, 2.0]), hm64, ys64)
        du_g, dv_g = pu - p0, pv - p0
        s_hand = torch.where(du_g[..., 0] * dv_g[..., 1] - du_g[..., 1] * dv_g[..., 0] >= 0.0, 1.0, -1.0)

        sh_ok = good[:, :, 5] & good[:, :, 6] & f_ok & ok0 & oku & okv
        hp_ok = good[:, :, 11] & good[:, :, 12] & f_ok & ok0 & oku & okv
        duv_sh = 0.5 * (kp[:, :, 5, :2] - kp[:, :, 6, :2])
        duv_hp = 0.5 * (kp[:, :, 11, :2] - kp[:, :, 12, :2])
        g_lsh, ok_ls = _map_uv_torch(uv_foot + duv_sh, hm64, ys64)
        g_rsh, ok_rs = _map_uv_torch(uv_foot - duv_sh, hm64, ys64)
        g_lhp, ok_lh = _map_uv_torch(uv_foot + duv_hp, hm64, ys64)
        g_rhp, ok_rh = _map_uv_torch(uv_foot - duv_hp, hm64, ys64)

        d_sh = g_lsh - g_rsh
        ok_cor_s = sh_ok & ok_ls & ok_rs & (torch.linalg.vector_norm(d_sh, dim=-1) <= 1.2)
        d_hp = g_lhp - g_rhp
        ok_cor_h = hp_ok & ok_lh & ok_rh & (torch.linalg.vector_norm(d_hp, dim=-1) <= 1.2)
        d_lr = torch.where(ok_cor_s[..., None], d_sh, torch.where(ok_cor_h[..., None], d_hp, 0.0))
        f_coronal = s_hand[..., None] * torch.stack((-d_lr[..., 1], d_lr[..., 0]), dim=-1)

        u_sh_mid = 0.5 * (kp[:, :, 5, 0] + kp[:, :, 6, 0])
        u_hp_mid = 0.5 * (kp[:, :, 11, 0] + kp[:, :, 12, 0])
        g_shm, ok_sm = _map_uv_torch(torch.stack((u_sh_mid, v_foot), dim=-1), hm64, ys64)
        g_hpm, ok_hm = _map_uv_torch(torch.stack((u_hp_mid, v_foot), dim=-1), hm64, ys64)
        lean = g_shm - g_hpm
        ok_ln = sh_ok & hp_ok & ok_sm & ok_hm & (torch.linalg.vector_norm(lean, dim=-1) <= 0.8)
        f_lean = torch.where(ok_ln[..., None], lean, 0.0)

    pose_vec_world = torch.stack((f_coronal, f_lean), dim=2)  # (B, T, 2, 2)
    return {
        "bio_even": bio_even,
        "q_bio": q_bio,
        "signed_knee": signed_knee,
        "pose_vec_world": pose_vec_world,
        "m_pose": m_pose,
    }


def local_frenet_with_pose(
    history: torch.Tensor,
    pose_vec_world: torch.Tensor | None = None,
    use_pose_backup: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    origin = history[:, -1:]
    back = (origin - history[:, :-1]).flip(1)
    lengths = torch.linalg.vector_norm(back, dim=2)
    elig = lengths >= (AXIS_THRESHOLD_M - 1e-12)
    valid_m = elig.any(1)
    rows = torch.arange(len(history), device=history.device)
    idx = elig.long().argmax(1)
    u_m = back[rows, idx] / lengths[rows, idx, None].clamp_min(1e-12)

    if use_pose_backup and pose_vec_world is not None:
        f_mean = pose_vec_world[:, :, 0].mean(dim=1)
        fnorm = torch.linalg.vector_norm(f_mean, dim=1)
        valid_p = fnorm >= 0.03
        u_p = f_mean / fnorm[:, None].clamp_min(1e-12)
        unit = torch.where(valid_m[:, None], u_m, torch.where(valid_p[:, None], u_p, 0.0))
        valid = valid_m | valid_p
    else:
        unit = torch.where(valid_m[:, None], u_m, 0.0)
        valid = valid_m
    left = torch.stack((-unit[:, 1], unit[:, 0]), dim=1)
    basis = torch.where(valid[:, None, None], torch.stack((unit, left), dim=2), 0.0)
    return origin, basis, valid
def polyline_crossing_rate_np(paths_rel: np.ndarray, p_env: np.ndarray) -> float:
    """Fraction of windows where predicted 3.0 s trajectory intersects any observed local polyline segment."""
    n = len(paths_rel)
    if n == 0:
        return 0.0
    crossed = np.zeros(n, dtype=bool)
    traj = np.concatenate((np.zeros((n, 1, 2), dtype=np.float64), paths_rel.astype(np.float64)), axis=1)
    a1, a2 = traj[:, :-1], traj[:, 1:]  # (N, 30, 2)
    for i in range(n):
        valid_poly = p_env[i, :, :, 10] > 0.5
        if not np.any(valid_poly):
            continue
        pts = p_env[i, :, :, 0:2].astype(np.float64)
        for k in range(pts.shape[0]):
            if valid_poly[k].sum() < 2:
                continue
            pk = pts[k, valid_poly[k]]
            b1, b2 = pk[:-1], pk[1:]  # (M, 2)
            da = a2[i, :, None, :] - a1[i, :, None, :]
            db = b2[None, :, :] - b1[None, :, :]
            dp = a1[i, :, None, :] - b1[None, :, :]
            denom = da[..., 0] * db[..., 1] - da[..., 1] * db[..., 0]
            ok = np.abs(denom) > 1e-9
            t = np.where(ok, (db[..., 0] * dp[..., 1] - db[..., 1] * dp[..., 0]) / np.where(ok, denom, 1.0), -1.0)
            u = np.where(ok, (da[..., 0] * dp[..., 1] - da[..., 1] * dp[..., 0]) / np.where(ok, denom, 1.0), -1.0)
            if np.any(ok & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)):
                crossed[i] = True
                break
    return float(crossed.mean())


class PPTMainlineO2Forecaster(nn.Module):
    """Unified O(2)-equivariant forecaster supporting all 11 PPT Slide 13 arms (M0 through M3)."""

    def __init__(self, arm: str = "M3", v_ref: float = 0.85) -> None:
        super().__init__()
        if arm not in PPT_ARMS:
            raise ValueError(f"Unknown PPT arm: {arm}")
        self.arm = arm
        self.v_ref = v_ref
        self.use_pose = arm != "M0"
        self.use_poly = arm in ("M2a", "M2", "M3a", "M3b", "M3")
        self.use_graph = arm in ("M2b_no_vrad", "M2b", "M2", "M3a", "M3b", "M3")
        self.keep_vrad = arm != "M2b_no_vrad"
        self.use_act_head = arm in ("M3a", "M3b", "M3")
        self.act_bridge_mode = "none" if arm not in ("M3b", "M3") else ("e2e" if arm == "M3b" else "stop_grad")

        self.motion = nn.GRU(5, 48, batch_first=True)
        self.ctx_proj = nn.Sequential(nn.LayerNorm(48), nn.Linear(48, 48), nn.SiLU())
        self.cv_low_bonus = nn.Parameter(torch.tensor(0.35))

        self.pose_in = nn.Linear(16, 24)
        self.pose_gru = nn.GRU(24, 24, batch_first=True)
        self.gate_no_q = nn.Sequential(nn.Linear(48 + 24, 24), nn.SiLU(), nn.Linear(24, 1))
        self.gate_with_q = nn.Sequential(nn.Linear(48 + 24 + 4, 24), nn.SiLU(), nn.Linear(24, 1))
        self.pose_proj = nn.Linear(24, 48, bias=False)
        nn.init.zeros_(self.pose_proj.weight)
        self.poly_pt = nn.Sequential(nn.Linear(10, 24, bias=False), nn.SiLU(), nn.Linear(24, 32, bias=False))
        self.poly_att = nn.Linear(48 + 32, 1)
        self.poly_null = nn.Parameter(torch.tensor(1.5))
        self.poly_proj = nn.Linear(32, 48, bias=False)
        nn.init.zeros_(self.poly_proj.weight)

        self.edge_mlp = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 48))
        self.edge_att = nn.Linear(96, 1)
        self.null_logit = nn.Parameter(torch.tensor(1.5))
        self.graph_proj = nn.Linear(48, 48, bias=False)
        nn.init.zeros_(self.graph_proj.weight)

        self.act_even = nn.Linear(48 + 24, 3)
        self.act_turn = nn.Linear(48, 1)
        self.bridge_even = nn.Linear(4, 48, bias=False)
        self.bridge_odd = nn.Linear(1, 48, bias=False)
        nn.init.zeros_(self.bridge_even.weight)
        nn.init.zeros_(self.bridge_odd.weight)

        self.central_dx = nn.Linear(48, 3 * 3)
        self.central_dy = nn.Linear(48, 3 * 3, bias=False)
        self.pair_knots = nn.Linear(48, 3 * 2)
        self.central_scores = nn.Linear(48, 4)
        self.pair_score = nn.Linear(48, 1)
        for m in (self.central_dx, self.pair_knots):
            nn.init.normal_(m.weight, std=0.01); nn.init.zeros_(m.bias)
        nn.init.normal_(self.central_dy.weight, std=0.01)

    def _poly_delta(self, ctx: torch.Tensor, p_feat: torch.Tensor, p_pt_mask: torch.Tensor) -> torch.Tensor:
        emb = self.poly_pt(p_feat) * p_pt_mask.unsqueeze(-1)
        cnt = p_pt_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        poly_vec = emb.sum(dim=2) / cnt
        poly_ok = p_pt_mask.sum(dim=2) >= 2.0
        att = self.poly_att(torch.cat((ctx[:, None, :].expand(-1, poly_vec.size(1), -1), poly_vec), dim=-1)).squeeze(-1)
        att = att.masked_fill(~poly_ok, -1e9)
        w = torch.softmax(torch.cat((att, self.poly_null.expand(len(ctx), 1)), dim=1), dim=1)[:, :16]
        return self.poly_proj((w[:, :, None] * poly_vec).sum(dim=1))

    def _graph_delta(self, ctx: torch.Tensor, e_feat: torch.Tensor, nbr_mask: torch.Tensor) -> torch.Tensor:
        emb = self.edge_mlp(e_feat)
        att = self.edge_att(torch.cat((ctx[:, None, :].expand_as(emb), emb), dim=-1)).squeeze(-1)
        att = att.masked_fill(nbr_mask < 0.5, -1e9)
        w = torch.softmax(torch.cat((att, self.null_logit.expand(len(ctx), 1)), dim=1), dim=1)[:, :12]
        return self.graph_proj((w[:, :, None] * emb).sum(dim=1))
    def forward(
        self,
        history: torch.Tensor,
        bio_even: torch.Tensor | None = None,
        pose_vec_world: torch.Tensor | None = None,
        signed_knee: torch.Tensor | None = None,
        q_vec: torch.Tensor | None = None,
        m_pose: torch.Tensor | None = None,
        p_env: torch.Tensor | None = None,
        m_map: torch.Tensor | None = None,
        edges: torch.Tensor | None = None,
        is_car: torch.Tensor | None = None,
        nbr_mask: torch.Tensor | None = None,
        obs_dt: torch.Tensor | None = None,
        fut_times_s: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, t_obs, _ = history.shape
        use_p_bkp = self.use_pose and (m_pose is not None)
        pvw_masked = (pose_vec_world * m_pose[:, None, None, None].to(pose_vec_world.dtype)) if (pose_vec_world is not None and m_pose is not None) else None
        origin, basis, valid = local_frenet_with_pose(history, pvw_masked, use_pose_backup=use_p_bkp)
        body = (history - origin) @ basis
        dt_step = history.new_full((bsz, t_obs - 1), 0.1) if obs_dt is None else obs_dt.to(history.dtype).clamp_min(1e-4)
        vel = torch.cat((torch.zeros_like(body[:, :1]), (body[:, 1:] - body[:, :-1]) / dt_step[:, :, None]), dim=1)
        spd = torch.linalg.vector_norm(vel, dim=2, keepdim=True)
        fp = torch.cat((body, vel, spd), dim=2).float()
        fn = fp * fp.new_tensor([1.0, -1.0, 1.0, -1.0, 1.0])
        _, hm = self.motion(torch.cat((fp, fn), dim=0))
        cb = self.ctx_proj(hm[0])
        ctx_pos, ctx_neg = cb[:bsz], cb[bsz:]

        last_v = (torch.linalg.vector_norm(history[:, -1] - history[:, -2], dim=-1) / dt_step[:, -1]).float()
        disp_all = torch.linalg.vector_norm(history[:, 1:] - history[:, :-1], dim=-1).sum(dim=-1).float()
        hp_even = last_v.new_zeros((bsz, 24))
        gate = last_v.new_zeros(bsz)

        if self.use_pose and bio_even is not None and pose_vec_world is not None and signed_knee is not None and m_pose is not None:
            pv_loc = torch.einsum("btvk,bks->btvs", pose_vec_world.double(), basis).float()
            fc_p, fl_p = pv_loc[:, :, 0], pv_loc[:, :, 1]
            flip = fc_p.new_tensor([1.0, -1.0])
            pp = torch.cat((bio_even, fc_p, fl_p, signed_knee), dim=-1)
            pn = torch.cat((bio_even, fc_p * flip, fl_p * flip, -signed_knee), dim=-1)
            _, hpb = self.pose_gru(F.silu(self.pose_in(torch.cat((pp, pn), dim=0))))
            hp_pos, hp_neg = hpb[0][:bsz], hpb[0][bsz:]
            hp_even = 0.5 * (hp_pos + hp_neg)
            ce0 = 0.5 * (ctx_pos + ctx_neg)
            if self.arm == "M1a":
                gate = m_pose
            elif self.arm == "M1b":
                gate = m_pose * torch.sigmoid(self.gate_no_q(torch.cat((ce0, hp_even), dim=-1)).squeeze(-1))
            else:
                qv = q_vec if q_vec is not None else last_v.new_zeros((bsz, 4))
                gate = m_pose * torch.sigmoid(self.gate_with_q(torch.cat((ce0, hp_even, qv), dim=-1)).squeeze(-1))
            ctx_pos = ctx_pos + gate[:, None] * self.pose_proj(hp_pos)
            ctx_neg = ctx_neg + gate[:, None] * self.pose_proj(hp_neg)

        if self.use_poly and p_env is not None and m_map is not None:
            pxy = torch.einsum("bkpw,bws->bkps", p_env[:, :, :, 0:2].double(), basis).float()
            pattr = p_env[:, :, :, 2:10].float()
            pmask = p_env[:, :, :, 10].float()
            pp_f = torch.cat((pxy, pattr), dim=-1)
            pn_f = torch.cat((pxy * pxy.new_tensor([1.0, -1.0]), pattr), dim=-1)
            mm = m_map.view(bsz, 1).float()
            ctx_pos = ctx_pos + mm * self._poly_delta(ctx_pos, pp_f, pmask)
            ctx_neg = ctx_neg + mm * self._poly_delta(ctx_neg, pn_f, pmask)

        if self.use_graph and edges is not None and is_car is not None and nbr_mask is not None:
            dp_l = torch.einsum("bmk,bks->bms", edges[:, :, 0:2].double(), basis).float()
            dv_l = torch.einsum("bmk,bks->bms", edges[:, :, 2:4].double(), basis).float()
            vrad = edges[:, :, 5] if self.keep_vrad else torch.zeros_like(edges[:, :, 5])
            inv = torch.stack((edges[:, :, 4] / 10.0, vrad, edges[:, :, 6], is_car), dim=-1).float()
            ep = torch.cat((dp_l, dv_l, inv), dim=-1)
            en = torch.cat((dp_l * ep.new_tensor([1.0, -1.0]), dv_l * ep.new_tensor([1.0, -1.0]), inv), dim=-1)
            ctx_pos = ctx_pos + self._graph_delta(ctx_pos, ep, nbr_mask)
            ctx_neg = ctx_neg + self._graph_delta(ctx_neg, en, nbr_mask)

        ctx_even, ctx_odd = 0.5 * (ctx_pos + ctx_neg), 0.5 * (ctx_pos - ctx_neg)
        ev_a = self.act_even(torch.cat((ctx_even, gate[:, None] * hp_even), dim=-1))
        act_logits = torch.cat((ev_a[:, :2], self.act_turn(ctx_pos), self.act_turn(ctx_neg), ev_a[:, 2:3]), dim=-1)

        if self.act_bridge_mode != "none":
            src_a = act_logits.detach() if self.act_bridge_mode == "stop_grad" else act_logits
            pa = torch.sigmoid(src_a)
            de = self.bridge_even(torch.stack((pa[:, 0], pa[:, 1], pa[:, 4], 0.5 * (pa[:, 2] + pa[:, 3])), dim=-1))
            do = self.bridge_odd(0.5 * (pa[:, 2:3] - pa[:, 3:4]))
            d_even, d_odd, d_pos, d_neg = ctx_even + de, ctx_odd + do, ctx_pos + de + do, ctx_neg + de - do
        else:
            d_even, d_odd, d_pos, d_neg = ctx_even, ctx_odd, ctx_pos, ctx_neg

        low_damp = torch.tanh((disp_all / 0.25).clamp_min(0.05))[:, None, None]
        c_dx = low_damp * self.central_dx(d_even).view(bsz, 3, 3)
        c_dy = low_damp * self.central_dy(d_odd).view(bsz, 3, 3)
        p_pos = low_damp * self.pair_knots(d_pos).view(bsz, 3, 2)
        p_neg = low_damp * self.pair_knots(d_neg).view(bsz, 3, 2)
        ck = torch.stack((c_dx, c_dy), dim=-1)
        m4 = torch.stack((p_neg[..., 0], -p_neg[..., 1]), dim=-1)[:, None]
        knots5 = torch.cat((ck[:, :2], p_pos[:, None], m4, ck[:, 2:]), dim=1).double()
        knots5 = torch.cat((torch.zeros_like(knots5[:, :, :1]), knots5), dim=2)

        csc = self.central_scores(d_even)
        c0 = csc[:, :1] + self.cv_low_bonus * torch.exp(-((disp_all / 0.25) ** 2))[:, None]
        raw_lg = torch.cat((c0, csc[:, 1:3], self.pair_score(d_pos), self.pair_score(d_neg), csc[:, 3:]), dim=1)
        logits = torch.where(valid[:, None], raw_lg, 0.0)

        t_sec = torch.arange(1, 31, dtype=history.dtype, device=history.device)[None, :] * 0.1 if fut_times_s is None else fut_times_s.to(history.dtype)
        idx = (t_sec - 1e-9).floor().long().clamp(0, 2)
        ph = (t_sec - idx.to(history.dtype))[:, None, :, None]
        bi = torch.arange(bsz, device=history.device)[:, None, None]
        mi = torch.arange(5, device=history.device)[None, :, None]
        delta = knots5[bi, mi, idx[:, None, :]] * (1.0 - ph) + knots5[bi, mi, (idx + 1)[:, None, :]] * ph
        world = torch.einsum("bkts,bws->bktw", delta, basis)
        cv = origin + ((history[:, -1:] - history[:, -2:-1]) / dt_step[:, -1:, None]) * t_sec[:, :, None]
        return {"paths": torch.cat((cv[:, None], cv[:, None] + world), dim=1), "logits": logits, "act_logits": act_logits, "valid": valid, "gate": gate}


def verify_pinns_ground_and_polyline_diagnostics(
    history: torch.Tensor,
    bbox: torch.Tensor,
    pose_vec_world: torch.Tensor,
    h_mats: torch.Tensor,
    y_sums: torch.Tensor,
    pt0_ego_all: np.ndarray | None = None,
    p_env: np.ndarray | None = None,
    p_env_bug_origin: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Hard runtime assertions preventing zeroed ground vectors or mis-centered 20 m polylines."""
    pvw = pose_vec_world.detach().cpu().numpy()
    fc_norm = np.linalg.norm(pvw[:, :, 0], axis=-1)
    fl_norm = np.linalg.norm(pvw[:, :, 1], axis=-1)
    total_pf = int(fc_norm.size)
    nz_fc = int((fc_norm > 1e-6).sum())
    nz_fl = int((fl_norm > 1e-6).sum())
    med_fc = float(np.median(fc_norm[fc_norm > 1e-6])) if nz_fc > 0 else 0.0
    if nz_fc < max(5, int(0.30 * total_pf)) or not (0.05 <= med_fc <= 0.90):
        raise RuntimeError(f"Ground coronal vector check failed: nz_fc={nz_fc}/{total_pf}, med_fc={med_fc:.4f} m")
    if nz_fl < max(3, int(0.10 * total_pf)):
        raise RuntimeError(f"Ground lean vector check failed: nz_fl={nz_fl}/{total_pf}")

    hist_np = history.detach().cpu().numpy()
    disp = hist_np[:, -1] - hist_np[:, 0]
    dn = np.linalg.norm(disp, axis=-1)
    fn_last = fc_norm[:, -1]
    mv = (dn >= 0.30) & (fn_last >= 0.08)
    mean_cos = float(np.mean(np.sum(disp[mv] * pvw[mv, -1, 0], axis=-1) / (dn[mv] * fn_last[mv]))) if mv.any() else 0.0
    if mv.sum() >= 30 and mean_cos <= 0.05:
        raise RuntimeError(f"Ground coronal facing orientation failed velocity alignment check: mean_cos={mean_cos:.4f}")

    diag: dict[str, float | int] = {
        "total_person_frames": total_pf,
        "nonzero_f_coronal_frames": nz_fc,
        "nonzero_f_lean_frames": nz_fl,
        "median_nonzero_f_coronal_norm_m": med_fc,
        "moving_coronal_velocity_mean_cos": mean_cos,
    }
    if pt0_ego_all is not None:
        bb_last = bbox[:, -1].double()
        uv_last = torch.stack((bb_last[:, 0], bb_last[:, 1] + 0.5 * bb_last[:, 3]), dim=-1)
        proj_xy, _ = _map_uv_torch(uv_last, h_mats.double(), y_sums.double())
        foot_err = np.linalg.norm(proj_xy.cpu().numpy() - pt0_ego_all, axis=-1)
        med_foot_err = float(np.median(foot_err))
        med_ego_norm = float(np.median(np.linalg.norm(pt0_ego_all, axis=-1)))
        if med_foot_err > 0.05 or med_ego_norm < 15.0:
            raise RuntimeError(f"Ego world calibration check failed: med_foot_err={med_foot_err:.6f} m, med_ego_norm={med_ego_norm:.3f} m")
        diag["footpoint_to_published_xy_median_err_m"] = med_foot_err
        diag["median_pt0_ego_world_norm_m"] = med_ego_norm
    if p_env is not None and p_env_bug_origin is not None:
        changed = int(np.any(np.abs(p_env - p_env_bug_origin) > 1e-5, axis=(1, 2, 3)).sum())
        if changed < 500:
            raise RuntimeError(f"Expected >500 map windows to differ from world-(0,0) center, got {changed}")
        diag["windows_changed_vs_origin_00_bug"] = changed
    return diag
