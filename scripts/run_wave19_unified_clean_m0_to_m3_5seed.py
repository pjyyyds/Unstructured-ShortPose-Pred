"""Wave 19 GPU1: Unified remediated O(2) chain (M0 -> M1_mono_q -> M2b_conflict -> M3_stopgrad_platt + fixed conformal) on PINNS & JRDB."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.stats import spearmanr
import torch
from torch import nn
import torch.nn.functional as F

from scripts.ppt_mainline_o2_models import PPTMainlineO2Forecaster, local_frenet_with_pose
from scripts.run_dair_o2_bioeq_external import SEEDS
from scripts.run_jrdb_rc1_pose_quality_and_initiation_5seed import _loc_bootstrap_ci
from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import _stage_gpu_dict, load_jrdb_10scene_rc2_rc3
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_conformal_and_events import conformal_quantile
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import _copy_matching, load_ppt_cohort_with_causal_maps
from scripts.run_wave18_action_calibration_and_rerank_5seed import (
    candidate_implied_actions_np,
    fit_platt_and_thresholds,
    rerank_top1_modes,
    score_grouped_actions,
)
from scripts.run_wave18_conflict_gated_interaction_5seed import (
    build_interaction_strata,
    compute_causal_conflict_risk,
    safe_site_clip_ci,
)
from src.models.bioeq_o2_joint import o2_joint_loss

UNIFIED_ARMS = ("U_M0", "U_M1_mono", "U_M2b_mono_conflict", "U_M3_mono_conflict")


class UnifiedCleanO2Forecaster(PPTMainlineO2Forecaster):
    """Unified remediated PPT O(2) forecaster: M1_mono_q + M2b_ttc_conflict_gate + M3 Stop-Grad action bridge."""

    def __init__(self, stage: str = "U_M3_mono_conflict") -> None:
        base_arm = {"U_M0": "M0", "U_M1_mono": "M1", "U_M2b_mono_conflict": "M2b", "U_M3_mono_conflict": "M3"}[stage]
        super().__init__(arm=base_arm)
        self.stage = stage
        self.use_pose = stage != "U_M0"
        self.use_graph = stage in ("U_M2b_mono_conflict", "U_M3_mono_conflict")
        self.use_act = stage == "U_M3_mono_conflict"
        self.mono_log_gamma = nn.Parameter(torch.tensor(0.35))
        self.mono_w_raw = nn.Parameter(torch.tensor([0.8, 0.8, 0.3, 0.2]))
        self.mono_bias = nn.Parameter(torch.tensor(-0.8))
        self.graph_scale = 1.0

    def compute_mono_gate(self, q_vec: torch.Tensor, m_pose: torch.Tensor) -> torch.Tensor:
        q = q_vec.clamp(0.0, 1.0)
        gamma = F.softplus(self.mono_log_gamma).clamp(0.1, 4.0)
        prod_q = (q[:, 0] * q[:, 1]).clamp(0.0, 1.0).pow(gamma)
        lin = self.mono_bias + (q * F.softplus(self.mono_w_raw)[None, :]).sum(dim=-1)
        return m_pose * prod_q * torch.sigmoid(lin)

    def forward(self, history: torch.Tensor, bio_even: torch.Tensor | None = None, pose_vec_world: torch.Tensor | None = None,
                signed_knee: torch.Tensor | None = None, q_vec: torch.Tensor | None = None, m_pose: torch.Tensor | None = None,
                edges: torch.Tensor | None = None, is_car: torch.Tensor | None = None, nbr_mask: torch.Tensor | None = None,
                obs_dt: torch.Tensor | None = None, fut_times_s: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        bsz, t_obs, _ = history.shape
        pvw_m = (pose_vec_world * m_pose[:, None, None, None].to(pose_vec_world.dtype)) if (self.use_pose and pose_vec_world is not None and m_pose is not None) else None
        origin, basis, valid = local_frenet_with_pose(history, pvw_m, use_pose_backup=self.use_pose)
        body = (history - origin) @ basis
        dt_s = history.new_full((bsz, t_obs - 1), 0.1) if obs_dt is None else obs_dt.to(history.dtype).clamp_min(1e-4)
        vel = torch.cat((torch.zeros_like(body[:, :1]), (body[:, 1:] - body[:, :-1]) / dt_s[:, :, None]), dim=1)
        fp = torch.cat((body, vel, torch.linalg.vector_norm(vel, dim=2, keepdim=True)), dim=2).float()
        _, hm = self.motion(torch.cat((fp, fp * fp.new_tensor([1.0, -1.0, 1.0, -1.0, 1.0])), dim=0))
        cb = self.ctx_proj(hm[0])
        cp, cn = cb[:bsz], cb[bsz:]
        disp_all = torch.linalg.vector_norm(history[:, 1:] - history[:, :-1], dim=-1).sum(dim=-1).float()
        gate = disp_all.new_zeros(bsz)
        hp_even = disp_all.new_zeros((bsz, 24))
        if self.use_pose and bio_even is not None and pose_vec_world is not None and signed_knee is not None and m_pose is not None:
            pvl = torch.einsum("btvk,bks->btvs", pose_vec_world.double(), basis).float()
            fc, fl = pvl[:, :, 0], pvl[:, :, 1]
            flp = fc.new_tensor([1.0, -1.0])
            _, hpb = self.pose_gru(F.silu(self.pose_in(torch.cat((torch.cat((bio_even, fc, fl, signed_knee), -1), torch.cat((bio_even, fc * flp, fl * flp, -signed_knee), -1)), 0))))
            hpp, hpn = hpb[0][:bsz], hpb[0][bsz:]
            hp_even = 0.5 * (hpp + hpn)
            gate = self.compute_mono_gate(q_vec if q_vec is not None else disp_all.new_zeros((bsz, 4)), m_pose)
            cp, cn = cp + gate[:, None] * self.pose_proj(hpp), cn + gate[:, None] * self.pose_proj(hpn)
        w_risk = disp_all.new_zeros(bsz)
        if self.use_graph and edges is not None and is_car is not None and nbr_mask is not None and self.graph_scale > 0.0:
            dp_l = torch.einsum("bmk,bks->bms", edges[:, :, 0:2].double(), basis).float()
            dv_l = torch.einsum("bmk,bks->bms", edges[:, :, 2:4].double(), basis).float()
            inv = torch.stack((edges[:, :, 4] / 10.0, edges[:, :, 5], edges[:, :, 6], is_car), dim=-1).float()
            ep = torch.cat((dp_l, dv_l, inv), dim=-1)
            en = torch.cat((dp_l * ep.new_tensor([1.0, -1.0]), dv_l * ep.new_tensor([1.0, -1.0]), inv), dim=-1)
            w_risk = compute_causal_conflict_risk(edges, nbr_mask)
            sc = float(self.graph_scale) * w_risk[:, None]
            cp = cp + sc * self._graph_delta(cp, ep, nbr_mask)
            cn = cn + sc * self._graph_delta(cn, en, nbr_mask)

        ce, co = 0.5 * (cp + cn), 0.5 * (cp - cn)
        ev_a = self.act_even(torch.cat((ce, gate[:, None] * hp_even), dim=-1))
        act_logits = torch.cat((ev_a[:, :2], self.act_turn(cp), self.act_turn(cn), ev_a[:, 2:3]), dim=-1)
        if self.use_act:
            pa = torch.sigmoid(act_logits.detach())
            de = self.bridge_even(torch.stack((pa[:, 0], pa[:, 1], pa[:, 4], 0.5 * (pa[:, 2] + pa[:, 3])), dim=-1))
            do = self.bridge_odd(0.5 * (pa[:, 2:3] - pa[:, 3:4]))
            de_c, do_c, dp_c, dn_c = ce + de, co + do, cp + de + do, cn + de - do
        else:
            de_c, do_c, dp_c, dn_c = ce, co, cp, cn

        ld = torch.tanh((disp_all / 0.25).clamp_min(0.05))[:, None, None]
        ck = torch.stack((ld * self.central_dx(de_c).view(bsz, 3, 3), ld * self.central_dy(do_c).view(bsz, 3, 3)), dim=-1)
        pp_k, pn_k = ld * self.pair_knots(dp_c).view(bsz, 3, 2), ld * self.pair_knots(dn_c).view(bsz, 3, 2)
        knots5 = torch.cat((ck[:, :2], pp_k[:, None], torch.stack((pn_k[..., 0], -pn_k[..., 1]), -1)[:, None], ck[:, 2:]), dim=1).double()
        knots5 = torch.cat((torch.zeros_like(knots5[:, :, :1]), knots5), dim=2)
        csc = self.central_scores(de_c)
        c0 = csc[:, :1] + self.cv_low_bonus * torch.exp(-((disp_all / 0.25) ** 2))[:, None]
        logits = torch.where(valid[:, None], torch.cat((c0, csc[:, 1:3], self.pair_score(dp_c), self.pair_score(dn_c), csc[:, 3:]), dim=1), 0.0)
        ts = torch.arange(1, 31, dtype=history.dtype, device=history.device)[None, :] * 0.1 if fut_times_s is None else fut_times_s.to(history.dtype)
        idx = (ts - 1e-9).floor().long().clamp(0, 2)
        ph = (ts - idx.to(history.dtype))[:, None, :, None]
        bi, mi = torch.arange(bsz, device=history.device)[:, None, None], torch.arange(5, device=history.device)[None, :, None]
        delta = knots5[bi, mi, idx[:, None, :]] * (1.0 - ph) + knots5[bi, mi, (idx + 1)[:, None, :]] * ph
        cv = origin + ((history[:, -1:] - history[:, -2:-1]) / dt_s[:, -1:, None]) * ts[:, :, None]
        return {"paths": torch.cat((cv[:, None], cv[:, None] + torch.einsum("bkts,bws->bktw", delta, basis)), dim=1),
                "logits": logits, "act_logits": act_logits, "valid": valid, "gate": gate, "w_risk": w_risk}


def _fwd_u(net: UnifiedCleanO2Forecaster, gd: dict, ids_g: torch.Tensor) -> dict[str, torch.Tensor]:
    return net(gd["history"][ids_g], gd["bio_even"][ids_g], gd["pose_vec_world"][ids_g], gd["signed_knee"][ids_g],
               gd["quality"][ids_g], gd["m_pose"][ids_g], gd["edges"][ids_g], gd["is_car"][ids_g], gd["nbr_mask"][ids_g],
               obs_dt=gd["obs_dt"][ids_g], fut_times_s=gd["fut_s"][ids_g])


def fit_unified_clean_fold(seed: int, grp_arr: np.ndarray, gd: dict, tr_ids: np.ndarray, bsz: int,
                           ep_s: tuple[int, int, int, int], lr_s: tuple[float, float, float, float],
                           device: torch.device, g_scale: float = 1.0) -> dict[str, UnifiedCleanO2Forecaster]:
    w_tr = make_site_weights(grp_arr, tr_ids).to(device)
    tr_t = torch.from_numpy(tr_ids).to(device)
    truth_tr, ya_tr, va_tr = gd["truth"][tr_t], gd["y_act"][tr_t], gd["valid_act"][tr_t]
    pw = (((~ya_tr) & va_tr).sum(0).float().clamp_min(1.0) / (ya_tr & va_tr).sum(0).float().clamp_min(1.0)).clamp(1.0, 12.0)

    def _tr(net: UnifiedCleanO2Forecaster, prefs: tuple[str, ...], ep: int, lr: float, wd: float, aw: float, soff: int) -> UnifiedCleanO2Forecaster:
        for nm, p in net.named_parameters():
            p.requires_grad = nm.startswith(prefs) or nm in prefs
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(seed + soff)
        net.train()
        for _ in range(ep):
            for ch in torch.randperm(len(tr_ids), generator=gen).split(bsz):
                cg = ch.to(device)
                loss = o2_joint_loss(_fwd_u(net, gd, tr_t[cg]), truth_tr[cg], w_tr[cg],
                                     ya_tr[cg] if aw > 0 else None, va_tr[cg] if aw > 0 else None, pw if aw > 0 else None, act_weight=aw)
                if loss is not None:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return net.eval()

    torch.manual_seed(seed)
    m0 = _tr(UnifiedCleanO2Forecaster("U_M0").to(device), ("motion.", "ctx_proj.", "cv_low_bonus", "central_", "pair_"), ep_s[0], lr_s[0], 1e-4, 0.0, 0)
    torch.manual_seed(seed); m1 = UnifiedCleanO2Forecaster("U_M1_mono").to(device); _copy_matching(m1, m0)
    m1 = _tr(m1, ("pose_", "mono_", "central_", "pair_", "cv_low_bonus"), ep_s[1], lr_s[1], 3e-4, 0.0, 11)
    torch.manual_seed(seed); m2b = UnifiedCleanO2Forecaster("U_M2b_mono_conflict").to(device); _copy_matching(m2b, m1)
    m2b = _tr(m2b, ("edge_", "graph_proj", "null_logit", "central_", "pair_", "cv_low_bonus"), ep_s[2], lr_s[2], 3e-4, 0.0, 22)
    m2b.graph_scale = g_scale
    torch.manual_seed(seed); m3 = UnifiedCleanO2Forecaster("U_M3_mono_conflict").to(device); _copy_matching(m3, m2b)
    m3.graph_scale = g_scale
    m3 = _tr(m3, ("ctx_proj.", "act_", "bridge_", "central_", "pair_", "cv_low_bonus"), ep_s[3], lr_s[3], 2e-4, 0.25, 33)
    return {"U_M0": m0, "U_M1_mono": m1, "U_M2b_mono_conflict": m2b, "U_M3_mono_conflict": m3}


def select_inner_unified_calibration(
    seed: int, grp_arr: np.ndarray, gd: dict, hist_np: np.ndarray, truth_np: np.ndarray,
    ya_np: np.ndarray, va_np: np.ndarray, dt: float, tr_ids: np.ndarray, bsz: int,
    ep_s: tuple[int, int, int, int], lr_s: tuple[float, float, float, float], device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    src_grps = sorted(set(grp_arr[tr_ids].tolist()))[:4]
    scales = (0.0, 0.25, 0.5, 1.0)
    sc_oof = {sc: [] for sc in scales}
    oof_mask = np.zeros(len(grp_arr), dtype=bool)
    oof_act_lg = np.zeros((len(grp_arr), 5), dtype=np.float64)
    oof_mode_lg = np.zeros((len(grp_arr), 6), dtype=np.float64)
    oof_paths = np.zeros((len(grp_arr), 6, truth_np.shape[1], 2), dtype=np.float64)

    for vg in src_grps:
        in_tr = tr_ids[grp_arr[tr_ids] != vg]
        in_va = tr_ids[grp_arr[tr_ids] == vg]
        va_g = torch.from_numpy(in_va).to(device)
        mods = fit_unified_clean_fold(seed, grp_arr, gd, in_tr, bsz, ep_s, lr_s, device, g_scale=1.0)
        m2b, m3 = mods["U_M2b_mono_conflict"], mods["U_M3_mono_conflict"]
        with torch.inference_mode():
            for sc in scales:
                m2b.graph_scale = sc
                oo = _fwd_u(m2b, gd, va_g)
                tp = oo["paths"][torch.arange(len(in_va), device=device), oo["logits"].argmax(1)]
                sc_oof[sc].append(float(torch.linalg.vector_norm(tp - gd["truth"][va_g], dim=-1).mean()))
            m3.graph_scale = 1.0
            o3 = _fwd_u(m3, gd, va_g)
            oof_mask[in_va] = True
            oof_act_lg[in_va] = o3["act_logits"].cpu().numpy()
            oof_mode_lg[in_va] = o3["logits"].cpu().numpy()
            oof_paths[in_va] = o3["paths"].cpu().numpy()

    best_sc = float(min(scales, key=lambda s: np.mean(sc_oof[s]) + (1e-7 if s > 0 else 0.0)))
    om_ids = np.flatnonzero(oof_mask)
    a_pl, b_pl, thr_pl = fit_platt_and_thresholds(oof_act_lg[om_ids], ya_np[om_ids], va_np[om_ids])
    p_oof = 1.0 / (1.0 + np.exp(-np.clip(oof_act_lg[om_ids] * a_pl[None, :] + b_pl[None, :], -30.0, 30.0)))
    cand_oof = candidate_implied_actions_np(hist_np[om_ids], oof_paths[om_ids] - hist_np[om_ids, -1:, None, :], dt=dt)
    betas = (0.0, 0.05, 0.10, 0.20, 0.35)
    best_b, best_err = 0.0, 1e9
    for bt in betas:
        k_idx = rerank_top1_modes(oof_mode_lg[om_ids], p_oof, cand_oof, bt)
        d = np.linalg.norm(oof_paths[om_ids][np.arange(len(om_ids)), k_idx] - truth_np[om_ids], axis=-1).mean()
        if d + (1e-7 if bt > 0 else 0.0) < best_err:
            best_err, best_b = float(d), float(bt)
    return best_sc, a_pl, b_pl, thr_pl, best_b


def run_unified_benchmark_on_cohort(
    tag: str, grp_arr: np.ndarray, unit_arr: np.ndarray, is_pinns: bool, gd: dict,
    hist_np: np.ndarray, truth_np: np.ndarray, ya_np: np.ndarray, va_np: np.ndarray,
    q_np: np.ndarray, m_pose_np: np.ndarray, cv_err: np.ndarray, cv_tube: np.ndarray,
    strata: dict[str, np.ndarray], dt: float, bsz: int, ep_s: tuple[int, int, int, int],
    lr_s: tuple[float, float, float, float], seeds: list[int], device: torch.device,
) -> dict:
    n = len(grp_arr)
    groups = sorted(set(grp_arr.tolist()))
    all_arms = (*UNIFIED_ARMS, "U_M3_mono_conflict_rerank")
    top_s = {a: np.zeros((len(seeds), n, 2), dtype=np.float64) for a in all_arms}
    min_s = {a: np.zeros((len(seeds), n, 2), dtype=np.float64) for a in UNIFIED_ARMS}
    tubes_s = {a: np.zeros((len(seeds), n, 2), dtype=np.float64) for a in UNIFIED_ARMS}
    gate_s = np.zeros((len(seeds), n), dtype=np.float64)
    raw_act_s = np.zeros((len(seeds), n, 5), dtype=np.float64)
    cal_act_s = np.zeros((len(seeds), n, 5), dtype=np.float64)
    thr_acc, sc_log, beta_log = [], {g: [] for g in groups}, {g: [] for g in groups}

    for si, seed in enumerate(seeds):
        for g in groups:
            tr = np.flatnonzero(grp_arr != g)
            te = np.flatnonzero(grp_arr == g)
            te_g = torch.from_numpy(te).to(device)
            sc, a_pl, b_pl, thr_pl, beta = select_inner_unified_calibration(
                seed, grp_arr, gd, hist_np, truth_np, ya_np, va_np, dt, tr, bsz, ep_s, lr_s, device,
            )
            sc_log[g].append(sc)
            beta_log[g].append(beta)
            thr_acc.append(thr_pl)
            mods = fit_unified_clean_fold(seed, grp_arr, gd, tr, bsz, ep_s, lr_s, device, g_scale=sc)
            with torch.inference_mode():
                for a in UNIFIED_ARMS:
                    oo = _fwd_u(mods[a], gd, te_g)
                    pw = oo["paths"].cpu().numpy()
                    lg = oo["logits"].cpu().numpy()
                    d_all = np.linalg.norm(pw - truth_np[te, None], axis=-1)
                    d_top = d_all[np.arange(len(te)), lg.argmax(1)]
                    top_s[a][si, te] = np.column_stack((d_top.mean(1), d_top[:, -1]))
                    min_s[a][si, te] = np.column_stack((d_all.mean(2).min(1), d_all[:, :, -1].min(1)))
                    tubes_s[a][si, te] = np.column_stack((d_top.max(1), d_all.max(2).min(1)))
                    if a == "U_M1_mono":
                        gate_s[si, te] = oo["gate"].cpu().numpy()
                    if a == "U_M3_mono_conflict":
                        alg = oo["act_logits"].cpu().numpy()
                        raw_act_s[si, te] = 1.0 / (1.0 + np.exp(-np.clip(alg, -30.0, 30.0)))
                        p_cal = 1.0 / (1.0 + np.exp(-np.clip(alg * a_pl[None, :] + b_pl[None, :], -30.0, 30.0)))
                        cal_act_s[si, te] = p_cal
                        c_act = candidate_implied_actions_np(hist_np[te], pw - hist_np[te, -1:, None, :], dt=dt)
                        rk = rerank_top1_modes(lg, p_cal, c_act, beta)
                        dr = d_all[np.arange(len(te)), rk]
                        top_s["U_M3_mono_conflict_rerank"][si, te] = np.column_stack((dr.mean(1), dr[:, -1]))
        print(f"[W19-GPU1 {tag}] Finished seed={seed}", flush=True)

    top_m = {a: top_s[a].mean(0) for a in all_arms}
    min_m = {a: min_s[a].mean(0) for a in UNIFIED_ARMS}
    ci_fn = (lambda d, m: safe_site_clip_ci(d[m], grp_arr[m], unit_arr[m])) if is_pinns else (lambda d, m: _loc_bootstrap_ci(d[m], grp_arr[m]))

    strata_out = {}
    for sk, sm in strata.items():
        if int(sm.sum()) == 0:
            continue
        gs = sorted(set(grp_arr[sm].tolist()))
        mods_d = {
            "Last2_CV": {
                "group_equal": {"ade1_m": float(np.mean([cv_err[(grp_arr == g) & sm, 0].mean() for g in gs])),
                                "fde1_m": float(np.mean([cv_err[(grp_arr == g) & sm, 1].mean() for g in gs]))},
                "window_equal": {"ade1_m": float(cv_err[sm, 0].mean()), "fde1_m": float(cv_err[sm, 1].mean())},
            }
        }
        for a in all_arms:
            ge = {"ade1_m": float(np.mean([top_m[a][(grp_arr == g) & sm, 0].mean() for g in gs])),
                  "fde1_m": float(np.mean([top_m[a][(grp_arr == g) & sm, 1].mean() for g in gs]))}
            we = {"ade1_m": float(top_m[a][sm, 0].mean()), "fde1_m": float(top_m[a][sm, 1].mean())}
            if a in min_m:
                ge["min_ade6_m"] = float(np.mean([min_m[a][(grp_arr == g) & sm, 0].mean() for g in gs]))
                ge["min_fde6_m"] = float(np.mean([min_m[a][(grp_arr == g) & sm, 1].mean() for g in gs]))
                we["min_ade6_m"] = float(min_m[a][sm, 0].mean())
                we["min_fde6_m"] = float(min_m[a][sm, 1].mean())
            mods_d[a] = {"group_equal": ge, "window_equal": we}
        paired = {
            "U_M1_mono_minus_U_M0": ci_fn(top_m["U_M1_mono"] - top_m["U_M0"], sm),
            "U_M2b_conflict_minus_U_M1_mono": ci_fn(top_m["U_M2b_mono_conflict"] - top_m["U_M1_mono"], sm),
            "U_M3_conflict_minus_U_M2b_conflict": ci_fn(top_m["U_M3_mono_conflict"] - top_m["U_M2b_mono_conflict"], sm),
            "U_M3_rerank_minus_U_M3_conflict": ci_fn(top_m["U_M3_mono_conflict_rerank"] - top_m["U_M3_mono_conflict"], sm),
            "U_M2b_conflict_minus_CV": ci_fn(top_m["U_M2b_mono_conflict"] - cv_err, sm),
            "U_M3_rerank_minus_CV": ci_fn(top_m["U_M3_mono_conflict_rerank"] - cv_err, sm),
        }
        strata_out[sk] = {"n_windows": int(sm.sum()), "n_groups_present": len(gs), "models": mods_d, "paired_top1_ci": paired}

    g_mean = gate_s.mean(0)
    vp = m_pose_np > 0.5
    rho_q1 = float(spearmanr(g_mean[vp], q_np[vp, 0]).statistic) if int(vp.sum()) >= 5 and np.std(q_np[vp, 0]) > 1e-8 else 0.0
    mean_thr = np.mean(thr_acc, axis=0) if thr_acc else np.full(5, 0.5)

    # Fixed-predictor zero-retrain within-group split-conformal on unified arms
    conf = {a: {l: {"cov_cv": [], "rad_cv": [], "cov_t1": [], "rad_t1": [], "cov_u6": [], "rad_u6": []} for l in ("90", "95")} for a in UNIFIED_ARMS}
    for si in range(len(seeds)):
        for g in groups:
            ids = np.flatnonzero(grp_arr == g)
            if len(ids) < 16:
                continue
            mid = len(ids) // 2
            ci, ti = ids[:mid], ids[mid:]
            for a in UNIFIED_ARMS:
                tb = tubes_s[a][si]
                for lvl, al in (("90", 0.10), ("95", 0.05)):
                    rcv, rt1, ru6 = conformal_quantile(cv_tube[ci], al), conformal_quantile(tb[ci, 0], al), conformal_quantile(tb[ci, 1], al)
                    conf[a][lvl]["cov_cv"].append(float((cv_tube[ti] <= rcv).mean())); conf[a][lvl]["rad_cv"].append(rcv)
                    conf[a][lvl]["cov_t1"].append(float((tb[ti, 0] <= rt1).mean())); conf[a][lvl]["rad_t1"].append(rt1)
                    conf[a][lvl]["cov_u6"].append(float((tb[ti, 1] <= ru6).mean())); conf[a][lvl]["rad_u6"].append(ru6)

    return {
        "n_windows": n,
        "selected_conflict_scales": sc_log,
        "selected_rerank_betas": beta_log,
        "spearman_mono_gate_vs_q1": rho_q1,
        "strata": strata_out,
        "action_raw_thr05": score_grouped_actions(ya_np, va_np, raw_act_s.mean(0), grp_arr, np.full(5, 0.5)),
        "action_platt_thr05": score_grouped_actions(ya_np, va_np, cal_act_s.mean(0), grp_arr, np.full(5, 0.5)),
        "action_platt_inner_opt_thr": score_grouped_actions(ya_np, va_np, cal_act_s.mean(0), grp_arr, mean_thr),
        "fixed_predictor_within_group_conformal": {a: {l: {k: float(np.mean(v)) for k, v in d.items()} for l, d in wd.items()} for a, wd in conf.items() if conf[a]["90"]["cov_u6"]},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = load_ppt_cohort_with_causal_maps(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin)
    np_w = len(dp["history"])
    gdp = {k: dp[k].to(dev) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask", "y_act", "valid_act")}
    gdp["obs_dt"] = torch.full((np_w, 4), 0.1, dtype=torch.float64, device=dev)
    gdp["fut_s"] = torch.arange(1, 31, dtype=torch.float64, device=dev)[None].expand(np_w, -1) * 0.1

    hp_np, tp_np = dp["history"].numpy(), dp["truth"].numpy()
    cv_p = hp_np[:, -1:] + ((hp_np[:, -1:] - hp_np[:, -2:-1]) / 0.1) * (np.arange(1, 31, dtype=np.float64)[None, :, None] * 0.1)
    d_cvp = np.linalg.norm(cv_p - tp_np, axis=-1)
    p_int = build_interaction_strata(dp["edges"].numpy(), dp["is_car"].numpy(), dp["nbr_mask"].numpy(), has_cars=True)
    p_strata = {
        "all_1482": np.ones(np_w, dtype=bool),
        "first_short_323": dp["strata"]["first_short_323"],
        "clear_pose": dp["strata"]["clear_pose"],
        "heavy_pose": dp["strata"]["heavy_pose"],
        "S1_non_approaching": p_int["S1_non_approaching"],
        "S3_ttc_lt4s": p_int["S3_ttc_lt4s"],
        "S4_approaching_car": p_int["S4_approaching_car"],
    }
    p_res = run_unified_benchmark_on_cohort(
        "pinns", dp["site"], dp["clip"], True, gdp, hp_np, tp_np, dp["y_act"].numpy(), dp["valid_act"].numpy(),
        dp["quality"].numpy(), dp["m_pose"].numpy(), np.column_stack((d_cvp.mean(1), d_cvp[:, -1])), d_cvp.max(1),
        p_strata, 0.1, 64, (45, 25, 26, 28), (2e-3, 1.2e-3, 1.5e-3, 1.2e-3), list(args.seeds), dev,
    )

    dj = load_jrdb_10scene_rc2_rc3(args.repo_root)
    gdj = _stage_gpu_dict(dj, dev)
    hj_np, tj_np = dj["history"].numpy(), dj["truth"].numpy()
    cv_j = hj_np[:, -1:] + ((hj_np[:, 1:2] - hj_np[:, 0:1]) / 0.4) * (np.arange(1, 8, dtype=np.float64)[None, :, None] * 0.4)
    d_cvj = np.linalg.norm(cv_j - tj_np, axis=-1)
    j_int = build_interaction_strata(dj["edges"].numpy(), dj["is_car"].numpy(), dj["nbr_mask"].numpy(), has_cars=False)
    j_strata = {
        "all_10scenes": np.ones(len(hj_np), dtype=bool),
        "S1_non_approaching": j_int["S1_non_approaching"],
        "S2_close_approaching": j_int["S2_close_approaching"],
        "S3_ttc_lt4s": j_int["S3_ttc_lt4s"],
    }
    j_res = run_unified_benchmark_on_cohort(
        "jrdb", dj["location"], dj["actors"], False, gdj, hj_np, tj_np, dj["y_act"].numpy(), dj["valid_act"].numpy(),
        dj["quality"].numpy(), dj["m_pose"].numpy(), np.column_stack((d_cvj.mean(1), d_cvj[:, -1])), d_cvj.max(1),
        j_strata, 0.4, 512, (45, 28, 28, 30), (2.5e-3, 1.5e-3, 1.8e-3, 1.5e-3), list(args.seeds), dev,
    )

    summary = {
        "experiment": "wave19_unified_clean_m0_to_m3_5seed",
        "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
        "seeds": list(args.seeds),
        "pinns_4site_loso_1482": p_res,
        "jrdb_9location_lolo_5836": j_res,
    }
    raw = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = (json.dumps({"summary.json": hashlib.sha256(raw).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw)
        zf.writestr("sha256.json", sh)
    print("Saved:", args.zip_path, flush=True)


if __name__ == "__main__":
    main()
