"""Wave 28A: PINNS Terminal-Mobility Viewing-Ray O(2) (TM-RKR6), Zero-Retrain Split-Conformal Safety Tubes, & Jitter Stress."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
import torch

from scripts.benchmark_heldout_sites import load_site_windows
from scripts.ppt_mainline_o2_models import build_ground_pose_o2_inputs, fit_pinns_ground_homography
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import load_ppt_cohort_with_causal_maps
from scripts.run_wave18_conflict_gated_interaction_5seed import safe_site_clip_ci
from scripts.run_wave23_pinns_video_audit_and_6slot_ranker import (
    _fit_cond_precursor_head,
    blind_pinns_homography_and_raw_video_audit,
)
from scripts.run_wave25_pinns_explicit_audit_and_emergence_safe import _pooled_clip_bootstrap_ci
from src.models.apa_precursor_leadtime import build_delayed_onset_start_stop_candidates
from src.models.cond_precursor_ranker import (
    build_conditioned_precursor_features,
    select_medoid_expected_ade,
    summarize_6candidate_5seed,
)
from src.models.emergence_kalman_residual_6slot import (
    apply_ekr6_regime_selector,
    compute_split_anatomical_covisibility,
    ekr6_decoupled_slot0_and_maneuver_loss,
    fit_ekr6_source_oof_regime_selector,
)
from src.models.emergence_maneuver_6slot import (
    audit_6candidate_generation_vs_selection,
    compute_action_labels_with_onset,
    fit_strict_source_platt_and_slot_selector,
    horizon_and_post_onset_errors,
)
from src.models.ray_kalman_and_cusum import compute_homography_jacobian_world
from src.models.tm_ray_rkr6 import O2TerminalMobilityRay6SlotNet

SEEDS = (42, 1024, 2025, 3407, 4096)


def _fwd_tmr6(
    model: O2TerminalMobilityRay6SlotNet,
    gd: dict[str, torch.Tensor],
    split_covis: dict[str, torch.Tensor] | None,
    jac_t: torch.Tensor | None,
    idx_t: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sc_sub = {k: v[idx_t] for k, v in split_covis.items()} if split_covis is not None else None
    return model(
        history=gd["history"][idx_t],
        bio_even=gd["bio_even"][idx_t],
        pose_vec_world=gd["pose_vec_world"][idx_t],
        signed_knee=gd["signed_knee"][idx_t],
        q_vec=gd["quality"][idx_t],
        m_pose=gd["m_pose"][idx_t],
        split_covis=sc_sub,
        jac_uv_world=jac_t[idx_t] if jac_t is not None else None,
        edges=gd["edges"][idx_t],
        is_car=gd["is_car"][idx_t],
        nbr_mask=gd["nbr_mask"][idx_t],
        obs_dt=gd["obs_dt"][idx_t],
        fut_times_s=gd["fut_s"][idx_t],
    )


def fit_tmr6_arm_fold(
    seed: int,
    use_pose: bool,
    use_nbr: bool,
    use_ray: bool,
    grp_arr: np.ndarray,
    gd: dict[str, torch.Tensor],
    split_covis: dict[str, torch.Tensor] | None,
    jac_t: torch.Tensor | None,
    tr_ids: np.ndarray,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> O2TerminalMobilityRay6SlotNet:
    torch.manual_seed(seed + (157 if use_pose else 0) + (311 if use_nbr else 0) + (17 if use_ray else 0))
    model = O2TerminalMobilityRay6SlotNet(use_pose=use_pose, use_interaction=use_nbr, use_ray_aniso_gain=use_ray).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    g_tr = grp_arr[tr_ids]
    u_g = sorted(set(g_tr.tolist()))
    w_np = np.zeros(len(tr_ids), dtype=np.float64)
    for g in u_g:
        m = g_tr == g
        w_np[m] = len(tr_ids) / (len(u_g) * max(int(m.sum()), 1))
    w_all = torch.zeros(gd["history"].size(0), dtype=torch.float64, device=device)
    w_all[torch.from_numpy(tr_ids).to(device)] = torch.from_numpy(w_np).to(device)

    tr_t = torch.from_numpy(tr_ids).to(device)
    ya_tr, va_tr = gd["y_act"][tr_t], gd["valid_act"][tr_t]
    pw = ((va_tr.float().sum(0) - ya_tr.float().sum(0)) / ya_tr.float().sum(0).clamp_min(1.0)).clamp(1.0, 14.0)
    rng = np.random.default_rng(seed + 103)
    for _ in range(epochs):
        perm = rng.permutation(tr_ids)
        for s in range(0, len(perm), batch_size):
            b_t = torch.from_numpy(perm[s : s + batch_size]).to(device)
            out = _fwd_tmr6(model, gd, split_covis, jac_t, b_t)
            loss = ekr6_decoupled_slot0_and_maneuver_loss(
                out, gd["truth"][b_t], gd["history"][b_t], w_all[b_t], gd["y_act"][b_t], gd["valid_act"][b_t], pw
            )
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sch.step()
    model.eval()
    return model


def eval_zero_retrain_split_conformal(
    oof_paths6_by_fold: dict[str, np.ndarray],
    te_paths6_5seed: np.ndarray,
    cv_world: np.ndarray,
    truth_np: np.ndarray,
    v_last_spd: np.ndarray,
    grp_arr: np.ndarray,
    unit_arr: np.ndarray,
    strata: dict[str, np.ndarray],
) -> dict:
    """Zero-retrain locally speed-spread weighted split-conformal tubes (90% & 95%, Top-1 & Union-6)."""
    n, _, tf, _ = te_paths6_5seed.shape
    t_scale = np.linspace(0.25, 1.0, tf, dtype=np.float64)[None, :]
    spread_all = np.linalg.norm(te_paths6_5seed - te_paths6_5seed[:, :1], axis=-1).mean(axis=(1, 2))
    sig_te = (0.22 + 0.55 * v_last_spd + 0.35 * spread_all)[:, None] * t_scale
    sig_cv = (0.22 + 0.65 * v_last_spd)[:, None] * t_scale

    err_top1 = np.linalg.norm(te_paths6_5seed[:, 0] - truth_np, axis=-1)
    err_uni6 = np.linalg.norm(te_paths6_5seed - truth_np[:, None], axis=-1).min(axis=1)
    err_cv = np.linalg.norm(cv_world - truth_np, axis=-1)

    # 1. Cross-country LOSO zero-retrain calibration (calibrated strictly on 3 source sites' OOF predictions)
    cov_loso = {a: {"top1_cov": np.zeros(n, dtype=bool), "uni6_cov": np.zeros(n, dtype=bool), "cv_cov": np.zeros(n, dtype=bool),
                    "top1_rad": np.zeros(n), "uni6_rad": np.zeros(n), "cv_rad": np.zeros(n)} for a in ("90", "95")}
    for held in sorted(set(grp_arr.tolist())):
        te, tr = np.flatnonzero(grp_arr == held), np.flatnonzero(grp_arr != held)
        op6 = oof_paths6_by_fold[str(held)]
        sp_tr = np.linalg.norm(op6 - op6[:, :1], axis=-1).mean(axis=(1, 2))
        sig_tr = (0.22 + 0.55 * v_last_spd[tr] + 0.35 * sp_tr)[:, None] * t_scale
        sig_cv_tr = (0.22 + 0.65 * v_last_spd[tr])[:, None] * t_scale
        s_t1 = (np.linalg.norm(op6[:, 0] - truth_np[tr], axis=-1) / sig_tr).max(axis=1)
        s_u6 = (np.linalg.norm(op6 - truth_np[tr, None], axis=-1).min(axis=1) / sig_tr).max(axis=1)
        s_cv = (err_cv[tr] / sig_cv_tr).max(axis=1)
        m_cal = len(tr)
        for a_nm, alpha in (("90", 0.10), ("95", 0.05)):
            k_q = min(max(int(math.ceil((m_cal + 1) * (1.0 - alpha))) - 1, 0), m_cal - 1)
            q1, qu, qc = float(np.sort(s_t1)[k_q]), float(np.sort(s_u6)[k_q]), float(np.sort(s_cv)[k_q])
            cov_loso[a_nm]["top1_cov"][te] = np.all(err_top1[te] <= q1 * sig_te[te], axis=1)
            cov_loso[a_nm]["uni6_cov"][te] = np.all(err_uni6[te] <= qu * sig_te[te], axis=1)
            cov_loso[a_nm]["cv_cov"][te] = np.all(err_cv[te] <= qc * sig_cv[te], axis=1)
            cov_loso[a_nm]["top1_rad"][te] = (q1 * sig_te[te]).mean(axis=1)
            cov_loso[a_nm]["uni6_rad"][te] = (qu * sig_te[te]).mean(axis=1)
            cov_loso[a_nm]["cv_rad"][te] = (qc * sig_cv[te]).mean(axis=1)

    # 2. Within-site clip-disjoint split-conformal (zero retraining of frozen source-trained predictor!)
    cov_ws = {a: {"top1_cov": [], "uni6_cov": [], "cv_cov": [], "top1_rad": [], "uni6_rad": [], "cv_rad": []} for a in ("90", "95")}
    for held in sorted(set(grp_arr.tolist())):
        te = np.flatnonzero(grp_arr == held)
        clips = sorted(set(unit_arr[te].tolist()))
        if len(clips) < 2:
            continue
        c_cal = set(clips[::2])
        cal_i = np.array([i for i in te if unit_arr[i] in c_cal], dtype=np.int64)
        ev_i = np.array([i for i in te if unit_arr[i] not in c_cal], dtype=np.int64)
        for cal_sub, ev_sub in ((cal_i, ev_i), (ev_i, cal_i)):
            s1 = (err_top1[cal_sub] / sig_te[cal_sub]).max(axis=1)
            su = (err_uni6[cal_sub] / sig_te[cal_sub]).max(axis=1)
            sc = (err_cv[cal_sub] / sig_cv[cal_sub]).max(axis=1)
            mc = len(cal_sub)
            for a_nm, alpha in (("90", 0.10), ("95", 0.05)):
                kq = min(max(int(math.ceil((mc + 1) * (1.0 - alpha))) - 1, 0), mc - 1)
                q1, qu, qc = float(np.sort(s1)[kq]), float(np.sort(su)[kq]), float(np.sort(sc)[kq])
                cov_ws[a_nm]["top1_cov"].extend(np.all(err_top1[ev_sub] <= q1 * sig_te[ev_sub], axis=1).tolist())
                cov_ws[a_nm]["uni6_cov"].extend(np.all(err_uni6[ev_sub] <= qu * sig_te[ev_sub], axis=1).tolist())
                cov_ws[a_nm]["cv_cov"].extend(np.all(err_cv[ev_sub] <= qc * sig_cv[ev_sub], axis=1).tolist())
                cov_ws[a_nm]["top1_rad"].extend((q1 * sig_te[ev_sub]).mean(axis=1).tolist())
                cov_ws[a_nm]["uni6_rad"].extend((qu * sig_te[ev_sub]).mean(axis=1).tolist())
                cov_ws[a_nm]["cv_rad"].extend((qc * sig_cv[ev_sub]).mean(axis=1).tolist())

    out_c = {"within_site_clip_split_zero_retrain": {}, "cross_country_loso_zero_retrain_by_stratum": {}}
    for a_nm in ("90", "95"):
        w = cov_ws[a_nm]
        out_c["within_site_clip_split_zero_retrain"][f"nominal_{a_nm}pct"] = {
            "tmr6_top1_joint_cov": float(np.mean(w["top1_cov"])),
            "tmr6_union6_joint_cov": float(np.mean(w["uni6_cov"])),
            "cv_joint_cov": float(np.mean(w["cv_cov"])),
            "tmr6_top1_mean_radius_m": float(np.mean(w["top1_rad"])),
            "tmr6_union6_mean_radius_m": float(np.mean(w["uni6_rad"])),
            "cv_mean_radius_m": float(np.mean(w["cv_rad"])),
        }
    for s_nm, msk in strata.items():
        if int(msk.sum()) == 0:
            continue
        out_c["cross_country_loso_zero_retrain_by_stratum"][s_nm] = {
            f"nominal_{a_nm}pct": {
                "tmr6_top1_joint_cov": float(cov_loso[a_nm]["top1_cov"][msk].mean()),
                "tmr6_union6_joint_cov": float(cov_loso[a_nm]["uni6_cov"][msk].mean()),
                "cv_joint_cov": float(cov_loso[a_nm]["cv_cov"][msk].mean()),
                "tmr6_top1_mean_radius_m": float(cov_loso[a_nm]["top1_rad"][msk].mean()),
                "tmr6_union6_mean_radius_m": float(cov_loso[a_nm]["uni6_rad"][msk].mean()),
                "cv_mean_radius_m": float(cov_loso[a_nm]["cv_rad"][msk].mean()),
            }
            for a_nm in ("90", "95")
        }
    return out_c


def eval_tmr6_pinns_5seed(
    grp_arr: np.ndarray,
    unit_arr: np.ndarray,
    gd: dict[str, torch.Tensor],
    kpts_t: torch.Tensor,
    bbox_t: torch.Tensor,
    hm_t: torch.Tensor,
    ys_t: torch.Tensor,
    jac_t: torch.Tensor,
    hist_np: np.ndarray,
    truth_np: np.ndarray,
    ya_np: np.ndarray,
    va_np: np.ndarray,
    onset_np: np.ndarray,
    strata: dict[str, np.ndarray],
    diag_mask: np.ndarray,
    device: torch.device,
    seeds: tuple[int, ...] = SEEDS,
    epochs: int = 34,
    batch_size: int = 128,
    lr: float = 1.8e-3,
) -> dict:
    dt = 0.1
    n, tf = len(hist_np), truth_np.shape[1]
    fut_s_np = gd["fut_s"].detach().cpu().numpy()
    cor_xy = gd["pose_vec_world"][:, -1, 0].detach().cpu().numpy()
    path_len_all = np.linalg.norm(np.diff(hist_np, axis=1), axis=-1).sum(axis=1)
    v_last = (hist_np[:, -1] - hist_np[:, -2]) / dt
    v_last_spd = np.linalg.norm(v_last, axis=-1)
    cv_world = hist_np[:, -1:, :] + v_last[:, None, :] * fut_s_np[:, :, None]
    err_cv = horizon_and_post_onset_errors(cv_world, truth_np, onset_np, dt=dt)
    cv_rel, truth_rel = cv_world - hist_np[:, -1:, :], truth_np - hist_np[:, -1:, :]

    gd["y_act"] = torch.from_numpy(ya_np).to(device)
    gd["valid_act"] = torch.from_numpy(va_np).to(device)
    jac_g = jac_t.to(device)
    with torch.inference_mode():
        split_covis = compute_split_anatomical_covisibility(
            kpts_t.to(device), bbox_t.to(device), gd["bio_even"], gd["pose_vec_world"], gd["signed_knee"], gd["quality"]
        )
        feat20 = build_conditioned_precursor_features(
            gd["history"], gd["bio_even"], gd["pose_vec_world"], gd["signed_knee"],
            gd["quality"], gd["m_pose"], gd["edges"], gd["is_car"], gd["nbr_mask"], gd["obs_dt"],
        )

    arms = (
        ("TMR6_M0_nopose_nonbr", False, False, True),
        ("TMR6_M1_splitpose_nonbr", True, False, True),
        ("TMR6_M2b_nopose_conflict", False, True, True),
        ("TMR6_NoRayGain_Full", True, True, False),
        ("TMR6_Full_Slot0", True, True, True),
    )
    all_names = [a[0] for a in arms] + ["TMR6_Full_OOFRegime"]
    err_s = {nm: np.zeros((len(seeds), n, 6), dtype=np.float64) for nm in all_names}
    audits_5s = {nm: [] for nm in ("TMR6_M0_nopose_nonbr", "TMR6_Full_Slot0")}
    oof_regime_by_fold: dict[str, list[dict[str, float]]] = {str(g): [] for g in sorted(set(grp_arr.tolist()))}
    oof_p6_sum: dict[str, np.ndarray] = {
        str(g): np.zeros((int((grp_arr != g).sum()), 6, tf, 2), dtype=np.float64) for g in sorted(set(grp_arr.tolist()))
    }
    te_p6_sum = np.zeros((n, 6, tf, 2), dtype=np.float64)
    jit_sigmas = (0.0, 5.0, 10.0, 20.0)
    jit_ade_sum = {s_px: np.zeros(n, dtype=np.float64) for s_px in jit_sigmas}

    groups = sorted(set(grp_arr.tolist()))
    ep_oof = max(16, epochs // 2)
    t_start = time.perf_counter()

    for si, seed in enumerate(seeds):
        pk_seed = {nm: np.zeros((n, 6, tf, 2), dtype=np.float64) for nm in audits_5s}
        ix_seed = {nm: np.zeros(n, dtype=np.int64) for nm in audits_5s}
        for held in groups:
            te, tr = np.flatnonzero(grp_arr == held), np.flatnonzero(grp_arr != held)
            te_g, tr_g = torch.from_numpy(te).to(device), torch.from_numpy(tr).to(device)
            kin_tr_path = cv_world[tr]
            kin_te_path = cv_world[te]

            for arm_nm, u_pose, u_nbr, u_ray in arms:
                m_f = fit_tmr6_arm_fold(seed, u_pose, u_nbr, u_ray, grp_arr, gd, split_covis, jac_g, tr, batch_size, epochs, lr, device)
                with torch.inference_mode():
                    ot = _fwd_tmr6(m_f, gd, split_covis, jac_g, te_g)
                    pwt, lgt = ot["paths"].cpu().numpy(), ot["logits"].cpu().numpy()
                err_s[arm_nm][si, te] = horizon_and_post_onset_errors(pwt[:, 0], truth_np[te], onset_np[te], dt=dt)
                if arm_nm in pk_seed:
                    pk_seed[arm_nm][te] = pwt - hist_np[te, -1:, None, :]
                    ix_seed[arm_nm][te] = 0
                if arm_nm == "TMR6_Full_Slot0":
                    te_p6_sum[te] += pwt / len(seeds)
                    # Keypoint pixel jitter stress test on held-out fold
                    rng_j = np.random.default_rng(seed + 701)
                    for s_px in jit_sigmas:
                        if s_px == 0.0:
                            jit_ade_sum[s_px][te] += err_s[arm_nm][si, te, 0] / len(seeds)
                        else:
                            kp_j = kpts_t[te].clone()
                            kp_j[:, :, :, :2] += torch.from_numpy(rng_j.normal(0.0, s_px, size=kp_j[:, :, :, :2].shape).astype(np.float32))
                            kp_j[:, :, :, 2] = (kp_j[:, :, :, 2] * max(0.25, 1.0 - 0.02 * s_px)).clamp(0.0, 1.0)
                            gp_j = build_ground_pose_o2_inputs(gd["history"][te_g].cpu(), kp_j, bbox_t[te], hm_t[te], ys_t[te])
                            q_j = (gd["quality"][te_g] * max(0.25, 1.0 - 0.025 * s_px)).clamp(0.0, 1.0)
                            sc_j = compute_split_anatomical_covisibility(
                                kp_j.to(device), bbox_t[te].to(device),
                                gp_j["bio_even"].to(device), gp_j["pose_vec_world"].to(device),
                                gp_j["signed_knee"].to(device), q_j,
                            )
                            with torch.inference_mode():
                                pj = m_f(
                                    gd["history"][te_g], gp_j["bio_even"].to(device), gp_j["pose_vec_world"].to(device),
                                    gp_j["signed_knee"].to(device), q_j, gp_j["m_pose"].to(device),
                                    sc_j, jac_g[te_g], None, None, gd["edges"][te_g], gd["is_car"][te_g], gd["nbr_mask"][te_g],
                                    gd["obs_dt"][te_g], gd["fut_s"][te_g],
                                )["paths"][:, 0].cpu().numpy()
                            jit_ade_sum[s_px][te] += np.linalg.norm(pj - truth_np[te], axis=-1).mean(axis=-1) / len(seeds)

                    src_gs = sorted(set(grp_arr[tr].tolist()))
                    oof_ml = np.zeros((len(tr), 6), dtype=np.float64)
                    oof_cal = np.zeros((len(tr), 5), dtype=np.float64)
                    oof_pk = np.zeros((len(tr), 6, tf, 2), dtype=np.float64)
                    for g_val in src_gs:
                        vm_l = grp_arr[tr] == g_val
                        in_tr, in_va = tr[~vm_l], tr[vm_l]
                        m_in = fit_tmr6_arm_fold(seed + 19, True, True, True, grp_arr, gd, split_covis, jac_g, in_tr, batch_size, ep_oof, lr, device)
                        with torch.inference_mode():
                            oo = _fwd_tmr6(m_in, gd, split_covis, jac_g, torch.from_numpy(in_va).to(device))
                            otr = _fwd_tmr6(m_in, gd, split_covis, jac_g, torch.from_numpy(in_tr).to(device))
                            oof_ml[vm_l], oof_pk[vm_l] = oo["logits"].cpu().numpy(), oo["paths"].cpu().numpy()
                        ch_in = _fit_cond_precursor_head(seed + 19, feat20, otr["act_logits"].detach(), gd["y_act"][in_tr], gd["valid_act"][in_tr], in_tr, device)
                        with torch.inference_mode():
                            oof_cal[vm_l] = ch_in(feat20[torch.from_numpy(in_va).to(device)], oo["act_logits"]).cpu().numpy()

                    oof_p6_sum[str(held)] += oof_pk / len(seeds)
                    med_st = float(np.median(onset_np[tr][ya_np[tr, 0], 0]) * dt) if ya_np[tr, 0].any() else 0.7
                    st_o, sp_o = build_delayed_onset_start_stop_candidates(hist_np[tr], cor_xy[tr], np.full(len(tr), med_st), fut_s_np[tr], dt_obs=dt)
                    oof_pk_a = oof_pk.copy(); oof_pk_a[:, 1], oof_pk_a[:, 2] = st_o, sp_o
                    cal_c = fit_strict_source_platt_and_slot_selector(oof_ml, oof_cal, oof_pk_a, truth_np[tr], ya_np[tr], va_np[tr])
                    pc_oof = 1.0 / (1.0 + np.exp(-np.clip(oof_cal * np.asarray(cal_c["a_pl"])[None] + np.asarray(cal_c["b_pl"])[None], -30.0, 30.0)))
                    ix_mo = select_medoid_expected_ade(oof_pk_a, oof_ml, pc_oof, va_np[tr], beta=max(float(cal_c["beta"]), 0.15))
                    oof_med = oof_pk_a[np.arange(len(tr)), ix_mo]
                    rcfg_sel = fit_ekr6_source_oof_regime_selector(kin_tr_path, oof_pk[:, 0], oof_med, truth_np[tr], path_len_all[tr])
                    oof_regime_by_fold[str(held)].append(rcfg_sel)

                    with torch.inference_mode():
                        otr_f = _fwd_tmr6(m_f, gd, split_covis, jac_g, tr_g)
                    ch_f = _fit_cond_precursor_head(seed, feat20, otr_f["act_logits"].detach(), gd["y_act"][tr], gd["valid_act"][tr], tr, device)
                    with torch.inference_mode():
                        cal_te = ch_f(feat20[te_g], ot["act_logits"]).cpu().numpy()
                    st_t, sp_t = build_delayed_onset_start_stop_candidates(hist_np[te], cor_xy[te], np.full(len(te), med_st), fut_s_np[te], dt_obs=dt)
                    pwt_a = pwt.copy(); pwt_a[:, 1], pwt_a[:, 2] = st_t, sp_t
                    pc_te = 1.0 / (1.0 + np.exp(-np.clip(cal_te * np.asarray(cal_c["a_pl"])[None] + np.asarray(cal_c["b_pl"])[None], -30.0, 30.0)))
                    ix_mt = select_medoid_expected_ade(pwt_a, lgt, pc_te, va_np[te], beta=max(float(cal_c["beta"]), 0.15))
                    p_reg = apply_ekr6_regime_selector(kin_te_path, pwt[:, 0], pwt_a[np.arange(len(te)), ix_mt], path_len_all[te], rcfg_sel)
                    err_s["TMR6_Full_OOFRegime"][si, te] = horizon_and_post_onset_errors(p_reg, truth_np[te], onset_np[te], dt=dt)

            print(f"[Wave 28A] Finished seed={seed} ({si+1}/{len(seeds)}) site={held} elapsed={time.perf_counter()-t_start:.1f}s", flush=True)

        for nm in audits_5s:
            audits_5s[nm].append(
                audit_6candidate_generation_vs_selection(hist_np, pk_seed[nm], ix_seed[nm], truth_rel, cv_rel, ya_np, va_np, diag_mask, dt=dt)
            )

    em = {nm: err_s[nm].mean(axis=0) for nm in all_names}
    keys6 = ("ade1_full_m", "fde1_full_m", "ade1_trig_0p5s_m", "ade1_trig_1p0s_m", "ade1_post_onset_0p5s_m", "ade1_post_onset_1p0s_m")

    def _agg_eq(arr6: np.ndarray, m: np.ndarray) -> dict[str, float]:
        gs = [arr6[m & (grp_arr == g)].mean(axis=0) for g in groups if np.any(m & (grp_arr == g))]
        v = np.mean(gs, axis=0)
        return {k: float(v[j]) for j, k in enumerate(keys6)}

    def _agg_pool(arr6: np.ndarray, m: np.ndarray) -> dict[str, float]:
        v = arr6[m].mean(axis=0)
        return {k: float(v[j]) for j, k in enumerate(keys6)}

    def _pair_both(d2: np.ndarray, m: np.ndarray) -> dict:
        if int(m.sum()) < 4:
            return {"n": int(m.sum())}
        return {
            "group_equal": safe_site_clip_ci(d2[m], grp_arr[m], unit_arr[m]),
            "pooled_window_weighted": _pooled_clip_bootstrap_ci(d2[m], unit_arr[m]),
        }

    strata_out = {}
    for s_nm, msk in strata.items():
        cnt = int(msk.sum())
        if cnt == 0:
            continue
        m_eq = {"Last2_CV": _agg_eq(err_cv, msk), **{nm: _agg_eq(em[nm], msk) for nm in all_names}}
        m_pl = {"Last2_CV": _agg_pool(err_cv, msk), **{nm: _agg_pool(em[nm], msk) for nm in all_names}}
        per_grp = {}
        for g in groups:
            gm = msk & (grp_arr == g)
            if not gm.any():
                continue
            per_grp[str(g)] = {
                "n_windows": int(gm.sum()),
                "n_units": len(set(unit_arr[gm].tolist())),
                "maneuver_counts": {k: int(ya_np[gm, c].sum()) for c, k in enumerate(("Start", "Stop", "Turn_Left", "Turn_Right", "Maintain"))},
                "models": {"Last2_CV": _agg_pool(err_cv, gm), **{nm: _agg_pool(em[nm], gm) for nm in all_names}},
            }
        strata_out[s_nm] = {
            "n_windows": cnt,
            "n_units": len(set(unit_arr[msk].tolist())),
            "n_groups": len(set(grp_arr[msk].tolist())),
            "maneuver_counts": {k: int(ya_np[msk, c].sum()) for c, k in enumerate(("Start", "Stop", "Turn_Left", "Turn_Right", "Maintain"))},
            "models_group_equal": m_eq,
            "models_pooled_window_weighted": m_pl,
            "per_group_breakdown": per_grp,
            "paired_ci": {
                "tmr6_m0_minus_CV": _pair_both(em["TMR6_M0_nopose_nonbr"][:, :2] - err_cv[:, :2], msk),
                "tmr6_m1_minus_CV": _pair_both(em["TMR6_M1_splitpose_nonbr"][:, :2] - err_cv[:, :2], msk),
                "tmr6_m2b_minus_CV": _pair_both(em["TMR6_M2b_nopose_conflict"][:, :2] - err_cv[:, :2], msk),
                "tmr6_noray_minus_CV": _pair_both(em["TMR6_NoRayGain_Full"][:, :2] - err_cv[:, :2], msk),
                "tmr6_full_slot0_minus_CV": _pair_both(em["TMR6_Full_Slot0"][:, :2] - err_cv[:, :2], msk),
                "tmr6_oof_regime_minus_CV": _pair_both(em["TMR6_Full_OOFRegime"][:, :2] - err_cv[:, :2], msk),
                "tmr6_pose_no_nbr_full": _pair_both(em["TMR6_M1_splitpose_nonbr"][:, :2] - em["TMR6_M0_nopose_nonbr"][:, :2], msk),
                "tmr6_pose_no_nbr_post_onset": _pair_both(em["TMR6_M1_splitpose_nonbr"][:, 4:6] - em["TMR6_M0_nopose_nonbr"][:, 4:6], msk),
                "tmr6_nbr_no_pose_full": _pair_both(em["TMR6_M2b_nopose_conflict"][:, :2] - em["TMR6_M0_nopose_nonbr"][:, :2], msk),
                "tmr6_ray_gain_minus_noray": _pair_both(em["TMR6_Full_Slot0"][:, :2] - em["TMR6_NoRayGain_Full"][:, :2], msk),
            },
        }

    conf_out = eval_zero_retrain_split_conformal(oof_p6_sum, te_p6_sum, cv_world, truth_np, v_last_spd, grp_arr, unit_arr, strata)
    jitter_out = {
        s_nm: {
            f"jitter_{int(s_px)}px_ade_pool_m": float(jit_ade_sum[s_px][msk].mean())
            for s_px in jit_sigmas
        }
        for s_nm, msk in strata.items() if int(msk.sum()) > 0
    }
    return {
        "n_windows": n,
        "source_oof_regime_selectors": oof_regime_by_fold,
        "strata": strata_out,
        "zero_retrain_split_conformal_safety_tubes": conf_out,
        "keypoint_pixel_jitter_stress_test": jitter_out,
        "candidate_generation_vs_selection_5seed": {nm: summarize_6candidate_5seed(audits_5s[nm]) for nm in audits_5s},
    }


def run_pinns_tmr6(args: argparse.Namespace, device: torch.device) -> dict:
    dp = load_ppt_cohort_with_causal_maps(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin)
    groups_raw, _, _ = load_site_windows(args.meta_dir)
    rows = [r for s in sorted(groups_raw) for r in groups_raw[s]]
    keys = [(str(r.clip), int(r.person_id), int(r.t0)) for r in rows]
    n = len(rows)
    with np.load(args.pose_npz, allow_pickle=False) as pz:
        ord_map = {k: i for i, k in enumerate(zip(pz["clip"].tolist(), pz["person_id"].tolist(), pz["t0"].tolist()))}
        ids = [ord_map[k] for k in keys]
        kpts_t = torch.from_numpy(pz["hist_kpts"][ids].astype(np.float32))
        bbox_t = torch.from_numpy(pz["hist_bbox"][ids].astype(np.float32))

    h_mats = np.zeros((n, 3, 3), dtype=np.float64)
    y_sums = np.zeros(n, dtype=np.float64)
    for c in sorted({r.clip for r in rows}):
        calib = json.loads((args.meta_dir / "videos" / c.rsplit("_", 1)[0] / "calib.json").read_text())
        hm, ys = fit_pinns_ground_homography(calib)
        for i in [idx for idx, r in enumerate(rows) if r.clip == c]:
            h_mats[i], y_sums[i] = hm, ys
    hm_t, ys_t = torch.from_numpy(h_mats), torch.from_numpy(y_sums)
    jac_t = compute_homography_jacobian_world(bbox_t, hm_t, ys_t)

    hist_np, truth_np = dp["history"].numpy(), dp["truth"].numpy()
    hm5, fm30 = np.ones(5, dtype=np.float32), np.ones(30, dtype=np.float32)
    ya_p, va_p, on_p = np.zeros((n, 5), dtype=bool), np.zeros((n, 5), dtype=bool), np.full((n, 5), -1, dtype=np.int64)
    for i in range(n):
        ya_p[i], va_p[i], on_p[i] = compute_action_labels_with_onset(
            hist_np[i] - hist_np[i, -1:], hm5, truth_np[i] - hist_np[i, -1:], fm30, dt=0.1
        )

    exp_labels = json.loads(args.explicit_audit_json.read_text(encoding="utf-8"))["labels"]
    with tempfile.TemporaryDirectory() as td:
        exp_flat = Path(td) / "exp_flat.json"
        exp_flat.write_text(json.dumps(exp_labels), encoding="utf-8")
        cen_w24 = blind_pinns_homography_and_raw_video_audit(args.meta_dir, args.pose_npz, dp, args.video_cache_dir, None, args.wave24_labels_json)
        cen_exp = blind_pinns_homography_and_raw_video_audit(args.meta_dir, args.pose_npz, dp, args.video_cache_dir, None, exp_flat)

    act_m = (ya_p[:, :4] & va_p[:, :4]).any(axis=1)
    turn_m = (ya_p[:, 2] & va_p[:, 2]) | (ya_p[:, 3] & va_p[:, 3])
    e_exp, e_w24, fs323 = cen_exp["e_phys_all"], cen_w24["e_phys_all"], dp["strata"]["first_short_323"]
    strata = {
        "all_1482": np.ones(n, dtype=bool),
        "explicit_phys_emergence_118": e_exp,
        "explicit_matched_visible_ctrl_118": cen_exp["matched_visible_ctrl"],
        "core_explicit_short_phys_lt0p5m": fs323 & e_exp,
        "core_wave24_short_phys_86": fs323 & e_w24,
        "explicit_phys_emergence_active_maneuvers": e_exp & act_m,
        "explicit_phys_emergence_turn_only": e_exp & turn_m,
        "first_short_323": fs323,
    }
    gd_p = {k: dp[k].to(device) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
    gd_p["obs_dt"] = torch.full((n, 4), 0.1, dtype=torch.float64, device=device)
    gd_p["fut_s"] = torch.arange(1, 31, dtype=torch.float64, device=device)[None].expand(n, -1) * 0.1
    return eval_tmr6_pinns_5seed(dp["site"], dp["clip"], gd_p, kpts_t, bbox_t, hm_t, ys_t, jac_t, hist_np, truth_np, ya_p, va_p, on_p, strata, fs323 | e_exp, device, SEEDS, args.epochs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--video-cache-dir", type=Path, default=Path("/tmp/w28_vids"))
    ap.add_argument("--explicit-audit-json", type=Path, default=Path("data/annotations/pinns_explicit_241_visual_audit.json"))
    ap.add_argument("--wave24-labels-json", type=Path, default=Path("data/annotations/pinns_blind_raw_video_emergence_labels.json"))
    ap.add_argument("--zip-path", type=Path, default=Path("wave28_pinns_tm_rkr6_and_conformal.zip"))
    ap.add_argument("--epochs", type=int, default=34)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        torch.set_num_threads(4)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Wave 28A] Running PINNS TM-RKR6 + Zero-Retrain Conformal + Jitter Stress on {dev}...", flush=True)
    res_p = run_pinns_tmr6(args, dev)
    summary = {"wave": "wave28_pinns_tm_rkr6_and_conformal_5seed", "pinns_4site_loso": res_p}
    raw_b = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sha_b = (json.dumps({"summary.json": hashlib.sha256(raw_b).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw_b)
        zf.writestr("sha256.json", sha_b)
    print(f"Saved {args.zip_path} ({args.zip_path.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
