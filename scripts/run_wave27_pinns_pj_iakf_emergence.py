"""Wave 27A: Projective-Jacobian Anisotropic Ray-Kalman (PJ-IAKF) & 2x2 Covariance-Guided RKR6 5-seed evaluation on PINNS."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
import torch

from scripts.benchmark_heldout_sites import load_site_windows
from scripts.ppt_mainline_o2_models import fit_pinns_ground_homography
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
from src.models.ray_kalman_and_cusum import (
    O2RayKalmanRes6SlotNet,
    anisotropic_ray_kalman_filter,
    calibrate_ray_kalman_params,
    compute_homography_jacobian_world,
)

SEEDS = (42, 1024, 2025, 3407, 4096)


def _fwd_rkr6(
    model: O2RayKalmanRes6SlotNet,
    gd: dict[str, torch.Tensor],
    split_covis: dict[str, torch.Tensor] | None,
    jac_t: torch.Tensor | None,
    bbox_h: torch.Tensor | None,
    ank_c: torch.Tensor | None,
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
        bbox_h=bbox_h[idx_t] if bbox_h is not None else None,
        ank_conf=ank_c[idx_t] if ank_c is not None else None,
        edges=gd["edges"][idx_t],
        is_car=gd["is_car"][idx_t],
        nbr_mask=gd["nbr_mask"][idx_t],
        obs_dt=gd["obs_dt"][idx_t],
        fut_times_s=gd["fut_s"][idx_t],
    )


def fit_rkr6_arm_fold(
    seed: int,
    use_pose: bool,
    use_nbr: bool,
    ray_cfg: dict[str, float],
    grp_arr: np.ndarray,
    gd: dict[str, torch.Tensor],
    split_covis: dict[str, torch.Tensor] | None,
    jac_t: torch.Tensor | None,
    bbox_h: torch.Tensor | None,
    ank_c: torch.Tensor | None,
    tr_ids: np.ndarray,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> O2RayKalmanRes6SlotNet:
    torch.manual_seed(seed + (151 if use_pose else 0) + (307 if use_nbr else 0))
    model = O2RayKalmanRes6SlotNet(use_pose=use_pose, use_interaction=use_nbr, ray_cfg=ray_cfg).to(device)
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
    rng = np.random.default_rng(seed + 97)
    for _ in range(epochs):
        perm = rng.permutation(tr_ids)
        for s in range(0, len(perm), batch_size):
            b_t = torch.from_numpy(perm[s : s + batch_size]).to(device)
            out = _fwd_rkr6(model, gd, split_covis, jac_t, bbox_h, ank_c, b_t)
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


def eval_rkr6_pinns_5seed(
    grp_arr: np.ndarray,
    unit_arr: np.ndarray,
    gd: dict[str, torch.Tensor],
    kpts_t: torch.Tensor,
    bbox_t: torch.Tensor,
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
    cv_world = hist_np[:, -1:, :] + v_last[:, None, :] * fut_s_np[:, :, None]
    err_cv = horizon_and_post_onset_errors(cv_world, truth_np, onset_np, dt=dt)
    cv_rel, truth_rel = cv_world - hist_np[:, -1:, :], truth_np - hist_np[:, -1:, :]

    gd["y_act"] = torch.from_numpy(ya_np).to(device)
    gd["valid_act"] = torch.from_numpy(va_np).to(device)
    bbox_h_g = bbox_t[..., 3].to(device)
    ank_c_g = (0.5 * (kpts_t[:, :, 15, 2] + kpts_t[:, :, 16, 2])).clamp(0.0, 1.0).to(device)
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
        ("RKR6_M0_nopose_nonbr", False, False),
        ("RKR6_M1_splitpose_nonbr", True, False),
        ("RKR6_M2b_nopose_conflict", False, True),
        ("RKR6_Full_Slot0", True, True),
    )
    kin_names = ["PJ_IAKF_IsoNoRay", "PJ_IAKF_AnisoRay_NoEB", "PJ_IAKF_Full_Kinematic"]
    all_names = kin_names + [a[0] for a in arms] + ["RKR6_Full_OOFRegime"]
    err_s = {nm: np.zeros((len(seeds), n, 6), dtype=np.float64) for nm in all_names}
    audits_5s = {nm: [] for nm in ("RKR6_M0_nopose_nonbr", "RKR6_Full_Slot0")}
    cal_iso_by_fold: dict[str, dict[str, float]] = {}
    cal_ray_by_fold: dict[str, dict[str, float]] = {}
    oof_regime_by_fold: dict[str, list[dict[str, float]]] = {}
    groups = sorted(set(grp_arr.tolist()))
    ep_oof = max(16, epochs // 2)

    for held in groups:
        tr = np.flatnonzero(grp_arr != held)
        tr_g = torch.from_numpy(tr).to(device)
        cal_iso_by_fold[str(held)] = calibrate_ray_kalman_params(
            gd["history"][tr_g], gd["obs_dt"][tr_g], gd["fut_s"][tr_g], gd["truth"][tr_g],
            src_grp=grp_arr[tr], jac_src=None, bbox_h_src=bbox_h_g[tr_g], ank_conf_src=ank_c_g[tr_g],
        )
        cal_ray_by_fold[str(held)] = calibrate_ray_kalman_params(
            gd["history"][tr_g], gd["obs_dt"][tr_g], gd["fut_s"][tr_g], gd["truth"][tr_g],
            src_grp=grp_arr[tr], jac_src=jac_g[tr_g], bbox_h_src=bbox_h_g[tr_g], ank_conf_src=ank_c_g[tr_g],
        )
        oof_regime_by_fold[str(held)] = []

    t_start = time.perf_counter()
    for si, seed in enumerate(seeds):
        pk_seed = {nm: np.zeros((n, 6, tf, 2), dtype=np.float64) for nm in audits_5s}
        ix_seed = {nm: np.zeros(n, dtype=np.int64) for nm in audits_5s}
        for held in groups:
            te, tr = np.flatnonzero(grp_arr == held), np.flatnonzero(grp_arr != held)
            te_g, tr_g = torch.from_numpy(te).to(device), torch.from_numpy(tr).to(device)
            icfg, rcfg_k = cal_iso_by_fold[str(held)], cal_ray_by_fold[str(held)]
            rcfg_noeb = {**rcfg_k, "tau_eb": 0.0, "anchor_p_blend": 0.0}

            with torch.inference_mode():
                o_iso = anisotropic_ray_kalman_filter(gd["history"][te_g], gd["obs_dt"][te_g], None, bbox_h_g[te_g], ank_c_g[te_g], **icfg)
                p_iso = o_iso["p_filt"].cpu().numpy()[:, None, :] + o_iso["v_filt"].cpu().numpy()[:, None, :] * fut_s_np[te, :, None]
                o_noeb = anisotropic_ray_kalman_filter(gd["history"][te_g], gd["obs_dt"][te_g], jac_g[te_g], bbox_h_g[te_g], ank_c_g[te_g], **rcfg_noeb)
                p_noeb = o_noeb["p_filt"].cpu().numpy()[:, None, :] + o_noeb["v_filt"].cpu().numpy()[:, None, :] * fut_s_np[te, :, None]
                o_rk = anisotropic_ray_kalman_filter(gd["history"][te_g], gd["obs_dt"][te_g], jac_g[te_g], bbox_h_g[te_g], ank_c_g[te_g], **rcfg_k)
                kin_te_path = o_rk["p_filt"].cpu().numpy()[:, None, :] + o_rk["v_filt"].cpu().numpy()[:, None, :] * fut_s_np[te, :, None]
                o_rk_tr = anisotropic_ray_kalman_filter(gd["history"][tr_g], gd["obs_dt"][tr_g], jac_g[tr_g], bbox_h_g[tr_g], ank_c_g[tr_g], **rcfg_k)
                kin_tr_path = o_rk_tr["p_filt"].cpu().numpy()[:, None, :] + o_rk_tr["v_filt"].cpu().numpy()[:, None, :] * fut_s_np[tr, :, None]

            err_s["PJ_IAKF_IsoNoRay"][si, te] = horizon_and_post_onset_errors(p_iso, truth_np[te], onset_np[te], dt=dt)
            err_s["PJ_IAKF_AnisoRay_NoEB"][si, te] = horizon_and_post_onset_errors(p_noeb, truth_np[te], onset_np[te], dt=dt)
            err_s["PJ_IAKF_Full_Kinematic"][si, te] = horizon_and_post_onset_errors(kin_te_path, truth_np[te], onset_np[te], dt=dt)

            for arm_nm, u_pose, u_nbr in arms:
                m_f = fit_rkr6_arm_fold(seed, u_pose, u_nbr, rcfg_k, grp_arr, gd, split_covis, jac_g, bbox_h_g, ank_c_g, tr, batch_size, epochs, lr, device)
                with torch.inference_mode():
                    ot = _fwd_rkr6(m_f, gd, split_covis, jac_g, bbox_h_g, ank_c_g, te_g)
                    pwt, lgt = ot["paths"].cpu().numpy(), ot["logits"].cpu().numpy()
                err_s[arm_nm][si, te] = horizon_and_post_onset_errors(pwt[:, 0], truth_np[te], onset_np[te], dt=dt)
                if arm_nm in pk_seed:
                    pk_seed[arm_nm][te] = pwt - hist_np[te, -1:, None, :]
                    ix_seed[arm_nm][te] = 0
                if arm_nm == "RKR6_Full_Slot0":
                    src_gs = sorted(set(grp_arr[tr].tolist()))
                    oof_ml = np.zeros((len(tr), 6), dtype=np.float64)
                    oof_cal = np.zeros((len(tr), 5), dtype=np.float64)
                    oof_pk = np.zeros((len(tr), 6, tf, 2), dtype=np.float64)
                    for g_val in src_gs:
                        vm_l = grp_arr[tr] == g_val
                        in_tr, in_va = tr[~vm_l], tr[vm_l]
                        m_in = fit_rkr6_arm_fold(seed + 19, True, True, rcfg_k, grp_arr, gd, split_covis, jac_g, bbox_h_g, ank_c_g, in_tr, batch_size, ep_oof, lr, device)
                        with torch.inference_mode():
                            oo = _fwd_rkr6(m_in, gd, split_covis, jac_g, bbox_h_g, ank_c_g, torch.from_numpy(in_va).to(device))
                            otr = _fwd_rkr6(m_in, gd, split_covis, jac_g, bbox_h_g, ank_c_g, torch.from_numpy(in_tr).to(device))
                            oof_ml[vm_l], oof_pk[vm_l] = oo["logits"].cpu().numpy(), oo["paths"].cpu().numpy()
                        ch_in = _fit_cond_precursor_head(seed + 19, feat20, otr["act_logits"].detach(), gd["y_act"][in_tr], gd["valid_act"][in_tr], in_tr, device)
                        with torch.inference_mode():
                            oof_cal[vm_l] = ch_in(feat20[torch.from_numpy(in_va).to(device)], oo["act_logits"]).cpu().numpy()

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
                        otr_f = _fwd_rkr6(m_f, gd, split_covis, jac_g, bbox_h_g, ank_c_g, tr_g)
                    ch_f = _fit_cond_precursor_head(seed, feat20, otr_f["act_logits"].detach(), gd["y_act"][tr], gd["valid_act"][tr], tr, device)
                    with torch.inference_mode():
                        cal_te = ch_f(feat20[te_g], ot["act_logits"]).cpu().numpy()
                    st_t, sp_t = build_delayed_onset_start_stop_candidates(hist_np[te], cor_xy[te], np.full(len(te), med_st), fut_s_np[te], dt_obs=dt)
                    pwt_a = pwt.copy(); pwt_a[:, 1], pwt_a[:, 2] = st_t, sp_t
                    pc_te = 1.0 / (1.0 + np.exp(-np.clip(cal_te * np.asarray(cal_c["a_pl"])[None] + np.asarray(cal_c["b_pl"])[None], -30.0, 30.0)))
                    ix_mt = select_medoid_expected_ade(pwt_a, lgt, pc_te, va_np[te], beta=max(float(cal_c["beta"]), 0.15))
                    p_reg = apply_ekr6_regime_selector(kin_te_path, pwt[:, 0], pwt_a[np.arange(len(te)), ix_mt], path_len_all[te], rcfg_sel)
                    err_s["RKR6_Full_OOFRegime"][si, te] = horizon_and_post_onset_errors(p_reg, truth_np[te], onset_np[te], dt=dt)

            print(f"[Wave 27A] Finished seed={seed} ({si+1}/{len(seeds)}) site={held} elapsed={time.perf_counter()-t_start:.1f}s", flush=True)

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
                "iso_kalman_minus_CV": _pair_both(em["PJ_IAKF_IsoNoRay"][:, :2] - err_cv[:, :2], msk),
                "aniso_ray_noeb_minus_CV": _pair_both(em["PJ_IAKF_AnisoRay_NoEB"][:, :2] - err_cv[:, :2], msk),
                "pj_iakf_full_kin_minus_CV": _pair_both(em["PJ_IAKF_Full_Kinematic"][:, :2] - err_cv[:, :2], msk),
                "aniso_ray_minus_iso_kalman": _pair_both(em["PJ_IAKF_AnisoRay_NoEB"][:, :2] - em["PJ_IAKF_IsoNoRay"][:, :2], msk),
                "rkr6_m0_minus_CV": _pair_both(em["RKR6_M0_nopose_nonbr"][:, :2] - err_cv[:, :2], msk),
                "rkr6_m1_minus_CV": _pair_both(em["RKR6_M1_splitpose_nonbr"][:, :2] - err_cv[:, :2], msk),
                "rkr6_m2b_minus_CV": _pair_both(em["RKR6_M2b_nopose_conflict"][:, :2] - err_cv[:, :2], msk),
                "rkr6_full_slot0_minus_CV": _pair_both(em["RKR6_Full_Slot0"][:, :2] - err_cv[:, :2], msk),
                "rkr6_oof_regime_minus_CV": _pair_both(em["RKR6_Full_OOFRegime"][:, :2] - err_cv[:, :2], msk),
                "rkr6_pose_no_nbr_full": _pair_both(em["RKR6_M1_splitpose_nonbr"][:, :2] - em["RKR6_M0_nopose_nonbr"][:, :2], msk),
                "rkr6_pose_no_nbr_post_onset": _pair_both(em["RKR6_M1_splitpose_nonbr"][:, 4:6] - em["RKR6_M0_nopose_nonbr"][:, 4:6], msk),
                "rkr6_nbr_no_pose_full": _pair_both(em["RKR6_M2b_nopose_conflict"][:, :2] - em["RKR6_M0_nopose_nonbr"][:, :2], msk),
                "rkr6_full_minus_pj_iakf_kin": _pair_both(em["RKR6_Full_Slot0"][:, :2] - em["PJ_IAKF_Full_Kinematic"][:, :2], msk),
            },
        }
    return {
        "n_windows": n,
        "source_calibrated_iso_params": cal_iso_by_fold,
        "source_calibrated_ray_kalman_params": cal_ray_by_fold,
        "source_oof_regime_selectors": oof_regime_by_fold,
        "strata": strata_out,
        "candidate_generation_vs_selection_5seed": {nm: summarize_6candidate_5seed(audits_5s[nm]) for nm in audits_5s},
    }


def run_pinns_pj_iakf(args: argparse.Namespace, device: torch.device) -> dict:
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
    jac_t = compute_homography_jacobian_world(bbox_t, torch.from_numpy(h_mats), torch.from_numpy(y_sums))
    ju_n = torch.linalg.vector_norm(jac_t[:, :, 0], dim=-1).numpy()
    jv_n = torch.linalg.vector_norm(jac_t[:, :, 1], dim=-1).numpy()
    aniso_ratio = float(np.median(jv_n[ju_n > 1e-6] / ju_n[ju_n > 1e-6]))
    print(f"[Wave 27A] PINNS median viewing-ray / azimuthal Jacobian norm ratio = {aniso_ratio:.3f}x", flush=True)

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
    out = eval_rkr6_pinns_5seed(dp["site"], dp["clip"], gd_p, kpts_t, bbox_t, jac_t, hist_np, truth_np, ya_p, va_p, on_p, strata, fs323 | e_exp, device, SEEDS, args.epochs)
    out["homography_anisotropy_diagnostics"] = {
        "median_jv_over_ju_ratio": aniso_ratio,
        "median_jv_m_per_px": float(np.median(jv_n[jv_n > 1e-6])),
        "median_ju_m_per_px": float(np.median(ju_n[ju_n > 1e-6])),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--video-cache-dir", type=Path, default=Path("/tmp/w27_vids"))
    ap.add_argument("--explicit-audit-json", type=Path, default=Path("data/annotations/pinns_explicit_241_visual_audit.json"))
    ap.add_argument("--wave24-labels-json", type=Path, default=Path("data/annotations/pinns_blind_raw_video_emergence_labels.json"))
    ap.add_argument("--zip-path", type=Path, default=Path("wave27_pinns_pj_iakf_emergence.zip"))
    ap.add_argument("--epochs", type=int, default=34)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        torch.set_num_threads(4)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Wave 27A] Running PINNS PJ-IAKF + RKR6 5-seed on {dev}...", flush=True)
    res_p = run_pinns_pj_iakf(args, dev)
    summary = {"wave": "wave27_pinns_pj_iakf_emergence_5seed", "pinns_4site_loso": res_p}
    raw_b = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sha_b = (json.dumps({"summary.json": hashlib.sha256(raw_b).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw_b)
        zf.writestr("sha256.json", sha_b)
    print(f"Saved {args.zip_path} ({args.zip_path.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
