"""Wave 25A: PINNS explicit-241 audit, single-factor CondMedoid ablation, per-site core 86/73 tables, & ES6."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.benchmark_heldout_sites import load_site_windows
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import load_ppt_cohort_with_causal_maps
from scripts.run_wave18_conflict_gated_interaction_5seed import safe_site_clip_ci
from scripts.run_wave21_pinns_true_emergence_and_6slot_diag import (
    _fwd_6slot,
    fit_6slot_arm_fold,
)
from scripts.run_wave23_pinns_video_audit_and_6slot_ranker import (
    _fit_cond_precursor_head,
    blind_pinns_homography_and_raw_video_audit,
)
from src.models.apa_precursor_leadtime import build_delayed_onset_start_stop_candidates
from src.models.cond_precursor_ranker import (
    build_conditioned_precursor_features,
    select_medoid_expected_ade,
    summarize_6candidate_5seed,
)
from src.models.emergence_maneuver_6slot import (
    apply_6slot_selector,
    audit_6candidate_generation_vs_selection,
    compute_action_labels_with_onset,
    fit_strict_source_platt_and_slot_selector,
    horizon_and_post_onset_errors,
)
from src.models.emergence_safe_6slot import (
    O2EmergenceSafe6SlotNet,
    calibrate_source_low_disp_shrink,
    compute_pose_covisibility_features,
)
from src.models.emergence_maneuver_6slot import slot_semantic_maneuver_loss

SEEDS = (42, 1024, 2025, 3407, 4096)


def _pooled_clip_bootstrap_ci(diff2: np.ndarray, clip_arr: np.ndarray, n_boot: int = 4000, seed: int = 20261128) -> dict:
    """Cluster bootstrap across clips with each window weighted equally (1/N), avoiding 1-window site inflation."""
    m0 = diff2.mean(axis=0)
    u_clips = sorted(set(clip_arr.tolist()))
    if len(u_clips) < 2:
        return {"mean": {"ade_m": float(m0[0]), "fde_m": float(m0[1])}, "n_clips": len(u_clips)}
    c_sums = np.array([diff2[clip_arr == c].sum(axis=0) for c in u_clips], dtype=np.float64)
    c_cnts = np.array([int((clip_arr == c).sum()) for c in u_clips], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u_clips), size=(n_boot, len(u_clips)))
    tot_cnt = np.maximum(c_cnts[idx].sum(axis=1, keepdims=True), 1.0)
    boots = c_sums[idx].sum(axis=1) / tot_cnt
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return {
        "mean": {"ade_m": float(m0[0]), "fde_m": float(m0[1])},
        "ci95": {"ade_m": [float(lo[0]), float(hi[0])], "fde_m": [float(lo[1]), float(hi[1])]},
        "n_clips": len(u_clips),
    }


def _fwd_es6(model: O2EmergenceSafe6SlotNet, gd: dict, covis: dict[str, torch.Tensor], idx_t: torch.Tensor) -> dict[str, torch.Tensor]:
    cv_sub = {k: v[idx_t] for k, v in covis.items()}
    return model(
        history=gd["history"][idx_t],
        bio_even=gd["bio_even"][idx_t],
        pose_vec_world=gd["pose_vec_world"][idx_t],
        signed_knee=gd["signed_knee"][idx_t],
        q_vec=gd["quality"][idx_t],
        m_pose=gd["m_pose"][idx_t],
        covis=cv_sub,
        edges=gd["edges"][idx_t],
        is_car=gd["is_car"][idx_t],
        nbr_mask=gd["nbr_mask"][idx_t],
        obs_dt=gd["obs_dt"][idx_t],
        fut_times_s=gd["fut_s"][idx_t],
    )


def fit_es6_arm_fold(
    seed: int,
    use_pose: bool,
    use_nbr: bool,
    low_shrink: float,
    grp_arr: np.ndarray,
    gd: dict[str, torch.Tensor],
    covis: dict[str, torch.Tensor],
    tr_ids: np.ndarray,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> O2EmergenceSafe6SlotNet:
    torch.manual_seed(seed + (101 if use_pose else 0) + (202 if use_nbr else 0))
    model = O2EmergenceSafe6SlotNet(use_pose=use_pose, use_interaction=use_nbr, low_disp_shrink=low_shrink).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    g_tr = grp_arr[tr_ids]
    u_g = sorted(set(g_tr.tolist()))
    w_np = np.zeros(len(tr_ids), dtype=np.float64)
    for g in u_g:
        m = g_tr == g
        w_np[m] = len(tr_ids) / (len(u_g) * max(int(m.sum()), 1))
    w_all = torch.zeros(gd["history"].size(0), dtype=torch.float64, device=device)
    w_all[torch.from_numpy(tr_ids).to(device)] = torch.from_numpy(w_np).to(device)

    ya_tr = gd["y_act"][torch.from_numpy(tr_ids).to(device)]
    va_tr = gd["valid_act"][torch.from_numpy(tr_ids).to(device)]
    pw = ((va_tr.float().sum(0) - ya_tr.float().sum(0)) / ya_tr.float().sum(0).clamp_min(1.0)).clamp(1.0, 14.0)
    rng = np.random.default_rng(seed + 77)
    for _ in range(epochs):
        perm = rng.permutation(tr_ids)
        for s in range(0, len(perm), batch_size):
            b_t = torch.from_numpy(perm[s : s + batch_size]).to(device)
            out = _fwd_es6(model, gd, covis, b_t)
            loss = slot_semantic_maneuver_loss(out, gd["truth"][b_t], w_all[b_t], gd["y_act"][b_t], gd["valid_act"][b_t], pw)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sch.step()
    model.eval()
    return model


def eval_pinns_wave25_5seed(
    grp_arr: np.ndarray,
    unit_arr: np.ndarray,
    gd: dict[str, torch.Tensor],
    kpts_t: torch.Tensor,
    bbox_t: torch.Tensor,
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
    batch_size: int = 64,
    lr: float = 1.6e-3,
) -> dict:
    n, tf = len(hist_np), truth_np.shape[1]
    dt = 0.1
    cor_xy = gd["pose_vec_world"][:, -1, 0].detach().cpu().numpy()
    fut_s_np = gd["fut_s"].detach().cpu().numpy()

    v_last = (hist_np[:, -1] - hist_np[:, -2]) / dt
    cv_world = hist_np[:, -1:, :] + v_last[:, None, :] * fut_s_np[:, :, None]
    err_cv = horizon_and_post_onset_errors(cv_world, truth_np, onset_np, dt=dt)
    cv_rel = cv_world - hist_np[:, -1:, :]
    truth_rel = truth_np - hist_np[:, -1:, :]

    gd["y_act"] = torch.from_numpy(ya_np).to(device)
    gd["valid_act"] = torch.from_numpy(va_np).to(device)
    with torch.inference_mode():
        covis = compute_pose_covisibility_features(
            kpts_t.to(device), bbox_t.to(device), gd["bio_even"], gd["pose_vec_world"], gd["signed_knee"]
        )
        feat20 = build_conditioned_precursor_features(
            gd["history"], gd["bio_even"], gd["pose_vec_world"], gd["signed_knee"],
            gd["quality"], gd["m_pose"], gd["edges"], gd["is_car"], gd["nbr_mask"], gd["obs_dt"],
        )

    base_arms = (
        ("S6_M0_nopose_nonbr", False, False),
        ("S6_M1_pose_nonbr", True, False),
        ("S6_M2b_nopose_nbr", False, True),
        ("S6_Full_pose_nbr", True, True),
    )
    es6_arms = (
        ("ES6_M0_nopose_nonbr", False, False),
        ("ES6_M1_vispose_nonbr", True, False),
        ("ES6_M2b_nopose_nbr", False, True),
        ("ES6_Full_vispose_nbr", True, True),
    )
    factor_arms = (
        "S6_Full_OnlyDelayedAPA",
        "S6_Full_OnlyCondScorer",
        "S6_Full_OnlyMedoid",
        "S6_Full_APA_CondScorer",
        "S6_Full_CondMedoid",
        "ES6_Full_CondMedoid",
    )
    all_names = [a[0] for a in base_arms] + list(factor_arms) + [a[0] for a in es6_arms]
    err_s = {nm: np.zeros((len(seeds), n, 6), dtype=np.float64) for nm in all_names}
    audits_per_seed = {nm: [] for nm in ("S6_M0_nopose_nonbr", "S6_Full_CondMedoid", "ES6_M0_nopose_nonbr", "ES6_Full_vispose_nbr", "ES6_Full_CondMedoid")}
    groups = sorted(set(grp_arr.tolist()))
    ep_oof = max(14, epochs // 2)
    ya_t, va_t = gd["y_act"], gd["valid_act"]

    for si, seed in enumerate(seeds):
        pk_seed = {nm: np.zeros((n, 6, tf, 2), dtype=np.float64) for nm in audits_per_seed}
        ix_seed = {nm: np.zeros(n, dtype=np.int64) for nm in audits_per_seed}
        for held in groups:
            te = np.flatnonzero(grp_arr == held)
            tr = np.flatnonzero(grp_arr != held)
            te_g = torch.from_numpy(te).to(device)
            src_gs = sorted(set(grp_arr[tr].tolist()))

            for arm_nm, u_pose, u_nbr in base_arms:
                oof_ml = np.zeros((len(tr), 6), dtype=np.float64)
                oof_al = np.zeros((len(tr), 5), dtype=np.float64)
                oof_cal = np.zeros((len(tr), 5), dtype=np.float64)
                oof_pk = np.zeros((len(tr), 6, tf, 2), dtype=np.float64)
                for in_h in src_gs:
                    vm_l = grp_arr[tr] == in_h
                    in_tr, in_va = tr[~vm_l], tr[vm_l]
                    m_in = fit_6slot_arm_fold(seed + 13, u_pose, u_nbr, grp_arr, gd, in_tr, batch_size, ep_oof, lr, device)
                    with torch.inference_mode():
                        oo = _fwd_6slot(m_in, gd, torch.from_numpy(in_va).to(device))
                        otr = _fwd_6slot(m_in, gd, torch.from_numpy(in_tr).to(device))
                        oof_ml[vm_l], oof_al[vm_l], oof_pk[vm_l] = oo["logits"].cpu().numpy(), oo["act_logits"].cpu().numpy(), oo["paths"].cpu().numpy()
                    if arm_nm == "S6_Full_pose_nbr":
                        ch_in = _fit_cond_precursor_head(seed + 13, feat20, otr["act_logits"].detach(), ya_t[in_tr], va_t[in_tr], in_tr, device)
                        with torch.inference_mode():
                            oof_cal[vm_l] = ch_in(feat20[torch.from_numpy(in_va).to(device)], oo["act_logits"]).cpu().numpy()

                cal_base = fit_strict_source_platt_and_slot_selector(oof_ml, oof_al, oof_pk, truth_np[tr], ya_np[tr], va_np[tr])
                m_f = fit_6slot_arm_fold(seed, u_pose, u_nbr, grp_arr, gd, tr, batch_size, epochs, lr, device)
                with torch.inference_mode():
                    ot = _fwd_6slot(m_f, gd, te_g)
                    otr_f = _fwd_6slot(m_f, gd, torch.from_numpy(tr).to(device))
                    pwt, lgt, alt = ot["paths"].cpu().numpy(), ot["logits"].cpu().numpy(), ot["act_logits"].cpu().numpy()
                sel_i, pc_base, _ = apply_6slot_selector(lgt, alt, va_np[te], cal_base)
                err_s[arm_nm][si, te] = horizon_and_post_onset_errors(pwt[np.arange(len(te)), sel_i], truth_np[te], onset_np[te], dt=dt)
                if arm_nm in pk_seed:
                    pk_seed[arm_nm][te], ix_seed[arm_nm][te] = pwt - hist_np[te, -1:, None, :], sel_i

                if arm_nm == "S6_Full_pose_nbr":
                    med_st = float(np.median(onset_np[tr][ya_np[tr, 0], 0]) * dt) if ya_np[tr, 0].any() else 0.7
                    med_sp = float(np.median(onset_np[tr][ya_np[tr, 1], 1]) * dt) if ya_np[tr, 1].any() else 0.6
                    st_o, sp_o = build_delayed_onset_start_stop_candidates(hist_np[tr], cor_xy[tr], np.full(len(tr), med_st), fut_s_np[tr], dt_obs=dt)
                    oof_pk_apa = oof_pk.copy(); oof_pk_apa[:, 1] = st_o; oof_pk_apa[:, 2] = sp_o
                    st_t, sp_t = build_delayed_onset_start_stop_candidates(hist_np[te], cor_xy[te], np.full(len(te), med_st), fut_s_np[te], dt_obs=dt)
                    pwt_apa = pwt.copy(); pwt_apa[:, 1] = st_t; pwt_apa[:, 2] = sp_t

                    # Factor 1 only: Delayed-APA curves + base scorer + base MAP selector
                    cal_f1 = fit_strict_source_platt_and_slot_selector(oof_ml, oof_al, oof_pk_apa, truth_np[tr], ya_np[tr], va_np[tr])
                    ix_f1, _, _ = apply_6slot_selector(lgt, alt, va_np[te], cal_f1)
                    err_s["S6_Full_OnlyDelayedAPA"][si, te] = horizon_and_post_onset_errors(pwt_apa[np.arange(len(te)), ix_f1], truth_np[te], onset_np[te], dt=dt)

                    # Factor 2 only: Conditioned scorer + base paths + base MAP selector
                    cal_f2 = fit_strict_source_platt_and_slot_selector(oof_ml, oof_cal, oof_pk, truth_np[tr], ya_np[tr], va_np[tr])
                    ch_f = _fit_cond_precursor_head(seed, feat20, otr_f["act_logits"].detach(), ya_t[tr], va_t[tr], tr, device)
                    with torch.inference_mode():
                        alt_c = ch_f(feat20[te_g], ot["act_logits"]).cpu().numpy()
                    ix_f2, pc_c_basep, _ = apply_6slot_selector(lgt, alt_c, va_np[te], cal_f2)
                    err_s["S6_Full_OnlyCondScorer"][si, te] = horizon_and_post_onset_errors(pwt[np.arange(len(te)), ix_f2], truth_np[te], onset_np[te], dt=dt)

                    # Factor 3 only: Medoid selection + base paths + base scorer
                    ix_f3 = select_medoid_expected_ade(pwt, lgt, pc_base, va_np[te], beta=max(float(cal_base["beta"]), 0.18), sw_thr=np.asarray(cal_base["sw_thr"]))
                    err_s["S6_Full_OnlyMedoid"][si, te] = horizon_and_post_onset_errors(pwt[np.arange(len(te)), ix_f3], truth_np[te], onset_np[te], dt=dt)

                    # Factor 1+2 (Delayed-APA + CondScorer, MAP selector) vs Factor 1+2+3 (CondMedoid)
                    cal_f12 = fit_strict_source_platt_and_slot_selector(oof_ml, oof_cal, oof_pk_apa, truth_np[tr], ya_np[tr], va_np[tr])
                    ix_f12, pc_c, _ = apply_6slot_selector(lgt, alt_c, va_np[te], cal_f12)
                    err_s["S6_Full_APA_CondScorer"][si, te] = horizon_and_post_onset_errors(pwt_apa[np.arange(len(te)), ix_f12], truth_np[te], onset_np[te], dt=dt)
                    ix_cm = select_medoid_expected_ade(pwt_apa, lgt, pc_c, va_np[te], beta=max(float(cal_f12["beta"]), 0.18), sw_thr=np.asarray(cal_f12["sw_thr"]))
                    err_s["S6_Full_CondMedoid"][si, te] = horizon_and_post_onset_errors(pwt_apa[np.arange(len(te)), ix_cm], truth_np[te], onset_np[te], dt=dt)
                    pk_seed["S6_Full_CondMedoid"][te], ix_seed["S6_Full_CondMedoid"][te] = pwt_apa - hist_np[te, -1:, None, :], ix_cm

            low_sh = calibrate_source_low_disp_shrink(hist_np, truth_np, tr, dt_obs=dt, fut_times_s=fut_s_np)
            disp_te = np.linalg.norm(np.diff(hist_np[te], axis=1), axis=-1).sum(axis=1)
            for arm_nm, u_pose, u_nbr in es6_arms:
                oof_ml = np.zeros((len(tr), 6), dtype=np.float64)
                oof_al = np.zeros((len(tr), 5), dtype=np.float64)
                oof_cal = np.zeros((len(tr), 5), dtype=np.float64)
                oof_pk = np.zeros((len(tr), 6, tf, 2), dtype=np.float64)
                for in_h in src_gs:
                    vm_l = grp_arr[tr] == in_h
                    in_tr, in_va = tr[~vm_l], tr[vm_l]
                    sh_in = calibrate_source_low_disp_shrink(hist_np, truth_np, in_tr, dt_obs=dt, fut_times_s=fut_s_np)
                    m_in = fit_es6_arm_fold(seed + 13, u_pose, u_nbr, sh_in, grp_arr, gd, covis, in_tr, batch_size, ep_oof, lr, device)
                    with torch.inference_mode():
                        oo = _fwd_es6(m_in, gd, covis, torch.from_numpy(in_va).to(device))
                        otr = _fwd_es6(m_in, gd, covis, torch.from_numpy(in_tr).to(device))
                        oof_ml[vm_l], oof_al[vm_l], oof_pk[vm_l] = oo["logits"].cpu().numpy(), oo["act_logits"].cpu().numpy(), oo["paths"].cpu().numpy()
                    if arm_nm == "ES6_Full_vispose_nbr":
                        ch_in = _fit_cond_precursor_head(seed + 13, feat20, otr["act_logits"].detach(), ya_t[in_tr], va_t[in_tr], in_tr, device)
                        with torch.inference_mode():
                            oof_cal[vm_l] = ch_in(feat20[torch.from_numpy(in_va).to(device)], oo["act_logits"]).cpu().numpy()

                cal_es = fit_strict_source_platt_and_slot_selector(oof_ml, oof_al, oof_pk, truth_np[tr], ya_np[tr], va_np[tr])
                m_es = fit_es6_arm_fold(seed, u_pose, u_nbr, low_sh, grp_arr, gd, covis, tr, batch_size, epochs, lr, device)
                with torch.inference_mode():
                    ot_e = _fwd_es6(m_es, gd, covis, te_g)
                    otr_e = _fwd_es6(m_es, gd, covis, torch.from_numpy(tr).to(device))
                    pwt_e, lgt_e, alt_e = ot_e["paths"].cpu().numpy(), ot_e["logits"].cpu().numpy(), ot_e["act_logits"].cpu().numpy()
                ix_e, _, _ = apply_6slot_selector(lgt_e, alt_e, va_np[te], cal_es)
                err_s[arm_nm][si, te] = horizon_and_post_onset_errors(pwt_e[np.arange(len(te)), ix_e], truth_np[te], onset_np[te], dt=dt)
                if arm_nm in pk_seed:
                    pk_seed[arm_nm][te], ix_seed[arm_nm][te] = pwt_e - hist_np[te, -1:, None, :], ix_e

                if arm_nm == "ES6_Full_vispose_nbr":
                    med_st = float(np.median(onset_np[tr][ya_np[tr, 0], 0]) * dt) if ya_np[tr, 0].any() else 0.7
                    med_sp = float(np.median(onset_np[tr][ya_np[tr, 1], 1]) * dt) if ya_np[tr, 1].any() else 0.6
                    st_o, sp_o = build_delayed_onset_start_stop_candidates(hist_np[tr], cor_xy[tr], np.full(len(tr), med_st), fut_s_np[tr], dt_obs=dt)
                    oof_pk_e = oof_pk.copy(); oof_pk_e[:, 1] = st_o; oof_pk_e[:, 2] = sp_o
                    cal_ec = fit_strict_source_platt_and_slot_selector(oof_ml, oof_cal, oof_pk_e, truth_np[tr], ya_np[tr], va_np[tr])
                    ch_e = _fit_cond_precursor_head(seed, feat20, otr_e["act_logits"].detach(), ya_t[tr], va_t[tr], tr, device)
                    with torch.inference_mode():
                        alt_ec = ch_e(feat20[te_g], ot_e["act_logits"]).cpu().numpy()
                    ix_map_ec, pc_ec, _ = apply_6slot_selector(lgt_e, alt_ec, va_np[te], cal_ec)
                    st_t, sp_t = build_delayed_onset_start_stop_candidates(hist_np[te], cor_xy[te], np.full(len(te), med_st), fut_s_np[te], dt_obs=dt)
                    pwt_ec = pwt_e.copy(); pwt_ec[:, 1] = st_t; pwt_ec[:, 2] = sp_t
                    ix_med_ec = select_medoid_expected_ade(pwt_ec, lgt_e, pc_ec, va_np[te], beta=max(float(cal_ec["beta"]), 0.18), sw_thr=np.asarray(cal_ec["sw_thr"]))
                    # Causal regime-aware selection: MAP under < 0.5 m emergence, Medoid under >= 0.5 m cruising
                    ix_hyb = np.where(disp_te < 0.50, ix_map_ec, ix_med_ec)
                    err_s["ES6_Full_CondMedoid"][si, te] = horizon_and_post_onset_errors(pwt_ec[np.arange(len(te)), ix_hyb], truth_np[te], onset_np[te], dt=dt)
                    pk_seed["ES6_Full_CondMedoid"][te], ix_seed["ES6_Full_CondMedoid"][te] = pwt_ec - hist_np[te, -1:, None, :], ix_hyb

        for nm_a in audits_per_seed:
            audits_per_seed[nm_a].append(
                audit_6candidate_generation_vs_selection(
                    hist_np, pk_seed[nm_a], ix_seed[nm_a], truth_rel, cv_rel, ya_np, va_np, diag_mask, dt=dt,
                )
            )
        print(f"[W25 PINNS] Finished seed={seed}", flush=True)

    em = {nm: err_s[nm].mean(axis=0) for nm in all_names}

    def _fmt6(vals: np.ndarray) -> dict[str, float]:
        return {
            "ade1_full_m": float(vals[0]), "fde1_full_m": float(vals[1]),
            "ade1_trig_0p5s_m": float(vals[2]), "ade1_trig_1p0s_m": float(vals[3]),
            "ade1_post_onset_0p5s_m": float(vals[4]), "ade1_post_onset_1p0s_m": float(vals[5]),
        }

    def _agg_site_eq(arr6: np.ndarray, sm: np.ndarray) -> dict[str, float]:
        gs = sorted(set(grp_arr[sm].tolist()))
        return _fmt6(np.mean([arr6[(grp_arr == g) & sm].mean(axis=0) for g in gs], axis=0))

    def _agg_pooled(arr6: np.ndarray, sm: np.ndarray) -> dict[str, float]:
        return _fmt6(arr6[sm].mean(axis=0))

    def _pair_both(d2: np.ndarray, sm: np.ndarray) -> dict:
        if int(sm.sum()) < 4:
            return {"n": int(sm.sum())}
        return {
            "site_equal": safe_site_clip_ci(d2[sm], grp_arr[sm], unit_arr[sm]),
            "pooled_clip_bootstrap": _pooled_clip_bootstrap_ci(d2[sm], unit_arr[sm]),
        }

    strata_out = {}
    for s_nm, msk in strata.items():
        cnt = int(msk.sum())
        if cnt == 0:
            strata_out[s_nm] = {"n_windows": 0}
            continue
        m_seq = {"Last2_CV": _agg_site_eq(err_cv, msk)}
        m_pol = {"Last2_CV": _agg_pooled(err_cv, msk)}
        for nm in all_names:
            m_seq[nm] = _agg_site_eq(em[nm], msk)
            m_pol[nm] = _agg_pooled(em[nm], msk)

        per_site = {}
        for g in sorted(set(grp_arr[msk].tolist())):
            gm = msk & (grp_arr == g)
            ps_m = {"Last2_CV": _agg_pooled(err_cv, gm)}
            for nm in all_names:
                ps_m[nm] = _agg_pooled(em[nm], gm)
            per_site[str(g)] = {
                "n_windows": int(gm.sum()),
                "n_clips": len(set(unit_arr[gm].tolist())),
                "maneuver_counts": {k: int(ya_np[gm, c].sum()) for c, k in enumerate(("Start", "Stop", "Turn_Left", "Turn_Right", "Maintain"))},
                "models": ps_m,
            }

        strata_out[s_nm] = {
            "n_windows": cnt,
            "n_clips": len(set(unit_arr[msk].tolist())),
            "n_sites": len(set(grp_arr[msk].tolist())),
            "maneuver_counts": {k: int(ya_np[msk, c].sum()) for c, k in enumerate(("Start", "Stop", "Turn_Left", "Turn_Right", "Maintain"))},
            "models_site_equal": m_seq,
            "models_pooled_window_weighted": m_pol,
            "per_site_breakdown": per_site,
            "paired_ci": {
                "cond_medoid_minus_CV_3s": _pair_both(em["S6_Full_CondMedoid"][:, :2] - err_cv[:, :2], msk),
                "es6_full_minus_CV_3s": _pair_both(em["ES6_Full_vispose_nbr"][:, :2] - err_cv[:, :2], msk),
                "es6_condmed_minus_CV_3s": _pair_both(em["ES6_Full_CondMedoid"][:, :2] - err_cv[:, :2], msk),
                "s6_pose_no_nbr_3s": _pair_both(em["S6_M1_pose_nonbr"][:, :2] - em["S6_M0_nopose_nonbr"][:, :2], msk),
                "s6_pose_no_nbr_post_onset": _pair_both(em["S6_M1_pose_nonbr"][:, 4:6] - em["S6_M0_nopose_nonbr"][:, 4:6], msk),
                "s6_nbr_no_pose_3s": _pair_both(em["S6_M2b_nopose_nbr"][:, :2] - em["S6_M0_nopose_nonbr"][:, :2], msk),
                "es6_pose_no_nbr_3s": _pair_both(em["ES6_M1_vispose_nonbr"][:, :2] - em["ES6_M0_nopose_nonbr"][:, :2], msk),
                "es6_pose_no_nbr_post_onset": _pair_both(em["ES6_M1_vispose_nonbr"][:, 4:6] - em["ES6_M0_nopose_nonbr"][:, 4:6], msk),
                "es6_nbr_no_pose_3s": _pair_both(em["ES6_M2b_nopose_nbr"][:, :2] - em["ES6_M0_nopose_nonbr"][:, :2], msk),
                "factor1_apa_minus_s6full_3s": _pair_both(em["S6_Full_OnlyDelayedAPA"][:, :2] - em["S6_Full_pose_nbr"][:, :2], msk),
                "factor2_cond_minus_s6full_3s": _pair_both(em["S6_Full_OnlyCondScorer"][:, :2] - em["S6_Full_pose_nbr"][:, :2], msk),
                "factor3_medoid_minus_s6full_3s": _pair_both(em["S6_Full_OnlyMedoid"][:, :2] - em["S6_Full_pose_nbr"][:, :2], msk),
                "condmedoid_minus_apacond_3s": _pair_both(em["S6_Full_CondMedoid"][:, :2] - em["S6_Full_APA_CondScorer"][:, :2], msk),
            },
        }
    return {
        "n_windows": n,
        "strata": strata_out,
        "candidate_generation_vs_selection_5seed": {nm: summarize_6candidate_5seed(audits_per_seed[nm]) for nm in audits_per_seed},
    }


def main() -> None:
    import tempfile

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--video-cache-dir", type=Path, default=Path("/tmp/w25_vids"))
    ap.add_argument("--explicit-audit-json", type=Path, default=Path("data/annotations/pinns_explicit_241_visual_audit.json"))
    ap.add_argument("--wave24-labels-json", type=Path, default=Path("data/annotations/pinns_blind_raw_video_emergence_labels.json"))
    ap.add_argument("--zip-path", type=Path, default=Path("wave25_pinns_explicit_audit_and_emergence_safe.zip"))
    ap.add_argument("--epochs", type=int, default=34)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = load_ppt_cohort_with_causal_maps(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin)
    groups_raw, _, _ = load_site_windows(args.meta_dir)
    rows = [r for s in sorted(groups_raw) for r in groups_raw[s]]
    keys = [(str(r.clip), int(r.person_id), int(r.t0)) for r in rows]
    with np.load(args.pose_npz, allow_pickle=False) as pz:
        ord_map = {k: i for i, k in enumerate(zip(pz["clip"].tolist(), pz["person_id"].tolist(), pz["t0"].tolist()))}
        ids = [ord_map[k] for k in keys]
        kpts_t = torch.from_numpy(pz["hist_kpts"][ids].astype(np.float32))
        bbox_t = torch.from_numpy(pz["hist_bbox"][ids].astype(np.float32))

    hist_np, truth_np = dp["history"].numpy(), dp["truth"].numpy()
    n = len(hist_np)
    hm5, fm30 = np.ones(5, dtype=np.float32), np.ones(30, dtype=np.float32)
    ya_p, va_p, on_p = np.zeros((n, 5), dtype=bool), np.zeros((n, 5), dtype=bool), np.full((n, 5), -1, dtype=np.int64)
    for i in range(n):
        ya_p[i], va_p[i], on_p[i] = compute_action_labels_with_onset(
            hist_np[i] - hist_np[i, -1:], hm5, truth_np[i] - hist_np[i, -1:], fm30, dt=0.1
        )

    exp_obj = json.loads(args.explicit_audit_json.read_text(encoding="utf-8"))
    exp_labels = exp_obj["labels"]
    with tempfile.TemporaryDirectory() as td:
        exp_flat = Path(td) / "exp_flat.json"
        exp_flat.write_text(json.dumps(exp_labels), encoding="utf-8")
        hom_dummy = Path(td) / "hom_dummy.json"
        hom_dummy.write_text(json.dumps({"__no_video_needed__": "open"}), encoding="utf-8")
        cen_w24 = blind_pinns_homography_and_raw_video_audit(args.meta_dir, args.pose_npz, dp, args.video_cache_dir, None, args.wave24_labels_json)
        cen_exp = blind_pinns_homography_and_raw_video_audit(args.meta_dir, args.pose_npz, dp, args.video_cache_dir, None, exp_flat)
        hom_only = blind_pinns_homography_and_raw_video_audit(args.meta_dir, args.pose_npz, dp, args.video_cache_dir, None, hom_dummy)

    fc_non_f0 = [
        f"{rows[i].clip}:{rows[i].person_id}:{rows[i].t0}"
        for i in range(n)
        if (not bool(cen_exp["e_clip_start"][i])) and (
            bool(cen_exp["e_phys_all"][i]) or bool(cen_exp["e_border"][i]) or bool(cen_exp["e_open_late"][i])
        )
    ]
    missing_exp = [k for k in fc_non_f0 if k not in exp_labels]
    if len(fc_non_f0) != 241 or missing_exp:
        raise KeyError(f"Zero-default assertion failed: count={len(fc_non_f0)} missing={missing_exp[:5]}")

    act_m = (ya_p[:, :4] & va_p[:, :4]).any(axis=1)
    turn_m = (ya_p[:, 2] & va_p[:, 2]) | (ya_p[:, 3] & va_p[:, 3])
    stsp_m = (ya_p[:, 0] & va_p[:, 0]) | (ya_p[:, 1] & va_p[:, 1])
    e_exp, e_w24 = cen_exp["e_phys_all"], cen_w24["e_phys_all"]
    fs323 = dp["strata"]["first_short_323"]
    old_bev_34 = hom_only["e_buggy_bev_veh"]

    strata = {
        "all_1482": np.ones(n, dtype=bool),
        "explicit_phys_emergence_118": e_exp,
        "explicit_matched_visible_ctrl_118": cen_exp["matched_visible_ctrl"],
        "core_explicit_short_phys_lt0p5m": fs323 & e_exp,
        "core_wave24_short_phys_86": fs323 & e_w24,
        "wave24_phys_emergence_140": e_w24,
        "explicit_phys_emergence_active_maneuvers": e_exp & act_m,
        "explicit_phys_emergence_turn_only": e_exp & turn_m,
        "explicit_phys_emergence_start_or_stop": e_exp & stsp_m,
        "first_short_323": fs323,
    }
    gd_p = {k: dp[k].to(device) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
    gd_p["obs_dt"] = torch.full((n, 4), 0.1, dtype=torch.float64, device=device)
    gd_p["fut_s"] = torch.arange(1, 31, dtype=torch.float64, device=device)[None].expand(n, -1) * 0.1

    res_p = eval_pinns_wave25_5seed(dp["site"], dp["clip"], gd_p, kpts_t, bbox_t, hist_np, truth_np, ya_p, va_p, on_p, strata, fs323 | e_exp, device, SEEDS, args.epochs)
    summary = {
        "wave": "wave25_pinns_explicit_audit_and_emergence_safe_5seed",
        "explicit_241_audit_census": {
            "explicit_241_sha256": hashlib.sha256(args.explicit_audit_json.read_bytes()).hexdigest(),
            "zero_default_verified_count": len(fc_non_f0),
            "class_counts_241": dict(Counter(exp_labels.values())),
            "old_bev_34_vs_homography_rule_overlap": int((old_bev_34 & hom_only["e_veh_cam"]).sum()),
            "old_bev_34_vs_visual_veh_label_overlap": int((old_bev_34 & cen_exp["e_veh_cam"]).sum()),
            "explicit_phys_118_by_site": {str(g): int((e_exp & (dp["site"] == g)).sum()) for g in sorted(set(dp["site"].tolist()))},
            "core_explicit_short_lt0p5m_by_site": {str(g): int((fs323 & e_exp & (dp["site"] == g)).sum()) for g in sorted(set(dp["site"].tolist()))},
            "core_wave24_short_86_by_site": {str(g): int((fs323 & e_w24 & (dp["site"] == g)).sum()) for g in sorted(set(dp["site"].tolist()))},
        },
        "pinns_4site_loso": res_p,
    }
    raw_b = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sha_b = (json.dumps({"summary.json": hashlib.sha256(raw_b).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw_b)
        zf.writestr("sha256.json", sha_b)
    print(f"Saved {args.zip_path} ({args.zip_path.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
