"""Wave 26A: Coherence-Kalman + Split-CoVis Pose-Conflict Continuous-Slot0 (EKR6) 5-seed evaluation on PINNS & JRDB."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch

from scripts.benchmark_heldout_sites import load_site_windows
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import load_ppt_cohort_with_causal_maps
from scripts.run_wave18_conflict_gated_interaction_5seed import _loc_bootstrap_ci, safe_site_clip_ci
from scripts.run_wave21_jrdb_and_jta_emergence_maneuvers import load_jrdb_with_blind_emergence
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
    O2EmergenceKalmanRes6SlotNet,
    apply_ekr6_regime_selector,
    calibrate_coherence_kalman_params,
    compute_coherence_kalman_velocity,
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

SEEDS = (42, 1024, 2025, 3407, 4096)


def _fwd_ekr6(
    model: O2EmergenceKalmanRes6SlotNet,
    gd: dict[str, torch.Tensor],
    split_covis: dict[str, torch.Tensor] | None,
    bbox_h: torch.Tensor | None,
    idx_t: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sc_sub = {k: v[idx_t] for k, v in split_covis.items()} if split_covis is not None else None
    bh_sub = bbox_h[idx_t] if bbox_h is not None else None
    return model(
        history=gd["history"][idx_t],
        bio_even=gd["bio_even"][idx_t],
        pose_vec_world=gd["pose_vec_world"][idx_t],
        signed_knee=gd["signed_knee"][idx_t],
        q_vec=gd["quality"][idx_t],
        m_pose=gd["m_pose"][idx_t],
        split_covis=sc_sub,
        bbox_h=bh_sub,
        edges=gd["edges"][idx_t],
        is_car=gd["is_car"][idx_t],
        nbr_mask=gd["nbr_mask"][idx_t],
        obs_dt=gd["obs_dt"][idx_t],
        fut_times_s=gd["fut_s"][idx_t],
    )


def fit_ekr6_arm_fold(
    seed: int,
    use_pose: bool,
    use_nbr: bool,
    kalman_cfg: dict[str, float],
    grp_arr: np.ndarray,
    gd: dict[str, torch.Tensor],
    split_covis: dict[str, torch.Tensor] | None,
    bbox_h: torch.Tensor | None,
    tr_ids: np.ndarray,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> O2EmergenceKalmanRes6SlotNet:
    torch.manual_seed(seed + (131 if use_pose else 0) + (263 if use_nbr else 0))
    model = O2EmergenceKalmanRes6SlotNet(use_pose=use_pose, use_interaction=use_nbr, kalman_cfg=kalman_cfg).to(device)
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
    rng = np.random.default_rng(seed + 91)
    for _ in range(epochs):
        perm = rng.permutation(tr_ids)
        for s in range(0, len(perm), batch_size):
            b_t = torch.from_numpy(perm[s : s + batch_size]).to(device)
            out = _fwd_ekr6(model, gd, split_covis, bbox_h, b_t)
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


def _inner_group_splits(src_groups: list[str], grp_tr: np.ndarray, max_splits: int = 3) -> list[np.ndarray]:
    if len(src_groups) <= max_splits:
        return [grp_tr == g for g in src_groups]
    buckets = [src_groups[i::max_splits] for i in range(max_splits)]
    return [np.isin(grp_tr, b) for b in buckets if len(b) > 0]


def eval_ekr6_cohort_5seed(
    grp_arr: np.ndarray,
    unit_arr: np.ndarray,
    gd: dict[str, torch.Tensor],
    kpts_t: torch.Tensor | None,
    bbox_t: torch.Tensor | None,
    hist_np: np.ndarray,
    truth_np: np.ndarray,
    ya_np: np.ndarray,
    va_np: np.ndarray,
    onset_np: np.ndarray,
    strata: dict[str, np.ndarray],
    diag_mask: np.ndarray,
    dt: float,
    device: torch.device,
    seeds: tuple[int, ...] = SEEDS,
    epochs: int = 34,
    batch_size: int = 64,
    lr: float = 1.6e-3,
    is_pinns: bool = True,
) -> dict:
    n, tf = len(hist_np), truth_np.shape[1]
    fut_s_np = gd["fut_s"].detach().cpu().numpy()
    cor_xy = gd["pose_vec_world"][:, -1, 0].detach().cpu().numpy()
    path_len_all = np.linalg.norm(np.diff(hist_np, axis=1), axis=-1).sum(axis=1)

    v_last = (hist_np[:, -1] - hist_np[:, -2]) / dt
    cv_world = hist_np[:, -1:, :] + v_last[:, None, :] * fut_s_np[:, :, None]
    err_cv = horizon_and_post_onset_errors(cv_world, truth_np, onset_np, dt=dt)
    cv_rel = cv_world - hist_np[:, -1:, :]
    truth_rel = truth_np - hist_np[:, -1:, :]

    gd["y_act"] = torch.from_numpy(ya_np).to(device)
    gd["valid_act"] = torch.from_numpy(va_np).to(device)
    bbox_h_g = bbox_t[..., 3].to(device) if bbox_t is not None else None
    with torch.inference_mode():
        split_covis = compute_split_anatomical_covisibility(
            kpts_t.to(device) if kpts_t is not None else None,
            bbox_t.to(device) if bbox_t is not None else None,
            gd["bio_even"], gd["pose_vec_world"], gd["signed_knee"], gd["quality"],
        )
        feat20 = build_conditioned_precursor_features(
            gd["history"], gd["bio_even"], gd["pose_vec_world"], gd["signed_knee"],
            gd["quality"], gd["m_pose"], gd["edges"], gd["is_car"], gd["nbr_mask"], gd["obs_dt"],
        )

    arms = (
        ("EKR6_M0_nopose_nonbr", False, False),
        ("EKR6_M1_splitpose_nonbr", True, False),
        ("EKR6_M2b_nopose_conflict", False, True),
        ("EKR6_Full_Slot0", True, True),
    )
    all_names = ["EKR6_KinCohFilter_Only"] + [a[0] for a in arms] + ["EKR6_Full_OOFRegime"]
    err_s = {nm: np.zeros((len(seeds), n, 6), dtype=np.float64) for nm in all_names}
    audits_5s = {nm: [] for nm in ("EKR6_M0_nopose_nonbr", "EKR6_Full_Slot0")}
    cal_kalman_by_fold: dict[str, dict[str, float]] = {}
    oof_regime_by_fold: dict[str, list[dict[str, float]]] = {}
    groups = sorted(set(grp_arr.tolist()))
    ep_oof = max(14, epochs // 2)

    for held in groups:
        tr = np.flatnonzero(grp_arr != held)
        tr_g = torch.from_numpy(tr).to(device)
        bh_tr = bbox_h_g[tr_g] if bbox_h_g is not None else None
        cal_kalman_by_fold[str(held)] = calibrate_coherence_kalman_params(
            gd["history"][tr_g], gd["obs_dt"][tr_g], gd["fut_s"][tr_g], gd["truth"][tr_g], bh_tr
        )
        oof_regime_by_fold[str(held)] = []

    for si, seed in enumerate(seeds):
        pk_seed = {nm: np.zeros((n, 6, tf, 2), dtype=np.float64) for nm in audits_5s}
        ix_seed = {nm: np.zeros(n, dtype=np.int64) for nm in audits_5s}
        for held in groups:
            te = np.flatnonzero(grp_arr == held)
            tr = np.flatnonzero(grp_arr != held)
            te_g, tr_g = torch.from_numpy(te).to(device), torch.from_numpy(tr).to(device)
            kcfg = cal_kalman_by_fold[str(held)]

            with torch.inference_mode():
                vf_te = compute_coherence_kalman_velocity(
                    gd["history"][te_g], gd["obs_dt"][te_g], bbox_h_g[te_g] if bbox_h_g is not None else None, **kcfg
                )["v_filt"].cpu().numpy()
                kin_te_path = hist_np[te, -1:, :] + vf_te[:, None, :] * fut_s_np[te, :, None]
                vf_tr = compute_coherence_kalman_velocity(
                    gd["history"][tr_g], gd["obs_dt"][tr_g], bbox_h_g[tr_g] if bbox_h_g is not None else None, **kcfg
                )["v_filt"].cpu().numpy()
                kin_tr_path = hist_np[tr, -1:, :] + vf_tr[:, None, :] * fut_s_np[tr, :, None]
            err_s["EKR6_KinCohFilter_Only"][si, te] = horizon_and_post_onset_errors(kin_te_path, truth_np[te], onset_np[te], dt=dt)

            for arm_nm, u_pose, u_nbr in arms:
                m_f = fit_ekr6_arm_fold(seed, u_pose, u_nbr, kcfg, grp_arr, gd, split_covis, bbox_h_g, tr, batch_size, epochs, lr, device)
                with torch.inference_mode():
                    ot = _fwd_ekr6(m_f, gd, split_covis, bbox_h_g, te_g)
                    pwt, lgt, alt = ot["paths"].cpu().numpy(), ot["logits"].cpu().numpy(), ot["act_logits"].cpu().numpy()
                err_s[arm_nm][si, te] = horizon_and_post_onset_errors(pwt[:, 0], truth_np[te], onset_np[te], dt=dt)
                if arm_nm in pk_seed:
                    pk_seed[arm_nm][te] = pwt - hist_np[te, -1:, None, :]
                    ix_seed[arm_nm][te] = 0

                if arm_nm == "EKR6_Full_Slot0":
                    src_gs = sorted(set(grp_arr[tr].tolist()))
                    oof_ml = np.zeros((len(tr), 6), dtype=np.float64)
                    oof_cal = np.zeros((len(tr), 5), dtype=np.float64)
                    oof_pk = np.zeros((len(tr), 6, tf, 2), dtype=np.float64)
                    for vm_l in _inner_group_splits(src_gs, grp_arr[tr], max_splits=3):
                        in_tr, in_va = tr[~vm_l], tr[vm_l]
                        m_in = fit_ekr6_arm_fold(seed + 19, True, True, kcfg, grp_arr, gd, split_covis, bbox_h_g, in_tr, batch_size, ep_oof, lr, device)
                        with torch.inference_mode():
                            oo = _fwd_ekr6(m_in, gd, split_covis, bbox_h_g, torch.from_numpy(in_va).to(device))
                            otr = _fwd_ekr6(m_in, gd, split_covis, bbox_h_g, torch.from_numpy(in_tr).to(device))
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
                    rcfg = fit_ekr6_source_oof_regime_selector(kin_tr_path, oof_pk[:, 0], oof_med, truth_np[tr], path_len_all[tr])
                    oof_regime_by_fold[str(held)].append(rcfg)

                    with torch.inference_mode():
                        otr_f = _fwd_ekr6(m_f, gd, split_covis, bbox_h_g, tr_g)
                    ch_f = _fit_cond_precursor_head(seed, feat20, otr_f["act_logits"].detach(), gd["y_act"][tr], gd["valid_act"][tr], tr, device)
                    with torch.inference_mode():
                        cal_te = ch_f(feat20[te_g], ot["act_logits"]).cpu().numpy()
                    st_t, sp_t = build_delayed_onset_start_stop_candidates(hist_np[te], cor_xy[te], np.full(len(te), med_st), fut_s_np[te], dt_obs=dt)
                    pwt_a = pwt.copy(); pwt_a[:, 1], pwt_a[:, 2] = st_t, sp_t
                    pc_te = 1.0 / (1.0 + np.exp(-np.clip(cal_te * np.asarray(cal_c["a_pl"])[None] + np.asarray(cal_c["b_pl"])[None], -30.0, 30.0)))
                    ix_mt = select_medoid_expected_ade(pwt_a, lgt, pc_te, va_np[te], beta=max(float(cal_c["beta"]), 0.15))
                    p_reg = apply_ekr6_regime_selector(kin_te_path, pwt[:, 0], pwt_a[np.arange(len(te)), ix_mt], path_len_all[te], rcfg)
                    err_s["EKR6_Full_OOFRegime"][si, te] = horizon_and_post_onset_errors(p_reg, truth_np[te], onset_np[te], dt=dt)

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
        eq_ci = safe_site_clip_ci(d2[m], grp_arr[m], unit_arr[m]) if is_pinns else _loc_bootstrap_ci(d2[m], grp_arr[m])
        return {
            "group_equal": eq_ci,
            "pooled_window_weighted": _pooled_clip_bootstrap_ci(d2[m], unit_arr[m]),
        }

    strata_out = {}
    for s_nm, msk in strata.items():
        cnt = int(msk.sum())
        if cnt == 0:
            continue
        m_eq = {"Last2_CV": _agg_eq(err_cv, msk)}
        m_pl = {"Last2_CV": _agg_pool(err_cv, msk)}
        for nm in all_names:
            m_eq[nm] = _agg_eq(em[nm], msk)
            m_pl[nm] = _agg_pool(em[nm], msk)
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
                "kin_coh_filter_minus_CV": _pair_both(em["EKR6_KinCohFilter_Only"][:, :2] - err_cv[:, :2], msk),
                "ekr6_m0_minus_CV": _pair_both(em["EKR6_M0_nopose_nonbr"][:, :2] - err_cv[:, :2], msk),
                "ekr6_m1_minus_CV": _pair_both(em["EKR6_M1_splitpose_nonbr"][:, :2] - err_cv[:, :2], msk),
                "ekr6_m2b_minus_CV": _pair_both(em["EKR6_M2b_nopose_conflict"][:, :2] - err_cv[:, :2], msk),
                "ekr6_full_slot0_minus_CV": _pair_both(em["EKR6_Full_Slot0"][:, :2] - err_cv[:, :2], msk),
                "ekr6_oof_regime_minus_CV": _pair_both(em["EKR6_Full_OOFRegime"][:, :2] - err_cv[:, :2], msk),
                "ekr6_pose_no_nbr_full": _pair_both(em["EKR6_M1_splitpose_nonbr"][:, :2] - em["EKR6_M0_nopose_nonbr"][:, :2], msk),
                "ekr6_pose_no_nbr_post_onset": _pair_both(em["EKR6_M1_splitpose_nonbr"][:, 4:6] - em["EKR6_M0_nopose_nonbr"][:, 4:6], msk),
                "ekr6_nbr_no_pose_full": _pair_both(em["EKR6_M2b_nopose_conflict"][:, :2] - em["EKR6_M0_nopose_nonbr"][:, :2], msk),
                "ekr6_full_minus_kin_filter": _pair_both(em["EKR6_Full_Slot0"][:, :2] - em["EKR6_KinCohFilter_Only"][:, :2], msk),
            },
        }
    return {
        "n_windows": n,
        "source_calibrated_kalman_params": cal_kalman_by_fold,
        "source_oof_regime_selectors": oof_regime_by_fold,
        "strata": strata_out,
        "candidate_generation_vs_selection_5seed": {nm: summarize_6candidate_5seed(audits_5s[nm]) for nm in audits_5s},
    }


def run_pinns_ekr6(args: argparse.Namespace, device: torch.device) -> dict:
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
    return eval_ekr6_cohort_5seed(dp["site"], dp["clip"], gd_p, kpts_t, bbox_t, hist_np, truth_np, ya_p, va_p, on_p, strata, fs323 | e_exp, 0.1, device, SEEDS, args.epochs, is_pinns=True)


def run_jrdb_ekr6(repo_root: Path, epochs: int, device: torch.device) -> dict:
    dj, cj = load_jrdb_with_blind_emergence(repo_root)
    hist_np, truth_np = dj["history"].numpy(), dj["truth"].numpy()
    n = len(hist_np)
    disp_j = np.linalg.norm(hist_np[:, 1] - hist_np[:, 0], axis=-1)
    act_m = (dj["y_act"][:, :4] & dj["valid_act"][:, :4]).any(axis=1)
    e_tr = cj["e_true"]
    strata_j = {
        "all_5836": np.ones(n, dtype=bool),
        "true_deocclusion_507": e_tr,
        "true_deocc_lowdisp_lt0p5m": e_tr & (disp_j < 0.50),
        "true_deocclusion_active_maneuvers": e_tr & act_m,
        "lowdisp_all_lt0p5m": disp_j < 0.50,
    }
    gd_j = {k: dj[k].to(device) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
    gd_j["obs_dt"] = torch.full((n, 1), 0.4, dtype=torch.float64, device=device)
    gd_j["fut_s"] = torch.arange(1, 8, dtype=torch.float64, device=device)[None].expand(n, -1) * 0.4
    return eval_ekr6_cohort_5seed(dj["location"], dj["actors"], gd_j, None, None, hist_np, truth_np, dj["y_act"], dj["valid_act"], dj["onset"], strata_j, e_tr, 0.4, device, SEEDS, max(26, epochs - 6), batch_size=96, is_pinns=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--video-cache-dir", type=Path, default=Path("/tmp/w26_vids"))
    ap.add_argument("--explicit-audit-json", type=Path, default=Path("data/annotations/pinns_explicit_241_visual_audit.json"))
    ap.add_argument("--wave24-labels-json", type=Path, default=Path("data/annotations/pinns_blind_raw_video_emergence_labels.json"))
    ap.add_argument("--zip-path", type=Path, default=Path("wave26_pinns_jrdb_ekr6_emergence.zip"))
    ap.add_argument("--epochs", type=int, default=34)
    args = ap.parse_args()

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu >= 2:
        print(f"[Wave 26A] Running PINNS on cuda:0 and JRDB on cuda:1 in parallel ({n_gpu} GPUs)...", flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_p = pool.submit(run_pinns_ekr6, args, torch.device("cuda:0"))
            fut_j = pool.submit(run_jrdb_ekr6, args.repo_root, args.epochs, torch.device("cuda:1"))
            res_p, res_j = fut_p.result(), fut_j.result()
    else:
        dev = torch.device("cuda:0" if n_gpu == 1 else "cpu")
        print(f"[Wave 26A] Running sequentially on {dev}...", flush=True)
        res_p = run_pinns_ekr6(args, dev)
        res_j = run_jrdb_ekr6(args.repo_root, args.epochs, dev)

    summary = {
        "wave": "wave26_pinns_jrdb_ekr6_emergence_5seed",
        "pinns_4site_loso": res_p,
        "jrdb_9loc_lolo": res_j,
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
