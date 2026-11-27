"""Wave 21A-GPU0: Blind PINNS physical emergence audit, 6-candidate diagnosis, & 6-slot 2x2 precursor."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.benchmark_heldout_sites import HISTORY, WINDOW, load_site_windows
from scripts.run_pinns_matched_o2_multiduration import make_site_weights
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import load_ppt_cohort_with_causal_maps
from scripts.run_wave18_conflict_gated_interaction_5seed import safe_site_clip_ci
from scripts.run_wave19_unified_clean_m0_to_m3_5seed import _fwd_u, fit_unified_clean_fold
from scripts.run_wave20_decoupled_m2b_m3_synergy_5seed import (
    SEEDS,
    candidate_implied_actions_np,
    fit_decoupled_action_head_only,
    inner_decoupled_calibration,
    rerank_top1_modes,
    score_grouped_actions,
)
from src.models.emergence_maneuver_6slot import (
    O2Maneuver6SlotPrecursorNet,
    apply_6slot_selector,
    audit_6candidate_generation_vs_selection,
    compute_action_labels_with_onset,
    fit_strict_source_platt_and_slot_selector,
    horizon_and_post_onset_errors,
    score_site_isolated_actions,
    slot_semantic_maneuver_loss,
)

VEH_PREFIXES = ("car", "bus", "tru", "van", "bic", "mot", "cyc")


def parse_raw_30hz_boxes(lbl_path: Path) -> tuple[dict[int, list[dict]], int, dict[int, int]]:
    """Parse raw 30Hz multi-object pixel boxes without touching future trajectories or predictions."""
    by_raw: dict[int, list[dict]] = defaultdict(list)
    first_person_raw: dict[int, int] = {}
    min_raw = 10**9
    with lbl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ps = ln.split()
            if len(ps) < 11:
                continue
            rf = int(float(ps[0]))
            min_raw = min(min_raw, rf)
            kind = ps[1].strip()
            oid = int(float(ps[2]))
            ox, oy, _, bw, bh = map(float, ps[3:8])
            by_raw[rf].append({"type": kind, "id": oid, "ox": ox, "oy": oy, "w": bw, "h": bh, "fixed": int(float(ps[10]))})
            if kind.lower().startswith("per"):
                first_person_raw[oid] = min(rf, first_person_raw.get(oid, rf))
    return by_raw, (0 if min_raw == 10**9 else min_raw), first_person_raw


def blind_pinns_physical_emergence_audit(
    meta_dir: Path,
    pose_npz: Path,
    dp: dict,
) -> dict:
    """Blind prediction-free physical occluder, torso/foot visibility, & matched-control audit on PINNS."""
    groups, _, _ = load_site_windows(meta_dir)
    rows = [r for s in sorted(groups) for r in groups[s]]
    n = len(rows)
    with np.load(pose_npz) as pz:
        kpts = pz["keypoints"].astype(np.float64)   # [N, 5, 17, 3]
        bbox = pz["bboxes"].astype(np.float64)      # [N, 5, 4] cx, cy, w, h

    clip_cache: dict[str, tuple[dict[int, list[dict]], int, tuple[float, float], list[tuple[float, float]]]] = {}
    for r in rows:
        if r.clip in clip_cache:
            continue
        cond, _ = r.clip.rsplit("_", 1)
        cal = json.loads((meta_dir / "videos" / cond / "calib.json").read_text())
        vw, vh = float(cal["v_size"][0]), float(cal["v_size"][1])
        by_raw, min_raw, first_p = parse_raw_30hz_boxes(meta_dir / "labels" / f"{r.clip}.txt")
        # Identify interior architectural/foliage occlusion zones where >=2 distinct tracks originate after clip start
        starts_xy: list[tuple[float, float]] = []
        for pid, rf0 in first_p.items():
            if rf0 <= min_raw + 6:
                continue
            for ob in by_raw.get(rf0, []):
                if ob["type"].lower().startswith("per") and ob["id"] == pid:
                    x1, y1 = ob["ox"] - 0.5 * ob["w"], ob["oy"] - 0.5 * ob["h"]
                    x2, y2 = ob["ox"] + 0.5 * ob["w"], ob["oy"] + 0.5 * ob["h"]
                    if x1 > 45.0 and y1 > 35.0 and x2 < vw - 45.0 and y2 < vh - 35.0:
                        starts_xy.append((ob["ox"], ob["oy"]))
                    break
        clip_cache[r.clip] = (by_raw, min_raw, (vw, vh), starts_xy)

    e_veh = np.zeros(n, dtype=bool)
    e_pers_scene = np.zeros(n, dtype=bool)
    e_border = np.zeros(n, dtype=bool)
    e_clip_start = np.zeros(n, dtype=bool)
    e_open_reid = np.zeros(n, dtype=bool)
    prior_20 = np.zeros(n, dtype=bool)

    for i, r in enumerate(rows):
        by_raw, min_raw, (vw, vh), starts_xy = clip_cache[r.clip]
        s_10 = r.t0 - HISTORY + 1
        r_start = 3 * s_10 + 1
        # Count contiguous prior 10Hz presence before s_10
        prev_cnt = 0
        while any(o["type"].lower().startswith("per") and o["id"] == r.person_id for o in by_raw.get(3 * (s_10 - prev_cnt - 1) + 1, [])):
            prev_cnt += 1
        prior_20[i] = prev_cnt >= 20
        if prev_cnt > 0:
            continue
        if r_start <= min_raw + 6:
            e_clip_start[i] = True
            continue

        cx, cy, bw, bh = bbox[i, 0]
        x1, y1, x2, y2 = cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh
        near_border = (x1 <= 45.0) or (y1 <= 30.0) or (x2 >= vw - 45.0) or (y2 >= vh - 30.0)

        veh_occ, pers_occ = False, False
        for rf in range(max(min_raw, r_start - 15), r_start + 4):
            for ob in by_raw.get(rf, []):
                if ob["type"].lower().startswith("per") and ob["id"] == r.person_id:
                    continue
                h_gap = abs(ob["ox"] - cx) - 0.55 * (ob["w"] + bw)
                ob_bot = ob["oy"] + 0.5 * ob["h"]
                if h_gap <= 20.0 and ob_bot >= y1 + 0.25 * bh:
                    if ob["type"].lower().startswith(VEH_PREFIXES):
                        veh_occ = True
                    elif ob["type"].lower().startswith("per") and ob_bot >= y2 - 0.15 * bh:
                        pers_occ = True
        arch_zone = (not near_border) and sum(np.hypot(sx - cx, sy - cy) <= max(65.0, 1.2 * bw) for sx, sy in starts_xy) >= 2
        if veh_occ:
            e_veh[i] = True
        elif pers_occ or arch_zone:
            e_pers_scene[i] = True
        elif near_border:
            e_border[i] = True
        else:
            e_open_reid[i] = True

    e_true = e_veh | e_pers_scene
    bh_med = np.median(bbox[:, :, 3], axis=1)
    sh_hip_ok = ((kpts[:, :, [5, 6, 11, 12], 2] >= 0.25).all(axis=-1).sum(axis=1) >= 3) & (bh_med >= 50.0)
    feet_ok = ((kpts[:, :, [15, 16], 2] >= 0.25).all(axis=-1).sum(axis=1) >= 3) & (bh_med >= 50.0)
    disp5 = np.linalg.norm(np.diff(dp["history"].numpy(), axis=1), axis=-1).sum(axis=1)
    v0 = np.linalg.norm(dp["history"].numpy()[:, -1] - dp["history"].numpy()[:, -2], axis=-1) / 0.1

    # 1-to-1 matched continuously-visible control per site matched on initial speed v0 and bbox height
    ctrl = np.zeros(n, dtype=bool)
    site_arr = dp["site"]
    for idx in np.flatnonzero(e_true):
        cand = np.flatnonzero(prior_20 & (~ctrl) & (site_arr == site_arr[idx]))
        if len(cand) > 0:
            cost = np.abs(v0[cand] - v0[idx]) / 0.5 + np.abs(bh_med[cand] - bh_med[idx]) / 40.0
            ctrl[cand[int(np.argmin(cost))]] = True

    # Whole-clip deterministic stratified Dev vs Locked Final Eval split within each country
    dev_clip_m = np.zeros(n, dtype=bool)
    clip_arr = dp["clip"]
    for s in sorted(set(site_arr.tolist())):
        s_clips = sorted(set(clip_arr[site_arr == s].tolist()))
        dev_set = set(s_clips[0::2])  # Strictly isolated whole video clips!
        dev_clip_m |= np.isin(clip_arr, list(dev_set))
    return {
        "e_true_vehicle": e_veh, "e_true_person_or_arch": e_pers_scene, "e_true_all": e_true,
        "e_border": e_border, "e_clip_start": e_clip_start, "e_open_reid": e_open_reid,
        "vis_torso": sh_hip_ok, "vis_feet": feet_ok, "disp5": disp5,
        "matched_visible_ctrl": ctrl, "dev_clips": dev_clip_m, "locked_eval_clips": ~dev_clip_m,
    }


def _fwd_6slot(net: O2Maneuver6SlotPrecursorNet, gd: dict, ids_g: torch.Tensor, g_scale: float = 1.0) -> dict[str, torch.Tensor]:
    return net(
        gd["history"][ids_g], gd["bio_even"][ids_g], gd["pose_vec_world"][ids_g], gd["signed_knee"][ids_g],
        gd["quality"][ids_g], gd["m_pose"][ids_g], gd["edges"][ids_g], gd["is_car"][ids_g], gd["nbr_mask"][ids_g],
        obs_dt=gd["obs_dt"][ids_g], fut_times_s=gd["fut_s"][ids_g], conflict_scale=g_scale,
    )


def fit_6slot_arm_fold(
    seed: int,
    use_pose: bool,
    use_interaction: bool,
    grp_arr: np.ndarray,
    gd: dict,
    tr_ids: np.ndarray,
    bsz: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> O2Maneuver6SlotPrecursorNet:
    torch.manual_seed(seed)
    net = O2Maneuver6SlotPrecursorNet(use_pose=use_pose, use_interaction=use_interaction).to(device)
    w_tr = make_site_weights(grp_arr, tr_ids).to(device)
    tr_t = torch.from_numpy(tr_ids).to(device)
    gt, ya, va = gd["truth"][tr_t], gd["y_act"][tr_t], gd["valid_act"][tr_t]
    pw = (((~ya) & va).sum(0).float().clamp_min(1.0) / (ya & va).sum(0).float().clamp_min(1.0)).clamp(1.0, 12.0)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=lr * 0.05)
    net.train()
    for ep in range(epochs):
        g_rng = torch.Generator().manual_seed(seed * 991 + ep)
        perm = torch.randperm(len(tr_ids), generator=g_rng).to(device)
        for chunk in perm.split(bsz):
            loss = slot_semantic_maneuver_loss(_fwd_6slot(net, gd, tr_t[chunk]), gt[chunk], w_tr[chunk], ya[chunk], va[chunk], pw)
            if loss is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
        sch.step()
    return net.eval()


def run_2x2_6slot_and_baseline_eval(
    tag: str,
    grp_arr: np.ndarray,
    unit_arr: np.ndarray,
    is_pinns: bool,
    gd: dict,
    hist_np: np.ndarray,
    truth_np: np.ndarray,
    cv_paths_rel: np.ndarray,
    ya_np: np.ndarray,
    va_np: np.ndarray,
    onset_np: np.ndarray,
    strata: dict[str, np.ndarray],
    diag_mask: np.ndarray,
    dt: float,
    bsz: int,
    ep_base: tuple[int, int, int, int],
    ep_slot: int,
    lr_base: tuple[float, float, float, float],
    lr_slot: float,
    seeds: list[int],
    device: torch.device,
    include_unified_baseline: bool = True,
) -> dict:
    n, tf, _ = truth_np.shape
    groups = sorted(set(grp_arr.tolist()))
    slot_arms = (
        ("S6_M0_nopose_nonbr", False, False),
        ("S6_M1_pose_nonbr", True, False),
        ("S6_M2b_nopose_nbr", False, True),
        ("S6_Full_pose_nbr", True, True),
    )
    all_names = (["U_M2b_mono_conflict", "U_M3_decoupled_site_thr"] if include_unified_baseline else []) + [a[0] for a in slot_arms]
    err_s = {nm: np.zeros((len(seeds), n, 6), dtype=np.float64) for nm in all_names}
    err_cv = horizon_and_post_onset_errors(cv_paths_rel + hist_np[:, -1:, :], truth_np, onset_np, dt=dt)

    # Seed-0 per-window 6-candidate buffers for exact 'not generated vs not selected' audit
    pk_s0 = {nm: np.zeros((n, 6, tf, 2), dtype=np.float64) for nm in all_names}
    idx_s0 = {nm: np.zeros(n, dtype=np.int64) for nm in all_names}

    act_prob_u = np.zeros((len(seeds), n, 5), dtype=np.float64)
    act_bin_u_site = np.zeros((len(seeds), n, 5), dtype=np.float64)
    act_thr_u_list = []
    act_prob_s6 = np.zeros((len(seeds), n, 5), dtype=np.float64)
    act_bin_s6 = np.zeros((len(seeds), n, 5), dtype=np.float64)

    for si, seed in enumerate(seeds):
        for g in groups:
            tr, te = np.flatnonzero(grp_arr != g), np.flatnonzero(grp_arr == g)
            te_g = torch.from_numpy(te).to(device)
            src_g = sorted(set(grp_arr[tr].tolist()))
            val_sets = [src_g[k :: min(3, max(1, len(src_g)))] for k in range(min(3, max(1, len(src_g))))] if len(src_g) >= 2 else [src_g]

            if include_unified_baseline:
                sc, a_pl, b_pl, thr_pl, beta = inner_decoupled_calibration(
                    seed, grp_arr, gd, hist_np, truth_np, ya_np, va_np, dt, tr, bsz, ep_base, lr_base, device,
                )
                act_thr_u_list.append(thr_pl)
                mods = fit_unified_clean_fold(seed, grp_arr, gd, tr, bsz, ep_base, lr_base, device, g_scale=sc)
                m2b = mods["U_M2b_mono_conflict"]
                m3_dec = fit_decoupled_action_head_only(seed, m2b, grp_arr, gd, tr, bsz, ep_base[3], lr_base[3], device)
                with torch.inference_mode():
                    o2 = _fwd_u(m2b, gd, te_g)
                    od = _fwd_u(m3_dec, gd, te_g)
                    pw2, lg2 = o2["paths"].cpu().numpy(), o2["logits"].cpu().numpy()
                    i2 = lg2.argmax(axis=1)
                    err_s["U_M2b_mono_conflict"][si, te] = horizon_and_post_onset_errors(pw2[np.arange(len(te)), i2], truth_np[te], onset_np[te], dt=dt)
                    pwd, lgd, alg = od["paths"].cpu().numpy(), od["logits"].cpu().numpy(), od["act_logits"].cpu().numpy()
                    p_cal = 1.0 / (1.0 + np.exp(-np.clip(alg * a_pl[None, :] + b_pl[None, :], -30.0, 30.0)))
                    act_prob_u[si, te] = p_cal
                    # Strictly site-isolated threshold (never averaged across test sites!)
                    act_bin_u_site[si, te] = ((p_cal >= thr_pl[None, :]) & va_np[te]).astype(np.float64)
                    c_act = candidate_implied_actions_np(hist_np[te], pwd - hist_np[te, -1:, None, :], dt=dt)
                    rk = rerank_top1_modes(lgd, p_cal, c_act, beta)
                    err_s["U_M3_decoupled_site_thr"][si, te] = horizon_and_post_onset_errors(pwd[np.arange(len(te)), rk], truth_np[te], onset_np[te], dt=dt)
                    if si == 0:
                        pk_s0["U_M2b_mono_conflict"][te] = pw2 - hist_np[te, -1:, None, :]
                        idx_s0["U_M2b_mono_conflict"][te] = i2
                        pk_s0["U_M3_decoupled_site_thr"][te] = pwd - hist_np[te, -1:, None, :]
                        idx_s0["U_M3_decoupled_site_thr"][te] = rk

            for arm_nm, u_pose, u_nbr in slot_arms:
                oof_ml = np.zeros((len(tr), 6), dtype=np.float64)
                oof_al = np.zeros((len(tr), 5), dtype=np.float64)
                oof_pk = np.zeros((len(tr), 6, tf, 2), dtype=np.float64)
                for v_locs in val_sets:
                    vm_l = np.isin(grp_arr[tr], v_locs) if len(val_sets) > 1 else (np.arange(len(tr)) % 2 == 0)
                    in_tr, in_va = tr[~vm_l], tr[vm_l]
                    m_in = fit_6slot_arm_fold(seed, u_pose, u_nbr, grp_arr, gd, in_tr, bsz, max(18, ep_slot - 8), lr_slot, device)
                    with torch.inference_mode():
                        oo = _fwd_6slot(m_in, gd, torch.from_numpy(in_va).to(device))
                        oof_ml[vm_l] = oo["logits"].cpu().numpy()
                        oof_al[vm_l] = oo["act_logits"].cpu().numpy()
                        oof_pk[vm_l] = oo["paths"].cpu().numpy()
                cal_s = fit_strict_source_platt_and_slot_selector(oof_ml, oof_al, oof_pk, truth_np[tr], ya_np[tr], va_np[tr])
                m_f = fit_6slot_arm_fold(seed, u_pose, u_nbr, grp_arr, gd, tr, bsz, ep_slot, lr_slot, device)
                with torch.inference_mode():
                    ot = _fwd_6slot(m_f, gd, te_g)
                    pwt, lgt, alt = ot["paths"].cpu().numpy(), ot["logits"].cpu().numpy(), ot["act_logits"].cpu().numpy()
                sel_i, pc_s, pb_s = apply_6slot_selector(lgt, alt, va_np[te], cal_s)
                err_s[arm_nm][si, te] = horizon_and_post_onset_errors(pwt[np.arange(len(te)), sel_i], truth_np[te], onset_np[te], dt=dt)
                if arm_nm == "S6_Full_pose_nbr":
                    act_prob_s6[si, te], act_bin_s6[si, te] = pc_s, pb_s.astype(np.float64)
                if si == 0:
                    pk_s0[arm_nm][te] = pwt - hist_np[te, -1:, None, :]
                    idx_s0[arm_nm][te] = sel_i
        print(f"[W21 {tag}] Finished seed={seed}", flush=True)

    em = {nm: err_s[nm].mean(axis=0) for nm in all_names}
    from scripts.run_wave18_conflict_gated_interaction_5seed import _loc_bootstrap_ci

    def _ci2(diff2: np.ndarray, msk: np.ndarray) -> dict:
        if int(msk.sum()) < 4:
            return {"n": int(msk.sum())}
        if is_pinns:
            return safe_site_clip_ci(diff2[msk], grp_arr[msk], unit_arr[msk])
        return _loc_bootstrap_ci(diff2[msk], grp_arr[msk]) if len(set(grp_arr[msk].tolist())) >= 2 else {"mean": {"ade_m": float(diff2[msk, 0].mean()), "fde_m": float(diff2[msk, 1].mean())}}

    def _agg_row(arr6: np.ndarray, sm: np.ndarray) -> dict[str, float]:
        gs = sorted(set(grp_arr[sm].tolist()))
        vals = np.mean([arr6[(grp_arr == g) & sm].mean(axis=0) for g in gs], axis=0)
        return {
            "ade1_full_m": float(vals[0]), "fde1_full_m": float(vals[1]),
            "ade1_trig_0p5s_m": float(vals[2]), "ade1_trig_1p0s_m": float(vals[3]),
            "ade1_post_onset_0p5s_m": float(vals[4]), "ade1_post_onset_1p0s_m": float(vals[5]),
        }

    strata_out = {}
    for sk, sm in strata.items():
        if int(sm.sum()) == 0:
            continue
        md = {"Last2_CV": _agg_row(err_cv, sm)}
        for nm in all_names:
            md[nm] = _agg_row(em[nm], sm)
        s_full = em["S6_Full_pose_nbr"]
        s_m0 = em["S6_M0_nopose_nonbr"]
        s_pose = em["S6_M1_pose_nonbr"]
        s_nbr = em["S6_M2b_nopose_nbr"]
        paired = {
            "full_minus_CV_3s": _ci2(s_full[:, 0:2] - err_cv[:, 0:2], sm),
            "full_minus_CV_post_onset_0p5_1p0s": _ci2(s_full[:, 4:6] - err_cv[:, 4:6], sm),
            "pose_effect_no_nbr_3s": _ci2(s_pose[:, 0:2] - s_m0[:, 0:2], sm),
            "pose_effect_no_nbr_post_onset_0p5_1p0s": _ci2(s_pose[:, 4:6] - s_m0[:, 4:6], sm),
            "pose_effect_with_nbr_3s": _ci2(s_full[:, 0:2] - s_nbr[:, 0:2], sm),
            "pose_effect_with_nbr_post_onset_0p5_1p0s": _ci2(s_full[:, 4:6] - s_nbr[:, 4:6], sm),
            "nbr_effect_no_pose_3s": _ci2(s_nbr[:, 0:2] - s_m0[:, 0:2], sm),
            "nbr_effect_with_pose_3s": _ci2(s_full[:, 0:2] - s_pose[:, 0:2], sm),
        }
        if include_unified_baseline:
            paired["full_6slot_minus_U_M3_decoupled_3s"] = _ci2(s_full[:, 0:2] - em["U_M3_decoupled_site_thr"][:, 0:2], sm)
            paired["full_6slot_minus_U_M3_post_onset_0p5_1p0s"] = _ci2(s_full[:, 4:6] - em["U_M3_decoupled_site_thr"][:, 4:6], sm)
        strata_out[sk] = {
            "n_windows": int(sm.sum()),
            "n_clips_or_units": int(len(set(unit_arr[sm].tolist()))),
            "n_sites_or_locs": int(len(set(grp_arr[sm].tolist()))),
            "models": md,
            "paired_ci": paired,
        }

    truth_rel = truth_np - hist_np[:, -1:, :]
    cand_diag = {nm: audit_6candidate_generation_vs_selection(hist_np, pk_s0[nm], idx_s0[nm], truth_rel, cv_paths_rel, ya_np, va_np, diag_mask, dt=dt) for nm in all_names}
    res_d = {
        "n_windows": n,
        "strata": strata_out,
        "candidate_generation_vs_selection_seed0": cand_diag,
        "s6_full_site_isolated_actions": score_site_isolated_actions(ya_np, va_np, act_prob_s6, act_bin_s6, grp_arr),
    }
    if include_unified_baseline and act_thr_u_list:
        res_d["u_m3_strict_site_isolated_actions"] = score_site_isolated_actions(ya_np, va_np, act_prob_u, act_bin_u_site, grp_arr)
        res_d["u_m3_legacy_pooled_thr_actions"] = score_grouped_actions(ya_np, va_np, act_prob_u.mean(0), grp_arr, np.mean(act_thr_u_list, axis=0))
    return res_d



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = load_ppt_cohort_with_causal_maps(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin)
    blind = blind_pinns_physical_emergence_audit(args.meta_dir, args.pose_npz, dp)

    hp_np, tp_np = dp["history"].numpy(), dp["truth"].numpy()
    n = len(hp_np)
    hm, fm = np.ones(5, dtype=np.float32), np.ones(30, dtype=np.float32)
    ya_np = np.zeros((n, 5), dtype=bool)
    va_np = np.zeros((n, 5), dtype=bool)
    on_np = np.full((n, 5), -1, dtype=np.int64)
    for i in range(n):
        ya_np[i], va_np[i], on_np[i] = compute_action_labels_with_onset(hp_np[i] - hp_np[i, -1:], hm, tp_np[i] - hp_np[i, -1:], fm, dt=0.1)

    gdp = {k: dp[k].to(dev) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
    gdp["y_act"] = torch.from_numpy(ya_np).to(dev)
    gdp["valid_act"] = torch.from_numpy(va_np).to(dev)
    gdp["obs_dt"] = torch.full((n, 4), 0.1, dtype=torch.float64, device=dev)
    gdp["fut_s"] = torch.arange(1, 31, dtype=torch.float64, device=dev)[None].expand(n, -1) * 0.1
    cv_rel = ((hp_np[:, -1:] - hp_np[:, -2:-1]) / 0.1) * (np.arange(1, 31)[None, :, None] * 0.1)

    f323 = dp["strata"]["first_short_323"]
    f449 = dp["strata"]["first_contiguous_449"]
    man_any = (ya_np[:, :4] & va_np[:, :4]).any(axis=1)
    strata = {
        "all_1482": np.ones(n, dtype=bool),
        "locked_eval_clips_all": blind["locked_eval_clips"],
        "dev_clips_all": blind["dev_clips"],
        "true_occluder_emergence_all": blind["e_true_all"],
        "locked_eval_true_emergence": blind["locked_eval_clips"] & blind["e_true_all"],
        "true_emergence_lowdisp_lt0p5m": blind["e_true_all"] & (blind["disp5"] < 0.50),
        "true_emergence_clear_torso_feet": blind["e_true_all"] & blind["vis_torso"] & blind["vis_feet"],
        "true_emergence_active_maneuver": blind["e_true_all"] & man_any,
        "matched_continuously_visible_ctrl": blind["matched_visible_ctrl"],
        "clip_start_artifact_non_emergence": blind["e_clip_start"],
        "first_short_323": f323,
        "first_short_323_true_occluder_only": f323 & blind["e_true_all"],
    }

    census = {}
    for k_c, m_c in (("first_contiguous_449", f449), ("first_short_323", f323), ("true_occluder_emergence", blind["e_true_all"]), ("matched_visible_ctrl", blind["matched_visible_ctrl"])):
        census[k_c] = {
            "n_windows": int(m_c.sum()),
            "n_clips": int(len(set(dp["clip"][m_c].tolist()))) if m_c.any() else 0,
            "n_sites": int(len(set(dp["site"][m_c].tolist()))) if m_c.any() else 0,
            "true_vehicle_occluder": int((m_c & blind["e_true_vehicle"]).sum()),
            "true_person_or_arch_occluder": int((m_c & blind["e_true_person_or_arch"]).sum()),
            "border_entry": int((m_c & blind["e_border"]).sum()),
            "clip_start_frame0_artifact": int((m_c & blind["e_clip_start"]).sum()),
            "open_field_reid": int((m_c & blind["e_open_reid"]).sum()),
            "vis_shoulders_hips_ge50px": int((m_c & blind["vis_torso"]).sum()),
            "vis_both_feet_ge50px": int((m_c & blind["vis_feet"]).sum()),
            "maneuver_counts": {nm: int((m_c & ya_np[:, ci] & va_np[:, ci]).sum()) for ci, nm in enumerate(("Start", "Stop", "Turn_Left", "Turn_Right", "Maintain"))},
        }

    eval_res = run_2x2_6slot_and_baseline_eval(
        "pinns", dp["site"], dp["clip"], True, gdp, hp_np, tp_np, cv_rel, ya_np, va_np, on_np, strata,
        blind["e_true_all"] | f323, 0.1, 64, (45, 25, 26, 28), 34, (2e-3, 1.2e-3, 1.5e-3, 1.5e-3), 1.6e-3, list(args.seeds), dev, True,
    )
    summary = {"experiment": "wave21_pinns_true_emergence_and_6slot_diag", "blind_emergence_census": census, "pinns_4site_loso": eval_res}
    raw = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = (json.dumps({"summary.json": hashlib.sha256(raw).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw)
        zf.writestr("sha256.json", sh)
    print("Saved:", args.zip_path, flush=True)


if __name__ == "__main__":
    main()
