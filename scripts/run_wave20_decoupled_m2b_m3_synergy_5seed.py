"""Wave 20 GPU0: Representation-decoupled M3 action head + Platt rerank on U_M2b_mono_conflict (eliminating multi-task knot interference)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from scripts.run_dair_o2_bioeq_external import SEEDS
from scripts.run_jrdb_rc1_pose_quality_and_initiation_5seed import _loc_bootstrap_ci
from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import _stage_gpu_dict, load_jrdb_10scene_rc2_rc3
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import _copy_matching, load_ppt_cohort_with_causal_maps
from scripts.run_wave18_action_calibration_and_rerank_5seed import (
    candidate_implied_actions_np,
    fit_platt_and_thresholds,
    rerank_top1_modes,
    score_grouped_actions,
)
from scripts.run_wave18_conflict_gated_interaction_5seed import build_interaction_strata, safe_site_clip_ci
from scripts.run_wave19_unified_clean_m0_to_m3_5seed import UnifiedCleanO2Forecaster, _fwd_u, fit_unified_clean_fold


def fit_decoupled_action_head_only(
    seed: int, m2b_frozen: UnifiedCleanO2Forecaster, grp_arr: np.ndarray, gd: dict,
    tr_ids: np.ndarray, bsz: int, ep: int, lr: float, device: torch.device,
) -> UnifiedCleanO2Forecaster:
    """Copy U_M2b_mono_conflict with use_act=False (0.0m trajectory drift!) and train ONLY act_even & act_turn."""
    torch.manual_seed(seed + 44)
    net = UnifiedCleanO2Forecaster("U_M2b_mono_conflict").to(device)
    _copy_matching(net, m2b_frozen)
    net.graph_scale = float(m2b_frozen.graph_scale)
    w_tr = make_site_weights(grp_arr, tr_ids).to(device)
    tr_t = torch.from_numpy(tr_ids).to(device)
    ya_tr, va_tr = gd["y_act"][tr_t].float(), gd["valid_act"][tr_t].float()
    pw = (((1.0 - ya_tr) * va_tr).sum(0).clamp_min(1.0) / (ya_tr * va_tr).sum(0).clamp_min(1.0)).clamp(1.0, 12.0)

    for nm, p in net.named_parameters():
        p.requires_grad = nm.startswith("act_")
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=2e-4)
    gen = torch.Generator().manual_seed(seed + 55)
    net.train()
    for _ in range(ep):
        for ch in torch.randperm(len(tr_ids), generator=gen).split(bsz):
            cg = ch.to(device)
            alg = _fwd_u(net, gd, tr_t[cg])["act_logits"]
            bce = F.binary_cross_entropy_with_logits(alg, ya_tr[cg], pos_weight=pw, reduction="none")
            denom = va_tr[cg].sum().clamp_min(1.0)
            loss = ((bce * va_tr[cg]) * w_tr[cg, None]).sum() / denom
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return net.eval()


def inner_decoupled_calibration(
    seed: int, grp_arr: np.ndarray, gd: dict, hist_np: np.ndarray, truth_np: np.ndarray,
    ya_np: np.ndarray, va_np: np.ndarray, dt: float, tr_ids: np.ndarray, bsz: int,
    ep_s: tuple[int, int, int, int], lr_s: tuple[float, float, float, float], device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    src_grps = sorted(set(grp_arr[tr_ids].tolist()))[:4]
    scales = (0.0, 0.25, 0.5, 1.0)
    sc_oof = {sc: [] for sc in scales}
    oof_mask = np.zeros(len(grp_arr), dtype=bool)
    oof_alg = np.zeros((len(grp_arr), 5), dtype=np.float64)
    oof_mlg = np.zeros((len(grp_arr), 6), dtype=np.float64)
    oof_pw = np.zeros((len(grp_arr), 6, truth_np.shape[1], 2), dtype=np.float64)

    for vg in src_grps:
        in_tr = tr_ids[grp_arr[tr_ids] != vg]
        in_va = tr_ids[grp_arr[tr_ids] == vg]
        va_g = torch.from_numpy(in_va).to(device)
        mods = fit_unified_clean_fold(seed, grp_arr, gd, in_tr, bsz, ep_s, lr_s, device, g_scale=1.0)
        m2b = mods["U_M2b_mono_conflict"]
        with torch.inference_mode():
            for sc in scales:
                m2b.graph_scale = sc
                oo = _fwd_u(m2b, gd, va_g)
                tp = oo["paths"][torch.arange(len(in_va), device=device), oo["logits"].argmax(1)]
                sc_oof[sc].append(float(torch.linalg.vector_norm(tp - gd["truth"][va_g], dim=-1).mean()))
        best_in_sc = float(min(scales, key=lambda s: sc_oof[s][-1] + (1e-7 if s > 0 else 0.0)))
        m2b.graph_scale = best_in_sc
        m3_dec = fit_decoupled_action_head_only(seed, m2b, grp_arr, gd, in_tr, bsz, ep_s[3], lr_s[3], device)
        with torch.inference_mode():
            od = _fwd_u(m3_dec, gd, va_g)
            oof_mask[in_va] = True
            oof_alg[in_va] = od["act_logits"].cpu().numpy()
            oof_mlg[in_va] = od["logits"].cpu().numpy()
            oof_pw[in_va] = od["paths"].cpu().numpy()

    best_sc = float(min(scales, key=lambda s: np.mean(sc_oof[s]) + (1e-7 if s > 0 else 0.0)))
    om = np.flatnonzero(oof_mask)
    a_pl, b_pl, thr_pl = fit_platt_and_thresholds(oof_alg[om], ya_np[om], va_np[om])
    p_oof = 1.0 / (1.0 + np.exp(-np.clip(oof_alg[om] * a_pl[None, :] + b_pl[None, :], -30.0, 30.0)))
    c_oof = candidate_implied_actions_np(hist_np[om], oof_pw[om] - hist_np[om, -1:, None, :], dt=dt)
    best_b, best_e = 0.0, 1e9
    for bt in (0.0, 0.05, 0.10, 0.20, 0.35):
        ki = rerank_top1_modes(oof_mlg[om], p_oof, c_oof, bt)
        e = float(np.linalg.norm(oof_pw[om][np.arange(len(om)), ki] - truth_np[om], axis=-1).mean())
        if e + (1e-7 if bt > 0 else 0.0) < best_e:
            best_e, best_b = e, float(bt)
    return best_sc, a_pl, b_pl, thr_pl, best_b


def run_decoupled_synergy_on_cohort(
    tag: str, grp_arr: np.ndarray, unit_arr: np.ndarray, is_pinns: bool, gd: dict,
    hist_np: np.ndarray, truth_np: np.ndarray, ya_np: np.ndarray, va_np: np.ndarray,
    cv_err: np.ndarray, strata: dict[str, np.ndarray], dt: float, bsz: int,
    ep_s: tuple[int, int, int, int], lr_s: tuple[float, float, float, float],
    seeds: list[int], device: torch.device,
) -> dict:
    n = len(grp_arr)
    groups = sorted(set(grp_arr.tolist()))
    arms = ("U_M2b_mono_conflict", "U_M3_coadapt_knots", "U_M3_decoupled_frozen_knots", "U_M3_decoupled_platt_rerank")
    top_s = {a: np.zeros((len(seeds), n, 2), dtype=np.float64) for a in arms}
    raw_a = np.zeros((len(seeds), n, 5), dtype=np.float64)
    cal_a = np.zeros((len(seeds), n, 5), dtype=np.float64)
    thr_l, beta_l, max_drift = [], [], 0.0

    for si, seed in enumerate(seeds):
        for g in groups:
            tr, te = np.flatnonzero(grp_arr != g), np.flatnonzero(grp_arr == g)
            te_g = torch.from_numpy(te).to(device)
            sc, a_pl, b_pl, thr_pl, beta = inner_decoupled_calibration(
                seed, grp_arr, gd, hist_np, truth_np, ya_np, va_np, dt, tr, bsz, ep_s, lr_s, device,
            )
            thr_l.append(thr_pl); beta_l.append(beta)
            mods = fit_unified_clean_fold(seed, grp_arr, gd, tr, bsz, ep_s, lr_s, device, g_scale=sc)
            m2b, m3_co = mods["U_M2b_mono_conflict"], mods["U_M3_mono_conflict"]
            m3_dec = fit_decoupled_action_head_only(seed, m2b, grp_arr, gd, tr, bsz, ep_s[3], lr_s[3], device)
            with torch.inference_mode():
                o2 = _fwd_u(m2b, gd, te_g)
                oc = _fwd_u(m3_co, gd, te_g)
                od = _fwd_u(m3_dec, gd, te_g)
                drift = float((od["paths"] - o2["paths"]).abs().max().item())
                max_drift = max(max_drift, drift)
                if drift > 1e-12:
                    raise RuntimeError(f"Decoupled M3 drifted from M2b paths by {drift}")
                for key, oo in (("U_M2b_mono_conflict", o2), ("U_M3_coadapt_knots", oc), ("U_M3_decoupled_frozen_knots", od)):
                    pw = oo["paths"].cpu().numpy()
                    d = np.linalg.norm(pw[np.arange(len(te)), oo["logits"].argmax(1).cpu().numpy()] - truth_np[te], axis=-1)
                    top_s[key][si, te] = np.column_stack((d.mean(1), d[:, -1]))
                pw_d = od["paths"].cpu().numpy()
                lg_d = od["logits"].cpu().numpy()
                alg = od["act_logits"].cpu().numpy()
                raw_a[si, te] = 1.0 / (1.0 + np.exp(-np.clip(alg, -30.0, 30.0)))
                p_cal = 1.0 / (1.0 + np.exp(-np.clip(alg * a_pl[None, :] + b_pl[None, :], -30.0, 30.0)))
                cal_a[si, te] = p_cal
                c_act = candidate_implied_actions_np(hist_np[te], pw_d - hist_np[te, -1:, None, :], dt=dt)
                rk = rerank_top1_modes(lg_d, p_cal, c_act, beta)
                dr = np.linalg.norm(pw_d[np.arange(len(te)), rk] - truth_np[te], axis=-1)
                top_s["U_M3_decoupled_platt_rerank"][si, te] = np.column_stack((dr.mean(1), dr[:, -1]))
        print(f"[W20-GPU0 {tag}] Finished seed={seed}", flush=True)

    tm = {a: top_s[a].mean(0) for a in arms}
    ci_fn = (lambda d, m: safe_site_clip_ci(d[m], grp_arr[m], unit_arr[m])) if is_pinns else (lambda d, m: _loc_bootstrap_ci(d[m], grp_arr[m]))
    strata_out = {}
    for sk, sm in strata.items():
        if int(sm.sum()) == 0:
            continue
        gs = sorted(set(grp_arr[sm].tolist()))
        mods_d = {"Last2_CV": {"group_equal_ade1_m": float(np.mean([cv_err[(grp_arr == g) & sm, 0].mean() for g in gs])),
                               "group_equal_fde1_m": float(np.mean([cv_err[(grp_arr == g) & sm, 1].mean() for g in gs]))}}
        for a in arms:
            mods_d[a] = {"group_equal_ade1_m": float(np.mean([tm[a][(grp_arr == g) & sm, 0].mean() for g in gs])),
                         "group_equal_fde1_m": float(np.mean([tm[a][(grp_arr == g) & sm, 1].mean() for g in gs]))}
        paired = {
            "decoupled_minus_coadapt": ci_fn(tm["U_M3_decoupled_frozen_knots"] - tm["U_M3_coadapt_knots"], sm),
            "decoupled_rerank_minus_coadapt": ci_fn(tm["U_M3_decoupled_platt_rerank"] - tm["U_M3_coadapt_knots"], sm),
            "decoupled_rerank_minus_m2b": ci_fn(tm["U_M3_decoupled_platt_rerank"] - tm["U_M2b_mono_conflict"], sm),
            "decoupled_rerank_minus_CV": ci_fn(tm["U_M3_decoupled_platt_rerank"] - cv_err, sm),
        }
        strata_out[sk] = {"n_windows": int(sm.sum()), "models": mods_d, "paired_top1_ci": paired}
    m_thr = np.mean(thr_l, axis=0) if thr_l else np.full(5, 0.5)
    return {
        "n_windows": n, "max_abs_traj_drift_vs_m2b_m": max_drift, "mean_selected_beta": float(np.mean(beta_l)),
        "strata": strata_out,
        "decoupled_action_raw_thr05": score_grouped_actions(ya_np, va_np, raw_a.mean(0), grp_arr, np.full(5, 0.5)),
        "decoupled_action_platt_thr05": score_grouped_actions(ya_np, va_np, cal_a.mean(0), grp_arr, np.full(5, 0.5)),
        "decoupled_action_platt_inner_opt_thr": score_grouped_actions(ya_np, va_np, cal_a.mean(0), grp_arr, m_thr),
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
    d_cvp = np.linalg.norm((hp_np[:, -1:] + ((hp_np[:, -1:] - hp_np[:, -2:-1]) / 0.1) * (np.arange(1, 31)[None, :, None] * 0.1)) - tp_np, axis=-1)
    p_int = build_interaction_strata(dp["edges"].numpy(), dp["is_car"].numpy(), dp["nbr_mask"].numpy(), has_cars=True)
    p_res = run_decoupled_synergy_on_cohort(
        "pinns", dp["site"], dp["clip"], True, gdp, hp_np, tp_np, dp["y_act"].numpy(), dp["valid_act"].numpy(),
        np.column_stack((d_cvp.mean(1), d_cvp[:, -1])),
        {"all_1482": np.ones(np_w, dtype=bool), "first_short_323": dp["strata"]["first_short_323"], "S3_ttc_lt4s": p_int["S3_ttc_lt4s"], "S4_approaching_car": p_int["S4_approaching_car"]},
        0.1, 64, (45, 25, 26, 28), (2e-3, 1.2e-3, 1.5e-3, 1.5e-3), list(args.seeds), dev,
    )

    dj = load_jrdb_10scene_rc2_rc3(args.repo_root)
    gdj = _stage_gpu_dict(dj, dev)
    hj_np, tj_np = dj["history"].numpy(), dj["truth"].numpy()
    d_cvj = np.linalg.norm((hj_np[:, -1:] + ((hj_np[:, 1:2] - hj_np[:, 0:1]) / 0.4) * (np.arange(1, 8)[None, :, None] * 0.4)) - tj_np, axis=-1)
    j_int = build_interaction_strata(dj["edges"].numpy(), dj["is_car"].numpy(), dj["nbr_mask"].numpy(), has_cars=False)
    j_res = run_decoupled_synergy_on_cohort(
        "jrdb", dj["location"], dj["actors"], False, gdj, hj_np, tj_np, dj["y_act"].numpy(), dj["valid_act"].numpy(),
        np.column_stack((d_cvj.mean(1), d_cvj[:, -1])),
        {"all_10scenes": np.ones(len(hj_np), dtype=bool), "S2_close_approaching": j_int["S2_close_approaching"], "S3_ttc_lt4s": j_int["S3_ttc_lt4s"]},
        0.4, 512, (45, 28, 28, 30), (2.5e-3, 1.5e-3, 1.8e-3, 1.5e-3), list(args.seeds), dev,
    )

    summary = {"experiment": "wave20_decoupled_m2b_m3_synergy_5seed", "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
               "seeds": list(args.seeds), "pinns_4site_loso_1482": p_res, "jrdb_9location_lolo_5836": j_res}
    raw = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = (json.dumps({"summary.json": hashlib.sha256(raw).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw); zf.writestr("sha256.json", sh)
    print("Saved:", args.zip_path, flush=True)


if __name__ == "__main__":
    main()
