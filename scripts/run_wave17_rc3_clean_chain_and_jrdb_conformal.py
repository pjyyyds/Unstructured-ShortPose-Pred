"""Wave 17 GPU1: Polylineless PINNS O(2) M0-M3 chain + JRDB 9-location (72-pair) fixed-predictor split-conformal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.ppt_mainline_o2_models import PPTMainlineO2Forecaster
from scripts.run_dair_o2_bioeq_external import SEEDS
from scripts.run_jrdb_rc1_pose_quality_and_initiation_5seed import _loc_bootstrap_ci
from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import (
    _fwd_gd,
    _stage_gpu_dict,
    fit_jrdb_rc2_rc3_fold,
    load_jrdb_10scene_rc2_rc3,
    score_jrdb_actions,
)
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_conformal_and_events import conformal_quantile
from scripts.run_pinns_hybrid_action_events_5seed import score_action_matrix
from scripts.run_pinns_matched_o2_multiduration import SITES, paired_site_clip_ci
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import (
    _agg_stratum,
    _copy_matching,
    load_ppt_cohort_with_causal_maps,
    predict_arm_numpy,
)
from src.models.bioeq_o2_joint import o2_joint_loss

CLEAN_PINNS_ARMS = ("M0", "M1", "M2b_no_vrad", "M2b", "M3a", "M3b", "M3")


def fit_clean_pinns_nopoly_fold(seed: int, data: dict, tr_ids: np.ndarray, device: torch.device) -> dict[str, PPTMainlineO2Forecaster]:
    """Train M0 -> M1 -> M2b(±v_rad) -> M3a/M3b/M3 with m_map=0 (eliminating noisy M2a visual contour interference)."""
    w_tr = make_site_weights(data["site"], tr_ids).to(device)
    truth_tr = data["truth"][tr_ids].to(device)
    ya_tr, va_tr = data["y_act"][tr_ids].to(device), data["valid_act"][tr_ids].to(device)
    pos = (ya_tr & va_tr).sum(0).float().clamp_min(1.0)
    neg = ((~ya_tr) & va_tr).sum(0).float().clamp_min(1.0)
    pw = (neg / pos).clamp(1.0, 12.0).to(device)
    tr_t = torch.from_numpy(tr_ids)

    def _fwd(net: PPTMainlineO2Forecaster, ids: torch.Tensor) -> dict:
        z_map = torch.zeros(len(ids), dtype=torch.float32, device=device)
        z_pe = torch.zeros((len(ids), 16, 10, 11), dtype=torch.float32, device=device)
        return net(
            data["history"][ids].to(device), data["bio_even"][ids].to(device), data["pose_vec_world"][ids].to(device),
            data["signed_knee"][ids].to(device), data["quality"][ids].to(device), data["m_pose"][ids].to(device),
            z_pe, z_map, data["edges"][ids].to(device), data["is_car"][ids].to(device), data["nbr_mask"][ids].to(device),
        )

    def _train(net: PPTMainlineO2Forecaster, prefs: tuple[str, ...], ep: int, lr: float, wd: float, aw: float, soff: int) -> PPTMainlineO2Forecaster:
        for nm, p in net.named_parameters():
            p.requires_grad = nm.startswith(prefs) or nm in prefs
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(seed + soff)
        net.train()
        for _ in range(ep):
            for ch in torch.randperm(len(tr_ids), generator=gen).split(64):
                loss = o2_joint_loss(_fwd(net, tr_t[ch]), truth_tr[ch], w_tr[ch], ya_tr[ch] if aw > 0 else None, va_tr[ch] if aw > 0 else None, pw if aw > 0 else None, act_weight=aw)
                if loss is not None:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return net.eval()

    torch.manual_seed(seed)
    m0 = _train(PPTMainlineO2Forecaster("M0").to(device), ("motion.", "ctx_proj.", "cv_low_bonus", "central_", "pair_"), 45, 2e-3, 1e-4, 0.0, 0)
    torch.manual_seed(seed); m1 = PPTMainlineO2Forecaster("M1").to(device); _copy_matching(m1, m0)
    m1 = _train(m1, ("pose_", "gate_", "central_", "pair_", "cv_low_bonus"), 25, 1.2e-3, 3e-4, 0.0, 10)
    models = {"M0": m0, "M1": m1}
    for i, arm in enumerate(("M2b_no_vrad", "M2b")):
        torch.manual_seed(seed); net = PPTMainlineO2Forecaster(arm).to(device); _copy_matching(net, m1)
        models[arm] = _train(net, ("edge_", "graph_proj", "null_logit"), 26, 1.5e-3, 3e-4, 0.0, 20 + i)
    for i, arm in enumerate(("M3a", "M3b", "M3")):
        torch.manual_seed(seed); net = PPTMainlineO2Forecaster(arm).to(device); _copy_matching(net, models["M2b"])
        models[arm] = _train(net, ("ctx_proj.", "act_", "bridge_", "central_", "pair_", "cv_low_bonus"), 28, 1.2e-3, 2e-4, 0.25, 30 + i)
    return models


@torch.inference_mode()
def _jrdb_tubes_gd(net: PPTMainlineO2Forecaster, gd: dict, truth_np: np.ndarray, ids: np.ndarray, device: torch.device) -> np.ndarray:
    o = _fwd_gd(net, gd, torch.from_numpy(ids).to(device))
    p, q = o["paths"].cpu().numpy(), o["logits"].softmax(dim=1).cpu().numpy()
    dst = np.linalg.norm(p - truth_np[ids][:, None], axis=-1)
    top = q.argmax(1); ii = np.arange(len(ids))
    return np.column_stack((dst[ii, top].max(1), dst.max(2).min(1), dst[:, :, -1].min(1)))


def evaluate_jrdb_fixed_conformal(
    oof_tubes_by_arm: dict[str, np.ndarray], cv_tube: np.ndarray, data_j: dict, gd: dict, seeds: list[int], device: torch.device,
) -> dict:
    """Strict non-retrained split-conformal on JRDB: (1) within-scene temporal split, (2) 72-pair cross-location fixed 7-loc predictor."""
    sc_arr, loc_arr = data_j["scene"], data_j["location"]
    truth_np = data_j["truth"].numpy()
    u_locs = sorted(set(loc_arr.tolist()))
    within = {a: {lvl: {"cov_t1": [], "cov_u6": [], "rad_u6": [], "cov_cv": [], "rad_cv": []} for lvl in ("90", "95")} for a in ("M0", "M1", "M2b", "M3")}
    for s_idx in range(len(seeds)):
        for sc in sorted(set(sc_arr.tolist())):
            ids = np.flatnonzero(sc_arr == sc)
            if len(ids) < 20:
                continue
            mid = len(ids) // 2
            c_ids, t_ids = ids[:mid], ids[mid:]
            for a in within:
                tb = oof_tubes_by_arm[a][s_idx]
                for lvl, alpha in (("90", 0.10), ("95", 0.05)):
                    r_cv = conformal_quantile(cv_tube[c_ids], alpha)
                    r_t1 = conformal_quantile(tb[c_ids, 0], alpha)
                    r_u6 = conformal_quantile(tb[c_ids, 1], alpha)
                    within[a][lvl]["cov_cv"].append(float((cv_tube[t_ids] <= r_cv).mean()))
                    within[a][lvl]["cov_t1"].append(float((tb[t_ids, 0] <= r_t1).mean()))
                    within[a][lvl]["cov_u6"].append(float((tb[t_ids, 1] <= r_u6).mean()))
                    within[a][lvl]["rad_u6"].append(r_u6); within[a][lvl]["rad_cv"].append(r_cv)

    # 36 unordered pairs of held-out locations -> 72 ordered (cal_loc -> test_loc) pairs with fixed 7-location M3 predictor!
    cross72 = {lvl: {"cov_cv": [], "cov_t1": [], "cov_u6": [], "rad_u6": [], "rad_cv": []} for lvl in ("90", "95")}
    loc_pairs = [(u_locs[i], u_locs[j]) for i in range(len(u_locs)) for j in range(i + 1, len(u_locs))]
    for seed in seeds[:3]:  # 3 seeds x 72 ordered pairs = 216 out-of-location evaluations
        for la, lb in loc_pairs:
            tr7 = np.flatnonzero((loc_arr != la) & (loc_arr != lb))
            m3_7 = _fit_single_jrdb_m3(seed, loc_arr, gd, tr7, device)
            ta = _jrdb_tubes_gd(m3_7, gd, truth_np, np.flatnonzero(loc_arr == la), device)
            tb = _jrdb_tubes_gd(m3_7, gd, truth_np, np.flatnonzero(loc_arr == lb), device)
            cva, cvb = cv_tube[loc_arr == la], cv_tube[loc_arr == lb]
            for (tc, td, cc, cd) in ((ta, tb, cva, cvb), (tb, ta, cvb, cva)):
                for lvl, alpha in (("90", 0.10), ("95", 0.05)):
                    r_cv, r_t1, r_u6 = conformal_quantile(cc, alpha), conformal_quantile(tc[:, 0], alpha), conformal_quantile(tc[:, 1], alpha)
                    cross72[lvl]["cov_cv"].append(float((cd <= r_cv).mean()))
                    cross72[lvl]["cov_t1"].append(float((td[:, 0] <= r_t1).mean()))
                    cross72[lvl]["cov_u6"].append(float((td[:, 1] <= r_u6).mean()))
                    cross72[lvl]["rad_u6"].append(r_u6); cross72[lvl]["rad_cv"].append(r_cv)

    return {
        "within_scene_temporal_split_9loc_lolo": {a: {l: {k: float(np.mean(v)) for k, v in d.items()} for l, d in wd.items()} for a, wd in within.items()},
        "cross_location_72pair_fixed_7loc_predictor_no_retrain": {l: {k: float(np.mean(v)) for k, v in d.items()} for l, d in cross72.items()},
    }


def _fit_single_jrdb_m3(seed: int, loc_arr: np.ndarray, gd: dict, tr_ids: np.ndarray, device: torch.device) -> PPTMainlineO2Forecaster:
    w_tr = make_site_weights(loc_arr, tr_ids).to(device)
    tr_t = torch.from_numpy(tr_ids).to(device)
    truth_tr, ya_tr, va_tr = gd["truth"][tr_t], gd["y_act"][tr_t], gd["valid_act"][tr_t]
    pos = (ya_tr & va_tr).sum(0).float().clamp_min(1.0)
    neg = ((~ya_tr) & va_tr).sum(0).float().clamp_min(1.0)
    pw = (neg / pos).clamp(1.0, 12.0)

    def _run(net: PPTMainlineO2Forecaster, prefs: tuple[str, ...], ep: int, lr: float, wd: float, aw: float, soff: int) -> PPTMainlineO2Forecaster:
        for nm, p in net.named_parameters():
            p.requires_grad = nm.startswith(prefs) or nm in prefs
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(seed + soff)
        net.train()
        for _ in range(ep):
            for ch in torch.randperm(len(tr_ids), generator=gen).split(512):
                cg = ch.to(device)
                loss = o2_joint_loss(_fwd_gd(net, gd, tr_t[cg]), truth_tr[cg], w_tr[cg], ya_tr[cg] if aw > 0 else None, va_tr[cg] if aw > 0 else None, pw if aw > 0 else None, act_weight=aw)
                if loss is not None:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return net.eval()

    torch.manual_seed(seed)
    m0 = _run(PPTMainlineO2Forecaster("M0").to(device), ("motion.", "ctx_proj.", "cv_low_bonus", "central_", "pair_"), 45, 2.5e-3, 1e-4, 0.0, 0)
    torch.manual_seed(seed); m1 = PPTMainlineO2Forecaster("M1").to(device); _copy_matching(m1, m0)
    m1 = _run(m1, ("pose_", "gate_", "central_", "pair_", "cv_low_bonus"), 28, 1.5e-3, 3e-4, 0.0, 12)
    torch.manual_seed(seed); m2b = PPTMainlineO2Forecaster("M2b").to(device); _copy_matching(m2b, m1)
    m2b = _run(m2b, ("edge_", "graph_proj", "null_logit"), 28, 1.8e-3, 3e-4, 0.0, 21)
    torch.manual_seed(seed); m3 = PPTMainlineO2Forecaster("M3").to(device); _copy_matching(m3, m2b)
    return _run(m3, ("ctx_proj.", "act_", "bridge_", "central_", "pair_", "cv_low_bonus"), 30, 1.5e-3, 2e-4, 0.25, 32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--output-zip", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = load_ppt_cohort_with_causal_maps(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin)
    dp_nopoly = {**dp, "p_env": torch.zeros_like(dp["p_env"]), "m_map": torch.zeros_like(dp["m_map"])}
    np_w = len(dp["site"])
    h_p, t_p = dp["history"].numpy(), dp["truth"].numpy()
    cv_p = h_p[:, -1:] + (h_p[:, -1:] - h_p[:, -2:-1]) * np.arange(1, 31, dtype=np.float64)[None, :, None]
    d_cvp = np.linalg.norm(cv_p - t_p, axis=-1)
    cv_mp = np.column_stack((d_cvp.mean(1), d_cvp[:, -1], d_cvp.mean(1), d_cvp[:, -1], d_cvp.mean(1), d_cvp[:, -1], np.zeros(np_w), np.zeros(np_w), np.zeros(np_w), np.zeros(np_w)))

    acc_p = {a: np.zeros((len(args.seeds), np_w, 10), dtype=np.float64) for a in CLEAN_PINNS_ARMS}
    act_p = {a: np.zeros((len(args.seeds), np_w, 5), dtype=np.float64) for a in ("M3a", "M3b", "M3")}
    for si, seed in enumerate(args.seeds):
        for s in SITES:
            tr, te = np.flatnonzero(dp["site"] != s), np.flatnonzero(dp["site"] == s)
            mods = fit_clean_pinns_nopoly_fold(seed, dp_nopoly, tr, dev)
            for a in CLEAN_PINNS_ARMS:
                pr = predict_arm_numpy(mods[a], dp_nopoly, te, dev)
                acc_p[a][si, te] = pr["metrics"]
                if a in act_p:
                    act_p[a][si, te] = pr["act_probs"]
        print(f"Finished clean polylineless PINNS M0-M3 seed={seed}", flush=True)

    mm_p = {"CV": cv_mp, **{a: acc_p[a].mean(0) for a in CLEAN_PINNS_ARMS}}
    p_cs = (("M2b_minus_M1", "M2b", "M1"), ("M2b_minus_M2b_no_vrad", "M2b", "M2b_no_vrad"),
            ("M3a_minus_M2b", "M3a", "M2b"), ("M3b_minus_M2b", "M3b", "M2b"), ("M3_minus_M3b", "M3", "M3b"), ("M3_minus_M2b", "M3", "M2b"))
    p_strata = {}
    for sn in ("all_1482", "first_short_323", "clear_pose", "heavy_pose"):
        sm = dp["strata"][sn]
        p_strata[sn] = {
            "models": {m: _agg_stratum(arr, sm, dp["site"]) for m, arr in mm_p.items()},
            "paired_top1_clip_ci": {cn: paired_site_clip_ci(mm_p[ma][sm, :2] - mm_p[mb][sm, :2], dp["site"][sm], dp["clip"][sm]) for cn, ma, mb in p_cs},
        }
    p_act = {a: score_action_matrix(dp["y_act"].numpy(), dp["valid_act"].numpy(), act_p[a].mean(0), dp["site"], np.full(5, 0.5)) for a in ("M3a", "M3b", "M3")}

    dj = load_jrdb_10scene_rc2_rc3(args.repo_root)
    gdj = _stage_gpu_dict(dj, dev)
    nj_w = len(dj["scene"])
    t_j = dj["truth"].numpy(); h_j = dj["history"].numpy()
    cv_j = (h_j[:, 1:] - h_j[:, :1]) * np.arange(1, 8, dtype=np.float64)[None, :, None]
    cv_tb_j = np.linalg.norm(cv_j - t_j, axis=-1).max(1)
    j_locs = sorted(set(dj["location"].tolist()))
    oof_tb_j = {a: np.zeros((len(args.seeds), nj_w, 3), dtype=np.float64) for a in ("M0", "M1", "M2b", "M3")}
    for si, seed in enumerate(args.seeds):
        for l in j_locs:
            tr, te = np.flatnonzero(dj["location"] != l), np.flatnonzero(dj["location"] == l)
            mods_j = fit_jrdb_rc2_rc3_fold(seed, dj["location"], gdj, tr, dev)
            for a in oof_tb_j:
                oof_tb_j[a][si, te] = _jrdb_tubes_gd(mods_j[a], gdj, t_j, te, dev)
    j_conf = evaluate_jrdb_fixed_conformal(oof_tb_j, cv_tb_j, dj, gdj, list(args.seeds), dev)
    print("Finished JRDB 72-pair fixed-predictor split-conformal", flush=True)

    report = {"experiment": "wave17_rc3_clean_chain_and_jrdb_conformal", "seeds": list(args.seeds),
              "pinns_clean_nopoly_m0_m3": p_strata, "pinns_clean_action_5class": p_act, "jrdb_fixed_predictor_split_conformal": j_conf}
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = json.dumps({"wave17_rc3_clean_chain_and_jrdb_conformal.json": hashlib.sha256(raw).hexdigest()}, indent=2).encode("utf-8")
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("wave17_rc3_clean_chain_and_jrdb_conformal.json", raw)
        zf.writestr("sha256.json", sh)
    print("Saved:", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
