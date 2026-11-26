"""Wave 14 GPU: Complete PPT Slide 13 O(2) M0-M3 controlled ablation chain + fixed-predictor split-conformal."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import torch

from scripts.benchmark_heldout_sites import load_site_windows
from scripts.ppt_mainline_o2_models import (
    PPT_ARMS,
    PPTMainlineO2Forecaster,
    build_ground_pose_o2_inputs,
    polyline_crossing_rate_np,
)
from scripts.run_emergence_and_action_sensitivity import build_emergence_masks
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_conformal_and_events import conformal_quantile
from scripts.run_pinns_hybrid_action_events_5seed import score_action_matrix
from scripts.run_pinns_matched_o2_multiduration import SITES, paired_site_clip_ci
from scripts.run_pinns_o2_joint_actions_5seed import ACTIONS, SEEDS
from scripts.run_unified_o2_full_gpu_benchmark import load_unified_cohort
from src.data.features import assign_quality_bucket
from src.data.pinns_causal_input import MAP_SHA256
from src.data.pinns_extractor import _check_right_handed_homography
from src.data.scene_graph import build_local_polylines
from src.models.bioeq_o2_joint import o2_joint_loss


def load_ppt_cohort_with_causal_maps(
    meta_dir: Path, pose_npz: Path, joint_zip: Path, causal_map_bin: Path
) -> dict:
    raw_map = causal_map_bin.read_bytes()
    if hashlib.sha256(raw_map).hexdigest() != MAP_SHA256:
        raise ValueError("Causal map archive SHA-256 mismatch")
    base = load_unified_cohort(meta_dir, pose_npz, joint_zip)
    groups, _, _ = load_site_windows(meta_dir)
    rows = [r for s in sorted(groups) for r in groups[s]]
    keys = [(str(r.clip), int(r.person_id), int(r.t0)) for r in rows]
    n = len(rows)

    with np.load(pose_npz, allow_pickle=False) as pz:
        ck = list(zip(pz["clip"].tolist(), pz["person_id"].tolist(), pz["t0"].tolist()))
        ord_map = {k: i for i, k in enumerate(ck)}
        ids = [ord_map[k] for k in keys]
        kpts = torch.from_numpy(pz["hist_kpts"][ids].copy())
        bbox = torch.from_numpy(pz["hist_bbox"][ids].copy())

    h_mats = np.zeros((n, 3, 3), dtype=np.float64)
    y_sums = np.zeros(n, dtype=np.float64)
    for c in sorted({r.clip for r in rows}):
        calib = json.loads((meta_dir / "videos" / c.rsplit("_", 1)[0] / "calib.json").read_text())
        hm = _check_right_handed_homography(calib)
        ys = float(calib["y_min"] + calib["y_max"])
        for i in [ idx for idx, r in enumerate(rows) if r.clip == c ]:
            h_mats[i] = hm
            y_sums[i] = ys
    from src.data.pinns_extractor import _parse_pinns_label_file

    clip_snaps: dict[str, dict] = {}
    for c in sorted({r.clip for r in rows}):
        calib = json.loads((meta_dir / "videos" / c.rsplit("_", 1)[0] / "calib.json").read_text())
        parsed = _parse_pinns_label_file(
            meta_dir / "labels" / f"{c}.txt",
            _check_right_handed_homography(calib),
            calib["y_min"],
            calib["y_max"],
        )
        clip_snaps[c] = {
            t: {int(o["id"]): np.array([o["wx"], o["wy"]], dtype=np.float32) for o in v if o["type"] == "Person"}
            for t, v in parsed.items()
        }

    p_env_arr = np.zeros((n, 16, 10, 11), dtype=np.float32)
    m_map_arr = np.zeros(n, dtype=np.float32)
    with np.load(io.BytesIO(raw_map), allow_pickle=False) as md:
        m_clips = md["clip"].tolist()
        m_pids = md["person_id"].tolist()
        m_t0s = md["t0"].tolist()
        m_lines = md["p_env_image"]
        m_elig = md["source_eligible"]
        by_clip: dict[str, list[tuple[int, int, int]]] = {}
        for j, (mc, mp, mt) in enumerate(zip(m_clips, m_pids, m_t0s)):
            if m_elig[j]:
                by_clip.setdefault(str(mc), []).append((int(mt), int(mp), j))
        for mc in by_clip:
            by_clip[mc].sort()

        for i, r in enumerate(rows):
            cands = [x for x in by_clip.get(r.clip, []) if x[0] <= int(r.t0)]
            if not cands:
                continue
            # Use the most recent strictly causal (<= t0) contour set for this clip
            mt, mp, mj = cands[-1]
            pt0_ref = clip_snaps[r.clip].get(mt, {}).get(mp)
            if pt0_ref is None:
                continue
            polys = []
            lk = m_lines[mj]
            for pk in range(lk.shape[0]):
                valid_pt = lk[pk, :, 10] > 0.5
                if valid_pt.sum() >= 2:
                    w_xy = lk[pk, valid_pt, 0:2] + pt0_ref[None, :]
                    polys.append((w_xy, lk[pk, 0, 2:10]))
            p_env_arr[i], m_map_arr[i] = build_local_polylines(polys, r.xy[4], map_available=True)

    gp = build_ground_pose_o2_inputs(
        base["history"], kpts, bbox, torch.from_numpy(h_mats), torch.from_numpy(y_sums)
    )
    q_np = base["quality"].numpy()
    q_buckets = np.array([assign_quality_bucket(q_np[i]) for i in range(n)], dtype=int)
    emg_masks = build_emergence_masks(meta_dir, base)
    return {
        **base,
        **gp,
        "p_env": torch.from_numpy(p_env_arr),
        "m_map": torch.from_numpy(m_map_arr),
        "q_buckets": q_buckets,
        "strata": {
            **emg_masks,
            "clear_pose": q_buckets == 0,
            "partial_pose": q_buckets == 1,
            "heavy_pose": q_buckets == 2,
        },
    }
def _forward_arm(net: PPTMainlineO2Forecaster, d: dict, ids: torch.Tensor | np.ndarray, device: torch.device) -> dict:
    return net(
        d["history"][ids].to(device),
        d["bio_even"][ids].to(device),
        d["pose_vec_world"][ids].to(device),
        d["signed_knee"][ids].to(device),
        d["quality"][ids].to(device),
        d["m_pose"][ids].to(device),
        d["p_env"][ids].to(device),
        d["m_map"][ids].to(device),
        d["edges"][ids].to(device),
        d["is_car"][ids].to(device),
        d["nbr_mask"][ids].to(device),
    )


def _copy_matching(dst: PPTMainlineO2Forecaster, src: PPTMainlineO2Forecaster) -> None:
    sd = dst.state_dict()
    for k, v in src.state_dict().items():
        if k in sd and sd[k].shape == v.shape:
            sd[k].copy_(v)
    dst.load_state_dict(sd)


def fit_ppt_chain_fold(seed: int, data: dict, tr_ids: np.ndarray, device: torch.device) -> dict[str, PPTMainlineO2Forecaster]:
    w_tr = make_site_weights(data["site"], tr_ids).to(device)
    truth_tr = data["truth"][tr_ids].to(device)
    ya_tr, va_tr = data["y_act"][tr_ids].to(device), data["valid_act"][tr_ids].to(device)
    pos = (ya_tr & va_tr).sum(0).float().clamp_min(1.0)
    neg = ((~ya_tr) & va_tr).sum(0).float().clamp_min(1.0)
    pw = (neg / pos).clamp(1.0, 12.0).to(device)
    tr_t = torch.from_numpy(tr_ids)

    def _run_epochs(net: PPTMainlineO2Forecaster, prefixes: tuple[str, ...], epochs: int, lr: float, wd: float, act_w: float, s_off: int) -> PPTMainlineO2Forecaster:
        for nm, p in net.named_parameters():
            p.requires_grad = nm.startswith(prefixes) or nm in prefixes
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(seed + s_off)
        net.train()
        for _ in range(epochs):
            for ch in torch.randperm(len(tr_ids), generator=gen).split(64):
                out = _forward_arm(net, data, tr_t[ch], device)
                loss = o2_joint_loss(out, truth_tr[ch], w_tr[ch], ya_tr[ch] if act_w > 0 else None, va_tr[ch] if act_w > 0 else None, pw if act_w > 0 else None, act_weight=act_w)
                if loss is not None:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return net.eval()

    torch.manual_seed(seed)
    m0 = _run_epochs(PPTMainlineO2Forecaster("M0").to(device), ("motion.", "ctx_proj.", "cv_low_bonus", "central_", "pair_"), 45, 2e-3, 1e-4, 0.0, 0)
    models: dict[str, PPTMainlineO2Forecaster] = {"M0": m0}

    for i, arm in enumerate(("M1a", "M1b", "M1")):
        torch.manual_seed(seed)
        net = PPTMainlineO2Forecaster(arm).to(device)
        _copy_matching(net, m0)
        models[arm] = _run_epochs(net, ("pose_", "gate_", "central_", "pair_", "cv_low_bonus"), 25, 1.2e-3, 3e-4, 0.0, 10 + i)

    for i, (arm, pref) in enumerate((
        ("M2a", ("poly_",)),
        ("M2b_no_vrad", ("edge_", "graph_proj", "null_logit")),
        ("M2b", ("edge_", "graph_proj", "null_logit")),
        ("M2", ("poly_", "edge_", "graph_proj", "null_logit")),
    )):
        torch.manual_seed(seed)
        net = PPTMainlineO2Forecaster(arm).to(device)
        _copy_matching(net, models["M1"])
        models[arm] = _run_epochs(net, pref, 26, 1.5e-3, 3e-4, 0.0, 20 + i)

    for i, arm in enumerate(("M3a", "M3b", "M3")):
        torch.manual_seed(seed)
        net = PPTMainlineO2Forecaster(arm).to(device)
        _copy_matching(net, models["M2"])
        models[arm] = _run_epochs(net, ("ctx_proj.", "act_", "bridge_", "central_", "pair_", "cv_low_bonus"), 28, 1.2e-3, 2e-4, 0.25, 30 + i)
    return models
@torch.inference_mode()
def predict_arm_numpy(net: PPTMainlineO2Forecaster, data: dict, ids: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    out = _forward_arm(net, data, torch.from_numpy(ids), device)
    paths = out["paths"].cpu().numpy()
    probs = out["logits"].softmax(dim=1).cpu().numpy()
    act_p = torch.sigmoid(out["act_logits"]).cpu().numpy()
    gate = out["gate"].cpu().numpy()
    truth = data["truth"][ids].numpy()
    dst = np.linalg.norm(paths - truth[:, None], axis=-1)
    ade_k, fde_k = dst.mean(axis=2), dst[:, :, -1]
    top = probs.argmax(axis=1)
    oracle = ade_k.argmin(axis=1)
    ii = np.arange(len(ids))
    top_paths = paths[ii, top]
    bayes_paths = np.einsum("nk,nktw->ntw", probs, paths)
    bd = np.linalg.norm(bayes_paths - truth, axis=-1)
    nll = -np.log(np.clip(probs[ii, oracle], 1e-12, 1.0))
    brier = ((probs - np.eye(6)[oracle]) ** 2).sum(axis=1)
    cross = np.array([
        polyline_crossing_rate_np(top_paths[j : j + 1], data["p_env"][ids[j : j + 1]].numpy())
        for j in range(len(ids))
    ], dtype=np.float64)
    # Columns: [ade1, fde1, minade6, minfde6, bayes_ade, bayes_fde, nll, brier, cross, gate]
    metrics = np.column_stack((
        ade_k[ii, top], fde_k[ii, top], ade_k.min(1), fde_k.min(1),
        bd.mean(1), bd[:, -1], nll, brier, cross, gate,
    ))
    tubes = np.column_stack((dst[ii, top].max(axis=1), dst.max(axis=2).min(axis=1), dst[:, :, -1].min(axis=1)))
    return {"metrics": metrics, "act_probs": act_p, "tubes": tubes}


def fit_single_m3(seed: int, data: dict, tr_ids: np.ndarray, device: torch.device) -> PPTMainlineO2Forecaster:
    w_tr = make_site_weights(data["site"], tr_ids).to(device)
    truth_tr = data["truth"][tr_ids].to(device)
    ya_tr, va_tr = data["y_act"][tr_ids].to(device), data["valid_act"][tr_ids].to(device)
    pos = (ya_tr & va_tr).sum(0).float().clamp_min(1.0)
    neg = ((~ya_tr) & va_tr).sum(0).float().clamp_min(1.0)
    pw = (neg / pos).clamp(1.0, 12.0).to(device)
    tr_t = torch.from_numpy(tr_ids)

    def _run(net: PPTMainlineO2Forecaster, prefs: tuple[str, ...], ep: int, lr: float, wd: float, aw: float, soff: int) -> PPTMainlineO2Forecaster:
        for nm, p in net.named_parameters():
            p.requires_grad = nm.startswith(prefs) or nm in prefs
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(seed + soff)
        net.train()
        for _ in range(ep):
            for ch in torch.randperm(len(tr_ids), generator=gen).split(64):
                loss = o2_joint_loss(_forward_arm(net, data, tr_t[ch], device), truth_tr[ch], w_tr[ch], ya_tr[ch] if aw > 0 else None, va_tr[ch] if aw > 0 else None, pw if aw > 0 else None, act_weight=aw)
                if loss is not None:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return net.eval()

    torch.manual_seed(seed)
    m0 = _run(PPTMainlineO2Forecaster("M0").to(device), ("motion.", "ctx_proj.", "cv_low_bonus", "central_", "pair_"), 45, 2e-3, 1e-4, 0.0, 0)
    torch.manual_seed(seed); m1 = PPTMainlineO2Forecaster("M1").to(device); _copy_matching(m1, m0)
    m1 = _run(m1, ("pose_", "gate_", "central_", "pair_", "cv_low_bonus"), 25, 1.2e-3, 3e-4, 0.0, 12)
    torch.manual_seed(seed); m2 = PPTMainlineO2Forecaster("M2").to(device); _copy_matching(m2, m1)
    m2 = _run(m2, ("poly_", "edge_", "graph_proj", "null_logit"), 26, 1.5e-3, 3e-4, 0.0, 23)
    torch.manual_seed(seed); m3 = PPTMainlineO2Forecaster("M3").to(device); _copy_matching(m3, m2)
    return _run(m3, ("ctx_proj.", "act_", "bridge_", "central_", "pair_", "cv_low_bonus"), 28, 1.2e-3, 2e-4, 0.25, 32)


def evaluate_fixed_predictor_conformal(
    oof_tubes_5seed: np.ndarray,
    cv_tube: np.ndarray,
    data: dict,
    seeds: list[int],
    device: torch.device,
) -> dict:
    """Audit-verified split-conformal WITHOUT retraining: (1) within-site clip split, (2) 12-pair fixed 2-site."""
    site_arr, clip_arr = data["site"], data["clip"]
    # Tube array averaged or per-seed: evaluate per-seed and average across seeds
    rng = np.random.default_rng(20261126)
    within = {lvl: {"cv_cov": [], "top1_cov": [], "u6_cov": [], "u6_rad": [], "cv_rad": []} for lvl in ("90", "95")}
    for s_idx in range(len(seeds)):
        tb = oof_tubes_5seed[s_idx]  # (N, 3): [top1_tube, u6_tube, u6_fde]
        for s in SITES:
            s_mask = site_arr == s
            u_clips = sorted(set(clip_arr[s_mask].tolist()))
            if len(u_clips) < 4:
                continue
            n_cal = len(u_clips) // 2
            for _ in range(100):
                perm = rng.permutation(len(u_clips))
                cal_set = {u_clips[k] for k in perm[:n_cal]}
                c_ids = np.flatnonzero(s_mask & np.isin(clip_arr, list(cal_set)))
                t_ids = np.flatnonzero(s_mask & (~np.isin(clip_arr, list(cal_set))))
                if len(c_ids) < 2 or len(t_ids) < 2:
                    continue
                for lvl, alpha in (("90", 0.10), ("95", 0.05)):
                    r_cv = conformal_quantile(cv_tube[c_ids], alpha)
                    r_t1 = conformal_quantile(tb[c_ids, 0], alpha)
                    r_u6 = conformal_quantile(tb[c_ids, 1], alpha)
                    within[lvl]["cv_cov"].append(float((cv_tube[t_ids] <= r_cv).mean()))
                    within[lvl]["top1_cov"].append(float((tb[t_ids, 0] <= r_t1).mean()))
                    within[lvl]["u6_cov"].append(float((tb[t_ids, 1] <= r_u6).mean()))
                    within[lvl]["u6_rad"].append(r_u6)
                    within[lvl]["cv_rad"].append(r_cv)

    # Protocol B: Fixed 2-site predictor calibrated on site C, tested on unseen site D (ZERO retraining!)
    cross12 = {lvl: {"cv_cov": [], "top1_cov": [], "u6_cov": [], "u6_fde_cov": [], "u6_rad": [], "cv_rad": []} for lvl in ("90", "95")}
    pairs_2site = [(SITES[i], SITES[j]) for i in range(4) for j in range(i + 1, 4)]
    for seed in seeds:
        for sa, sb in pairs_2site:
            fit_ids = np.flatnonzero((site_arr == sa) | (site_arr == sb))
            rem = [s for s in SITES if s not in (sa, sb)]
            net2 = fit_single_m3(seed, data, fit_ids, device)
            p_rem = {s: predict_arm_numpy(net2, data, np.flatnonzero(site_arr == s), device)["tubes"] for s in rem}
            cv_rem = {s: cv_tube[site_arr == s] for s in rem}
            for sc, sd in ((rem[0], rem[1]), (rem[1], rem[0])):
                tc, td = p_rem[sc], p_rem[sd]
                for lvl, alpha in (("90", 0.10), ("95", 0.05)):
                    r_cv = conformal_quantile(cv_rem[sc], alpha)
                    r_t1 = conformal_quantile(tc[:, 0], alpha)
                    r_u6 = conformal_quantile(tc[:, 1], alpha)
                    r_f6 = conformal_quantile(tc[:, 2], alpha)
                    cross12[lvl]["cv_cov"].append(float((cv_rem[sd] <= r_cv).mean()))
                    cross12[lvl]["top1_cov"].append(float((td[:, 0] <= r_t1).mean()))
                    cross12[lvl]["u6_cov"].append(float((td[:, 1] <= r_u6).mean()))
                    cross12[lvl]["u6_fde_cov"].append(float((td[:, 2] <= r_f6).mean()))
                    cross12[lvl]["u6_rad"].append(r_u6)
                    cross12[lvl]["cv_rad"].append(r_cv)
    return {
        "within_site_clip_split_fixed_3site_predictor": {
            lvl: {k: float(np.mean(v)) for k, v in d.items()} for lvl, d in within.items()
        },
        "cross_site_12pair_fixed_2site_predictor_no_retrain": {
            lvl: {k: float(np.mean(v)) for k, v in d.items()} for lvl, d in cross12.items()
        },
    }
def _agg_stratum(arr: np.ndarray, mask: np.ndarray, site_arr: np.ndarray) -> dict:
    sub, s_sub = arr[mask], site_arr[mask]
    cols = ("ade1_m", "fde1_m", "min_ade6_m", "min_fde6_m", "bayes_ade_m", "bayes_fde_m", "nll", "brier", "polyline_crossing_rate", "mean_gate")
    by_s = {}
    for s in SITES:
        sm = s_sub == s
        if sm.any():
            by_s[s] = {c: float(sub[sm, j].mean()) for j, c in enumerate(cols)}
    return {
        "n_windows": int(mask.sum()),
        "site_equal": {c: float(np.mean([v[c] for v in by_s.values()])) for c in cols},
        "pooled": {c: float(sub[:, j].mean()) for j, c in enumerate(cols)},
        "by_site": by_s,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("meta_dir", type=Path)
    ap.add_argument("pose_npz", type=Path)
    ap.add_argument("joint_zip", type=Path)
    ap.add_argument("causal_map_bin", type=Path)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_ppt_cohort_with_causal_maps(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin)
    n = len(data["site"])
    site_arr, clip_arr = data["site"], data["clip"]
    truth_np = data["truth"].numpy()
    hist_np = data["history"].numpy()

    cv_paths = hist_np[:, -1:] + (hist_np[:, -1:] - hist_np[:, -2:-1]) * np.arange(1, 31, dtype=np.float64)[None, :, None]
    d_cv = np.linalg.norm(cv_paths - truth_np, axis=-1)
    cv_cross = np.array([polyline_crossing_rate_np(cv_paths[j : j + 1], data["p_env"][j : j + 1].numpy()) for j in range(n)])
    cv_metrics = np.column_stack((d_cv.mean(1), d_cv[:, -1], d_cv.mean(1), d_cv[:, -1], d_cv.mean(1), d_cv[:, -1], np.zeros(n), np.zeros(n), cv_cross, np.zeros(n)))
    cv_tube = d_cv.max(axis=1)

    arm_acc = {a: np.zeros((len(args.seeds), n, 10), dtype=np.float64) for a in PPT_ARMS}
    act_acc = {a: np.zeros((len(args.seeds), n, 5), dtype=np.float64) for a in ("M3a", "M3b", "M3")}
    m3_tubes = np.zeros((len(args.seeds), n, 3), dtype=np.float64)

    for si, seed in enumerate(args.seeds):
        for s in SITES:
            tr_ids = np.flatnonzero(site_arr != s)
            te_ids = np.flatnonzero(site_arr == s)
            models = fit_ppt_chain_fold(seed, data, tr_ids, device)
            for arm in PPT_ARMS:
                pr = predict_arm_numpy(models[arm], data, te_ids, device)
                arm_acc[arm][si, te_ids] = pr["metrics"]
                if arm in act_acc:
                    act_acc[arm][si, te_ids] = pr["act_probs"]
                if arm == "M3":
                    m3_tubes[si, te_ids] = pr["tubes"]
        print(f"Finished PPT M0-M3 LOSO chain seed={seed}", flush=True)

    mean_m = {"CV": cv_metrics, **{a: arm_acc[a].mean(axis=0) for a in PPT_ARMS}}
    ya_np, va_np = data["y_act"].numpy(), data["valid_act"].numpy()
    th_fixed = np.full(5, 0.5, dtype=np.float64)
    act_summary = {
        a: score_action_matrix(ya_np, va_np, act_acc[a].mean(axis=0), site_arr, th_fixed)
        for a in ("M3a", "M3b", "M3")
    }

    contrasts = (
        ("M0_minus_CV", "M0", "CV"), ("M1_minus_CV", "M1", "CV"), ("M3_minus_CV", "M3", "CV"),
        ("M1a_minus_M0", "M1a", "M0"), ("M1b_minus_M1a", "M1b", "M1a"),
        ("M1_minus_M1b", "M1", "M1b"), ("M1_minus_M0", "M1", "M0"),
        ("M2a_minus_M1", "M2a", "M1"), ("M2b_minus_M1", "M2b", "M1"),
        ("M2b_minus_M2b_no_vrad", "M2b", "M2b_no_vrad"), ("M2_minus_M1", "M2", "M1"),
        ("M3a_minus_M2", "M3a", "M2"), ("M3b_minus_M2", "M3b", "M2"),
        ("M3_minus_M3b", "M3", "M3b"), ("M3_minus_M2", "M3", "M2"),
    )
    by_stratum = {}
    for st_name, mask in data["strata"].items():
        by_stratum[st_name] = {
            "models": {m: _agg_stratum(arr, mask, site_arr) for m, arr in mean_m.items()},
            "paired_top1_clip_ci": {
                cname: paired_site_clip_ci(mean_m[ma][mask, :2] - mean_m[mb][mask, :2], site_arr[mask], clip_arr[mask])
                for cname, ma, mb in contrasts
            },
        }

    conf = evaluate_fixed_predictor_conformal(m3_tubes, cv_tube, data, list(args.seeds), device)
    print("Finished fixed-predictor split-conformal evaluation", flush=True)

    report = {
        "experiment": "ppt_mainline_o2_m0_m3_and_fixed_conformal",
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "n_windows": n,
        "seeds": list(args.seeds),
        "param_counts": {a: sum(p.numel() for p in PPTMainlineO2Forecaster(a).parameters()) for a in PPT_ARMS},
        "map_available_windows": int((data["m_map"].numpy() > 0.5).sum()),
        "nonempty_polyline_windows": int((data["p_env"].numpy()[:, :, :, 10].sum(axis=(1, 2)) > 0).sum()),
        "strata": by_stratum,
        "action_5class_fixed05": act_summary,
        "fixed_predictor_split_conformal": conf,
    }
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = json.dumps({"ppt_mainline_o2_m0_m3_and_fixed_conformal.json": hashlib.sha256(raw).hexdigest()}, indent=2).encode("utf-8")
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ppt_mainline_o2_m0_m3_and_fixed_conformal.json", raw)
        zf.writestr("sha256.json", sh)
    print("Saved:", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
