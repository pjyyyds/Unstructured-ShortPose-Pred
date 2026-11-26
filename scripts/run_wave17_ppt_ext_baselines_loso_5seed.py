"""Wave 17 GPU2-A (Slide 13): 5-seed cross-site/location Ext1_MCENET, Ext2_HST, Ext3_SocialTransmotion (Raw & Frenet)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.ppt_mainline_o2_models import polyline_crossing_rate_np
from scripts.run_dair_o2_bioeq_external import SEEDS
from scripts.run_jrdb_rc1_pose_quality_and_initiation_5seed import _loc_bootstrap_ci
from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import fast_person_bootstrap_ci, load_jrdb_10scene_rc2_rc3
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_matched_o2_multiduration import SITES, paired_site_clip_ci
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import load_ppt_cohort_with_causal_maps
from src.data.features import build_motion_features, build_pose_and_quality_features
from src.models.baselines import HSTBaseline, MCENETBaseline, SocialTransmotionBaseline
from src.models.bioeq_neural import local_frenet_torch

COCO17_FROM_BLAZE33 = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
EXT_SPECS = (
    ("Ext1_MCENET", "mcenet", False),
    ("Ext2_HST", "hst", False),
    ("Ext3_SocialTransmotion", "st", False),
    ("Ext1_MCENET_Frenet", "mcenet", True),
    ("Ext2_HST_Frenet", "hst", True),
    ("Ext3_SocialTransmotion_Frenet", "st", True),
)


def compute_so2_frenet_basis(history: torch.Tensor) -> torch.Tensor:
    """Return exact SO(2) basis (B, 2, 2) for every window (identity fallback if disp < 0.05m)."""
    _, basis, valid = local_frenet_torch(history.double())
    eye = torch.eye(2, dtype=torch.float64, device=history.device)[None].expand(len(history), -1, -1)
    return torch.where(valid[:, None, None], basis, eye)


def rotate_world_to_local(vec: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    return torch.einsum("b...w,bwc->b...c", vec.double(), rot).to(vec.dtype)


def rotate_local_to_world(vec: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    return torch.einsum("b...c,bwc->b...w", vec.double(), rot).to(vec.dtype)


def build_pinns_ext_tensors(data: dict, pose_npz: Path, meta_dir: Path) -> dict[str, dict[str, torch.Tensor]]:
    """Build scene-axis and SO(2) Frenet-canonicalized input dicts for Ext1..Ext3 on PINNS (1,482)."""
    from scripts.benchmark_heldout_sites import load_site_windows

    groups, _, _ = load_site_windows(meta_dir)
    rows = [r for s in sorted(groups) for r in groups[s]]
    keys = [(str(r.clip), int(r.person_id), int(r.t0)) for r in rows]
    with np.load(pose_npz, allow_pickle=False) as pz:
        ord_map = {k: i for i, k in enumerate(zip(pz["clip"].tolist(), pz["person_id"].tolist(), pz["t0"].tolist()))}
        ids = [ord_map[k] for k in keys]
        kpts_np = pz["hist_kpts"][ids].astype(np.float32)
        bbox_np = pz["hist_bbox"][ids].astype(np.float32)

    hist_np = data["history"].numpy()
    n = len(hist_np)
    pm5 = np.ones(5, dtype=np.float32)
    xm_raw = np.stack([build_motion_features(hist_np[i], pm5, dt=0.1) for i in range(n)])
    xp_raw = np.stack([build_pose_and_quality_features(kpts_np[i], bbox_np[i], pm5)[0] for i in range(n)])

    rot = compute_so2_frenet_basis(data["history"])
    hist_f = rotate_world_to_local(data["history"] - data["history"][:, -1:], rot).numpy()
    xm_fr = np.stack([build_motion_features(hist_f[i], pm5, dt=0.1) for i in range(n)])

    p_env_raw = data["p_env"].clone()
    p_env_fr = p_env_raw.clone()
    pt_m = (p_env_fr[..., 10:11] > 0.5).float()
    p_env_fr[..., 0:2] = rotate_world_to_local(p_env_raw[..., 0:2], rot) * pt_m

    edg_raw = data["edges"].clone()
    edg_fr = edg_raw.clone()
    edg_fr[..., 0:2] = rotate_world_to_local(edg_raw[..., 0:2], rot)
    edg_fr[..., 2:4] = rotate_world_to_local(edg_raw[..., 2:4], rot)

    car = data["is_car"]
    nbr_oh = torch.zeros((n, 12, 8), dtype=torch.float32)
    nbr_oh[:, :, 0] = (1.0 - car) * data["nbr_mask"]
    nbr_oh[:, :, 1] = car * data["nbr_mask"]
    mm_col = data["m_map"].float().unsqueeze(-1)
    return {
        "raw": {
            "x_m": torch.from_numpy(xm_raw), "x_p": torch.from_numpy(xp_raw),
            "p_env": p_env_raw, "m_map": mm_col,
            "nbr_edges": edg_raw, "nbr_mask": data["nbr_mask"].float(), "nbr_type_oh": nbr_oh,
            "fut_xy": data["truth"].float(), "rot": torch.eye(2, dtype=torch.float64)[None].expand(n, -1, -1),
        },
        "frenet": {
            "x_m": torch.from_numpy(xm_fr), "x_p": torch.from_numpy(xp_raw),
            "p_env": p_env_fr, "m_map": mm_col,
            "nbr_edges": edg_fr, "nbr_mask": data["nbr_mask"].float(), "nbr_type_oh": nbr_oh,
            "fut_xy": rotate_world_to_local(data["truth"], rot).float(), "rot": rot,
        },
    }


def build_jrdb_ext_tensors(data: dict) -> dict[str, dict[str, torch.Tensor]]:
    """Build scene-axis and SO(2) Frenet-canonicalized input dicts for Ext1..Ext3 on JRDB (5,836, T=2, Tf=7)."""
    hist_np = data["history"].numpy()
    n = len(hist_np)
    pm2 = np.ones(2, dtype=np.float32)
    xm_raw = np.stack([build_motion_features(hist_np[i], pm2, dt=0.4) for i in range(n)])
    rot = compute_so2_frenet_basis(data["history"])
    hist_f = rotate_world_to_local(data["history"] - data["history"][:, -1:], rot).numpy()
    xm_fr = np.stack([build_motion_features(hist_f[i], pm2, dt=0.4) for i in range(n)])

    j2 = data["joints_3d"][:, [0, 4]][:, :, COCO17_FROM_BLAZE33, :]  # (N, 2, 17, 3) aligned with 2.5Hz steps
    fin = np.isfinite(j2).all(axis=-1)
    xyz = np.where(fin[..., None], j2, 0.0)
    hip_ok = fin[:, :, 11] & fin[:, :, 12]
    root = np.where(hip_ok[:, :, None, None], 0.5 * (xyz[:, :, 11:12] + xyz[:, :, 12:13]), xyz.mean(axis=2, keepdims=True))
    rel_xy = np.where(fin[..., None], (xyz - root)[..., :2], 0.0)
    vel_xy = np.zeros_like(rel_xy)
    both_ok = fin[:, 1] & fin[:, 0]
    vel_xy[:, 1] = np.where(both_ok[..., None], (rel_xy[:, 1] - rel_xy[:, 0]) / 0.4, 0.0)
    mf = fin.astype(np.float32)[..., None]
    xp_raw_t = torch.from_numpy(np.concatenate([rel_xy, vel_xy, mf, mf], axis=-1).astype(np.float32))
    xp_fr_t = xp_raw_t.clone()
    xp_fr_t[..., 0:2] = rotate_world_to_local(xp_raw_t[..., 0:2], rot) * xp_fr_t[..., 5:6]
    xp_fr_t[..., 2:4] = rotate_world_to_local(xp_raw_t[..., 2:4], rot) * xp_fr_t[..., 5:6]

    edg_raw = data["edges"].clone()
    edg_fr = edg_raw.clone()
    edg_fr[..., 0:2] = rotate_world_to_local(edg_raw[..., 0:2], rot)
    edg_fr[..., 2:4] = rotate_world_to_local(edg_raw[..., 2:4], rot)
    nbr_oh = torch.zeros((n, 12, 8), dtype=torch.float32)
    nbr_oh[:, :, 0] = data["nbr_mask"].float()
    z_env = torch.zeros((n, 16, 10, 11), dtype=torch.float32)
    z_map = torch.zeros((n, 1), dtype=torch.float32)
    return {
        "raw": {
            "x_m": torch.from_numpy(xm_raw), "x_p": xp_raw_t, "p_env": z_env, "m_map": z_map,
            "nbr_edges": edg_raw, "nbr_mask": data["nbr_mask"].float(), "nbr_type_oh": nbr_oh,
            "fut_xy": data["truth"].float(), "rot": torch.eye(2, dtype=torch.float64)[None].expand(n, -1, -1),
        },
        "frenet": {
            "x_m": torch.from_numpy(xm_fr), "x_p": xp_fr_t, "p_env": z_env, "m_map": z_map,
            "nbr_edges": edg_fr, "nbr_mask": data["nbr_mask"].float(), "nbr_type_oh": nbr_oh,
            "fut_xy": rotate_world_to_local(data["truth"], rot).float(), "rot": rot,
        },
    }


def _make_ext_model(kind: str, fut_steps: int, dt: float) -> torch.nn.Module:
    if kind == "mcenet":
        return MCENETBaseline(hidden_dim=128, fut_steps=fut_steps, dt=dt)
    if kind == "hst":
        return HSTBaseline(hidden_dim=128, fut_steps=fut_steps, dt=dt)
    if kind == "st":
        return SocialTransmotionBaseline(hidden_dim=128, fut_steps=fut_steps, dt=dt)
    raise ValueError(f"Unknown external baseline kind: {kind}")


def weighted_shared_winner_loss(pred: dict[str, torch.Tensor], fut_xy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Site/location-weighted Eq. (10) shared-winner Smooth-L1 + soft/hard mode CE."""
    import torch.nn.functional as F

    trajs, logits = pred["trajs"], pred["mode_logits"]
    bsz = trajs.shape[0]
    ade_k = torch.linalg.vector_norm(trajs - fut_xy[:, None], dim=-1).mean(dim=-1)
    k_star = ade_k.detach().argmin(dim=-1)
    win = trajs[torch.arange(bsz, device=trajs.device), k_star]
    reg = F.smooth_l1_loss(win, fut_xy, beta=0.50, reduction="none").sum(dim=-1).mean(dim=-1)
    soft_r = torch.softmax(-ade_k.detach() / 0.15, dim=-1)
    ce_h = F.cross_entropy(logits, k_star, label_smoothing=0.04, reduction="none")
    ce_s = -(soft_r * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    tot = reg + 0.25 * (0.65 * ce_h + 0.35 * ce_s)
    return (tot * w).sum() / w.sum().clamp_min(1e-6)


def fit_and_eval_ext_fold(
    seed: int,
    kind: str,
    gpu_pack: dict[str, torch.Tensor],
    truth_world_np: np.ndarray,
    tr_ids: np.ndarray,
    te_ids: np.ndarray,
    w_tr: torch.Tensor,
    fut_steps: int,
    dt: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    net = _make_ext_model(kind, fut_steps=fut_steps, dt=dt).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=2e-4)
    tr_t = torch.from_numpy(tr_ids).to(device)
    gen = torch.Generator().manual_seed(seed)
    b_keys = ("x_m", "x_p", "p_env", "m_map", "nbr_edges", "nbr_mask", "nbr_type_oh")
    net.train()
    for _ in range(epochs):
        for ch in torch.randperm(len(tr_ids), generator=gen).split(batch_size):
            ch_g = ch.to(device)
            ids_g = tr_t[ch_g]
            pred = net({k: gpu_pack[k][ids_g] for k in b_keys})
            loss = weighted_shared_winner_loss(pred, gpu_pack["fut_xy"][ids_g], w_tr[ch_g])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
    net.eval()
    with torch.inference_mode():
        te_t = torch.from_numpy(te_ids).to(device)
        out = net({k: gpu_pack[k][te_t] for k in b_keys})
        tr_w = rotate_local_to_world(out["trajs"].double(), gpu_pack["rot"][te_t]).cpu().numpy()
        probs = out["mode_probs"].cpu().numpy()
    d = np.linalg.norm(tr_w - truth_world_np[te_ids][:, None], axis=-1)
    top = probs.argmax(axis=1)
    ii = np.arange(len(te_ids))
    top_err = np.column_stack((d[ii, top].mean(axis=1), d[ii, top, -1]))
    min6_err = np.column_stack((d.mean(axis=2).min(axis=1), d[:, :, -1].min(axis=1)))
    return top_err, min6_err, tr_w[ii, top]


def _agg_group_top_min(top_5s: np.ndarray, min_5s: np.ndarray, grp: np.ndarray, mask: np.ndarray) -> dict:
    tm, mm, gs = top_5s.mean(0)[mask], min_5s.mean(0)[mask], grp[mask]
    by_g = {}
    for g in sorted(set(gs.tolist())):
        gm = gs == g
        by_g[str(g)] = {"ade1_m": float(tm[gm, 0].mean()), "fde1_m": float(tm[gm, 1].mean()), "min_ade6_m": float(mm[gm, 0].mean()), "min_fde6_m": float(mm[gm, 1].mean())}
    keys = ("ade1_m", "fde1_m", "min_ade6_m", "min_fde6_m")
    return {
        "n_windows": int(mask.sum()),
        "group_equal": {k: float(np.mean([v[k] for v in by_g.values()])) for k in keys},
        "window_equal": {"ade1_m": float(tm[:, 0].mean()), "fde1_m": float(tm[:, 1].mean()), "min_ade6_m": float(mm[:, 0].mean()), "min_fde6_m": float(mm[:, 1].mean())},
        "seed_std_ade1_m": float(top_5s[:, mask, 0].mean(1).std(ddof=1)) if len(top_5s) > 1 else 0.0,
    }


def run_pinns_ext_loso(meta_dir: Path, pose_npz: Path, joint_zip: Path, causal_map_bin: Path, device: torch.device) -> tuple[dict, dict[str, np.ndarray]]:
    from scripts.run_wave17_rc3_clean_chain_and_jrdb_conformal import fit_clean_pinns_nopoly_fold

    data = load_ppt_cohort_with_causal_maps(meta_dir, pose_npz, joint_zip, causal_map_bin)
    packs_cpu = build_pinns_ext_tensors(data, pose_npz, meta_dir)
    packs = {m: {k: v.to(device) for k, v in d.items()} for m, d in packs_cpu.items()}
    n = len(data["history"])
    truth_np = data["truth"].numpy()
    p_env_np = data["p_env"].numpy()
    m_map_ok = data["m_map"].numpy() > 0.5
    t_s = np.arange(1, 31, dtype=np.float64)[None, :, None] * 0.1
    cv_p = ((data["history"][:, -1:] - data["history"][:, -2:-1]).numpy() / 0.1) * t_s
    cv_d = np.linalg.norm(cv_p - truth_np, axis=-1)
    cv_err = np.column_stack((cv_d.mean(1), cv_d[:, -1]))

    names = [nm for nm, _, _ in EXT_SPECS] + ["O2_M2b_ref", "O2_M3_ref"]
    top_s = {nm: np.zeros((len(SEEDS), n, 2)) for nm in names}
    min_s = {nm: np.zeros((len(SEEDS), n, 2)) for nm in names}
    cross_s = {nm: np.zeros((len(SEEDS), 2)) for nm in names}

    for si, seed in enumerate(SEEDS):
        paths_seed = {nm: np.zeros((n, 30, 2)) for nm in names}
        for site in SITES:
            tr_ids = np.flatnonzero(data["site"] != site)
            te_ids = np.flatnonzero(data["site"] == site)
            w_tr = make_site_weights(data["site"], tr_ids).to(device)
            for nm, kind, use_fr in EXT_SPECS:
                gp = packs["frenet" if use_fr else "raw"]
                te_top, te_min, p_top = fit_and_eval_ext_fold(
                    seed, kind, gp, truth_np, tr_ids, te_ids, w_tr, fut_steps=30, dt=0.1, epochs=32, batch_size=64, device=device
                )
                top_s[nm][si, te_ids], min_s[nm][si, te_ids], paths_seed[nm][te_ids] = te_top, te_min, p_top
            o2_mods = fit_clean_pinns_nopoly_fold(seed, data, tr_ids, device)
            with torch.inference_mode():
                id_t = torch.from_numpy(te_ids)
                zm = torch.zeros(len(te_ids), dtype=torch.float32, device=device)
                zp = torch.zeros((len(te_ids), 16, 10, 11), dtype=torch.float32, device=device)
                for r_nm, arm in (("O2_M2b_ref", "M2b"), ("O2_M3_ref", "M3")):
                    oo = o2_mods[arm](
                        data["history"][id_t].to(device), data["bio_even"][id_t].to(device), data["pose_vec_world"][id_t].to(device),
                        data["signed_knee"][id_t].to(device), data["quality"][id_t].to(device), data["m_pose"][id_t].to(device),
                        zp, zm, data["edges"][id_t].to(device), data["is_car"][id_t].to(device), data["nbr_mask"][id_t].to(device),
                    )
                    pp, qq = oo["paths"].cpu().numpy(), oo["logits"].softmax(1).cpu().numpy()
                    dd = np.linalg.norm(pp - truth_np[te_ids][:, None], axis=-1)
                    tp, ii = qq.argmax(1), np.arange(len(te_ids))
                    top_s[r_nm][si, te_ids] = np.column_stack((dd[ii, tp].mean(1), dd[ii, tp, -1]))
                    min_s[r_nm][si, te_ids] = np.column_stack((dd.mean(2).min(1), dd[:, :, -1].min(1)))
                    paths_seed[r_nm][te_ids] = pp[ii, tp]
        for nm in names:
            cross_s[nm][si] = [polyline_crossing_rate_np(paths_seed[nm], p_env_np), polyline_crossing_rate_np(paths_seed[nm][m_map_ok], p_env_np[m_map_ok])]
        print(f"[W17-EXT PINNS] seed={seed} MCENET={top_s['Ext1_MCENET'][si].mean(0)[0]:.4f} MCENET_Fr={top_s['Ext1_MCENET_Frenet'][si].mean(0)[0]:.4f} O2_M2b={top_s['O2_M2b_ref'][si].mean(0)[0]:.4f}", flush=True)

    tm = {nm: v.mean(0) for nm, v in top_s.items()}
    mm = {nm: v.mean(0) for nm, v in min_s.items()}
    strata_out = {}
    for s_nm, msk in data["strata"].items():
        strata_out[s_nm] = {"n_windows": int(msk.sum()), "Last2_CV": {"ade_m": float(cv_err[msk, 0].mean()), "fde_m": float(cv_err[msk, 1].mean())}}
        for nm in names:
            strata_out[s_nm][nm] = _agg_group_top_min(top_s[nm], min_s[nm], data["site"], msk)
    comps = {}
    for nm, _, _ in EXT_SPECS:
        comps[f"O2_M2b_minus_{nm}_all"] = paired_site_clip_ci(tm["O2_M2b_ref"][:, 0] - tm[nm][:, 0], tm["O2_M2b_ref"][:, 1] - tm[nm][:, 1], data["site"], data["clip"])
        comps[f"{nm}_minus_CV_all"] = paired_site_clip_ci(tm[nm][:, 0] - cv_err[:, 0], tm[nm][:, 1] - cv_err[:, 1], data["site"], data["clip"])
    for b_nm in ("Ext1_MCENET", "Ext2_HST", "Ext3_SocialTransmotion"):
        comps[f"{b_nm}_Frenet_minus_Raw"] = paired_site_clip_ci(tm[f"{b_nm}_Frenet"][:, 0] - tm[b_nm][:, 0], tm[f"{b_nm}_Frenet"][:, 1] - tm[b_nm][:, 1], data["site"], data["clip"])
    summary = {
        "n_windows": n,
        "param_counts": {nm: int(sum(p.numel() for p in _make_ext_model(kind, 30, 0.1).parameters())) for nm, kind, _ in EXT_SPECS[:3]},
        "polyline_crossing_rate_mean": {nm: {"all_1482": float(cross_s[nm][:, 0].mean()), "map_eligible_835": float(cross_s[nm][:, 1].mean())} for nm in names},
        "strata": strata_out,
        "paired_comparisons": comps,
    }
    npz = {f"pinns_{nm}_top1": tm[nm] for nm in names} | {f"pinns_{nm}_min6": mm[nm] for nm in names}
    return summary, npz


def run_jrdb_ext_lolo(repo_root: Path, device: torch.device) -> tuple[dict, dict[str, np.ndarray]]:
    from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import _eval_gd, _stage_gpu_dict, fit_jrdb_rc2_rc3_fold

    data = load_jrdb_10scene_rc2_rc3(repo_root)
    packs_cpu = build_jrdb_ext_tensors(data)
    packs = {m: {k: v.to(device) for k, v in d.items()} for m, d in packs_cpu.items()}
    gd = _stage_gpu_dict(data, device)
    n = len(data["history"])
    truth_np = data["truth"].numpy()
    locs = sorted(set(data["location"].tolist()))
    t_s = np.arange(1, 8, dtype=np.float64)[None, :, None] * 0.4
    cv_p = ((data["history"][:, 1:2] - data["history"][:, 0:1]).numpy() / 0.4) * t_s
    cv_d = np.linalg.norm(cv_p - truth_np, axis=-1)
    cv_err = np.column_stack((cv_d.mean(1), cv_d[:, -1]))

    names = [nm for nm, _, _ in EXT_SPECS] + ["O2_M2b_ref", "O2_M3a_ref"]
    top_s = {nm: np.zeros((len(SEEDS), n, 2)) for nm in names}
    min_s = {nm: np.zeros((len(SEEDS), n, 2)) for nm in names}

    for si, seed in enumerate(SEEDS):
        for loc in locs:
            tr_ids = np.flatnonzero(data["location"] != loc)
            te_ids = np.flatnonzero(data["location"] == loc)
            w_tr = make_site_weights(data["location"], tr_ids).to(device)
            for nm, kind, use_fr in EXT_SPECS:
                gp = packs["frenet" if use_fr else "raw"]
                te_top, te_min, _ = fit_and_eval_ext_fold(
                    seed, kind, gp, truth_np, tr_ids, te_ids, w_tr, fut_steps=7, dt=0.4, epochs=24, batch_size=128, device=device
                )
                top_s[nm][si, te_ids], min_s[nm][si, te_ids] = te_top, te_min
            o2_mods = fit_jrdb_rc2_rc3_fold(seed, data["location"], gd, tr_ids, device)
            for r_nm, arm in (("O2_M2b_ref", "M2b"), ("O2_M3a_ref", "M3a")):
                top_s[r_nm][si, te_ids], min_s[r_nm][si, te_ids] = _eval_gd(o2_mods[arm], gd, truth_np, te_ids, device)
        print(f"[W17-EXT JRDB] seed={seed} MCENET={top_s['Ext1_MCENET'][si].mean(0)[0]:.4f} ST_Fr={top_s['Ext3_SocialTransmotion_Frenet'][si].mean(0)[0]:.4f} O2_M3a={top_s['O2_M3a_ref'][si].mean(0)[0]:.4f}", flush=True)

    tm = {nm: v.mean(0) for nm, v in top_s.items()}
    mm = {nm: v.mean(0) for nm, v in min_s.items()}
    strata_out = {}
    for s_nm, msk in data["strata"].items():
        strata_out[s_nm] = {"n_windows": int(msk.sum()), "Last2_CV": {"ade_m": float(cv_err[msk, 0].mean()), "fde_m": float(cv_err[msk, 1].mean())}}
        for nm in names:
            strata_out[s_nm][nm] = _agg_group_top_min(top_s[nm], min_s[nm], data["location"], msk)
    all_m = data["strata"]["all_10scenes"]
    comps = {}
    for nm, _, _ in EXT_SPECS:
        d_m3 = tm["O2_M3a_ref"] - tm[nm]
        d_cv = tm[nm] - cv_err
        comps[f"O2_M3a_minus_{nm}_all"] = {"location_equal": _loc_bootstrap_ci(d_m3, all_m, data["location"]), **fast_person_bootstrap_ci(d_m3, all_m, data["actors"])}
        comps[f"{nm}_minus_CV_all"] = {"location_equal": _loc_bootstrap_ci(d_cv, all_m, data["location"]), **fast_person_bootstrap_ci(d_cv, all_m, data["actors"])}
    for b_nm in ("Ext1_MCENET", "Ext2_HST", "Ext3_SocialTransmotion"):
        df = tm[f"{b_nm}_Frenet"] - tm[b_nm]
        comps[f"{b_nm}_Frenet_minus_Raw"] = {"location_equal": _loc_bootstrap_ci(df, all_m, data["location"]), **fast_person_bootstrap_ci(df, all_m, data["actors"])}
    npz = {f"jrdb_{nm}_top1": tm[nm] for nm in names} | {f"jrdb_{nm}_min6": mm[nm] for nm in names}
    return {"n_windows": n, "n_locations": len(locs), "strata": strata_out, "paired_comparisons": comps}, npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pinns_res, pinns_npz = run_pinns_ext_loso(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin, device)
    jrdb_res, jrdb_npz = run_jrdb_ext_lolo(args.repo_root, device)
    summary = {
        "experiment": "wave17_ppt_slide13_ext_baselines_loso_5seed",
        "device": str(device),
        "seeds": list(SEEDS),
        "note": "Unified-head architectural proxies Ext1_MCENET, Ext2_HST, Ext3_SocialTransmotion evaluated in both scene-axis Raw and exact SO(2) Frenet-canonicalized frames under strict 5-seed cross-site/location splits.",
        "pinns_4site_loso_1482": pinns_res,
        "jrdb_9location_lolo_5836": jrdb_res,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(args.out_dir / "per_window_ext_5seed.npz", **pinns_npz, **jrdb_npz)
    dig = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out_dir.iterdir()) if p.is_file()}
    (args.out_dir / "sha256.json").write_text(json.dumps(dig, indent=2), encoding="utf-8")
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(args.out_dir.iterdir()):
            zf.write(p, arcname=p.name)
    print(f"DONE: {args.zip_path} sha256={hashlib.sha256(args.zip_path.read_bytes()).hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
