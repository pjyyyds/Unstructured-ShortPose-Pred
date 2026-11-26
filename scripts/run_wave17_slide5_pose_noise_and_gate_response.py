"""Wave 17 GPU2-B (Slides 5, 9, 17): Social-Pose Clean/VarQ train x controlled pose-degradation test & gate response."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.ppt_mainline_o2_models import PPTMainlineO2Forecaster, build_ground_pose_o2_inputs, fit_pinns_ground_homography
from scripts.run_dair_o2_bioeq_external import SEEDS
from scripts.run_jrdb_rc1_pose_quality_and_initiation_5seed import _loc_bootstrap_ci, build_jrdb_3d_pose_o2_features
from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import fast_person_bootstrap_ci, load_jrdb_10scene_rc2_rc3
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_matched_o2_multiduration import SITES, paired_site_clip_ci
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import _copy_matching, load_ppt_cohort_with_causal_maps
from src.data.features import build_pose_and_quality_features
from src.models.bioeq_o2_joint import o2_joint_loss

ARMS = ("M0", "M1a", "M1b", "M1")
TEST_CONDS = ("Test_Natural", "Test_Jitter020", "Test_LowerOcc", "Test_FullDrop")


def _spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 1e-12 else 0.0


def summarize_gate_response(gate_5s: np.ndarray, q_vec: np.ndarray, m_pose: np.ndarray, q_buckets: np.ndarray) -> dict:
    gm = gate_5s.mean(axis=0)
    valid_m = m_pose > 0.5
    out: dict[str, object] = {
        "spearman_gate_vs_q1_valid_joint_ratio": _spearman_np(gm[valid_m], q_vec[valid_m, 0]) if valid_m.any() else 0.0,
        "spearman_gate_vs_q2_mean_conf": _spearman_np(gm[valid_m], q_vec[valid_m, 1]) if valid_m.any() else 0.0,
        "max_abs_gate_when_mpose_zero": float(np.max(np.abs(gm[~valid_m]))) if (~valid_m).any() else 0.0,
    }
    for b_idx, b_nm in ((0, "Clear"), (1, "Partial"), (2, "Heavy")):
        bm = q_buckets == b_idx
        if bm.any():
            vals = gm[bm]
            q25, q50, q75 = np.percentile(vals, [25.0, 50.0, 75.0])
            out[b_nm] = {"n": int(bm.sum()), "mean": float(vals.mean()), "median": float(q50), "p25": float(q25), "p75": float(q75)}
    return out


def build_pinns_stress_packs(data: dict, meta_dir: Path, pose_npz: Path, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    from scripts.benchmark_heldout_sites import load_site_windows

    groups, _, _ = load_site_windows(meta_dir)
    rows = [r for s in sorted(groups) for r in groups[s]]
    keys = [(str(r.clip), int(r.person_id), int(r.t0)) for r in rows]
    n = len(rows)
    with np.load(pose_npz, allow_pickle=False) as pz:
        ord_map = {k: i for i, k in enumerate(zip(pz["clip"].tolist(), pz["person_id"].tolist(), pz["t0"].tolist()))}
        ids = [ord_map[k] for k in keys]
        kpts_np = pz["hist_kpts"][ids].astype(np.float32)
        bbox_np = pz["hist_bbox"][ids].astype(np.float32)

    h_mats = np.zeros((n, 3, 3), dtype=np.float64)
    y_sums = np.zeros(n, dtype=np.float64)
    for c in sorted({r.clip for r in rows}):
        calib = json.loads((meta_dir / "videos" / c.rsplit("_", 1)[0] / "calib.json").read_text())
        hm, ys = fit_pinns_ground_homography(calib)
        for idx in [i for i, r in enumerate(rows) if r.clip == c]:
            h_mats[idx], y_sums[idx] = hm, ys
    hm_t, ys_t, bbox_t = torch.from_numpy(h_mats), torch.from_numpy(y_sums), torch.from_numpy(bbox_np)
    pm5 = np.ones(5, dtype=np.float32)

    def _pack_from_kpts(kp_arr: np.ndarray) -> dict[str, torch.Tensor]:
        gp = build_ground_pose_o2_inputs(data["history"], torch.from_numpy(kp_arr), bbox_t, hm_t, ys_t)
        qv = np.stack([build_pose_and_quality_features(kp_arr[i], bbox_np[i], pm5)[1] for i in range(n)])
        return {
            "bio_even": gp["bio_even"].to(device), "pose_vec_world": gp["pose_vec_world"].to(device),
            "signed_knee": gp["signed_knee"].to(device), "quality": torch.from_numpy(qv).to(device), "m_pose": gp["m_pose"].to(device),
        }

    rng = np.random.default_rng(20261126)
    kp_jit = kpts_np.copy()
    hb = np.maximum(bbox_np[:, :, 3:4], 1.0)[..., None]
    kp_jit[..., :2] += rng.normal(0.0, 0.20, size=kp_jit[..., :2].shape).astype(np.float32) * hb
    drop_m = rng.random(size=kp_jit[..., 2].shape) < 0.30
    kp_jit[..., 2] = np.where(drop_m, 0.05, np.clip(kp_jit[..., 2] * rng.uniform(0.35, 0.70, size=kp_jit[..., 2].shape), 0.0, 1.0)).astype(np.float32)

    kp_occ = kpts_np.copy()
    kp_occ[:, :, [13, 14, 15, 16], :2] = 0.0
    kp_occ[:, :, [13, 14, 15, 16], 2] = 0.02

    kp_drop = np.zeros_like(kpts_np)
    return {
        "Test_Natural": {k: data[k].to(device) for k in ("bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose")},
        "Test_Jitter020": _pack_from_kpts(kp_jit),
        "Test_LowerOcc": _pack_from_kpts(kp_occ),
        "Test_FullDrop": _pack_from_kpts(kp_drop),
    }


def build_jrdb_stress_packs(data: dict, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    j3d = data["joints_3d"]
    rng = np.random.default_rng(20261127)
    j_jit = j3d + rng.normal(0.0, 0.20, size=j3d.shape)
    j_jit[rng.random(size=j3d.shape[:3]) < 0.35] = np.nan
    j_occ = j3d.copy()
    j_occ[:, :, [25, 26, 27, 28, 29, 30, 31, 32], :] = np.nan
    j_drop = np.full_like(j3d, np.nan)

    def _to_dev(pf: dict) -> dict[str, torch.Tensor]:
        return {k: pf[k].to(device) for k in ("bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose")}

    return {
        "Test_Natural": {k: data[k].to(device) for k in ("bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose")},
        "Test_Jitter020": _to_dev(build_jrdb_3d_pose_o2_features(j_jit)),
        "Test_LowerOcc": _to_dev(build_jrdb_3d_pose_o2_features(j_occ)),
        "Test_FullDrop": _to_dev(build_jrdb_3d_pose_o2_features(j_drop)),
    }


def fit_m0_m1_fold(
    seed: int,
    grp_arr: np.ndarray,
    hist_g: torch.Tensor,
    truth_g: torch.Tensor,
    nat_pack: dict[str, torch.Tensor],
    obs_dt_g: torch.Tensor,
    fut_s_g: torch.Tensor,
    tr_ids: np.ndarray,
    batch_size: int,
    ep_m0: int,
    ep_m1: int,
    device: torch.device,
) -> dict[str, PPTMainlineO2Forecaster]:
    w_tr = make_site_weights(grp_arr, tr_ids).to(device)
    tr_t = torch.from_numpy(tr_ids).to(device)

    def _fwd(net: PPTMainlineO2Forecaster, ids_g: torch.Tensor) -> dict[str, torch.Tensor]:
        return net(
            hist_g[ids_g], nat_pack["bio_even"][ids_g], nat_pack["pose_vec_world"][ids_g],
            nat_pack["signed_knee"][ids_g], nat_pack["quality"][ids_g], nat_pack["m_pose"][ids_g],
            obs_dt=obs_dt_g[ids_g], fut_times_s=fut_s_g[ids_g],
        )

    def _run(net: PPTMainlineO2Forecaster, prefs: tuple[str, ...], ep: int, lr: float, wd: float, soff: int) -> PPTMainlineO2Forecaster:
        for nm, p in net.named_parameters():
            p.requires_grad = nm.startswith(prefs) or nm in prefs
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(seed + soff)
        net.train()
        for _ in range(ep):
            for ch in torch.randperm(len(tr_ids), generator=gen).split(batch_size):
                ch_g = ch.to(device)
                loss = o2_joint_loss(_fwd(net, tr_t[ch_g]), truth_g[tr_t[ch_g]], w_tr[ch_g], act_weight=0.0)
                if loss is not None:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
        return net.eval()

    torch.manual_seed(seed)
    m0 = _run(PPTMainlineO2Forecaster("M0").to(device), ("motion.", "ctx_proj.", "cv_low_bonus", "central_", "pair_"), ep_m0, 2.0e-3, 1e-4, 0)
    models = {"M0": m0}
    for i, arm in enumerate(("M1a", "M1b", "M1")):
        torch.manual_seed(seed)
        net = PPTMainlineO2Forecaster(arm).to(device)
        _copy_matching(net, m0)
        models[arm] = _run(net, ("pose_", "gate_", "central_", "pair_", "cv_low_bonus"), ep_m1, 1.3e-3, 3e-4, 10 + i)
    return models


@torch.inference_mode()
def eval_model_on_pack(
    net: PPTMainlineO2Forecaster,
    hist_g: torch.Tensor,
    pack: dict[str, torch.Tensor],
    obs_dt_g: torch.Tensor,
    fut_s_g: torch.Tensor,
    truth_np: np.ndarray,
    te_ids: np.ndarray,
    device: torch.device,
    force_mpose_one: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids_g = torch.from_numpy(te_ids).to(device)
    mp = torch.ones_like(pack["m_pose"][ids_g]) if force_mpose_one else pack["m_pose"][ids_g]
    out = net(
        hist_g[ids_g], pack["bio_even"][ids_g], pack["pose_vec_world"][ids_g],
        pack["signed_knee"][ids_g], pack["quality"][ids_g], mp,
        obs_dt=obs_dt_g[ids_g], fut_times_s=fut_s_g[ids_g],
    )
    paths = out["paths"].cpu().numpy()
    probs = out["logits"].softmax(dim=1).cpu().numpy()
    gate = out["gate"].cpu().numpy()
    d = np.linalg.norm(paths - truth_np[te_ids][:, None], axis=-1)
    top = probs.argmax(axis=1)
    ii = np.arange(len(te_ids))
    return np.column_stack((d[ii, top].mean(1), d[ii, top, -1])), np.column_stack((d.mean(2).min(1), d[:, :, -1].min(1))), gate


def evaluate_dataset_matrix(
    tag: str,
    grp_arr: np.ndarray,
    unit_arr: np.ndarray,
    is_pinns: bool,
    hist_g: torch.Tensor,
    truth_g: torch.Tensor,
    truth_np: np.ndarray,
    packs: dict[str, dict[str, torch.Tensor]],
    q_buckets: np.ndarray,
    obs_dt_g: torch.Tensor,
    fut_s_g: torch.Tensor,
    batch_size: int,
    ep_m0: int,
    ep_m1: int,
    device: torch.device,
) -> tuple[dict, dict[str, np.ndarray]]:
    n = len(truth_np)
    groups = sorted(set(grp_arr.tolist()))
    regimes = ("VarQTrain", "CleanTrain")
    top_res = {(rg, tc, arm): np.zeros((len(SEEDS), n, 2)) for rg in regimes for tc in TEST_CONDS for arm in ARMS}
    min_res = {(rg, tc, arm): np.zeros((len(SEEDS), n, 2)) for rg in regimes for tc in TEST_CONDS for arm in ARMS}
    gate_res = {(rg, tc, arm): np.zeros((len(SEEDS), n)) for rg in regimes for tc in TEST_CONDS for arm in ("M1a", "M1b", "M1")}
    noshut_top = {rg: np.zeros((len(SEEDS), n, 2)) for rg in regimes}

    for si, seed in enumerate(SEEDS):
        for g in groups:
            tr_all = np.flatnonzero(grp_arr != g)
            tr_clr = np.flatnonzero((grp_arr != g) & (q_buckets == 0))
            te_ids = np.flatnonzero(grp_arr == g)
            for rg, tr_ids in (("VarQTrain", tr_all), ("CleanTrain", tr_clr)):
                mods = fit_m0_m1_fold(seed, grp_arr, hist_g, truth_g, packs["Test_Natural"], obs_dt_g, fut_s_g, tr_ids, batch_size, ep_m0, ep_m1, device)
                for tc in TEST_CONDS:
                    for arm in ARMS:
                        tp, mn, gt = eval_model_on_pack(mods[arm], hist_g, packs[tc], obs_dt_g, fut_s_g, truth_np, te_ids, device)
                        top_res[(rg, tc, arm)][si, te_ids], min_res[(rg, tc, arm)][si, te_ids] = tp, mn
                        if arm != "M0":
                            gate_res[(rg, tc, arm)][si, te_ids] = gt
                tp_ns, _, _ = eval_model_on_pack(mods["M1a"], hist_g, packs["Test_FullDrop"], obs_dt_g, fut_s_g, truth_np, te_ids, device, force_mpose_one=True)
                noshut_top[rg][si, te_ids] = tp_ns
        print(f"[W17-SLIDE5 {tag}] seed={seed} Clean->Jit M1a={top_res[('CleanTrain','Test_Jitter020','M1a')][si].mean(0)[0]:.4f} VarQ->Jit M1={top_res[('VarQTrain','Test_Jitter020','M1')][si].mean(0)[0]:.4f}", flush=True)

    def _pair_ci(diff_2d: np.ndarray, mask: np.ndarray) -> dict:
        if is_pinns:
            return paired_site_clip_ci(diff_2d[mask], grp_arr[mask], unit_arr[mask])
        return {"location_equal": _loc_bootstrap_ci(diff_2d, mask, grp_arr), **fast_person_bootstrap_ci(diff_2d, mask, unit_arr)}

    all_m = np.ones(n, dtype=bool)
    matrix_out: dict[str, dict] = {}
    for rg in regimes:
        matrix_out[rg] = {}
        for tc in TEST_CONDS:
            cell: dict[str, object] = {}
            for arm in ARMS:
                tm, mm = top_res[(rg, tc, arm)].mean(0), min_res[(rg, tc, arm)].mean(0)
                by_g = [tm[grp_arr == g, 0].mean() for g in groups]
                by_gf = [tm[grp_arr == g, 1].mean() for g in groups]
                cell[arm] = {
                    "group_equal": {"ade1_m": float(np.mean(by_g)), "fde1_m": float(np.mean(by_gf)), "min_ade6_m": float(np.mean([mm[grp_arr == g, 0].mean() for g in groups])), "min_fde6_m": float(np.mean([mm[grp_arr == g, 1].mean() for g in groups]))},
                    "window_equal": {"ade1_m": float(tm[:, 0].mean()), "fde1_m": float(tm[:, 1].mean()), "min_ade6_m": float(mm[:, 0].mean()), "min_fde6_m": float(mm[:, 1].mean())},
                }
            cell["M1_minus_M1a"] = _pair_ci(top_res[(rg, tc, "M1")].mean(0) - top_res[(rg, tc, "M1a")].mean(0), all_m)
            cell["M1_minus_M1b"] = _pair_ci(top_res[(rg, tc, "M1")].mean(0) - top_res[(rg, tc, "M1b")].mean(0), all_m)
            cell["M1_minus_M0"] = _pair_ci(top_res[(rg, tc, "M1")].mean(0) - top_res[(rg, tc, "M0")].mean(0), all_m)
            if tc == "Test_FullDrop":
                ns_tm = noshut_top[rg].mean(0)
                cell["M1a_no_mpose_shutoff_ade1_m"] = float(np.mean([ns_tm[grp_arr == g, 0].mean() for g in groups]))
                cell["M1_shutoff_minus_M1a_no_shutoff"] = _pair_ci(top_res[(rg, tc, "M1")].mean(0) - ns_tm, all_m)
            matrix_out[rg][tc] = cell

    q_nat_np = packs["Test_Natural"]["quality"].cpu().numpy()
    mp_nat_np = packs["Test_Natural"]["m_pose"].cpu().numpy()
    gate_audit = {
        rg: {
            "natural_strata_M1": summarize_gate_response(gate_res[(rg, "Test_Natural", "M1")], q_nat_np, mp_nat_np, q_buckets),
            "natural_strata_M1b": summarize_gate_response(gate_res[(rg, "Test_Natural", "M1b")], q_nat_np, mp_nat_np, q_buckets),
            "mean_gate_across_test_conditions": {
                tc: {arm: float(gate_res[(rg, tc, arm)].mean()) for arm in ("M1a", "M1b", "M1")} for tc in TEST_CONDS
            },
        }
        for rg in regimes
    }
    slide5_cross = {
        f"VarQTrain_minus_CleanTrain_M1_{tc}": _pair_ci(top_res[("VarQTrain", tc, "M1")].mean(0) - top_res[("CleanTrain", tc, "M1")].mean(0), all_m)
        for tc in TEST_CONDS
    } | {
        f"VarQTrain_M1_minus_CleanTrain_M1a_{tc}": _pair_ci(top_res[("VarQTrain", tc, "M1")].mean(0) - top_res[("CleanTrain", tc, "M1a")].mean(0), all_m)
        for tc in ("Test_Jitter020", "Test_LowerOcc")
    }
    npz = {f"{tag}_{rg}_{tc}_{arm}_top1": top_res[(rg, tc, arm)].mean(0) for rg in regimes for tc in TEST_CONDS for arm in ARMS}
    npz |= {f"{tag}_VarQ_M1_gate_{tc}": gate_res[("VarQTrain", tc, "M1")].mean(0) for tc in TEST_CONDS}
    return {
        "n_windows": n,
        "n_clear_windows": int((q_buckets == 0).sum()),
        "train_test_matrix": matrix_out,
        "gate_response_audit": gate_audit,
        "slide5_social_pose_comparisons": slide5_cross,
    }, npz


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

    p_data = load_ppt_cohort_with_causal_maps(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin)
    np_p = len(p_data["history"])
    p_packs = build_pinns_stress_packs(p_data, args.meta_dir, args.pose_npz, device)
    p_res, p_npz = evaluate_dataset_matrix(
        "pinns", p_data["site"], p_data["clip"], True,
        p_data["history"].to(device), p_data["truth"].to(device), p_data["truth"].numpy(),
        p_packs, p_data["q_buckets"],
        torch.full((np_p, 4), 0.1, dtype=torch.float64, device=device),
        torch.arange(1, 31, dtype=torch.float64, device=device)[None].expand(np_p, -1) * 0.1,
        64, 45, 25, device,
    )

    j_data = load_jrdb_10scene_rc2_rc3(args.repo_root)
    nj = len(j_data["history"])
    j_packs = build_jrdb_stress_packs(j_data, device)
    j_res, j_npz = evaluate_dataset_matrix(
        "jrdb", j_data["location"], j_data["actors"], False,
        j_data["history"].to(device), j_data["truth"].to(device), j_data["truth"].numpy(),
        j_packs, j_data["q_buckets"],
        torch.full((nj, 1), 0.4, dtype=torch.float64, device=device),
        torch.arange(1, 8, dtype=torch.float64, device=device)[None].expand(nj, -1) * 0.4,
        256, 45, 28, device,
    )

    summary = {
        "experiment": "wave17_slide5_pose_noise_and_gate_response_5seed",
        "device": str(device),
        "seeds": list(SEEDS),
        "scope_note": "Test_Jitter020, Test_LowerOcc, and Test_FullDrop are controlled synthetic keypoint-corruption stress tests (mirroring Slide 5 Social-Pose Table IV), strictly separated from natural Test_Natural Clear/Partial/Heavy strata.",
        "pinns_4site_loso_1482": p_res,
        "jrdb_9location_lolo_5836": j_res,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(args.out_dir / "per_window_slide5_stress_5seed.npz", **p_npz, **j_npz)
    dig = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out_dir.iterdir()) if p.is_file()}
    (args.out_dir / "sha256.json").write_text(json.dumps(dig, indent=2), encoding="utf-8")
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(args.out_dir.iterdir()):
            zf.write(p, arcname=p.name)
    print(f"DONE: {args.zip_path} sha256={hashlib.sha256(args.zip_path.read_bytes()).hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
