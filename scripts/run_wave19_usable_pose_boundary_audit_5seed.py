"""Wave 19b GPU0: Minimum usable pixel box-height & joint-completeness (q1) boundary audit on PINNS (1,482) & JRDB (5,836)."""
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
from scripts.benchmark_heldout_sites import load_site_windows
from scripts.run_jrdb_rc1_pose_quality_and_initiation_5seed import _loc_bootstrap_ci
from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import _stage_gpu_dict, load_jrdb_10scene_rc2_rc3
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import _copy_matching, load_ppt_cohort_with_causal_maps
from scripts.run_wave18_conflict_gated_interaction_5seed import safe_site_clip_ci
from scripts.run_wave18_t4_latency_and_monotonic_gate import MonotonicQualityM1Forecaster
from src.models.bioeq_o2_joint import o2_joint_loss

BOUNDARY_ARMS = ("M0", "M1a", "M1_unconstrained", "M1_mono_q")


def extract_pinns_bbox_heights_px(meta_dir: Path, pose_npz: Path) -> np.ndarray:
    groups, _, _ = load_site_windows(meta_dir)
    rows = [r for s in sorted(groups) for r in groups[s]]
    keys = [(str(r.clip), int(r.person_id), int(r.t0)) for r in rows]
    with np.load(pose_npz, allow_pickle=False) as pz:
        ck = list(zip(pz["clip"].tolist(), pz["person_id"].tolist(), pz["t0"].tolist()))
        ord_map = {k: i for i, k in enumerate(ck)}
        bbox = pz["hist_bbox"][[ord_map[k] for k in keys]].astype(np.float64)
    h = bbox[:, :, 3]
    out = np.zeros(len(rows), dtype=np.float64)
    for i in range(len(rows)):
        pos = h[i][h[i] > 1.0]
        out[i] = float(pos.mean()) if len(pos) else 0.0
    return out


def _fwd_pose_arm(net: torch.nn.Module, gd: dict, ids_g: torch.Tensor, m_pose_override: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    mp = gd["m_pose"][ids_g] if m_pose_override is None else m_pose_override
    return net(
        gd["history"][ids_g], gd["bio_even"][ids_g], gd["pose_vec_world"][ids_g],
        gd["signed_knee"][ids_g], gd["quality"][ids_g], mp,
        obs_dt=gd["obs_dt"][ids_g], fut_times_s=gd["fut_s"][ids_g],
    )


def fit_boundary_arms_fold(
    seed: int, grp_arr: np.ndarray, gd: dict, tr_ids: np.ndarray,
    bsz: int, ep_s: tuple[int, int], lr_s: tuple[float, float], device: torch.device,
) -> dict[str, torch.nn.Module]:
    w_tr = make_site_weights(grp_arr, tr_ids).to(device)
    tr_t = torch.from_numpy(tr_ids).to(device)
    truth_tr = gd["truth"][tr_t]

    def _tr(net: torch.nn.Module, prefs: tuple[str, ...], ep: int, lr: float, wd: float, soff: int) -> torch.nn.Module:
        for nm, p in net.named_parameters():
            p.requires_grad = nm.startswith(prefs) or nm in prefs
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(seed + soff)
        net.train()
        for _ in range(ep):
            for ch in torch.randperm(len(tr_ids), generator=gen).split(bsz):
                cg = ch.to(device)
                loss = o2_joint_loss(_fwd_pose_arm(net, gd, tr_t[cg]), truth_tr[cg], w_tr[cg], act_weight=0.0)
                if loss is not None:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return net.eval()

    torch.manual_seed(seed)
    m0 = _tr(PPTMainlineO2Forecaster("M0").to(device), ("motion.", "ctx_proj.", "cv_low_bonus", "central_", "pair_"), ep_s[0], lr_s[0], 1e-4, 0)
    torch.manual_seed(seed); m1a = PPTMainlineO2Forecaster("M1a").to(device); _copy_matching(m1a, m0)
    m1a = _tr(m1a, ("pose_", "central_", "pair_", "cv_low_bonus"), ep_s[1], lr_s[1], 3e-4, 10)
    torch.manual_seed(seed); m1u = PPTMainlineO2Forecaster("M1").to(device); _copy_matching(m1u, m0)
    m1u = _tr(m1u, ("pose_", "gate_", "central_", "pair_", "cv_low_bonus"), ep_s[1], lr_s[1], 3e-4, 11)
    torch.manual_seed(seed); m1m = MonotonicQualityM1Forecaster().to(device); _copy_matching(m1m, m0)
    m1m = _tr(m1m, ("pose_", "mono_", "central_", "pair_", "cv_low_bonus"), ep_s[1], lr_s[1], 3e-4, 12)
    return {"M0": m0, "M1a": m1a, "M1_unconstrained": m1u, "M1_mono_q": m1m}


def run_boundary_audit_on_cohort(
    tag: str, grp_arr: np.ndarray, unit_arr: np.ndarray, is_pinns: bool, gd: dict,
    truth_np: np.ndarray, q_np: np.ndarray, m_pose_np: np.ndarray, cv_err: np.ndarray,
    bbox_h_px: np.ndarray | None, bsz: int, ep_s: tuple[int, int], lr_s: tuple[float, float],
    seeds: list[int], device: torch.device,
) -> dict:
    n = len(grp_arr)
    groups = sorted(set(grp_arr.tolist()))
    top_s = {a: np.zeros((len(seeds), n, 2), dtype=np.float64) for a in BOUNDARY_ARMS}
    gate_s = {a: np.zeros((len(seeds), n), dtype=np.float64) for a in BOUNDARY_ARMS}
    q_taus = (0.0, 0.25, 0.50, 0.75)
    h_taus = (0.0, 45.0, 65.0, 85.0, 110.0) if bbox_h_px is not None else ()
    q_sweep = {t: {a: np.zeros((len(seeds), n, 2), dtype=np.float64) for a in ("M1a", "M1_unconstrained", "M1_mono_q")} for t in q_taus}
    h_sweep = {t: {a: np.zeros((len(seeds), n, 2), dtype=np.float64) for a in ("M1a", "M1_unconstrained", "M1_mono_q")} for t in h_taus}

    for si, seed in enumerate(seeds):
        for g in groups:
            tr = np.flatnonzero(grp_arr != g)
            te = np.flatnonzero(grp_arr == g)
            te_g = torch.from_numpy(te).to(device)
            mods = fit_boundary_arms_fold(seed, grp_arr, gd, tr, bsz, ep_s, lr_s, device)
            with torch.inference_mode():
                for a in BOUNDARY_ARMS:
                    oo = _fwd_pose_arm(mods[a], gd, te_g)
                    pw = oo["paths"].cpu().numpy()
                    top_p = pw[np.arange(len(te)), oo["logits"].argmax(1).cpu().numpy()]
                    d = np.linalg.norm(top_p - truth_np[te], axis=-1)
                    top_s[a][si, te] = np.column_stack((d.mean(1), d[:, -1]))
                    gate_s[a][si, te] = oo["gate"].cpu().numpy()
                for tq in q_taus:
                    mp_q = gd["m_pose"][te_g] * (gd["quality"][te_g, 0] >= tq).float()
                    for a in ("M1a", "M1_unconstrained", "M1_mono_q"):
                        oo = _fwd_pose_arm(mods[a], gd, te_g, m_pose_override=mp_q)
                        pw = oo["paths"].cpu().numpy()
                        d = np.linalg.norm(pw[np.arange(len(te)), oo["logits"].argmax(1).cpu().numpy()] - truth_np[te], axis=-1)
                        q_sweep[tq][a][si, te] = np.column_stack((d.mean(1), d[:, -1]))
                if bbox_h_px is not None:
                    h_te = torch.from_numpy(bbox_h_px[te].astype(np.float32)).to(device)
                    for th in h_taus:
                        mp_h = gd["m_pose"][te_g] * (h_te >= th).float()
                        for a in ("M1a", "M1_unconstrained", "M1_mono_q"):
                            oo = _fwd_pose_arm(mods[a], gd, te_g, m_pose_override=mp_h)
                            pw = oo["paths"].cpu().numpy()
                            d = np.linalg.norm(pw[np.arange(len(te)), oo["logits"].argmax(1).cpu().numpy()] - truth_np[te], axis=-1)
                            h_sweep[th][a][si, te] = np.column_stack((d.mean(1), d[:, -1]))
        print(f"[W19b-GPU0 {tag}] Finished seed={seed}", flush=True)

    tm = {a: top_s[a].mean(0) for a in BOUNDARY_ARMS}
    gm = {a: gate_s[a].mean(0) for a in BOUNDARY_ARMS}
    ci_fn = (lambda d, m: safe_site_clip_ci(d[m], grp_arr[m], unit_arr[m])) if is_pinns else (lambda d, m: _loc_bootstrap_ci(d[m], grp_arr[m]))

    q1 = q_np[:, 0]
    strata: dict[str, np.ndarray] = {
        "all": np.ones(n, dtype=bool),
        "Q0_blind_mpose0": m_pose_np < 0.5,
        "Q1_sparse_q1_lt40": (m_pose_np >= 0.5) & (q1 < 0.40),
        "Q2_moderate_q1_40_75": (m_pose_np >= 0.5) & (q1 >= 0.40) & (q1 < 0.75),
        "Q3_complete_q1_ge75": (m_pose_np >= 0.5) & (q1 >= 0.75),
    }
    if bbox_h_px is not None:
        strata["B0_tiny_lt50px"] = (m_pose_np >= 0.5) & (bbox_h_px < 50.0)
        strata["B1_small_50_80px"] = (m_pose_np >= 0.5) & (bbox_h_px >= 50.0) & (bbox_h_px < 80.0)
        strata["B2_medium_80_130px"] = (m_pose_np >= 0.5) & (bbox_h_px >= 80.0) & (bbox_h_px < 130.0)
        strata["B3_large_ge130px"] = (m_pose_np >= 0.5) & (bbox_h_px >= 130.0)

    strata_out = {}
    for sk, sm in strata.items():
        if int(sm.sum()) < 3:
            continue
        gs = sorted(set(grp_arr[sm].tolist()))
        mods_d = {"Last2_CV": {"group_equal_ade1_m": float(np.mean([cv_err[(grp_arr == g) & sm, 0].mean() for g in gs])),
                               "group_equal_fde1_m": float(np.mean([cv_err[(grp_arr == g) & sm, 1].mean() for g in gs]))}}
        for a in BOUNDARY_ARMS:
            mods_d[a] = {
                "group_equal_ade1_m": float(np.mean([tm[a][(grp_arr == g) & sm, 0].mean() for g in gs])),
                "group_equal_fde1_m": float(np.mean([tm[a][(grp_arr == g) & sm, 1].mean() for g in gs])),
                "mean_gate": float(gm[a][sm].mean()),
            }
        paired = {
            "M1a_minus_M0": ci_fn(tm["M1a"] - tm["M0"], sm),
            "M1_unconstrained_minus_M0": ci_fn(tm["M1_unconstrained"] - tm["M0"], sm),
            "M1_mono_q_minus_M0": ci_fn(tm["M1_mono_q"] - tm["M0"], sm),
            "M1_mono_q_minus_M1a": ci_fn(tm["M1_mono_q"] - tm["M1a"], sm),
            "M1_mono_q_minus_M1_unconstrained": ci_fn(tm["M1_mono_q"] - tm["M1_unconstrained"], sm),
        }
        strata_out[sk] = {"n_windows": int(sm.sum()), "n_groups_present": len(gs), "models": mods_d, "paired_top1_ci": paired}

    all_m = np.ones(n, dtype=bool)
    q_sw_out = {f"tau_q1_{t:.2f}": {a: {"minus_M0_ci": ci_fn(q_sweep[t][a].mean(0) - tm["M0"], all_m)} for a in ("M1a", "M1_unconstrained", "M1_mono_q")} for t in q_taus}
    h_sw_out = {f"tau_h_{int(t)}px": {a: {"minus_M0_ci": ci_fn(h_sweep[t][a].mean(0) - tm["M0"], all_m)} for a in ("M1a", "M1_unconstrained", "M1_mono_q")} for t in h_taus}
    return {"n_windows": n, "strata": strata_out, "q1_cutoff_sweep_vs_M0": q_sw_out, "bbox_height_px_cutoff_sweep_vs_M0": h_sw_out}


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
    gdp = {k: dp[k].to(dev) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose")}
    gdp["obs_dt"] = torch.full((np_w, 4), 0.1, dtype=torch.float64, device=dev)
    gdp["fut_s"] = torch.arange(1, 31, dtype=torch.float64, device=dev)[None].expand(np_w, -1) * 0.1
    hp_np, tp_np = dp["history"].numpy(), dp["truth"].numpy()
    cv_p = hp_np[:, -1:] + ((hp_np[:, -1:] - hp_np[:, -2:-1]) / 0.1) * (np.arange(1, 31, dtype=np.float64)[None, :, None] * 0.1)
    d_cvp = np.linalg.norm(cv_p - tp_np, axis=-1)
    bbox_h = extract_pinns_bbox_heights_px(args.meta_dir, args.pose_npz)
    p_res = run_boundary_audit_on_cohort(
        "pinns", dp["site"], dp["clip"], True, gdp, tp_np, dp["quality"].numpy(), dp["m_pose"].numpy(),
        np.column_stack((d_cvp.mean(1), d_cvp[:, -1])), bbox_h, 64, (45, 26), (2e-3, 1.2e-3), list(args.seeds), dev,
    )

    dj = load_jrdb_10scene_rc2_rc3(args.repo_root)
    gdj = _stage_gpu_dict(dj, dev)
    hj_np, tj_np = dj["history"].numpy(), dj["truth"].numpy()
    cv_j = hj_np[:, -1:] + ((hj_np[:, 1:2] - hj_np[:, 0:1]) / 0.4) * (np.arange(1, 8, dtype=np.float64)[None, :, None] * 0.4)
    d_cvj = np.linalg.norm(cv_j - tj_np, axis=-1)
    j_res = run_boundary_audit_on_cohort(
        "jrdb", dj["location"], dj["actors"], False, gdj, tj_np, dj["quality"].numpy(), dj["m_pose"].numpy(),
        np.column_stack((d_cvj.mean(1), d_cvj[:, -1])), None, 512, (45, 28), (2.5e-3, 1.5e-3), list(args.seeds), dev,
    )

    summary = {"experiment": "wave19_usable_pose_boundary_audit_5seed", "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
               "seeds": list(args.seeds), "pinns_4site_loso_1482": p_res, "jrdb_9location_lolo_5836": j_res}
    raw = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = (json.dumps({"summary.json": hashlib.sha256(raw).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw)
        zf.writestr("sha256.json", sh)
    print("Saved:", args.zip_path, flush=True)


if __name__ == "__main__":
    main()
