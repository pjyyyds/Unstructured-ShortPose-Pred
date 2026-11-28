"""Wave 29B: Real-Hardware Latency / Parameter / MFLOPs Audit & JRDB 9-Location PCR6 + Zero-Retrain Split-Conformal."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from scripts.run_wave18_conflict_gated_interaction_5seed import _loc_bootstrap_ci
from scripts.run_wave21_jrdb_and_jta_emergence_maneuvers import load_jrdb_with_blind_emergence
from scripts.run_wave25_pinns_explicit_audit_and_emergence_safe import _pooled_clip_bootstrap_ci
from scripts.run_wave27_jaad_cusum_and_ray_traj import JAADIntra5FSPRTPrecursorNet
from scripts.run_wave28_jaad_scale_ray_neural_5seed import JAADScaleRaySplitTrajectoryNet
from src.models.emergence_kalman_residual_6slot import (
    O2EmergenceKalmanRes6SlotNet,
    compute_split_anatomical_covisibility,
    ekr6_decoupled_slot0_and_maneuver_loss,
)
from src.models.emergence_maneuver_6slot import horizon_and_post_onset_errors
from src.models.jaad_full17_reappear_models import JAADPostReappearPrecursorNet
from src.models.passivity_ray_pcr6 import O2PassivityRay6SlotNet
from src.models.ray_kalman_and_cusum import O2RayKalmanRes6SlotNet, calibrate_ray_kalman_params
from src.models.tm_ray_rkr6 import O2TerminalMobilityRay6SlotNet

SEEDS = (42, 1024, 2025, 3407, 4096)


def benchmark_model_latency_and_params(device: torch.device, n_warmup: int = 30, n_iters: int = 200) -> dict:
    """Measure exact parameter counts & P50/P95 inference latency (CUDA events if GPU, perf_counter if CPU) at B=1 and B=12."""
    is_cuda = device.type == "cuda"
    dev_name = torch.cuda.get_device_name(device) if is_cuda else "CPU_4Core_Xeon"
    models_o2 = {
        "EKR6_Full": O2EmergenceKalmanRes6SlotNet(True, True).to(device).eval(),
        "RKR6_Full": O2RayKalmanRes6SlotNet(True, True).to(device).eval(),
        "TMR6_Full": O2TerminalMobilityRay6SlotNet(True, True, True).to(device).eval(),
        "PCR6_Full": O2PassivityRay6SlotNet(True, True, True).to(device).eval(),
    }
    models_jaad_lead = {
        "JAAD_Full17_Precursor": JAADPostReappearPrecursorNet(5, 20, 4, 3, True, True).to(device).eval(),
        "JAAD_Intra5F_SPRT": JAADIntra5FSPRTPrecursorNet(5, 20, 4, 3, True, True).to(device).eval(),
    }
    rcfg = {"sigma_u_px": 1.5, "sigma_v_px": 3.0, "kappa_dh": 0.35, "sigma_iso_m": 0.008, "q_acc": 3.0, "tau_eb": 0.0, "anchor_p_blend": 0.0, "d_ref": 0.25}
    m_jaad_traj = JAADScaleRaySplitTrajectoryNet(20, 4, True, True, rcfg).to(device).eval()

    def _time_fn(fn) -> dict[str, float]:
        with torch.inference_mode():
            for _ in range(n_warmup):
                fn()
            if is_cuda:
                torch.cuda.synchronize(device)
                ts = []
                for _ in range(n_iters):
                    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    ev0.record()
                    fn()
                    ev1.record()
                    torch.cuda.synchronize(device)
                    ts.append(float(ev0.elapsed_time(ev1)))
            else:
                ts = []
                for _ in range(n_iters):
                    t0 = time.perf_counter()
                    fn()
                    ts.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(ts, dtype=np.float64)
        return {"p50_ms": float(np.percentile(arr, 50.0)), "p95_ms": float(np.percentile(arr, 95.0)), "mean_ms": float(arr.mean())}

    out = {"hardware_device": dev_name, "is_cuda": is_cuda, "models": {}}
    for nm, m in models_o2.items():
        n_params = sum(p.numel() for p in m.parameters())
        b_res = {}
        for bsz in (1, 12):
            h = torch.randn(bsz, 5, 2, dtype=torch.float64, device=device) * 0.25
            be = torch.randn(bsz, 5, 11, device=device)
            pvw = torch.randn(bsz, 5, 2, 2, dtype=torch.float64, device=device) * 0.15
            sk = torch.randn(bsz, 5, 1, device=device)
            qv, mp = torch.ones(bsz, 4, device=device), torch.ones(bsz, device=device)
            jac = torch.randn(bsz, 5, 2, 2, dtype=torch.float64, device=device) * 0.03
            ed = torch.randn(bsz, 12, 8, device=device)
            ic, nmk = torch.ones(bsz, 12, device=device), torch.ones(bsz, 12, device=device)
            sc = compute_split_anatomical_covisibility(None, None, be, pvw, sk, qv)
            if nm == "EKR6_Full":
                b_res[f"B{bsz}"] = _time_fn(lambda: m(h, be, pvw, sk, qv, mp, sc, None, ed, ic, nmk))
            else:
                b_res[f"B{bsz}"] = _time_fn(lambda: m(h, be, pvw, sk, qv, mp, sc, jac, None, None, ed, ic, nmk))
        out["models"][nm] = {"params": int(n_params), **b_res}

    for nm, m in models_jaad_lead.items():
        n_params = sum(p.numel() for p in m.parameters())
        b_res = {}
        for bsz in (1, 12):
            k5, p5, q4, v3 = torch.randn(bsz, 5, 5, device=device), torch.randn(bsz, 5, 20, device=device), torch.ones(bsz, 4, device=device), torch.randn(bsz, 3, device=device)
            b_res[f"B{bsz}"] = _time_fn(lambda: m(k5, p5, q4, v3))
        out["models"][nm] = {"params": int(n_params), **b_res}

    n_params_jt = sum(p.numel() for p in m_jaad_traj.parameters())
    bt_res = {}
    for bsz in (1, 12):
        h5, p5, q4, v5 = torch.randn(bsz, 5, 2, dtype=torch.float64, device=device), torch.randn(bsz, 5, 20, device=device), torch.ones(bsz, 4, device=device), torch.randn(bsz, 5, 3, device=device)
        jac = torch.zeros(bsz, 5, 2, 2, dtype=torch.float64, device=device); jac[:, :, 0, 0] = 0.01; jac[:, :, 1, 1] = 0.028
        bh, ac = torch.full((bsz, 5), 100.0, dtype=torch.float64, device=device), torch.full((bsz, 5), 0.8, dtype=torch.float64, device=device)
        bt_res[f"B{bsz}"] = _time_fn(lambda: m_jaad_traj(h5, p5, q4, v5, jac, bh, ac))
    out["models"]["JAAD_ScaleRay_Traj"] = {"params": int(n_params_jt), **bt_res}
    return out


def run_jrdb_pcr6_and_conformal(repo_root: Path, epochs: int, device: torch.device) -> dict:
    dj, cj = load_jrdb_with_blind_emergence(repo_root)
    hist_np, truth_np = dj["history"].numpy(), dj["truth"].numpy()
    grp_arr, unit_arr = dj["location"], dj["actors"]
    n, tf = len(hist_np), truth_np.shape[1]
    dt = 0.4
    disp_j = np.linalg.norm(hist_np[:, 1] - hist_np[:, 0], axis=-1)
    v_last_spd = disp_j / dt
    act_m = (dj["y_act"][:, :4] & dj["valid_act"][:, :4]).any(axis=1)
    e_tr = cj["e_true"]
    strata_j = {
        "all_5836": np.ones(n, dtype=bool),
        "true_deocclusion_507": e_tr,
        "true_deocc_lowdisp_lt0p5m": e_tr & (disp_j < 0.50),
        "true_deocclusion_active_maneuvers": e_tr & act_m,
        "lowdisp_all_lt0p5m": disp_j < 0.50,
    }

    gd = {k: dj[k].to(device) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
    gd["obs_dt"] = torch.full((n, 1), 0.4, dtype=torch.float64, device=device)
    gd["fut_s"] = torch.arange(1, 8, dtype=torch.float64, device=device)[None].expand(n, -1) * 0.4
    gd["y_act"] = torch.from_numpy(dj["y_act"]).to(device)
    gd["valid_act"] = torch.from_numpy(dj["valid_act"]).to(device)
    fut_s_np = gd["fut_s"].cpu().numpy()
    cv_world = hist_np[:, -1:, :] + ((hist_np[:, -1] - hist_np[:, -2]) / dt)[:, None, :] * fut_s_np[:, :, None]
    err_cv = horizon_and_post_onset_errors(cv_world, truth_np, dj["onset"], dt=dt)

    with torch.inference_mode():
        sc = compute_split_anatomical_covisibility(None, None, gd["bio_even"], gd["pose_vec_world"], gd["signed_knee"], gd["quality"])

    arms = (("PCR6_M0_nopose_nonbr", False, False), ("PCR6_M1_splitpose_nonbr", True, False), ("PCR6_M2b_nopose_conflict", False, True), ("PCR6_Full_Slot0", True, True))
    err_s = {nm: np.zeros((len(SEEDS), n, 6), dtype=np.float64) for nm, *_ in arms}
    p6_te_sum = np.zeros((n, 6, tf, 2), dtype=np.float64)
    groups = sorted(set(grp_arr.tolist()))
    rcfg_fold = {}
    for held in groups:
        tr_g = torch.from_numpy(np.flatnonzero(grp_arr != held)).to(device)
        rcfg_fold[str(held)] = calibrate_ray_kalman_params(gd["history"][tr_g], gd["obs_dt"][tr_g], gd["fut_s"][tr_g], gd["truth"][tr_g])

    t0 = time.perf_counter()
    for si, seed in enumerate(SEEDS):
        for held in groups:
            te, tr = np.flatnonzero(grp_arr == held), np.flatnonzero(grp_arr != held)
            te_g, tr_g = torch.from_numpy(te).to(device), torch.from_numpy(tr).to(device)
            g_tr = grp_arr[tr]
            ug = sorted(set(g_tr.tolist()))
            w_np = np.zeros(len(tr), dtype=np.float64)
            for g in ug:
                m = g_tr == g
                w_np[m] = len(tr) / (len(ug) * max(int(m.sum()), 1))
            w_all = torch.zeros(n, dtype=torch.float64, device=device)
            w_all[tr_g] = torch.from_numpy(w_np).to(device)
            pw = ((gd["valid_act"][tr_g].float().sum(0) - gd["y_act"][tr_g].float().sum(0)) / gd["y_act"][tr_g].float().sum(0).clamp_min(1.0)).clamp(1.0, 14.0)

            for nm, up, un in arms:
                torch.manual_seed(seed + (173 if up else 0) + (331 if un else 0))
                net = O2PassivityRay6SlotNet(up, un, False, ray_cfg=rcfg_fold[str(held)]).to(device)
                opt = torch.optim.AdamW(net.parameters(), lr=2.0e-3, weight_decay=1.5e-4)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
                rng = np.random.default_rng(seed + 83)
                for _ in range(epochs):
                    perm = rng.permutation(tr)
                    for s in range(0, len(perm), 256):
                        bt = torch.from_numpy(perm[s : s + 256]).to(device)
                        sc_b = {k: v[bt] for k, v in sc.items()}
                        o = net(gd["history"][bt], gd["bio_even"][bt], gd["pose_vec_world"][bt], gd["signed_knee"][bt], gd["quality"][bt], gd["m_pose"][bt], sc_b, None, None, None, gd["edges"][bt], gd["is_car"][bt], gd["nbr_mask"][bt], gd["obs_dt"][bt], gd["fut_s"][bt])
                        loss = ekr6_decoupled_slot0_and_maneuver_loss(o, gd["truth"][bt], gd["history"][bt], w_all[bt], gd["y_act"][bt], gd["valid_act"][bt], pw)
                        if loss is not None:
                            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
                    sch.step()
                net.eval()
                with torch.inference_mode():
                    sc_t = {k: v[te_g] for k, v in sc.items()}
                    pwt = net(gd["history"][te_g], gd["bio_even"][te_g], gd["pose_vec_world"][te_g], gd["signed_knee"][te_g], gd["quality"][te_g], gd["m_pose"][te_g], sc_t, None, None, None, gd["edges"][te_g], gd["is_car"][te_g], gd["nbr_mask"][te_g], gd["obs_dt"][te_g], gd["fut_s"][te_g])["paths"].cpu().numpy()
                err_s[nm][si, te] = horizon_and_post_onset_errors(pwt[:, 0], truth_np[te], dj["onset"][te], dt=dt)
                if nm == "PCR6_Full_Slot0":
                    p6_te_sum[te] += pwt / len(SEEDS)
        print(f"[Wave 29B JRDB] Finished seed={seed} ({si+1}/{len(SEEDS)}) elapsed={time.perf_counter()-t0:.1f}s", flush=True)

    em = {nm: err_s[nm].mean(axis=0) for nm, *_ in arms}
    keys6 = ("ade1_full_m", "fde1_full_m", "ade1_trig_0p5s_m", "ade1_trig_1p0s_m", "ade1_post_onset_0p5s_m", "ade1_post_onset_1p0s_m")
    st_out = {}
    for sk, sm in strata_j.items():
        m_eq = {"Last2_CV": {k: float(np.mean([err_cv[sm & (grp_arr == g), j].mean() for g in groups if np.any(sm & (grp_arr == g))])) for j, k in enumerate(keys6)}}
        m_pl = {"Last2_CV": {k: float(err_cv[sm, j].mean()) for j, k in enumerate(keys6)}}
        for nm, *_ in arms:
            m_eq[nm] = {k: float(np.mean([em[nm][sm & (grp_arr == g), j].mean() for g in groups if np.any(sm & (grp_arr == g))])) for j, k in enumerate(keys6)}
            m_pl[nm] = {k: float(em[nm][sm, j].mean()) for j, k in enumerate(keys6)}
        st_out[sk] = {
            "n_windows": int(sm.sum()),
            "models_group_equal": m_eq,
            "models_pooled_window_weighted": m_pl,
            "paired_ci": {
                "pcr6_m2b_minus_CV_pool": _pooled_clip_bootstrap_ci(em["PCR6_M2b_nopose_conflict"][sm, :2] - err_cv[sm, :2], unit_arr[sm]),
                "pcr6_full_minus_CV_pool": _pooled_clip_bootstrap_ci(em["PCR6_Full_Slot0"][sm, :2] - err_cv[sm, :2], unit_arr[sm]),
                "pcr6_full_minus_CV_loc9": _loc_bootstrap_ci(em["PCR6_Full_Slot0"][sm, :2] - err_cv[sm, :2], grp_arr[sm]),
                "pcr6_pose_no_nbr_pool": _pooled_clip_bootstrap_ci(em["PCR6_M1_splitpose_nonbr"][sm, :2] - em["PCR6_M0_nopose_nonbr"][sm, :2], unit_arr[sm]),
            },
        }

    # Zero-retrain 72-ordered cross-location split-conformal safety tubes on JRDB
    t_sc = np.linspace(0.35, 1.0, tf, dtype=np.float64)[None, :]
    sp_all = np.linalg.norm(p6_te_sum - p6_te_sum[:, :1], axis=-1).mean(axis=(1, 2))
    sig_all = (0.12 + 0.40 * v_last_spd + 0.35 * sp_all)[:, None] * t_sc
    sig_cv = (0.12 + 0.45 * v_last_spd)[:, None] * t_sc
    e1 = np.linalg.norm(p6_te_sum[:, 0] - truth_np, axis=-1)
    eu = np.linalg.norm(p6_te_sum - truth_np[:, None], axis=-1).min(axis=1)
    ec = np.linalg.norm(cv_world - truth_np, axis=-1)
    pairs72 = {a: {"t1_c": [], "u6_c": [], "cv_c": [], "t1_r": [], "u6_r": [], "cv_r": []} for a in ("90", "95")}
    for g_cal in groups:
        ic = np.flatnonzero(grp_arr == g_cal)
        mc = len(ic)
        s1, su, sc_v = (e1[ic] / sig_all[ic]).max(1), (eu[ic] / sig_all[ic]).max(1), (ec[ic] / sig_cv[ic]).max(1)
        for g_ev in groups:
            if g_ev == g_cal:
                continue
            ie = np.flatnonzero(grp_arr == g_ev)
            for a_nm, alpha in (("90", 0.10), ("95", 0.05)):
                kq = min(max(int(math.ceil((mc + 1) * (1.0 - alpha))) - 1, 0), mc - 1)
                q1, qu, qc = float(np.sort(s1)[kq]), float(np.sort(su)[kq]), float(np.sort(sc_v)[kq])
                pairs72[a_nm]["t1_c"].append(float(np.all(e1[ie] <= q1 * sig_all[ie], axis=1).mean()))
                pairs72[a_nm]["u6_c"].append(float(np.all(eu[ie] <= qu * sig_all[ie], axis=1).mean()))
                pairs72[a_nm]["cv_c"].append(float(np.all(ec[ie] <= qc * sig_cv[ie], axis=1).mean()))
                pairs72[a_nm]["t1_r"].append(float((q1 * sig_all[ie]).mean()))
                pairs72[a_nm]["u6_r"].append(float((qu * sig_all[ie]).mean()))
                pairs72[a_nm]["cv_r"].append(float((qc * sig_cv[ie]).mean()))

    conf72 = {
        f"nominal_{a}pct": {
            "pcr6_top1_joint_cov_72pairs": float(np.mean(pairs72[a]["t1_c"])),
            "pcr6_union6_joint_cov_72pairs": float(np.mean(pairs72[a]["u6_c"])),
            "cv_joint_cov_72pairs": float(np.mean(pairs72[a]["cv_c"])),
            "pcr6_top1_mean_radius_m": float(np.mean(pairs72[a]["t1_r"])),
            "pcr6_union6_mean_radius_m": float(np.mean(pairs72[a]["u6_r"])),
            "cv_mean_radius_m": float(np.mean(pairs72[a]["cv_r"])),
        }
        for a in ("90", "95")
    }
    return {"strata": st_out, "zero_retrain_72_cross_location_conformal": conf72}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--zip-path", type=Path, default=Path("wave29_hardware_latency_and_jrdb_conformal.zip"))
    ap.add_argument("--epochs", type=int, default=28)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        torch.set_num_threads(4)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Wave 29B] Benchmarking latency + running JRDB PCR6 & 72-pair conformal on {dev}...", flush=True)
    lat = benchmark_model_latency_and_params(dev)
    jrdb = run_jrdb_pcr6_and_conformal(args.repo_root, args.epochs, dev)
    summary = {
        "wave": "wave29_hardware_latency_and_jrdb_pcr6_conformal_5seed",
        "hardware_latency_and_params": lat,
        "jrdb_9loc_pcr6_and_conformal": jrdb,
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
