"""Wave 30: True 20-Frame Multi-Duration (T=2,5,8,10,20) PCR6 Matrix & Zero-Shot IMPTC (564) Transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.stats import spearmanr
import torch

from scripts.ppt_mainline_o2_models import fit_pinns_ground_homography
from scripts.run_imptc_and_dair_o2_full_external import ensure_imptc_sample, load_imptc_with_neighbors
from scripts.run_jrdb_rc1_pose_quality_and_initiation_5seed import _loc_bootstrap_ci
from scripts.run_pinns_matched_o2_multiduration import DURATIONS, SEEDS, SITES
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import load_ppt_cohort_with_causal_maps
from scripts.run_ppt_rc1_3x3_duration_quality_matrix import load_matched_623_rc1_20frame
from scripts.run_wave18_conflict_gated_interaction_5seed import safe_site_clip_ci
from scripts.run_wave19_dut_imptc_conflict_and_conformal_5seed import ROT5, adapt_imptc_cohort
from scripts.run_wave25_pinns_explicit_audit_and_emergence_safe import _pooled_clip_bootstrap_ci
from src.models.emergence_kalman_residual_6slot import (
    compute_split_anatomical_covisibility,
    ekr6_decoupled_slot0_and_maneuver_loss,
)
from src.models.passivity_ray_pcr6 import O2PassivityRay6SlotNet
from src.models.ray_kalman_and_cusum import compute_homography_jacobian_world


def run_true20f_pcr6_multiduration(
    matched_npz: Path,
    meta_dir: Path,
    pose_npz: Path,
    matched_obs_npz: Path,
    device: torch.device,
    epochs: int = 30,
) -> dict:
    data = load_matched_623_rc1_20frame(matched_npz, meta_dir, pose_npz, matched_obs_npz)
    with np.load(matched_npz, allow_pickle=False) as m_ar:
        m = {k: m_ar[k] for k in m_ar.files}
    with np.load(matched_obs_npz, allow_pickle=False) as o_ar:
        obs = {k: o_ar[k] for k in o_ar.files}
    sel = sorted(np.flatnonzero(m["strict_disjoint_20"]).tolist(), key=lambda i: (str(m["clip"][i]), int(m["person_id"][i]), int(m["t0"][i])))
    keys = [(str(m["clip"][i]), int(m["person_id"][i]), int(m["t0"][i])) for i in sel]
    o_map = {(str(c), int(pid), int(t)): i for i, (c, pid, t) in enumerate(zip(obs["clip"], obs["person_id"], obs["t0"]))}
    o_idx = np.array([o_map[k] for k in keys], dtype=int)
    kpts20 = torch.from_numpy(obs["hist_kpts"][o_idx].astype(np.float32))
    bbox20 = torch.from_numpy(obs["hist_bbox"][o_idx].astype(np.float32))

    n = len(keys)
    clips, site_arr = data["clip"], data["site"]
    h_mats, y_sums = np.zeros((n, 3, 3), dtype=np.float64), np.zeros(n, dtype=np.float64)
    for c in sorted(set(clips.tolist())):
        calib = json.loads((meta_dir / "videos" / c.rsplit("_", 1)[0] / "calib.json").read_text())
        hm, ys = fit_pinns_ground_homography(calib)
        for i in np.flatnonzero(clips == c):
            h_mats[i], y_sums[i] = hm, ys
    jac20 = compute_homography_jacobian_world(bbox20, torch.from_numpy(h_mats), torch.from_numpy(y_sums)).to(device)
    truth_t = data["truth"].to(device)
    truth_np = data["truth"].numpy()
    h5 = data["hist20"][:, -5:].numpy()
    cv_p = (h5[:, -1:] - h5[:, -2:-1]) * np.arange(1, 31, dtype=np.float64)[None, :, None]
    d_cv = np.linalg.norm(cv_p - truth_np, axis=-1)
    err_cv = np.column_stack((d_cv.mean(1), d_cv[:, -1]))
    qb5 = data["per_dur"][5]["q_buckets"]

    ya_z = torch.zeros((n, 5), dtype=torch.bool, device=device); ya_z[:, 4] = True
    va_z = torch.ones((n, 5), dtype=torch.bool, device=device)
    pw_z = torch.ones(5, device=device)
    by_dur = {}

    for dur in DURATIONS:
        dd = data["per_dur"][dur]
        h_d = data["hist20"][:, -dur:].to(device)
        be_d, pv_d = dd["bio_even"].to(device), dd["pose_vec_world"].to(device)
        sk_d, q_d, mp_d = dd["signed_knee"].to(device), dd["quality"].to(device), dd["m_pose"].to(device)
        jac_d = jac20[:, -dur:]
        with torch.inference_mode():
            sc_d = compute_split_anatomical_covisibility(kpts20[:, -dur:].to(device), bbox20[:, -dur:].to(device), be_d, pv_d, sk_d, q_d)

        err_m0 = np.zeros((len(SEEDS), n, 2), dtype=np.float64)
        err_m1 = np.zeros((len(SEEDS), n, 2), dtype=np.float64)
        gates_m1 = np.zeros((len(SEEDS), n), dtype=np.float64)

        for si, seed in enumerate(SEEDS):
            for s in sorted(set(site_arr.tolist())):
                tr, te = np.flatnonzero(site_arr != s), np.flatnonzero(site_arr == s)
                tr_g, te_g = torch.from_numpy(tr).to(device), torch.from_numpy(te).to(device)
                g_tr = site_arr[tr]
                ug = sorted(set(g_tr.tolist()))
                w_np = np.zeros(len(tr), dtype=np.float64)
                for g in ug:
                    m_g = g_tr == g
                    w_np[m_g] = len(tr) / (len(ug) * max(int(m_g.sum()), 1))
                w_all = torch.zeros(n, dtype=torch.float64, device=device); w_all[tr_g] = torch.from_numpy(w_np).to(device)

                for u_pose, arr_e in ((False, err_m0), (True, err_m1)):
                    torch.manual_seed(seed + dur * 13 + (97 if u_pose else 0))
                    net = O2PassivityRay6SlotNet(u_pose, False, True).to(device)
                    opt = torch.optim.AdamW(net.parameters(), lr=1.8e-3, weight_decay=1.5e-4)
                    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
                    rng = np.random.default_rng(seed + dur)
                    for _ in range(epochs):
                        perm = rng.permutation(tr)
                        for st in range(0, len(perm), 96):
                            bt = torch.from_numpy(perm[st : st + 96]).to(device)
                            sc_b = {k: v[bt] for k, v in sc_d.items()}
                            # Note: truth_t is relative to hist20[:, -1] (origin==0), so pass h_d - h_d[:, -1:]
                            h_rel = h_d[bt] - h_d[bt, -1:]
                            o = net(h_rel, be_d[bt], pv_d[bt], sk_d[bt], q_d[bt], mp_d[bt], sc_b, jac_d[bt])
                            loss = ekr6_decoupled_slot0_and_maneuver_loss(o, truth_t[bt], h_rel, w_all[bt], ya_z[bt], va_z[bt], pw_z, act_weight=0.0)
                            if loss is not None:
                                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                        sch.step()
                    net.eval()
                    with torch.inference_mode():
                        sc_t = {k: v[te_g] for k, v in sc_d.items()}
                        ot = net(h_d[te_g] - h_d[te_g, -1:], be_d[te_g], pv_d[te_g], sk_d[te_g], q_d[te_g], mp_d[te_g], sc_t, jac_d[te_g])
                        p0 = ot["paths"][:, 0].cpu().numpy()
                        d_p = np.linalg.norm(p0 - truth_np[te], axis=-1)
                        arr_e[si, te] = np.column_stack((d_p.mean(1), d_p[:, -1]))
                        if u_pose:
                            gates_m1[si, te] = ot["gate_upper"].cpu().numpy()

        em0, em1, gm1 = err_m0.mean(0), err_m1.mean(0), gates_m1.mean(0)
        q1_np = dd["quality"][:, 0].numpy()
        rho, _ = spearmanr(gm1, q1_np)

        def _s4(arr2: np.ndarray, msk: np.ndarray) -> dict[str, float]:
            gs = [arr2[msk & (site_arr == g)].mean(0) for g in sorted(set(site_arr.tolist())) if np.any(msk & (site_arr == g))]
            v = np.mean(gs, axis=0)
            return {"ade_4site_m": float(v[0]), "fde_4site_m": float(v[1]), "ade_pool_m": float(arr2[msk, 0].mean()), "fde_pool_m": float(arr2[msk, 1].mean())}

        all_m = np.ones(n, dtype=bool)
        by_dur[f"T={dur}"] = {
            "spearman_rho_gate_upper_vs_q1": float(rho),
            "all_623": {
                "Last2_CV": _s4(err_cv, all_m), "PCR6_M0": _s4(em0, all_m), "PCR6_M1": _s4(em1, all_m),
                "paired_m1_minus_m0_pool": _pooled_clip_bootstrap_ci(em1 - em0, clips),
                "paired_m1_minus_CV_4site": safe_site_clip_ci(em1 - err_cv, site_arr, clips),
            },
            "quality_buckets": {
                bn: {"n": int((qb5 == bi).sum()), "Last2_CV": _s4(err_cv, qb5 == bi), "PCR6_M0": _s4(em0, qb5 == bi), "PCR6_M1": _s4(em1, qb5 == bi)}
                for bi, bn in enumerate(("Clear", "Partial", "Heavy")) if np.any(qb5 == bi)
            },
        }
        print(f"[Wave 30] Finished True-20f duration T={dur}: PCR6_M1 4Site={by_dur[f'T={dur}']['all_623']['PCR6_M1']['ade_4site_m']:.4f} rho={rho:.4f}", flush=True)
    return by_dur


def run_zeroshot_imptc_pcr6(
    meta_dir: Path,
    pose_npz: Path,
    joint_zip: Path,
    causal_map_bin: Path,
    imptc_tar_path: Path,
    device: torch.device,
    epochs: int = 32,
) -> dict:
    dp = load_ppt_cohort_with_causal_maps(meta_dir, pose_npz, joint_zip, causal_map_bin)
    np_w = len(dp["history"])
    gd = {k: dp[k].to(device) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
    gd["obs_dt"] = torch.full((np_w, 4), 0.1, dtype=torch.float64, device=device)
    gd["fut_s"] = torch.arange(1, 31, dtype=torch.float64, device=device)[None].expand(np_w, -1) * 0.1
    ya_z = torch.zeros((np_w, 5), dtype=torch.bool, device=device); ya_z[:, 4] = True
    va_z = torch.ones((np_w, 5), dtype=torch.bool, device=device)
    pw_z = torch.ones(5, device=device)
    with torch.inference_mode():
        sc_p = compute_split_anatomical_covisibility(None, None, gd["bio_even"], gd["pose_vec_world"], gd["signed_knee"], gd["quality"])

    imptc_tar = ensure_imptc_sample(imptc_tar_path)
    imp = adapt_imptc_cohort(load_imptc_with_neighbors(imptc_tar))
    ni = len(imp["history"])
    hi = torch.from_numpy(imp["history"]).to(device)
    odi = torch.from_numpy(imp["obs_dt"]).to(device)
    fsi = torch.from_numpy(imp["fut_s"]).to(device)
    edi = torch.from_numpy(imp["edges"][:, :, :8]).to(device)
    ici = torch.from_numpy(imp["is_car"]).to(device)
    nmi = torch.from_numpy(imp["nbr_mask"]).to(device)
    be_i = torch.zeros((ni, 5, 11), device=device)
    pv_i = torch.zeros((ni, 5, 2, 2), dtype=torch.float64, device=device)
    sk_i = torch.zeros((ni, 5, 1), device=device)
    qv_i = torch.zeros((ni, 4), device=device)
    mp_i = torch.zeros(ni, device=device)  # strict zero-pose mask!

    err_m0 = np.zeros((len(SEEDS), ni, 2), dtype=np.float64)
    err_m2b = np.zeros((len(SEEDS), ni, 2), dtype=np.float64)
    max_pose_swing = 0.0
    max_o2_rot_err = 0.0

    for si, seed in enumerate(SEEDS):
        g_tr = dp["site"]
        ug = sorted(set(g_tr.tolist()))
        w_np = np.zeros(np_w, dtype=np.float64)
        for g in ug:
            mg = g_tr == g
            w_np[mg] = np_w / (len(ug) * max(int(mg.sum()), 1))
        w_all = torch.from_numpy(w_np).to(device)
        tr_all = np.arange(np_w)

        for u_nbr, arr_i in ((False, err_m0), (True, err_m2b)):
            torch.manual_seed(seed + (211 if u_nbr else 0))
            net = O2PassivityRay6SlotNet(True, u_nbr, False).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=1.8e-3, weight_decay=1.5e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            rng = np.random.default_rng(seed + 55)
            for _ in range(epochs):
                perm = rng.permutation(tr_all)
                for st in range(0, np_w, 128):
                    bt = torch.from_numpy(perm[st : st + 128]).to(device)
                    sc_b = {k: v[bt] for k, v in sc_p.items()}
                    o = net(gd["history"][bt], gd["bio_even"][bt], gd["pose_vec_world"][bt], gd["signed_knee"][bt], gd["quality"][bt], gd["m_pose"][bt], sc_b, None, None, None, gd["edges"][bt], gd["is_car"][bt], gd["nbr_mask"][bt], gd["obs_dt"][bt], gd["fut_s"][bt])
                    loss = ekr6_decoupled_slot0_and_maneuver_loss(o, gd["truth"][bt], gd["history"][bt], w_all[bt], ya_z[bt], va_z[bt], pw_z, act_weight=0.0)
                    if loss is not None:
                        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                sch.step()
            net.eval()
            with torch.inference_mode():
                p_imp = net(hi, be_i, pv_i, sk_i, qv_i, mp_i, None, None, None, None, edi, ici, nmi, odi, fsi)["paths"][:, 0].cpu().numpy()
                d_i = np.linalg.norm(p_imp - imp["truth"], axis=-1)
                arr_i[si] = np.column_stack((d_i.mean(1), d_i[:, -1]))
                if si == 0 and u_nbr:
                    # 5-axis synthetic pose swing check when m_pose == 0 (must be exact 0.0m!)
                    p_sw = net(hi, torch.randn_like(be_i) * 5.0, torch.randn_like(pv_i) * 2.0, torch.randn_like(sk_i) * 3.0, torch.ones_like(qv_i), mp_i, None, None, None, None, edi, ici, nmi, odi, fsi)["paths"][:, 0].cpu().numpy()
                    max_pose_swing = max(max_pose_swing, float(np.abs(p_sw - p_imp).max()))
                    # 5-angle O(2) equivariance check on IMPTC
                    for _, rm in ROT5:
                        rt = torch.from_numpy(rm).to(device)
                        ed_r = edi.clone()
                        ed_r[:, :, 0:2] = (edi[:, :, 0:2].double() @ rt.T).float()
                        ed_r[:, :, 2:4] = (edi[:, :, 2:4].double() @ rt.T).float()
                        pr = net(hi @ rt.T, be_i, pv_i, sk_i, qv_i, mp_i, None, None, None, None, ed_r, ici, nmi, odi, fsi)["paths"][:, 0].cpu().numpy()
                        max_o2_rot_err = max(max_o2_rot_err, float(np.abs(p_imp @ rm.T - pr).max()))

    em0, em2 = err_m0.mean(0), err_m2b.mean(0)
    ecv, grp_i = imp["cv_err"], imp["group"]
    pl_i = np.linalg.norm(np.diff(imp["history"], axis=1), axis=-1).sum(axis=1)
    car_app = ((imp["nbr_mask"] * imp["is_car"] * (imp["edges"][:, :, 5] > 0.20)).max(axis=1)) > 0.5
    strata_i = {"all_564": np.ones(ni, dtype=bool), "car_approaching": car_app, "lowdisp_lt0p5m": pl_i < 0.50}
    out_i = {}
    for sk, sm in strata_i.items():
        if not sm.any():
            continue
        out_i[sk] = {
            "n_windows": int(sm.sum()),
            "Last2_CV": {"ade_m": float(ecv[sm, 0].mean()), "fde_m": float(ecv[sm, 1].mean())},
            "PCR6_M0_ZeroShot": {"ade_m": float(em0[sm, 0].mean()), "fde_m": float(em0[sm, 1].mean())},
            "PCR6_M2b_Conflict_ZeroShot": {"ade_m": float(em2[sm, 0].mean()), "fde_m": float(em2[sm, 1].mean())},
            "paired_m2b_minus_CV": _loc_bootstrap_ci(em2[sm] - ecv[sm], grp_i[sm]),
            "paired_m2b_minus_m0": _loc_bootstrap_ci(em2[sm] - em0[sm], grp_i[sm]),
        }
    return {
        "zero_pose_5axis_max_swing_m": max_pose_swing,
        "o2_5angle_equivariance_max_err_m": max_o2_rot_err,
        "strata": out_i,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matched-npz", type=Path, required=True)
    ap.add_argument("--matched-obs-npz", type=Path, required=True)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--imptc-tar", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, default=Path("wave30_multidur_and_imptc_pcr6.zip"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    if not torch.cuda.is_available():
        torch.set_num_threads(4)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Wave 30] Running True-20f Multi-Duration (T=2,5,8,10,20) + Zero-Shot IMPTC PCR6 on {dev}...", flush=True)
    md = run_true20f_pcr6_multiduration(args.matched_npz, args.meta_dir, args.pose_npz, args.matched_obs_npz, dev)
    im = run_zeroshot_imptc_pcr6(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin, args.imptc_tar, dev)
    summary = {
        "wave": "wave30_true20f_multiduration_and_imptc_zeroshot_pcr6_5seed",
        "elapsed_s": float(time.perf_counter() - t0),
        "true20f_multiduration_623": md,
        "imptc_564_zeroshot_transfer": im,
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
