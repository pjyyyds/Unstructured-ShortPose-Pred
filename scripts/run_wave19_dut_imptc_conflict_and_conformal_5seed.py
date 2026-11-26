"""Wave 19 GPU0: Zero-shot causal conflict-gated O(2) interaction & zero-retrain split-conformal on DUT (3,066) & IMPTC (564)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.run_dair_o2_bioeq_external import SEEDS
from scripts.run_imptc_and_dair_o2_full_external import ensure_imptc_sample, load_imptc_with_neighbors
from scripts.run_jrdb_rc1_pose_quality_and_initiation_5seed import _loc_bootstrap_ci
from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import _stage_gpu_dict
from scripts.run_pinns_conformal_and_events import conformal_quantile
from scripts.run_pinns_matched_o2_multiduration import SITES
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import load_ppt_cohort_with_causal_maps
from scripts.run_wave18_conflict_gated_interaction_5seed import (
    ConflictGatedM2bForecaster,
    _fit_base_and_conflict,
    build_interaction_strata,
    select_inner_conflict_scale,
)

ROT5 = (
    ("0deg", np.array([[1.0, 0.0], [0.0, 1.0]])),
    ("90deg", np.array([[0.0, -1.0], [1.0, 0.0]])),
    ("180deg", np.array([[-1.0, 0.0], [0.0, -1.0]])),
    ("270deg", np.array([[0.0, 1.0], [-1.0, 0.0]])),
    ("mirror_y", np.array([[1.0, 0.0], [0.0, -1.0]])),
)


def load_dut_graph_cohort(dut_zip: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(dut_zip) as zf:
        pg = np.load(io.BytesIO(zf.read("private_graph.npz")), allow_pickle=False)
        et = np.load(io.BytesIO(zf.read("errors_and_targets.npz")), allow_pickle=False)
        keys = json.loads(str(pg["keys_json"]))
        clips = np.array([str(k[0]) for k in keys], dtype=str)
        h_rel = (pg["history"] - pg["history"][:, 4:5]).astype(np.float64)
        frames = pg["frames"].astype(np.int64)
        e_raw = pg["edges"].astype(np.float32)
        kind = pg["source"]
        truth = et["truth"].astype(np.float64)
    if e_raw.shape[-1] < 9:
        pad = np.zeros((*e_raw.shape[:-1], 9 - e_raw.shape[-1]), dtype=np.float32)
        e_raw = np.concatenate((e_raw, pad), axis=-1)
    offsets = np.array([(j * 23976 + 4999) // 10000 for j in range(35)], dtype=np.int64)
    obs_dt = np.diff(frames, axis=1) * (1000.0 / 23976.0)
    fut_s = np.broadcast_to((offsets[5:] - offsets[4]) * (1000.0 / 23976.0), (len(frames), 30)).copy()
    v_last = (h_rel[:, -1:] - h_rel[:, -2:-1]) / np.maximum(obs_dt[:, -1:, None], 1e-4)
    cv_pred = v_last * fut_s[:, :, None]
    d_cv = np.linalg.norm(cv_pred - truth, axis=-1)
    return {
        "group": clips,
        "history": h_rel,
        "truth": truth,
        "obs_dt": obs_dt,
        "fut_s": fut_s,
        "edges": e_raw,
        "is_car": (kind == 1).astype(np.float32),
        "nbr_mask": (kind >= 0).astype(np.float32),
        "cv_err": np.column_stack((d_cv.mean(1), d_cv[:, -1])),
        "cv_tube": d_cv.max(1),
    }


def adapt_imptc_cohort(raw: dict) -> dict[str, np.ndarray]:
    ts = raw["ts_us"]
    obs_dt = ((ts[:, 1:5] - ts[:, 0:4]) / 1e6).astype(np.float64)
    fut_s = ((ts[:, 5:35] - ts[:, 4:5]) / 1e6).astype(np.float64)
    h_rel = raw["history"].astype(np.float64)
    truth = raw["truth"].astype(np.float64)
    v_last = (h_rel[:, -1:] - h_rel[:, -2:-1]) / np.maximum(obs_dt[:, -1:, None], 1e-4)
    d_cv = np.linalg.norm(v_last * fut_s[:, :, None] - truth, axis=-1)
    return {
        "group": raw["seq"].astype(str),
        "history": h_rel,
        "truth": truth,
        "obs_dt": obs_dt,
        "fut_s": fut_s,
        "edges": raw["edges"].astype(np.float32),
        "is_car": raw["is_car"].astype(np.float32),
        "nbr_mask": raw["nbr_mask"].astype(np.float32),
        "cv_err": np.column_stack((d_cv.mean(1), d_cv[:, -1])),
        "cv_tube": d_cv.max(1),
    }


def _eval_zero_retrain_conformal(tubes_by_arm: dict[str, np.ndarray], cv_tube: np.ndarray, grp: np.ndarray) -> dict:
    """Strict zero-retrain split-conformal on external benchmark: within-group 50/50 split & all ordered cross-group pairs."""
    u_grps = [g for g in sorted(set(grp.tolist())) if int((grp == g).sum()) >= 16]
    n_seeds = next(iter(tubes_by_arm.values())).shape[0]
    within = {a: {l: {"cov_cv": [], "rad_cv": [], "cov_t1": [], "rad_t1": [], "cov_u6": [], "rad_u6": []} for l in ("90", "95")} for a in tubes_by_arm}
    cross = {a: {l: {"cov_cv": [], "rad_cv": [], "cov_t1": [], "rad_t1": [], "cov_u6": [], "rad_u6": []} for l in ("90", "95")} for a in tubes_by_arm}

    for si in range(n_seeds):
        for g in u_grps:
            ids = np.flatnonzero(grp == g)
            mid = len(ids) // 2
            ci, ti = ids[:mid], ids[mid:]
            for a, tb_all in tubes_by_arm.items():
                tb = tb_all[si]
                for lvl, alpha in (("90", 0.10), ("95", 0.05)):
                    rcv = conformal_quantile(cv_tube[ci], alpha)
                    rt1 = conformal_quantile(tb[ci, 0], alpha)
                    ru6 = conformal_quantile(tb[ci, 1], alpha)
                    within[a][lvl]["cov_cv"].append(float((cv_tube[ti] <= rcv).mean()))
                    within[a][lvl]["rad_cv"].append(rcv)
                    within[a][lvl]["cov_t1"].append(float((tb[ti, 0] <= rt1).mean()))
                    within[a][lvl]["rad_t1"].append(rt1)
                    within[a][lvl]["cov_u6"].append(float((tb[ti, 1] <= ru6).mean()))
                    within[a][lvl]["rad_u6"].append(ru6)

        for i in range(len(u_grps)):
            for j in range(len(u_grps)):
                if i == j:
                    continue
                ci = np.flatnonzero(grp == u_grps[i])
                ti = np.flatnonzero(grp == u_grps[j])
                for a, tb_all in tubes_by_arm.items():
                    tb = tb_all[si]
                    for lvl, alpha in (("90", 0.10), ("95", 0.05)):
                        rcv = conformal_quantile(cv_tube[ci], alpha)
                        rt1 = conformal_quantile(tb[ci, 0], alpha)
                        ru6 = conformal_quantile(tb[ci, 1], alpha)
                        cross[a][lvl]["cov_cv"].append(float((cv_tube[ti] <= rcv).mean()))
                        cross[a][lvl]["rad_cv"].append(rcv)
                        cross[a][lvl]["cov_t1"].append(float((tb[ti, 0] <= rt1).mean()))
                        cross[a][lvl]["rad_t1"].append(rt1)
                        cross[a][lvl]["cov_u6"].append(float((tb[ti, 1] <= ru6).mean()))
                        cross[a][lvl]["rad_u6"].append(ru6)

    return {
        "n_eligible_groups": len(u_grps),
        "n_ordered_cross_pairs": len(u_grps) * max(len(u_grps) - 1, 0),
        "within_group_temporal_50_50": {a: {l: {k: float(np.mean(v)) for k, v in d.items()} for l, d in wd.items()} for a, wd in within.items()},
        "cross_group_ordered_pairs_no_retrain": {a: {l: {k: float(np.mean(v)) for k, v in d.items()} for l, d in cd.items()} for a, cd in cross.items()},
    }


@torch.inference_mode()
def evaluate_external_conflict_and_conformal(
    cohort: dict[str, np.ndarray],
    seed_ensembles: list[list[dict[str, torch.nn.Module]]],
    device: torch.device,
) -> dict:
    n = len(cohort["group"])
    grp = cohort["group"]
    truth = cohort["truth"]
    obs_dt = torch.from_numpy(cohort["obs_dt"]).to(device)
    fut_s = torch.from_numpy(cohort["fut_s"]).to(device)
    car_t = torch.from_numpy(cohort["is_car"]).to(device)
    msk_t = torch.from_numpy(cohort["nbr_mask"]).to(device)
    arms = ("M1", "M2b_frozen_head", "M2b_ttc_conflict_gate")

    top_s = {a: np.zeros((len(seed_ensembles), n, 2), dtype=np.float64) for a in arms}
    min_s = {a: np.zeros((len(seed_ensembles), n, 2), dtype=np.float64) for a in arms}
    tubes_s = {a: np.zeros((len(seed_ensembles), n, 2), dtype=np.float64) for a in arms}

    h0 = torch.from_numpy(cohort["history"]).to(device)
    e0 = torch.from_numpy(cohort["edges"]).to(device)
    for si, fold_list in enumerate(seed_ensembles):
        for a in arms:
            p_acc, l_acc = [], []
            for fm in fold_list:
                oo = fm[a](h0, edges=e0 if a != "M1" else None, is_car=car_t if a != "M1" else None,
                           nbr_mask=msk_t if a != "M1" else None, obs_dt=obs_dt, fut_times_s=fut_s)
                p_acc.append(oo["paths"].cpu().numpy())
                l_acc.append(oo["logits"].softmax(1).cpu().numpy())
            pw = np.mean(p_acc, axis=0)
            pr = np.mean(l_acc, axis=0)
            top_idx = pr.argmax(axis=1)
            d_all = np.linalg.norm(pw - truth[:, None], axis=-1)
            d_top = d_all[np.arange(n), top_idx]
            top_s[a][si] = np.column_stack((d_top.mean(1), d_top[:, -1]))
            min_s[a][si] = np.column_stack((d_all.mean(2).min(1), d_all[:, :, -1].min(1)))
            tubes_s[a][si] = np.column_stack((d_top.max(1), d_all.max(2).min(1)))

    # 5-axis exact O(2) equivariance swing check on seed-0 fold-0 M2b_ttc_conflict_gate
    cg0 = seed_ensembles[0][0]["M2b_ttc_conflict_gate"]
    ax_ades = []
    for _, op in ROT5:
        hr = torch.from_numpy(cohort["history"] @ op.T).to(device)
        er = cohort["edges"].copy()
        er[:, :, 0:2] = (er[:, :, 0:2].astype(np.float64) @ op.T).astype(np.float32)
        er[:, :, 2:4] = (er[:, :, 2:4].astype(np.float64) @ op.T).astype(np.float32)
        oo = cg0(hr, edges=torch.from_numpy(er).to(device), is_car=car_t, nbr_mask=msk_t, obs_dt=obs_dt, fut_times_s=fut_s)
        pw = oo["paths"].cpu().numpy() @ op
        top_p = pw[np.arange(n), oo["logits"].argmax(1).cpu().numpy()]
        ax_ades.append(float(np.linalg.norm(top_p - truth, axis=-1).mean()))
    five_axis_swing = float(max(ax_ades) - min(ax_ades))

    top_m = {a: top_s[a].mean(0) for a in arms}
    min_m = {a: min_s[a].mean(0) for a in arms}
    strata = build_interaction_strata(cohort["edges"], cohort["is_car"], cohort["nbr_mask"], has_cars=True)
    strata_out = {}
    for sk, sm in strata.items():
        if int(sm.sum()) == 0:
            continue
        gs = sorted(set(grp[sm].tolist()))
        mods = {
            "Last2_CV": {
                "group_equal": {"ade1_m": float(np.mean([cohort["cv_err"][(grp == g) & sm, 0].mean() for g in gs])),
                                "fde1_m": float(np.mean([cohort["cv_err"][(grp == g) & sm, 1].mean() for g in gs]))},
                "window_equal": {"ade1_m": float(cohort["cv_err"][sm, 0].mean()), "fde1_m": float(cohort["cv_err"][sm, 1].mean())},
            }
        }
        for a in arms:
            mods[a] = {
                "group_equal": {
                    "ade1_m": float(np.mean([top_m[a][(grp == g) & sm, 0].mean() for g in gs])),
                    "fde1_m": float(np.mean([top_m[a][(grp == g) & sm, 1].mean() for g in gs])),
                    "min_ade6_m": float(np.mean([min_m[a][(grp == g) & sm, 0].mean() for g in gs])),
                    "min_fde6_m": float(np.mean([min_m[a][(grp == g) & sm, 1].mean() for g in gs])),
                },
                "window_equal": {
                    "ade1_m": float(top_m[a][sm, 0].mean()),
                    "fde1_m": float(top_m[a][sm, 1].mean()),
                    "min_ade6_m": float(min_m[a][sm, 0].mean()),
                    "min_fde6_m": float(min_m[a][sm, 1].mean()),
                },
            }
        paired = {
            "M2b_frozen_minus_M1": _loc_bootstrap_ci(top_m["M2b_frozen_head"][sm] - top_m["M1"][sm], grp[sm]),
            "M2b_conflict_gate_minus_M1": _loc_bootstrap_ci(top_m["M2b_ttc_conflict_gate"][sm] - top_m["M1"][sm], grp[sm]),
            "M2b_conflict_gate_minus_M2b_frozen": _loc_bootstrap_ci(top_m["M2b_ttc_conflict_gate"][sm] - top_m["M2b_frozen_head"][sm], grp[sm]),
            "M2b_conflict_gate_minus_CV": _loc_bootstrap_ci(top_m["M2b_ttc_conflict_gate"][sm] - cohort["cv_err"][sm], grp[sm]),
        }
        strata_out[sk] = {"n_windows": int(sm.sum()), "n_groups_present": len(gs), "models": mods, "paired_top1_ci": paired}

    return {
        "n_windows": n,
        "n_groups": len(set(grp.tolist())),
        "max_five_axis_ade1_swing_m": five_axis_swing,
        "strata": strata_out,
        "zero_retrain_split_conformal": _eval_zero_retrain_conformal(tubes_s, cohort["cv_tube"], grp),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--dut-zip", type=Path, required=True)
    ap.add_argument("--imptc-tar", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = load_ppt_cohort_with_causal_maps(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin)
    np_w = len(dp["history"])
    gdp = {k: dp[k].to(device) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
    gdp["obs_dt"] = torch.full((np_w, 4), 0.1, dtype=torch.float64, device=device)
    gdp["fut_s"] = torch.arange(1, 31, dtype=torch.float64, device=device)[None].expand(np_w, -1) * 0.1
    gdp["p_env"] = torch.zeros((np_w, 16, 10, 11), dtype=torch.float32, device=device)
    gdp["m_map"] = torch.zeros(np_w, dtype=torch.float32, device=device)

    seed_ensembles: list[list[dict[str, torch.nn.Module]]] = []
    sel_scales: dict[str, dict[str, float]] = {}
    for seed in args.seeds:
        fold_list = []
        sel_scales[str(seed)] = {}
        for s in SITES:
            tr = np.flatnonzero(dp["site"] != s)
            sc = select_inner_conflict_scale(seed, dp["site"], gdp, tr, 64, (45, 28, 28), (2.5e-3, 1.5e-3, 1.8e-3), device)
            mods = _fit_base_and_conflict(seed, dp["site"], gdp, tr, 64, (45, 28, 28), (2.5e-3, 1.5e-3, 1.8e-3), device)
            cg = mods["M2b_ttc_conflict_gate"]
            assert isinstance(cg, ConflictGatedM2bForecaster)
            cg.graph_scale = sc
            sel_scales[str(seed)][s] = sc
            fold_list.append({k: mods[k] for k in ("M1", "M2b_frozen_head", "M2b_ttc_conflict_gate")})
        seed_ensembles.append(fold_list)
        print(f"[W19-GPU0] Trained PINNS 4-site ensemble seed={seed}", flush=True)

    dut_res = evaluate_external_conflict_and_conformal(load_dut_graph_cohort(args.dut_zip), seed_ensembles, device)
    imptc_tar = ensure_imptc_sample(args.imptc_tar)
    imptc_res = evaluate_external_conflict_and_conformal(adapt_imptc_cohort(load_imptc_with_neighbors(imptc_tar)), seed_ensembles, device)

    summary = {
        "experiment": "wave19_dut_imptc_conflict_and_conformal_5seed",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "seeds": list(args.seeds),
        "pinns_source_selected_scales": sel_scales,
        "dut_3066_zero_shot": dut_res,
        "imptc_564_zero_shot": imptc_res,
    }
    raw = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = (json.dumps({"summary.json": hashlib.sha256(raw).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw)
        zf.writestr("sha256.json", sh)
    print("Saved:", args.zip_path, flush=True)


if __name__ == "__main__":
    main()
