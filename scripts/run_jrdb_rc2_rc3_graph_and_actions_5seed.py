"""Wave 16 GPU: Fast CUDA-prestaged JRDB 10-scene (5,836 windows, 9-location LOLO, 5 seeds) RC2 & RC3 (M1->M2b(±v_rad)->M3a/M3b/M3)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

from scripts.ppt_mainline_o2_models import PPTMainlineO2Forecaster
from scripts.run_dair_o2_bioeq_external import SEEDS
from scripts.run_jrdb_outdoor_o2_full_5seed import _jrdb_edges


def fast_person_bootstrap_ci(diff: np.ndarray, mask: np.ndarray, actors: np.ndarray, seed: int = 2026, n_boot: int = 20000) -> dict:
    ids = np.flatnonzero(mask)
    uniq, inv = np.unique(actors[ids], return_inverse=True)
    k = len(uniq)
    d_sub = diff[ids].astype(np.float64)
    cnts = np.bincount(inv, minlength=k).astype(np.float64)[:, None]
    sums = np.column_stack([np.bincount(inv, weights=d_sub[:, j], minlength=k) for j in range(2)])
    means = sums / cnts
    pt_w, pt_p = d_sub.mean(axis=0), means.mean(axis=0)
    rng = np.random.default_rng(seed)
    dw = np.empty((n_boot, 2), dtype=np.float64)
    dp = np.empty((n_boot, 2), dtype=np.float64)
    for s in range(0, n_boot, 2000):
        e = min(n_boot, s + 2000)
        idx = rng.integers(0, k, size=(e - s, k))
        dw[s:e] = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
        dp[s:e] = means[idx].mean(axis=1)
    lw, hw = np.percentile(dw, [2.5, 97.5], axis=0)
    lp, hp = np.percentile(dp, [2.5, 97.5], axis=0)
    return {
        "window_equal": {"mean": {"ade_m": float(pt_w[0]), "fde_m": float(pt_w[1])}, "ci95": {"ade_m": [float(lw[0]), float(hw[0])], "fde_m": [float(lw[1]), float(hw[1])]}},
        "person_equal": {"mean": {"ade_m": float(pt_p[0]), "fde_m": float(pt_p[1])}, "ci95": {"ade_m": [float(lp[0]), float(hp[0])], "fde_m": [float(lp[1]), float(hp[1])]}},
    }
from scripts.run_jrdb_rc1_pose_quality_and_initiation_5seed import (
    FUTURE_STEPS,
    OUTDOOR_SCENES,
    POSE_OFFSETS,
    STEP,
    _loc_bootstrap_ci,
    _scene_location,
    build_jrdb_3d_pose_o2_features,
)
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_o2_joint_actions_5seed import ACTIONS
from src.data.actions import compute_action_labels
from src.metrics.metrics_and_stats import _binary_average_precision
from src.models.bioeq_o2_joint import o2_joint_loss


def load_jrdb_10scene_rc2_rc3(repo_root: Path) -> dict:
    jrdb_dir = repo_root / "reviews/jrdb_development"
    if str(jrdb_dir) not in sys.path:
        sys.path.insert(0, str(jrdb_dir))
    import check_training_coordinates as ctc  # type: ignore
    import evaluate_outdoor_motion as eom  # type: ignore

    odom_spec = json.loads((jrdb_dir / "train_split_odometry.json").read_text())["recordings"]
    tr_arc = ctc.TrainingArchive(*eom.ARCHIVES["tracks"])
    ps_arc = ctc.TrainingArchive(*eom.ARCHIVES["poses"])
    od_arc = ctc.TrainingArchive(*eom.ARCHIVES["odom"])

    hist_l, truth_l, joints_l, edges_l, nmsk_l = [], [], [], [], []
    ya_l, va_l, scene_l, loc_l, actor_l = [], [], [], [], []
    hm2 = np.ones(2, dtype=np.float32)
    fm7 = np.ones(FUTURE_STEPS, dtype=np.float32)

    for sc, meta in sorted(odom_spec.items()):
        b_od = od_arc.member(eom.MEMBERS["odom"].format(sc))
        if hashlib.sha256(b_od).hexdigest() != meta["member_sha256"]:
            raise ValueError(f"Odometry SHA-256 mismatch on {sc}")
        tracks, _ = eom.read_tracks(tr_arc.member(eom.MEMBERS["tracks"].format(sc)))
        poses: dict[tuple[int, int], np.ndarray] = {}
        for img_nm, plist in json.loads(ps_arc.member(eom.MEMBERS["poses"].format(sc)))["labels"].items():
            fr = int(Path(img_nm).stem)
            for obj in plist:
                if obj["label_id"].startswith("pedestrian:"):
                    poses[(int(obj["label_id"].split(":")[1]), fr)] = np.asarray(obj["keypoints"], dtype=np.float64).reshape(33, 3)

        by_fr: dict[int, dict[int, np.ndarray]] = {}
        for p, rows in tracks.items():
            for f, xy in rows.items():
                by_fr.setdefault(int(f), {})[int(p)] = np.asarray(xy, dtype=np.float64)

        for person, rows in sorted(tracks.items()):
            times = np.asarray(sorted(rows))
            for run in np.split(times, np.flatnonzero(np.diff(times) != STEP) + 1):
                for off in range(0, len(run) - FUTURE_STEPS - 1, 2 + FUTURE_STEPS):
                    ch = run[off : off + 2 + FUTURE_STEPS]
                    xy = np.stack([np.asarray(rows[int(t)], dtype=np.float64) for t in ch])
                    t0 = int(ch[1])
                    j5 = np.full((5, 33, 3), np.nan, dtype=np.float64)
                    for k, d_off in enumerate(POSE_OFFSETS):
                        if (int(person), t0 + d_off) in poses:
                            j5[k] = poses[(int(person), t0 + d_off)]
                    h_rel, f_rel = xy[:2] - xy[1:2], xy[2:] - xy[1:2]
                    ec, mc = _jrdb_edges(by_fr, int(ch[0]), t0, int(person))
                    yi, mi = compute_action_labels(h_rel, hm2, f_rel, fm7, dt=0.4)
                    v_pre = float(np.linalg.norm(h_rel[1] - h_rel[0]) / 0.4)
                    va = np.zeros(5, dtype=bool)
                    if mi > 0.5:
                        va[:] = True
                        va[0] = v_pre <= 0.20
                        va[1] = v_pre >= 0.60
                    hist_l.append(h_rel); truth_l.append(f_rel); joints_l.append(j5)
                    edges_l.append(ec); nmsk_l.append(mc)
                    ya_l.append(yi > 0.5); va_l.append(va)
                    scene_l.append(sc); loc_l.append(_scene_location(sc)); actor_l.append(f"{sc}:{person}")

    hist_np, truth_np = np.stack(hist_l), np.stack(truth_l)
    j3d_np = np.stack(joints_l)
    pf = build_jrdb_3d_pose_o2_features(j3d_np)
    n = len(hist_np)
    sc_arr = np.asarray(scene_l, dtype=str)
    return {
        "history": torch.from_numpy(hist_np), "truth": torch.from_numpy(truth_np), "joints_3d": j3d_np,
        "edges": torch.from_numpy(np.stack(edges_l)), "is_car": torch.zeros((n, 12), dtype=torch.float32),
        "nbr_mask": torch.from_numpy(np.stack(nmsk_l)),
        "y_act": torch.from_numpy(np.stack(ya_l)), "valid_act": torch.from_numpy(np.stack(va_l)),
        "scene": sc_arr, "location": np.asarray(loc_l, dtype=str), "actors": np.asarray(actor_l, dtype=str),
        **pf,
        "strata": {
            "all_10scenes": np.ones(n, dtype=bool),
            "outdoor_3scenes_1677": np.isin(sc_arr, list(OUTDOOR_SCENES)),
            "with_neighbors": (np.stack(nmsk_l).sum(axis=1) > 0),
            "short_disp_lt05": np.linalg.norm(hist_np[:, 1] - hist_np[:, 0], axis=-1) < 0.50,
            "clear_pose": pf["q_buckets"] == 0,
            "partial_pose": pf["q_buckets"] == 1,
            "missing_pose": pf["q_buckets"] == 2,
            "short_disp_clear": (np.linalg.norm(hist_np[:, 1] - hist_np[:, 0], axis=-1) < 0.50) & (pf["q_buckets"] == 0),
        },
    }


def _stage_gpu_dict(data: dict, device: torch.device) -> dict:
    keys = ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask", "y_act", "valid_act")
    gd = {k: data[k].to(device) for k in keys}
    n = len(data["history"])
    gd["obs_dt"] = torch.full((n, 1), 0.4, dtype=torch.float64, device=device)
    gd["fut_s"] = torch.arange(1, 8, dtype=torch.float64, device=device)[None, :].expand(n, -1) * 0.4
    gd["p_env"] = torch.zeros((n, 16, 10, 11), dtype=torch.float32, device=device)
    gd["m_map"] = torch.zeros(n, dtype=torch.float32, device=device)
    return gd


def _fwd_gd(net: PPTMainlineO2Forecaster, gd: dict, ids: torch.Tensor) -> dict:
    return net(
        gd["history"][ids], gd["bio_even"][ids], gd["pose_vec_world"][ids], gd["signed_knee"][ids],
        gd["quality"][ids], gd["m_pose"][ids], gd["p_env"][ids], gd["m_map"][ids],
        gd["edges"][ids], gd["is_car"][ids], gd["nbr_mask"][ids],
        obs_dt=gd["obs_dt"][ids], fut_times_s=gd["fut_s"][ids],
    )


def fit_jrdb_rc2_rc3_fold(seed: int, loc_arr: np.ndarray, gd: dict, tr_ids: np.ndarray, device: torch.device) -> dict[str, PPTMainlineO2Forecaster]:
    w_tr = make_site_weights(loc_arr, tr_ids).to(device)
    tr_t = torch.from_numpy(tr_ids).to(device)
    truth_tr = gd["truth"][tr_t]
    ya_tr, va_tr = gd["y_act"][tr_t], gd["valid_act"][tr_t]
    pos = (ya_tr & va_tr).sum(0).float().clamp_min(1.0)
    neg = ((~ya_tr) & va_tr).sum(0).float().clamp_min(1.0)
    pw = (neg / pos).clamp(1.0, 12.0)

    def _copy(dst: PPTMainlineO2Forecaster, src: PPTMainlineO2Forecaster) -> None:
        sd = dst.state_dict()
        for k, v in src.state_dict().items():
            if k in sd and sd[k].shape == v.shape:
                sd[k].copy_(v)
        dst.load_state_dict(sd)

    def _train(net: PPTMainlineO2Forecaster, prefs: tuple[str, ...], ep: int, lr: float, wd: float, aw: float, soff: int) -> PPTMainlineO2Forecaster:
        for nm, p in net.named_parameters():
            p.requires_grad = nm.startswith(prefs) or nm in prefs
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(seed + soff)
        net.train()
        for _ in range(ep):
            for ch in torch.randperm(len(tr_ids), generator=gen).split(512):
                id_g = tr_t[ch.to(device)]
                out = _fwd_gd(net, gd, id_g)
                ch_g = ch.to(device)
                loss = o2_joint_loss(out, truth_tr[ch_g], w_tr[ch_g], ya_tr[ch_g] if aw > 0 else None, va_tr[ch_g] if aw > 0 else None, pw if aw > 0 else None, act_weight=aw)
                if loss is not None:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return net.eval()

    torch.manual_seed(seed)
    m0 = _train(PPTMainlineO2Forecaster("M0").to(device), ("motion.", "ctx_proj.", "cv_low_bonus", "central_", "pair_"), 45, 2.5e-3, 1e-4, 0.0, 0)
    models = {"M0": m0}
    for i, arm in enumerate(("M1a", "M1b", "M1")):
        torch.manual_seed(seed); net = PPTMainlineO2Forecaster(arm).to(device); _copy(net, m0)
        models[arm] = _train(net, ("pose_", "gate_", "central_", "pair_", "cv_low_bonus"), 28, 1.5e-3, 3e-4, 0.0, 10 + i)
    for i, arm in enumerate(("M2b_no_vrad", "M2b")):
        torch.manual_seed(seed); net = PPTMainlineO2Forecaster(arm).to(device); _copy(net, models["M1"])
        models[arm] = _train(net, ("edge_", "graph_proj", "null_logit"), 28, 1.8e-3, 3e-4, 0.0, 20 + i)
    for i, arm in enumerate(("M3a", "M3b", "M3")):
        torch.manual_seed(seed); net = PPTMainlineO2Forecaster(arm).to(device); _copy(net, models["M2b"])
        models[arm] = _train(net, ("ctx_proj.", "act_", "bridge_", "central_", "pair_", "cv_low_bonus"), 30, 1.5e-3, 2e-4, 0.25, 30 + i)
    return models


@torch.inference_mode()
def _eval_gd(net: PPTMainlineO2Forecaster, gd: dict, truth_np: np.ndarray, ids: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    out = _fwd_gd(net, gd, torch.from_numpy(ids).to(device))
    p = out["paths"].cpu().numpy()
    q = out["logits"].softmax(dim=1).cpu().numpy()
    ap = torch.sigmoid(out["act_logits"]).cpu().numpy()
    tr = truth_np[ids]
    dst = np.linalg.norm(p - tr[:, None], axis=-1)
    ade_k, fde_k = dst.mean(2), dst[:, :, -1]
    top = q.argmax(1); ii = np.arange(len(ids))
    bay = np.einsum("nk,nktw->ntw", q, p)
    bd = np.linalg.norm(bay - tr, axis=-1)
    m = np.column_stack((ade_k[ii, top], fde_k[ii, top], ade_k.min(1), fde_k.min(1), bd.mean(1), bd[:, -1], out["gate"].cpu().numpy()))
    return m, ap


def score_jrdb_actions(y_true: np.ndarray, v_mask: np.ndarray, probs: np.ndarray, loc_arr: np.ndarray) -> dict:
    pooled = {}
    for ci, cname in enumerate(ACTIONS):
        vm = v_mask[:, ci]
        yt, ps = y_true[vm, ci], probs[vm, ci]
        pr = ps >= 0.5
        tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
        f1 = (2.0 * tp) / max(2 * tp + fp + fn, 1)
        ap = float(_binary_average_precision(yt.astype(np.float64), ps)) if yt.sum() > 0 else 0.0
        pooled[cname] = {"eligible": int(vm.sum()), "pos": int(yt.sum()), "ap": ap, "f1": float(f1), "brier": float(((ps - yt.astype(np.float64)) ** 2).mean()) if len(yt) else 0.0}
    loc_aps, loc_f1s = [], []
    for l in sorted(set(loc_arr.tolist())):
        lm = loc_arr == l
        aps, f1s = [], []
        for ci in range(5):
            vm = lm & v_mask[:, ci]
            if not vm.any():
                continue
            yt, ps = y_true[vm, ci], probs[vm, ci]
            if yt.sum() > 0:
                aps.append(float(_binary_average_precision(yt.astype(np.float64), ps)))
            pr = ps >= 0.5
            tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
            f1s.append((2.0 * tp) / max(2 * tp + fp + fn, 1))
        if aps:
            loc_aps.append(float(np.mean(aps)))
        if f1s:
            loc_f1s.append(float(np.mean(f1s)))
    return {
        "pooled_macro_ap": float(np.mean([v["ap"] for v in pooled.values()])),
        "pooled_macro_f1": float(np.mean([v["f1"] for v in pooled.values()])),
        "location_equal_macro_ap": float(np.mean(loc_aps)) if loc_aps else 0.0,
        "location_equal_macro_f1": float(np.mean(loc_f1s)) if loc_f1s else 0.0,
        "by_class": pooled,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--output-zip", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_jrdb_10scene_rc2_rc3(args.repo_root)
    gd = _stage_gpu_dict(data, device)
    n = len(data["scene"])
    loc_arr, act_arr = data["location"], data["actors"]
    locs = sorted(set(loc_arr.tolist()))
    arms = ("M0", "M1a", "M1b", "M1", "M2b_no_vrad", "M2b", "M3a", "M3b", "M3")

    hist_np, truth_np = data["history"].numpy(), data["truth"].numpy()
    cv_p = (hist_np[:, 1:] - hist_np[:, :1]) * np.arange(1, 8, dtype=np.float64)[None, :, None]
    d_cv = np.linalg.norm(cv_p - truth_np, axis=-1)
    cv_m = np.column_stack((d_cv.mean(1), d_cv[:, -1], d_cv.mean(1), d_cv[:, -1], d_cv.mean(1), d_cv[:, -1], np.zeros(n)))

    acc = {a: np.zeros((len(args.seeds), n, 7), dtype=np.float64) for a in arms}
    act_acc = {a: np.zeros((len(args.seeds), n, 5), dtype=np.float64) for a in ("M3a", "M3b", "M3")}
    for si, seed in enumerate(args.seeds):
        for loc in locs:
            tr_ids, te_ids = np.flatnonzero(loc_arr != loc), np.flatnonzero(loc_arr == loc)
            mods = fit_jrdb_rc2_rc3_fold(seed, loc_arr, gd, tr_ids, device)
            for a in arms:
                m_out, ap_out = _eval_gd(mods[a], gd, truth_np, te_ids, device)
                acc[a][si, te_ids] = m_out
                if a in act_acc:
                    act_acc[a][si, te_ids] = ap_out
        print(f"Finished fast JRDB M0-M3 chain seed={seed}", flush=True)

    mean_m = {"CV": cv_m, **{a: acc[a].mean(0) for a in arms}}
    cols = ("ade1_m", "fde1_m", "min_ade6_m", "min_fde6_m", "bayes_ade_m", "bayes_fde_m", "mean_gate")
    contrasts = (("M0_minus_CV", "M0", "CV"), ("M1_minus_CV", "M1", "CV"), ("M1a_minus_M0", "M1a", "M0"), ("M1b_minus_M1a", "M1b", "M1a"), ("M1_minus_M1b", "M1", "M1b"), ("M1_minus_M0", "M1", "M0"), ("M2b_minus_M1", "M2b", "M1"), ("M2b_minus_M2b_no_vrad", "M2b", "M2b_no_vrad"), ("M3a_minus_M2b", "M3a", "M2b"), ("M3b_minus_M2b", "M3b", "M2b"), ("M3_minus_M3b", "M3", "M3b"), ("M3_minus_M2b", "M3", "M2b"))

    st_out = {}
    for sn, sm in data["strata"].items():
        if not sm.any():
            continue
        sub_l = loc_arr[sm]; u_locs = sorted(set(sub_l.tolist()))
        md = {}
        for m, arr in mean_m.items():
            sub = arr[sm]
            byl = {l: {c: float(sub[sub_l == l, j].mean()) for j, c in enumerate(cols)} for l in u_locs}
            md[m] = {"location_equal": {c: float(np.mean([v[c] for v in byl.values()])) for c in cols}, "pooled": {c: float(sub[:, j].mean()) for j, c in enumerate(cols)}}
        st_out[sn] = {
            "n_windows": int(sm.sum()), "models": md,
            "paired_top1_location_ci": {cn: _loc_bootstrap_ci(mean_m[ma][sm, :2] - mean_m[mb][sm, :2], sub_l) for cn, ma, mb in contrasts},
            "paired_top1_person_ci": {cn: fast_person_bootstrap_ci(mean_m[ma][:, :2] - mean_m[mb][:, :2], sm, act_arr) for cn, ma, mb in contrasts},
        }

    ya_np, va_np = data["y_act"].numpy(), data["valid_act"].numpy()
    report = {
        "experiment": "jrdb_rc2_rc3_graph_and_actions_5seed",
        "n_windows": n,
        "locations": locs,
        "action_5class": {a: score_jrdb_actions(ya_np, va_np, act_acc[a].mean(0), loc_arr) for a in ("M3a", "M3b", "M3")},
        "strata": st_out,
    }
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = json.dumps({"jrdb_rc2_rc3_graph_and_actions_5seed.json": hashlib.sha256(raw).hexdigest()}, indent=2).encode("utf-8")
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("jrdb_rc2_rc3_graph_and_actions_5seed.json", raw)
        zf.writestr("sha256.json", sh)
    print("Saved:", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
