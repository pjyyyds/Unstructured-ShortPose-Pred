"""Wave 14 CPU/GPU: PPT Research Content 1 (M0->M1a->M1b->M1) on 10 stationary JRDB scenes (9-location LOLO, 5 seeds)."""
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
from scripts.run_dair_o2_bioeq_external import SEEDS, person_bootstrap_ci
from scripts.run_imptc_and_dair_o2_full_external import TRANSFORMS
from src.models.bioeq_o2_joint import o2_joint_loss

POSE_OFFSETS = (-6, -5, -3, -2, 0)
FUTURE_STEPS = 7
STEP = 6
OUTDOOR_SCENES = {
    "hewlett-packard-intersection-2019-01-24_0",
    "huang-lane-2019-02-12_0",
    "tressider-2019-03-16_1",
}


def _scene_location(scene: str) -> str:
    return scene.rsplit("-2019-", 1)[0]


def build_jrdb_3d_pose_o2_features(joints_np: np.ndarray) -> dict[str, torch.Tensor]:
    """Convert (N, 5, 33, 3) BlazePose 3D landmarks into 11-D even + O(2) 2D vectors + odd knee + 4-D q."""
    j = torch.from_numpy(joints_np.astype(np.float64))
    fin = torch.isfinite(j).all(dim=-1)  # (N, 5, 33)
    xyz = torch.where(fin[..., None], j, 0.0)
    m_pose = fin[:, :, [11, 12, 23, 24]].all(dim=-1).any(dim=-1).float()

    def _kcos(hip: int, knee: int, ank: int) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = xyz[:, :, hip] - xyz[:, :, knee], xyz[:, :, ank] - xyz[:, :, knee]
        nu, nv = torch.linalg.vector_norm(u, dim=-1), torch.linalg.vector_norm(v, dim=-1)
        ok = fin[:, :, hip] & fin[:, :, knee] & fin[:, :, ank] & (nu > 1e-6) & (nv > 1e-6)
        c = ((u * v).sum(dim=-1) / torch.where(ok, nu * nv, 1.0)).clamp(-1.0, 1.0)
        return torch.where(ok, c, 0.0), ok

    ank_ok = fin[:, :, 27] & fin[:, :, 28]
    ank_span = torch.where(ank_ok, torch.linalg.vector_norm(xyz[:, :, 27, :2] - xyz[:, :, 28, :2], dim=-1), 0.0)
    cl, okl = _kcos(23, 25, 27)
    cr, okr = _kcos(24, 26, 28)
    kok = okl & okr
    k_mean = torch.where(kok, 0.5 * (cl + cr), 0.0)
    k_diff = torch.where(kok, (cl - cr).abs(), 0.0)
    signed_knee = torch.where(kok, cl - cr, 0.0).float().unsqueeze(-1)

    lok = fin[:, :, [23, 24, 27, 28]].all(dim=-1)
    leg_ext = torch.where(lok, 0.5 * (torch.linalg.vector_norm(xyz[:, :, 23] - xyz[:, :, 27], dim=-1) + torch.linalg.vector_norm(xyz[:, :, 24] - xyz[:, :, 28], dim=-1)), 0.0)
    tok = fin[:, :, [11, 12, 23, 24]].all(dim=-1)
    tr_up = torch.where(tok, 0.5 * (xyz[:, :, 11, 2] + xyz[:, :, 12, 2] - xyz[:, :, 23, 2] - xyz[:, :, 24, 2]), 0.0)
    vals = torch.stack((ank_span, k_mean, k_diff, leg_ext, tr_up), dim=-1)
    msks = torch.stack((ank_ok, kok, kok, lok, tok), dim=-1)
    dts = torch.tensor([1.0 / 15.0, 2.0 / 15.0, 1.0 / 15.0, 2.0 / 15.0], dtype=torch.float64)[None, :, None]
    step_d = torch.where(msks[:, 1:] & msks[:, :-1], (vals[:, 1:] - vals[:, :-1]) / dts, 0.0)
    bio_even = torch.cat((vals.float(), torch.cat((torch.zeros_like(step_d[:, :1]), step_d), dim=1).float(), msks[:, :, :4].float().mean(dim=-1, keepdim=True)), dim=-1)

    # Coronal shoulder & hip facing J(p_L - p_R) + foot toe-minus-heel direction in JRDB (x fwd, y left)
    d_sh = xyz[:, :, 11, :2] - xyz[:, :, 12, :2]
    d_hp = xyz[:, :, 23, :2] - xyz[:, :, 24, :2]
    f_sh = torch.where((fin[:, :, 11] & fin[:, :, 12])[..., None], torch.stack((d_sh[..., 1], -d_sh[..., 0]), dim=-1), 0.0)
    f_hp = torch.where((fin[:, :, 23] & fin[:, :, 24])[..., None], torch.stack((d_hp[..., 1], -d_hp[..., 0]), dim=-1), 0.0)
    ft_ok = fin[:, :, [29, 30, 31, 32]].all(dim=-1)
    f_ft = torch.where(ft_ok[..., None], 0.5 * ((xyz[:, :, 31, :2] - xyz[:, :, 29, :2]) + (xyz[:, :, 32, :2] - xyz[:, :, 30, :2])), 0.0)
    f_cor = f_sh + f_hp + f_ft

    sh_m, hp_m = 0.5 * (xyz[:, :, 11, :2] + xyz[:, :, 12, :2]), 0.5 * (xyz[:, :, 23, :2] + xyz[:, :, 24, :2])
    f_lean = torch.where(tok[..., None], sh_m - hp_m, 0.0)
    pose_vec_world = torch.stack((f_cor, f_lean), dim=2)  # (N, 5, 2, 2)

    q1 = fin.float().mean(dim=(1, 2))
    q2 = tok.float().mean(dim=1)
    sh_w = torch.where(fin[:, :, 11] & fin[:, :, 12], torch.linalg.vector_norm(d_sh, dim=-1), 0.0).mean(dim=1).float()
    q3 = torch.exp(-F.relu(sh_w - 0.65)) * (q2 > 0).float() if hasattr(torch, "nn") else q2
    q4 = torch.ones_like(q1)
    q_vec = torch.stack((q1, q2, q3.clamp(0.0, 1.0), q4), dim=-1)
    all5_ok = fin.all(dim=(1, 2)).numpy()
    any_ok = (m_pose.numpy() > 0.5)
    bucket = np.where(all5_ok, 0, np.where(any_ok, 1, 2))  # 0=Clear, 1=Partial, 2=Heavy_Missing
    return {
        "bio_even": bio_even,
        "pose_vec_world": pose_vec_world,
        "signed_knee": signed_knee,
        "quality": q_vec,
        "m_pose": m_pose,
        "q_buckets": bucket,
    }


import torch.nn.functional as F  # noqa: E402


def load_jrdb_10scene_cohort(repo_root: Path) -> dict:
    jrdb_dir = repo_root / "reviews/jrdb_development"
    if str(jrdb_dir) not in sys.path:
        sys.path.insert(0, str(jrdb_dir))
    import check_training_coordinates as ctc  # type: ignore
    import evaluate_outdoor_motion as eom  # type: ignore

    odom_spec = json.loads((jrdb_dir / "train_split_odometry.json").read_text())["recordings"]
    tr_arc = ctc.TrainingArchive(*eom.ARCHIVES["tracks"])
    ps_arc = ctc.TrainingArchive(*eom.ARCHIVES["poses"])
    od_arc = ctc.TrainingArchive(*eom.ARCHIVES["odom"])

    hist_l, truth_l, joints_l, scene_l, loc_l, actor_l = [], [], [], [], [], []
    for sc, meta in sorted(odom_spec.items()):
        b_od = od_arc.member(eom.MEMBERS["odom"].format(sc))
        if hashlib.sha256(b_od).hexdigest() != meta["member_sha256"]:
            raise ValueError(f"Odometry SHA-256 mismatch on {sc}")
        b_tr = tr_arc.member(eom.MEMBERS["tracks"].format(sc))
        b_ps = ps_arc.member(eom.MEMBERS["poses"].format(sc))
        tracks, _ = eom.read_tracks(b_tr)
        poses: dict[tuple[int, int], np.ndarray] = {}
        for img_nm, plist in json.loads(b_ps)["labels"].items():
            fr = int(Path(img_nm).stem)
            for obj in plist:
                if obj["label_id"].startswith("pedestrian:"):
                    pid = int(obj["label_id"].split(":")[1])
                    poses[(pid, fr)] = np.asarray(obj["keypoints"], dtype=np.float64).reshape(33, 3)

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
                    hist_l.append(xy[:2] - xy[1:2])
                    truth_l.append(xy[2:] - xy[1:2])
                    joints_l.append(j5)
                    scene_l.append(sc)
                    loc_l.append(_scene_location(sc))
                    actor_l.append(f"{sc}:{person}")
    hist_np = np.stack(hist_l)
    truth_np = np.stack(truth_l)
    joints_np = np.stack(joints_l)
    pf = build_jrdb_3d_pose_o2_features(joints_np)
    disp = np.linalg.norm(hist_np[:, 1] - hist_np[:, 0], axis=-1)
    fut_disp = np.linalg.norm(truth_np[:, -1], axis=-1)
    sc_arr = np.asarray(scene_l, dtype=str)
    return {
        "history": torch.from_numpy(hist_np),
        "truth": torch.from_numpy(truth_np),
        "scene": sc_arr,
        "location": np.asarray(loc_l, dtype=str),
        "actors": np.asarray(actor_l, dtype=str),
        **pf,
        "strata": {
            "all_10scenes": np.ones(len(hist_np), dtype=bool),
            "outdoor_3scenes_1677": np.isin(sc_arr, list(OUTDOOR_SCENES)),
            "clear_5pose": pf["q_buckets"] == 0,
            "partial_pose": pf["q_buckets"] == 1,
            "heavy_missing_pose": pf["q_buckets"] == 2,
            "any_pose_present": pf["q_buckets"] < 2,
            "short_disp_lt05": disp < 0.50,
            "short_disp_lt05_pose": (disp < 0.50) & (pf["q_buckets"] < 2),
            "quasi_static_lt015_pose": (disp < 0.15) & (pf["q_buckets"] < 2),
            "stationary_start_pose": (disp < 0.15) & (fut_disp >= 0.80) & (pf["q_buckets"] < 2),
        },
    }
def _fwd_jrdb(net: PPTMainlineO2Forecaster, d: dict, ids: torch.Tensor, device: torch.device) -> dict:
    n = len(ids)
    obs_dt = torch.full((n, 1), 0.4, dtype=torch.float64, device=device)
    fut_s = torch.arange(1, 8, dtype=torch.float64, device=device)[None, :].expand(n, -1) * 0.4
    return net(
        d["history"][ids].to(device),
        d["bio_even"][ids].to(device),
        d["pose_vec_world"][ids].to(device),
        d["signed_knee"][ids].to(device),
        d["quality"][ids].to(device),
        d["m_pose"][ids].to(device),
        obs_dt=obs_dt,
        fut_times_s=fut_s,
    )


def fit_jrdb_rc1_fold(seed: int, data: dict, tr_ids: np.ndarray, device: torch.device) -> dict[str, PPTMainlineO2Forecaster]:
    from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights

    w_tr = make_site_weights(data["location"], tr_ids).to(device)
    truth_tr = data["truth"][tr_ids].to(device)
    tr_t = torch.from_numpy(tr_ids)

    def _train(net: PPTMainlineO2Forecaster, prefs: tuple[str, ...], ep: int, lr: float, wd: float, soff: int) -> PPTMainlineO2Forecaster:
        for nm, p in net.named_parameters():
            p.requires_grad = nm.startswith(prefs) or nm in prefs
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(seed + soff)
        net.train()
        for _ in range(ep):
            for ch in torch.randperm(len(tr_ids), generator=gen).split(64):
                out = _fwd_jrdb(net, data, tr_t[ch], device)
                loss = o2_joint_loss(out, truth_tr[ch], w_tr[ch], act_weight=0.0)
                if loss is not None:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return net.eval()

    torch.manual_seed(seed)
    m0 = _train(PPTMainlineO2Forecaster("M0").to(device), ("motion.", "ctx_proj.", "cv_low_bonus", "central_", "pair_"), 40, 2e-3, 1e-4, 0)
    models = {"M0": m0}
    for i, arm in enumerate(("M1a", "M1b", "M1")):
        torch.manual_seed(seed)
        net = PPTMainlineO2Forecaster(arm).to(device)
        sd = net.state_dict()
        for k, v in m0.state_dict().items():
            if k in sd and sd[k].shape == v.shape:
                sd[k].copy_(v)
        net.load_state_dict(sd)
        models[arm] = _train(net, ("pose_", "gate_", "central_", "pair_", "cv_low_bonus"), 26, 1.2e-3, 3e-4, 10 + i)
    return models


@torch.inference_mode()
def eval_jrdb_arm(net: PPTMainlineO2Forecaster, data: dict, ids: np.ndarray, device: torch.device) -> np.ndarray:
    out = _fwd_jrdb(net, data, torch.from_numpy(ids), device)
    paths = out["paths"].cpu().numpy()
    probs = out["logits"].softmax(dim=1).cpu().numpy()
    truth = data["truth"][ids].numpy()
    dst = np.linalg.norm(paths - truth[:, None], axis=-1)
    ade_k, fde_k = dst.mean(axis=2), dst[:, :, -1]
    top = probs.argmax(axis=1)
    ii = np.arange(len(ids))
    bay = np.einsum("nk,nktw->ntw", probs, paths)
    bd = np.linalg.norm(bay - truth, axis=-1)
    return np.column_stack((ade_k[ii, top], fde_k[ii, top], ade_k.min(1), fde_k.min(1), bd.mean(1), bd[:, -1], out["gate"].cpu().numpy()))
def _loc_bootstrap_ci(diff_2d: np.ndarray, loc_arr: np.ndarray, n_boot: int = 10000, seed: int = 20261126) -> dict:
    locs = sorted(set(loc_arr.tolist()))
    if len(locs) < 2:
        return {"mean": {"ade_m": float(diff_2d[:, 0].mean()), "fde_m": float(diff_2d[:, 1].mean())}, "ci95": None}
    by_l = [diff_2d[loc_arr == l].mean(axis=0) for l in locs]
    mat = np.stack(by_l, axis=0)
    pt = mat.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(locs), size=(n_boot, len(locs)))
    bs = mat[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return {
        "mean": {"ade_m": float(pt[0]), "fde_m": float(pt[1])},
        "ci95": {"ade_m": [float(lo[0]), float(hi[0])], "fde_m": [float(lo[1]), float(hi[1])]},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--output-zip", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_jrdb_10scene_cohort(args.repo_root)
    n = len(data["scene"])
    loc_arr, sc_arr, act_arr = data["location"], data["scene"], data["actors"]
    locs = sorted(set(loc_arr.tolist()))
    arms = ("M0", "M1a", "M1b", "M1")

    hist_np, truth_np = data["history"].numpy(), data["truth"].numpy()
    cv_paths = (hist_np[:, 1:] - hist_np[:, :1]) * np.arange(1, 8, dtype=np.float64)[None, :, None]
    d_cv = np.linalg.norm(cv_paths - truth_np, axis=-1)
    cv_m = np.column_stack((d_cv.mean(1), d_cv[:, -1], d_cv.mean(1), d_cv[:, -1], d_cv.mean(1), d_cv[:, -1], np.zeros(n)))

    acc = {a: np.zeros((len(args.seeds), n, 7), dtype=np.float64) for a in arms}
    sample_m1 = None
    for si, seed in enumerate(args.seeds):
        for loc in locs:
            tr_ids = np.flatnonzero(loc_arr != loc)
            te_ids = np.flatnonzero(loc_arr == loc)
            models = fit_jrdb_rc1_fold(seed, data, tr_ids, device)
            for a in arms:
                acc[a][si, te_ids] = eval_jrdb_arm(models[a], data, te_ids, device)
            if sample_m1 is None:
                sample_m1 = models["M1"]
        print(f"Finished JRDB 9-location LOLO seed={seed} (n={n})", flush=True)

    mean_m = {"CV": cv_m, **{a: acc[a].mean(axis=0) for a in arms}}
    cols = ("ade1_m", "fde1_m", "min_ade6_m", "min_fde6_m", "bayes_ade_m", "bayes_fde_m", "mean_gate")
    contrasts = (("M0_minus_CV", "M0", "CV"), ("M1_minus_CV", "M1", "CV"), ("M1a_minus_M0", "M1a", "M0"), ("M1b_minus_M1a", "M1b", "M1a"), ("M1_minus_M1b", "M1", "M1b"), ("M1_minus_M0", "M1", "M0"))

    strata_out = {}
    for st_name, mask in data["strata"].items():
        if not mask.any():
            continue
        sub_l = loc_arr[mask]
        u_locs = sorted(set(sub_l.tolist()))
        mod_d = {}
        for m, arr in mean_m.items():
            sub = arr[mask]
            by_l = {l: {c: float(sub[sub_l == l, j].mean()) for j, c in enumerate(cols)} for l in u_locs}
            mod_d[m] = {
                "location_equal": {c: float(np.mean([v[c] for v in by_l.values()])) for c in cols},
                "pooled": {c: float(sub[:, j].mean()) for j, c in enumerate(cols)},
                "by_location": by_l,
            }
        strata_out[st_name] = {
            "n_windows": int(mask.sum()),
            "n_locations": len(u_locs),
            "models": mod_d,
            "paired_top1_location_ci": {cn: _loc_bootstrap_ci(mean_m[ma][mask, :2] - mean_m[mb][mask, :2], sub_l) for cn, ma, mb in contrasts},
            "paired_top1_person_ci": {cn: person_bootstrap_ci(mean_m[ma][:, 0] - mean_m[mb][:, 0], mask, act_arr) for cn, ma, mb in contrasts},
        }

    # 5-axis O(2) invariance check on trained M1 across first 64 windows
    sub_id = torch.arange(min(64, n))
    sw_vals = []
    with torch.inference_mode():
        for _, op, is_ref in TRANSFORMS:
            op_t = torch.from_numpy(op)
            d_tf = {**data, "history": data["history"] @ op_t.T, "pose_vec_world": data["pose_vec_world"] @ op_t.T, "signed_knee": -data["signed_knee"] if is_ref else data["signed_knee"]}
            oo = _fwd_jrdb(sample_m1, d_tf, sub_id, device)
            pw = oo["paths"].cpu().numpy() @ op
            top = oo["logits"].argmax(1).cpu().numpy()
            sw_vals.append(float(np.linalg.norm(pw[np.arange(len(sub_id)), top] - truth_np[: len(sub_id)], axis=-1).mean()))

    report = {
        "experiment": "jrdb_rc1_pose_quality_and_initiation_5seed",
        "n_windows": n,
        "locations": locs,
        "scene_counts": {s: int((sc_arr == s).sum()) for s in sorted(set(sc_arr.tolist()))},
        "quality_counts": {"clear_5pose": int((data["q_buckets"] == 0).sum()), "partial_pose": int((data["q_buckets"] == 1).sum()), "heavy_missing_pose": int((data["q_buckets"] == 2).sum())},
        "five_axis_top1_ade_swing_m": float(max(sw_vals) - min(sw_vals)),
        "strata": strata_out,
    }
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = json.dumps({"jrdb_rc1_pose_quality_and_initiation_5seed.json": hashlib.sha256(raw).hexdigest()}, indent=2).encode("utf-8")
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("jrdb_rc1_pose_quality_and_initiation_5seed.json", raw)
        zf.writestr("sha256.json", sh)
    print("Saved:", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
