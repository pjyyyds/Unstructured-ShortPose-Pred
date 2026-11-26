"""PPT Slide 9 & 13 RC1 True 3x3 Duration (T=5,10,20) x Quality (Clear,Partial,Heavy) Matrix with 20-frame poses & 3x3 homography."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.ppt_mainline_o2_models import (
    PPTMainlineO2Forecaster,
    build_ground_pose_o2_inputs,
    fit_pinns_ground_homography,
    verify_pinns_ground_and_polyline_diagnostics,
)
from scripts.run_pinns_bioeq_v2_benchmark import make_site_weights
from scripts.run_pinns_matched_o2_multiduration import DURATIONS, SEEDS, SITES, load_matched_623, paired_site_clip_ci
from src.data.features import build_pose_and_quality_features
from src.models.bioeq_o2_joint import o2_joint_loss


def load_matched_623_rc1_20frame(matched_npz: Path, meta_dir: Path, pose_npz: Path, matched_obs_npz: Path) -> dict:
    m623 = load_matched_623(matched_npz, pose_npz)
    with np.load(matched_npz, allow_pickle=False) as m_ar:
        m = {k: m_ar[k] for k in m_ar.files}
    with np.load(pose_npz, allow_pickle=False) as p_ar:
        p = {k: p_ar[k] for k in p_ar.files}
    with np.load(matched_obs_npz, allow_pickle=False) as o_ar:
        obs = {k: o_ar[k] for k in o_ar.files}

    sel = sorted(
        np.flatnonzero(m["strict_disjoint_20"]).tolist(),
        key=lambda i: (str(m["clip"][i]), int(m["person_id"][i]), int(m["t0"][i])),
    )
    keys = [(str(m["clip"][i]), int(m["person_id"][i]), int(m["t0"][i])) for i in sel]
    p_map = {(str(c), int(pid), int(t)): i for i, (c, pid, t) in enumerate(zip(p["clip"], p["person_id"], p["t0"]))}
    o_map = {(str(c), int(pid), int(t)): i for i, (c, pid, t) in enumerate(zip(obs["clip"], obs["person_id"], obs["t0"]))}
    p_idx = np.array([p_map[k] for k in keys], dtype=int)
    o_idx = np.array([o_map[k] for k in keys], dtype=int)

    kpts20 = obs["hist_kpts"][o_idx].astype(np.float32)
    bbox20 = obs["hist_bbox"][o_idx].astype(np.float32)
    imgh = obs["img_h"][o_idx]
    if kpts20.shape != (len(keys), 20, 17, 3) or bbox20.shape != (len(keys), 20, 4):
        raise ValueError(f"Expected 20-frame pose arrays, got {kpts20.shape} and {bbox20.shape}")
    np.testing.assert_allclose(kpts20[:, -5:], p["hist_kpts"][p_idx], atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(bbox20[:, -5:], p["hist_bbox"][p_idx], atol=1e-5, rtol=0.0)

    n = len(keys)
    clips = m623["clip"]
    h_mats, y_sums = np.zeros((n, 3, 3), dtype=np.float64), np.zeros(n, dtype=np.float64)
    for c in sorted(set(clips.tolist())):
        calib = json.loads((meta_dir / "videos" / c.rsplit("_", 1)[0] / "calib.json").read_text())
        hm, ys = fit_pinns_ground_homography(calib)
        for i in np.flatnonzero(clips == c):
            h_mats[i] = hm; y_sums[i] = ys

    hm_t, ys_t = torch.from_numpy(h_mats), torch.from_numpy(y_sums)
    k20_t, b20_t = torch.from_numpy(kpts20), torch.from_numpy(bbox20)
    gp20 = build_ground_pose_o2_inputs(m623["hist20"], k20_t, b20_t, hm_t, ys_t)
    diag20 = verify_pinns_ground_and_polyline_diagnostics(m623["hist20"], b20_t, gp20["pose_vec_world"], hm_t, ys_t)
    print("Verified 623x20 matched ground pose diagnostics:", json.dumps(diag20), flush=True)

    per_dur = {}
    for dur in DURATIONS:
        q_list, b_list = [], []
        obs_m = np.ones(dur, dtype=np.float32)
        for j in range(n):
            _, qv, _, bk = build_pose_and_quality_features(kpts20[j, -dur:], bbox20[j, -dur:], obs_m, float(imgh[j]))
            q_list.append(qv); b_list.append(bk)
        gpd = build_ground_pose_o2_inputs(m623["hist20"][:, -dur:], k20_t[:, -dur:], b20_t[:, -dur:], hm_t, ys_t)
        per_dur[dur] = {**gpd, "quality": torch.from_numpy(np.stack(q_list)), "q_buckets": np.asarray(b_list, dtype=int)}
    return {"site": m623["site"], "clip": clips, "hist20": m623["hist20"], "truth": m623["truth30"], "per_dur": per_dur, "coord_diag_20f": diag20}


def fit_rc1_dur_fold(seed: int, data: dict, dur: int, tr_ids: np.ndarray, device: torch.device) -> dict[str, PPTMainlineO2Forecaster]:
    w_tr = make_site_weights(data["site"], tr_ids).to(device)
    truth_tr = data["truth"][tr_ids].to(device)
    tr_t = torch.from_numpy(tr_ids)
    dd = data["per_dur"][dur]

    def _fwd(net: PPTMainlineO2Forecaster, ids: torch.Tensor) -> dict:
        return net(
            data["hist20"][ids, -dur:].to(device), dd["bio_even"][ids].to(device), dd["pose_vec_world"][ids].to(device),
            dd["signed_knee"][ids].to(device), dd["quality"][ids].to(device), dd["m_pose"][ids].to(device),
        )

    def _train(net: PPTMainlineO2Forecaster, prefs: tuple[str, ...], ep: int, lr: float, wd: float, soff: int) -> PPTMainlineO2Forecaster:
        for nm, p in net.named_parameters():
            p.requires_grad = nm.startswith(prefs) or nm in prefs
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        gen = torch.Generator().manual_seed(seed + soff)
        net.train()
        for _ in range(ep):
            for ch in torch.randperm(len(tr_ids), generator=gen).split(64):
                loss = o2_joint_loss(_fwd(net, tr_t[ch]), truth_tr[ch], w_tr[ch], act_weight=0.0)
                if loss is not None:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return net.eval()

    torch.manual_seed(seed)
    m0 = _train(PPTMainlineO2Forecaster("M0").to(device), ("motion.", "ctx_proj.", "cv_low_bonus", "central_", "pair_"), 42, 2e-3, 1e-4, 0)
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
def eval_rc1_dur(net: PPTMainlineO2Forecaster, data: dict, dur: int, ids: np.ndarray, device: torch.device) -> np.ndarray:
    tid = torch.from_numpy(ids)
    dd = data["per_dur"][dur]
    out = net(
        data["hist20"][tid, -dur:].to(device), dd["bio_even"][tid].to(device), dd["pose_vec_world"][tid].to(device),
        dd["signed_knee"][tid].to(device), dd["quality"][tid].to(device), dd["m_pose"][tid].to(device),
    )
    p, q = out["paths"].cpu().numpy(), out["logits"].softmax(dim=1).cpu().numpy()
    tr = data["truth"][ids].numpy()
    dst = np.linalg.norm(p - tr[:, None], axis=-1)
    ade_k, fde_k = dst.mean(2), dst[:, :, -1]
    top = q.argmax(1); ii = np.arange(len(ids))
    bay = np.einsum("nk,nktw->ntw", q, p)
    bd = np.linalg.norm(bay - tr, axis=-1)
    return np.column_stack((ade_k[ii, top], fde_k[ii, top], ade_k.min(1), fde_k.min(1), bd.mean(1), bd[:, -1], out["gate"].cpu().numpy()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("matched_npz", type=Path)
    ap.add_argument("meta_dir", type=Path)
    ap.add_argument("pose_npz", type=Path)
    ap.add_argument("matched_obs_npz", type=Path)
    ap.add_argument("output_zip", type=Path)
    ap.add_argument("--scratch-dir", type=Path, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    if not args.matched_obs_npz.is_file() and args.scratch_dir is not None:
        import subprocess, sys
        from scripts.run_heldout_cloud import pip_install, stage
        from scripts.run_pinns_matched_durations import collect
        probe = subprocess.run([sys.executable, "-c", "import onnxruntime as ort; raise SystemExit('CUDAExecutionProvider' not in ort.get_available_providers())"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if probe.returncode:
            pip_install("onnxruntime-gpu")
        ds_dir, onnx_p = stage(args.scratch_dir)
        collect(ds_dir, onnx_p, args.pose_npz, args.matched_obs_npz)

    torch.set_num_threads(4)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_matched_623_rc1_20frame(args.matched_npz, args.meta_dir, args.pose_npz, args.matched_obs_npz)
    n, site_arr, clip_arr = len(data["site"]), data["site"], data["clip"]
    qb5 = data["per_dur"][5]["q_buckets"]
    h5 = data["hist20"][:, -5:].numpy()
    cv_p = (h5[:, -1:] - h5[:, -2:-1]) * np.arange(1, 31, dtype=np.float64)[None, :, None]
    d_cv = np.linalg.norm(cv_p - data["truth"].numpy(), axis=-1)
    cv_m = np.column_stack((d_cv.mean(1), d_cv[:, -1], d_cv.mean(1), d_cv[:, -1], d_cv.mean(1), d_cv[:, -1], np.zeros(n)))

    arms = ("M0", "M1a", "M1b", "M1")
    cols = ("ade1_m", "fde1_m", "min_ade6_m", "min_fde6_m", "bayes_ade_m", "bayes_fde_m", "mean_gate")
    contrasts = (("M1a_minus_M0", "M1a", "M0"), ("M1b_minus_M1a", "M1b", "M1a"), ("M1_minus_M1b", "M1", "M1b"), ("M1_minus_M0", "M1", "M0"), ("M1_minus_CV", "M1", "CV"))

    by_dur = {}
    for dur in DURATIONS:
        acc = {a: np.zeros((len(args.seeds), n, 7), dtype=np.float64) for a in arms}
        for si, seed in enumerate(args.seeds):
            for s in SITES:
                tr_ids, te_ids = np.flatnonzero(site_arr != s), np.flatnonzero(site_arr == s)
                mods = fit_rc1_dur_fold(seed, data, dur, tr_ids, dev)
                for a in arms:
                    acc[a][si, te_ids] = eval_rc1_dur(mods[a], data, dur, te_ids, dev)
        mm = {"CV": cv_m, **{a: acc[a].mean(0) for a in arms}}
        qbd = data["per_dur"][dur]["q_buckets"]
        slices = {"Overall_623": np.ones(n, dtype=bool), "Clear": qbd == 0, "Partial": qbd == 1, "Heavy": qbd == 2, "Clear_T5fixed": qb5 == 0, "Partial_T5fixed": qb5 == 1, "Heavy_T5fixed": qb5 == 2}
        sl_out = {}
        for sn, sm in slices.items():
            sub_s, sub_c = site_arr[sm], clip_arr[sm]
            md = {}
            for m, arr in mm.items():
                sub = arr[sm]
                bys = {s: {c: float(sub[sub_s == s, j].mean()) for j, c in enumerate(cols)} for s in SITES if (sub_s == s).any()}
                md[m] = {"site_equal": {c: float(np.mean([v[c] for v in bys.values()])) for c in cols}, "pooled": {c: float(sub[:, j].mean()) for j, c in enumerate(cols)}}
            sl_out[sn] = {"n_windows": int(sm.sum()), "models": md, "paired_top1_clip_ci": {cn: paired_site_clip_ci(mm[ma][sm, :2] - mm[mb][sm, :2], sub_s, sub_c) for cn, ma, mb in contrasts}}
        by_dur[f"T{dur}"] = sl_out
        print(f"Finished true-20frame 3x3 RC1 duration T={dur}", flush=True)

    report = {"experiment": "ppt_rc1_3x3_duration_quality_matrix_v2_true20f", "n_windows": n, "seeds": list(args.seeds), "coordinate_diagnostics_20f": data["coord_diag_20f"], "durations": by_dur}
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = json.dumps({"ppt_rc1_3x3_duration_quality_matrix.json": hashlib.sha256(raw).hexdigest()}, indent=2).encode("utf-8")
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ppt_rc1_3x3_duration_quality_matrix.json", raw)
        zf.writestr("sha256.json", sh)
    print("Saved:", args.output_zip, flush=True)


if __name__ == "__main__":
    main()
