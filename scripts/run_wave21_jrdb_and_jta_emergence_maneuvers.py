"""Wave 21B-GPU0: Blind physical emergence, 6-candidate diagnosis, & 2x2 6-slot precursors on JRDB & JTA-228."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

from scripts.cloud_experiment_runner import ST_RELEASE_URL, _download_verified_zip
from scripts.run_jrdb_outdoor_o2_full_5seed import _jrdb_edges
from scripts.run_jrdb_rc1_pose_quality_and_initiation_5seed import (
    FUTURE_STEPS,
    POSE_OFFSETS,
    STEP,
    _scene_location,
    build_jrdb_3d_pose_o2_features,
)
from scripts.run_wave21_pinns_true_emergence_and_6slot_diag import (
    SEEDS,
    run_2x2_6slot_and_baseline_eval,
)
from src.data.jta_builder import JTA22_TO_COCO17
from src.data.scene_graph import build_neighbor_edges
from src.models.emergence_maneuver_6slot import compute_action_labels_with_onset


def load_jrdb_with_blind_emergence(repo_root: Path) -> tuple[dict, dict]:
    """Load JRDB 9-location 5,836 windows + blind ray-occluder emergence & matched visible controls."""
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
    ya_l, va_l, on_l, loc_l, act_l = [], [], [], [], []
    e_occ_l, e_start_l, prior_long_l = [], [], []
    hm2, fm7 = np.ones(2, dtype=np.float32), np.ones(FUTURE_STEPS, dtype=np.float32)

    for sc, meta in sorted(odom_spec.items()):
        if hashlib.sha256(od_arc.member(eom.MEMBERS["odom"].format(sc))).hexdigest() != meta["member_sha256"]:
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
        min_sc_f = min(by_fr) if by_fr else 0

        for person, rows in sorted(tracks.items()):
            times = np.asarray(sorted(rows))
            for r_idx, run in enumerate(np.split(times, np.flatnonzero(np.diff(times) != STEP) + 1)):
                for off in range(0, len(run) - FUTURE_STEPS - 1, 2 + FUTURE_STEPS):
                    f0, f1 = int(run[off]), int(run[off + 1])
                    fut_f = [int(x) for x in run[off + 2 : off + 2 + FUTURE_STEPS]]
                    h_xy = np.stack([by_fr[f0][person], by_fr[f1][person]], axis=0)
                    t_xy = np.stack([by_fr[ff][person] for ff in fut_f], axis=0)
                    j5 = np.full((5, 33, 3), np.nan, dtype=np.float64)
                    for si_o, fo in enumerate(POSE_OFFSETS):
                        if (person, f1 + fo) in poses:
                            j5[si_o] = poses[(person, f1 + fo)]
                    ed, nm = _jrdb_edges(by_fr, f0, f1, person)
                    yk, vk, ok = compute_action_labels_with_onset(h_xy - h_xy[-1:], hm2, t_xy - h_xy[-1:], fm7, dt=0.4, min_run=2)

                    # Blind physical occlusion check: closer agent along panoramic ray before/at f0 or mid-clip track re-entry
                    is_clip_st = (off == 0) and (r_idx == 0) and (f0 <= min_sc_f + STEP)
                    ray_occ = False
                    if off == 0 and not is_clip_st:
                        p_ego = h_xy[0]
                        r_ego = float(np.hypot(p_ego[0], p_ego[1]))
                        ang_ego = float(np.arctan2(p_ego[1], p_ego[0]))
                        for fc in (f0 - 2 * STEP, f0 - STEP, f0):
                            for pid_o, xy_o in by_fr.get(fc, {}).items():
                                if pid_o == person:
                                    continue
                                r_o = float(np.hypot(xy_o[0], xy_o[1]))
                                da = abs(((np.arctan2(xy_o[1], xy_o[0]) - ang_ego + np.pi) % (2 * np.pi)) - np.pi)
                                if r_o < r_ego - 0.25 and da <= np.deg2rad(20.0):
                                    ray_occ = True
                        if r_idx > 0:
                            ray_occ = True  # Re-emerged after mid-clip track occlusion gap!
                    pose_emerge = (not np.isfinite(j5[0]).all()) and np.isfinite(j5[-1, [11, 12, 23, 24]]).all()
                    hist_l.append(h_xy - h_xy[-1:])
                    truth_l.append(t_xy - h_xy[-1:])
                    joints_l.append(j5)
                    edges_l.append(ed); nmsk_l.append(nm)
                    ya_l.append(yk); va_l.append(vk); on_l.append(ok)
                    loc_l.append(_scene_location(sc)); act_l.append(f"{sc}:{person}")
                    e_occ_l.append(bool(ray_occ or pose_emerge))
                    e_start_l.append(bool(is_clip_st))
                    prior_long_l.append(bool(off >= 2 * (2 + FUTURE_STEPS)))

    hist = np.asarray(hist_l, dtype=np.float64)
    truth = np.asarray(truth_l, dtype=np.float64)
    joints = np.asarray(joints_l, dtype=np.float64)
    pf = build_jrdb_3d_pose_o2_features(joints)
    loc_arr = np.asarray(loc_l)
    e_true = np.asarray(e_occ_l, dtype=bool)
    prior_m = np.asarray(prior_long_l, dtype=bool)
    vis_torso = np.isfinite(joints[:, :, [11, 12, 23, 24]]).all(axis=(2, 3)).sum(axis=1) >= 3
    vis_feet = np.isfinite(joints[:, :, [27, 28, 31, 32]]).all(axis=(2, 3)).sum(axis=1) >= 3
    v0 = np.linalg.norm(hist[:, 1] - hist[:, 0], axis=-1) / 0.4
    ctrl = np.zeros(len(hist), dtype=bool)
    for idx in np.flatnonzero(e_true):
        cand = np.flatnonzero(prior_m & (~ctrl) & (loc_arr == loc_arr[idx]))
        if len(cand) > 0:
            ctrl[cand[int(np.argmin(np.abs(v0[cand] - v0[idx])))]] = True
    locs_sorted = sorted(set(loc_arr.tolist()))
    dev_loc = np.isin(loc_arr, locs_sorted[0::2])
    return {
        "history": torch.from_numpy(hist), "truth": torch.from_numpy(truth), **pf,
        "edges": torch.from_numpy(np.asarray(edges_l, dtype=np.float32)),
        "is_car": torch.zeros((len(hist), 12), dtype=torch.float32),
        "nbr_mask": torch.from_numpy(np.asarray(nmsk_l, dtype=np.float32)),
        "y_act": np.asarray(ya_l, dtype=bool), "valid_act": np.asarray(va_l, dtype=bool),
        "onset": np.asarray(on_l, dtype=np.int64), "location": loc_arr, "actors": np.asarray(act_l),
    }, {
        "e_true": e_true, "e_clip_start": np.asarray(e_start_l, dtype=bool),
        "vis_torso": vis_torso, "vis_feet": vis_feet, "matched_ctrl": ctrl,
        "dev_mask": dev_loc, "locked_eval_mask": ~dev_loc,
    }


def _jta_ground_pose_o2(kpts_px: np.ndarray, bbox_px: np.ndarray, hist_xy: np.ndarray) -> dict[str, torch.Tensor]:
    """Extract O(2) biomechanical invariants + 2D facing/lean projected from uncorrupted JTA skeletons."""
    from scripts.ppt_mainline_o2_models import _even_pose_invariants

    kp_t = torch.from_numpy(kpts_px.astype(np.float32))
    bb_t = torch.from_numpy(bbox_px.astype(np.float32))
    bio_even, signed_knee, m_pose, q_vec = _even_pose_invariants(kp_t, bb_t)
    # Construct O(2)-consistent ground coronal & lean vectors from shoulder/hip screen geometry aligned to scene scale
    l_sh, r_sh = kpts_px[:, :, 5, :2], kpts_px[:, :, 6, :2]
    l_hp, r_hp = kpts_px[:, :, 11, :2], kpts_px[:, :, 12, :2]
    bh = np.maximum(bbox_px[:, :, 3:4], 20.0)
    d_sh = (l_sh - r_sh) / bh
    d_hp = (l_hp - r_hp) / bh
    # Screen x-right, y-down -> coronal normal in ground (x, y) proxy + sagittal lean
    f_cor = 0.5 * np.stack((d_sh[..., 1] + d_hp[..., 1], -(d_sh[..., 0] + d_hp[..., 0])), axis=-1)
    f_lean = ((0.5 * (l_sh + r_sh) - 0.5 * (l_hp + r_hp)) / bh) * np.array([1.0, -1.0])
    # Orient coronal sign with moving displacement on average across scene
    disp = hist_xy[:, -1] - hist_xy[:, 0]
    mv = np.linalg.norm(disp, axis=-1) > 0.25
    if mv.sum() >= 10 and float((disp[mv] * f_cor[mv, -1]).sum()) < 0.0:
        f_cor = -f_cor
    pvw = np.stack((f_cor, f_lean), axis=2).astype(np.float64)
    return {"bio_even": bio_even, "pose_vec_world": torch.from_numpy(pvw), "signed_knee": signed_knee, "quality": q_vec, "m_pose": m_pose}


def load_jta228_native_emergence_cohort(st_zip: Path, max_train_files: int = 80) -> tuple[dict, dict]:
    """Mine uncorrupted 0.4s (T=2) emergence, foreground-box occlusion, & maneuvers across JTA sequences."""
    from scripts.ppt_mainline_o2_models import build_ground_pose_o2_inputs

    hist_l, truth_l, kp_l, bb_l, pvw_l = [], [], [], [], []
    ed_l, nm_l, ya_l, va_l, on_l, grp_l, unit_l = [], [], [], [], [], [], []
    e_occ_l, prior_l, split_l = [], [], []
    hm2, fm8 = np.ones(2, dtype=np.float32), np.ones(8, dtype=np.float32)

    with zipfile.ZipFile(st_zip) as zf:
        tr_files = sorted(n for n in zf.namelist() if n.startswith("jta/data/train/") and n.endswith(".ndjson"))[:max_train_files]
        va_files = sorted(n for n in zf.namelist() if n.startswith("jta/data/val/") and n.endswith(".ndjson"))
        te_files = sorted(n for n in zf.namelist() if n.startswith("jta/data/test/") and n.endswith(".ndjson"))
        all_files = [("train", f) for f in tr_files] + [("val", f) for f in va_files] + [("test", f) for f in te_files]

        for sp_tag, fn in all_files:
            seq_id = Path(fn).stem
            by_f: dict[int, dict[int, dict]] = {}
            for ln in zf.read(fn).decode("utf-8", errors="ignore").splitlines():
                if not ln.strip():
                    continue
                ob = json.loads(ln)
                if "track" in ob:
                    t = ob["track"]
                    by_f.setdefault(int(t["f"]), {})[int(t["p"])] = t
            frames = sorted(by_f)
            if len(frames) < 12:
                continue
            f_step = int(np.median(np.diff(frames)))
            min_f = frames[0]

            # Fit causal screen-footpoint -> world-XY 2x2 Jacobian for this camera view
            duv_all, dxy_all = [], []
            for fi in frames[: min(len(frames) - 1, 25)]:
                f_nx = fi + f_step
                if f_nx not in by_f:
                    continue
                for pid, r0 in by_f[fi].items():
                    r1 = by_f[f_nx].get(pid)
                    if r1 is None:
                        continue
                    u0 = float(r0.get("bb_left", 0.0)) + 0.5 * float(r0.get("bb_width", 30.0))
                    v0 = float(r0.get("bb_top", 0.0)) + float(r0.get("bb_height", 60.0))
                    u1 = float(r1.get("bb_left", 0.0)) + 0.5 * float(r1.get("bb_width", 30.0))
                    v1 = float(r1.get("bb_top", 0.0)) + float(r1.get("bb_height", 60.0))
                    if abs(u1 - u0) + abs(v1 - v0) > 2.0:
                        duv_all.append([u1 - u0, v1 - v0])
                        dxy_all.append([float(r1["x"]) - float(r0["x"]), float(r1["y"]) - float(r0["y"])])
            j_mat = np.linalg.lstsq(np.asarray(duv_all), np.asarray(dxy_all), rcond=None)[0] if len(duv_all) >= 8 else np.eye(2) * 0.02

            p_tracks: dict[int, list[int]] = {}
            for fi in frames:
                for pid in by_f[fi]:
                    p_tracks.setdefault(pid, []).append(fi)

            for pid, p_fs in sorted(p_tracks.items()):
                fts = np.asarray(sorted(p_fs))
                for r_i, run in enumerate(np.split(fts, np.flatnonzero(np.diff(fts) != f_step) + 1)):
                    if len(run) < 10:
                        continue
                    # Sample first emergence window (off=0) + at most one steady window (off=12) per run
                    for off in (0, 12):
                        if off + 10 > len(run):
                            continue
                        f0, f1 = int(run[off]), int(run[off + 1])
                        futs = [int(x) for x in run[off + 2 : off + 10]]
                        h_xy = np.array([[float(by_f[f0][pid]["x"]), float(by_f[f0][pid]["y"])], [float(by_f[f1][pid]["x"]), float(by_f[f1][pid]["y"])]], dtype=np.float64)
                        t_xy = np.array([[float(by_f[ff][pid]["x"]), float(by_f[ff][pid]["y"])] for ff in futs], dtype=np.float64)
                        yk, vk, ok = compute_action_labels_with_onset(h_xy - h_xy[-1:], hm2, t_xy - h_xy[-1:], fm8, dt=0.4, min_run=2)

                        kp2 = np.zeros((2, 17, 3), dtype=np.float32)
                        bb2 = np.zeros((2, 4), dtype=np.float32)
                        pvw2 = np.zeros((2, 2, 2), dtype=np.float64)
                        for ti_h, fh in enumerate((f0, f1)):
                            rc = by_f[fh][pid]
                            bw, bh = max(float(rc.get("bb_width", 35.0)), 8.0), max(float(rc.get("bb_height", 65.0)), 16.0)
                            cx, cy = float(rc.get("bb_left", 400.0)) + 0.5 * bw, float(rc.get("bb_top", 270.0)) + 0.5 * bh
                            bb2[ti_h] = [cx, cy, bw, bh]
                            xx = np.array([float(rc.get(f"xx{j}", cx)) for j in JTA22_TO_COCO17], dtype=np.float32)
                            yy = np.array([float(rc.get(f"yy{j}", cy)) for j in JTA22_TO_COCO17], dtype=np.float32)
                            kp2[ti_h, :, 0], kp2[ti_h, :, 1], kp2[ti_h, :, 2] = xx, yy, 0.90
                            sh_uv = np.array([xx[5] - xx[6], yy[5] - yy[6]], dtype=np.float64)
                            hp_uv = np.array([xx[11] - xx[12], yy[11] - yy[12]], dtype=np.float64)
                            lat_w = 0.5 * (sh_uv @ j_mat + hp_uv @ j_mat)
                            ln_n = np.linalg.norm(lat_w)
                            f_cor = (np.array([-lat_w[1], lat_w[0]]) / max(ln_n, 1e-6)) * min(ln_n, 0.45)
                            lean_uv = np.array([0.5 * (xx[5] + xx[6] - xx[11] - xx[12]), 0.0], dtype=np.float64)
                            pvw2[ti_h, 0], pvw2[ti_h, 1] = f_cor, (lean_uv @ j_mat) * 0.5

                        # Neighbor edges at f1 from contemporaneous JTA pedestrians
                        nb_p, nb_v, nb_vv = [], [], []
                        for po, ro in by_f[f1].items():
                            if po == pid:
                                continue
                            xy_o = np.array([float(ro["x"]), float(ro["y"])], dtype=np.float64)
                            nb_p.append(xy_o)
                            rp = by_f.get(f0, {}).get(po)
                            if rp is not None:
                                nb_v.append((xy_o - np.array([float(rp["x"]), float(rp["y"])])) / 0.4); nb_vv.append(1.0)
                            else:
                                nb_v.append(np.zeros(2)); nb_vv.append(0.0)
                        if nb_p:
                            nt = np.zeros(len(nb_p), dtype=np.int64)
                            ed, _, nmk = build_neighbor_edges(h_xy[1], (h_xy[1] - h_xy[0]) / 0.4, 1.0, 1.0, np.asarray(nb_p), np.asarray(nb_v), np.asarray(nb_vv), np.ones(len(nb_p)), nt)
                        else:
                            ed, nmk = np.zeros((12, 9), dtype=np.float32), np.zeros(12, dtype=np.float32)

                        # Blind foreground occlusion emergence check in screen space
                        fg_occ = False
                        if off == 0 and f0 > min_f + f_step:
                            rc0 = by_f[f0][pid]
                            cx0 = float(rc0.get("bb_left", 0.0)) + 0.5 * float(rc0.get("bb_width", 30.0))
                            yb0 = float(rc0.get("bb_top", 0.0)) + float(rc0.get("bb_height", 60.0))
                            for fc in (f0 - f_step, f0):
                                for po, ro in by_f.get(fc, {}).items():
                                    if po == pid:
                                        continue
                                    cx_o = float(ro.get("bb_left", 0.0)) + 0.5 * float(ro.get("bb_width", 30.0))
                                    yb_o = float(ro.get("bb_top", 0.0)) + float(ro.get("bb_height", 60.0))
                                    if abs(cx_o - cx0) <= 0.8 * (float(ro.get("bb_width", 30.0)) + float(rc0.get("bb_width", 30.0))) and yb_o >= yb0:
                                        fg_occ = True
                            if r_i > 0:
                                fg_occ = True

                        hist_l.append(h_xy - h_xy[-1:]); truth_l.append(t_xy - h_xy[-1:])
                        kp_l.append(kp2); bb_l.append(bb2); pvw_l.append(pvw2)
                        ed_l.append(ed); nm_l.append(nmk)
                        ya_l.append(yk); va_l.append(vk); on_l.append(ok)
                        grp_l.append(f"{sp_tag}_{int(hashlib.md5(seq_id.encode()).hexdigest()[:4], 16) % 4}")
                        unit_l.append(seq_id); e_occ_l.append(fg_occ); prior_l.append(off >= 12); split_l.append(sp_tag)

    hist = np.asarray(hist_l, dtype=np.float64)
    truth = np.asarray(truth_l, dtype=np.float64)
    kp_np = np.asarray(kp_l, dtype=np.float32)
    bb_np = np.asarray(bb_l, dtype=np.float32)
    pvw = np.asarray(pvw_l, dtype=np.float64)
    disp = hist[:, -1] - hist[:, 0]
    mv = np.linalg.norm(disp, axis=-1) > 0.25
    if mv.sum() >= 20 and float((disp[mv] * pvw[mv, -1, 0]).sum()) < 0.0:
        pvw[:, :, 0] = -pvw[:, :, 0]
    gp_j = build_ground_pose_o2_inputs(torch.from_numpy(hist), torch.from_numpy(kp_np), torch.from_numpy(bb_np))
    return {
        "history": torch.from_numpy(hist), "truth": torch.from_numpy(truth),
        "bio_even": gp_j["bio_even"], "pose_vec_world": torch.from_numpy(pvw),
        "signed_knee": gp_j["signed_knee"], "quality": gp_j["q_bio"], "m_pose": gp_j["m_pose"],
        "edges": torch.from_numpy(np.asarray(ed_l, dtype=np.float32)),
        "is_car": torch.zeros((len(hist), 12), dtype=torch.float32),
        "nbr_mask": torch.from_numpy(np.asarray(nm_l, dtype=np.float32)),
        "y_act": np.asarray(ya_l, dtype=bool), "valid_act": np.asarray(va_l, dtype=bool),
        "onset": np.asarray(on_l, dtype=np.int64), "group": np.asarray(grp_l),
        "seq": np.asarray(unit_l), "split": np.asarray(split_l),
    }, {"e_true": np.asarray(e_occ_l, dtype=bool), "prior_steady": np.asarray(prior_l, dtype=bool)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--scratch-dir", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) JRDB 9-location 5,836 windows with blind ray/pose de-occlusion & matched controls
    dj, bj = load_jrdb_with_blind_emergence(args.repo_root)
    nj = len(dj["history"])
    hj_np, tj_np = dj["history"].numpy(), dj["truth"].numpy()
    gdj = {k: dj[k].to(dev) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
    gdj["y_act"] = torch.from_numpy(dj["y_act"]).to(dev)
    gdj["valid_act"] = torch.from_numpy(dj["valid_act"]).to(dev)
    gdj["obs_dt"] = torch.full((nj, 1), 0.4, dtype=torch.float64, device=dev)
    gdj["fut_s"] = torch.arange(1, FUTURE_STEPS + 1, dtype=torch.float64, device=dev)[None].expand(nj, -1) * 0.4
    cv_j = ((hj_np[:, 1:2] - hj_np[:, 0:1]) / 0.4) * (np.arange(1, FUTURE_STEPS + 1)[None, :, None] * 0.4)
    disp_j = np.linalg.norm(hj_np[:, 1] - hj_np[:, 0], axis=-1)
    man_j = (dj["y_act"][:, :4] & dj["valid_act"][:, :4]).any(axis=1)
    j_strata = {
        "all_5836": np.ones(nj, dtype=bool),
        "locked_eval_locations_all": bj["locked_eval_mask"],
        "true_deocclusion_emergence": bj["e_true"],
        "locked_eval_true_emergence": bj["locked_eval_mask"] & bj["e_true"],
        "true_emergence_lowdisp_lt0p5m": bj["e_true"] & (disp_j < 0.50),
        "true_emergence_clear_torso_feet": bj["e_true"] & bj["vis_torso"] & bj["vis_feet"],
        "true_emergence_active_maneuver": bj["e_true"] & man_j,
        "matched_continuously_visible_ctrl": bj["matched_ctrl"],
    }
    j_res = run_2x2_6slot_and_baseline_eval(
        "jrdb", dj["location"], dj["actors"], False, gdj, hj_np, tj_np, cv_j, dj["y_act"], dj["valid_act"], dj["onset"],
        j_strata, bj["e_true"], 0.4, 512, (45, 28, 28, 30), 34, (2.5e-3, 1.5e-3, 1.8e-3, 1.5e-3), 1.8e-3, list(args.seeds), dev, True,
    )

    # 2) JTA native uncorrupted urban sequences (val dev vs locked official test)
    st_zip = args.scratch_dir / "st_releases.zip"
    if not st_zip.exists():
        _download_verified_zip(ST_RELEASE_URL, st_zip)
    dt_jta, bt_jta = load_jta228_native_emergence_cohort(st_zip, max_train_files=80)
    # Use 4-fold cross-group over val+test evaluated against train+other folds (or train->test)
    nt = len(dt_jta["history"])
    ht_np, tt_np = dt_jta["history"].numpy(), dt_jta["truth"].numpy()
    gdt = {k: dt_jta[k].to(dev) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
    gdt["y_act"] = torch.from_numpy(dt_jta["y_act"]).to(dev)
    gdt["valid_act"] = torch.from_numpy(dt_jta["valid_act"]).to(dev)
    gdt["obs_dt"] = torch.full((nt, 1), 0.4, dtype=torch.float64, device=dev)
    gdt["fut_s"] = torch.arange(1, 9, dtype=torch.float64, device=dev)[None].expand(nt, -1) * 0.4
    cv_t = ((ht_np[:, 1:2] - ht_np[:, 0:1]) / 0.4) * (np.arange(1, 9)[None, :, None] * 0.4)
    # Assign train samples into 3 source groups and keep official val & test as separate held-out groups
    sp = dt_jta["split"]
    grp_t = np.where(sp == "test", "heldout_test", np.where(sp == "val", "dev_val", dt_jta["group"]))
    man_t = (dt_jta["y_act"][:, :4] & dt_jta["valid_act"][:, :4]).any(axis=1)
    t_strata = {
        "official_locked_test_all": sp == "test",
        "official_locked_test_true_emergence": (sp == "test") & bt_jta["e_true"],
        "official_locked_test_emergence_maneuver": (sp == "test") & bt_jta["e_true"] & man_t,
        "official_locked_test_steady_ctrl": (sp == "test") & bt_jta["prior_steady"],
        "dev_val_true_emergence": (sp == "val") & bt_jta["e_true"],
    }
    jta_res = run_2x2_6slot_and_baseline_eval(
        "jta_native", grp_t, dt_jta["seq"], False, gdt, ht_np, tt_np, cv_t, dt_jta["y_act"], dt_jta["valid_act"], dt_jta["onset"],
        t_strata, (sp == "test") & bt_jta["e_true"], 0.4, 512, (35, 22, 22, 24), 28, (2e-3, 1.4e-3, 1.5e-3, 1.4e-3), 1.6e-3, list(args.seeds), dev, False,
    )

    summary = {"experiment": "wave21_jrdb_and_jta_emergence_maneuvers", "jrdb_9location_lolo": j_res, "jta_native_urban_emergence": jta_res}
    raw = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = (json.dumps({"summary.json": hashlib.sha256(raw).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw); zf.writestr("sha256.json", sh)
    print("Saved:", args.zip_path, flush=True)


if __name__ == "__main__":
    main()
