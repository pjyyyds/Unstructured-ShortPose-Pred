"""Wave 23A: Camera-homography + raw-video (ori_*.mp4) blind PINNS emergence audit & 5-seed conditioned 6-slot ranker."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None

from scripts.benchmark_heldout_sites import HISTORY
from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import load_ppt_cohort_with_causal_maps
from scripts.run_wave18_conflict_gated_interaction_5seed import _loc_bootstrap_ci, safe_site_clip_ci
from scripts.run_wave21_jrdb_and_jta_emergence_maneuvers import ST_RELEASE_URL, _download_verified_zip, load_jrdb_with_blind_emergence
from scripts.run_wave21_pinns_true_emergence_and_6slot_diag import (
    SEEDS,
    VEH_PREFIXES,
    _fwd_6slot,
    fit_6slot_arm_fold,
)
from scripts.stage_pinns_metadata import RemoteDirectory, remote_member
from src.data.pinns_camera_occlusion import (
    camera_box_occludes_pedestrian,
    fit_verified_ground_to_camera_homography,
    project_vehicle_3d_to_camera_box,
)
from src.models.apa_precursor_leadtime import build_delayed_onset_start_stop_candidates
from src.models.cond_precursor_ranker import (
    ParityExactCondPrecursorNet,
    build_conditioned_precursor_features,
    select_medoid_expected_ade,
    summarize_6candidate_5seed,
)
from src.models.emergence_maneuver_6slot import (
    apply_6slot_selector,
    audit_6candidate_generation_vs_selection,
    compute_action_labels_with_onset,
    fit_strict_source_platt_and_slot_selector,
    horizon_and_post_onset_errors,
    score_site_isolated_actions,
)


def _fetch_needed_ori_videos(needed_clips: list[str], video_cache_dir: Path) -> dict[str, Path]:
    """Stream ONLY the required ori_*.mp4 clips from PINNS_URL via HTTP byte-range."""
    video_cache_dir.mkdir(parents=True, exist_ok=True)
    clip_to_member = {}
    for cl in needed_clips:
        cond, num = cl.rsplit("_", 1)
        clip_to_member[cl] = f"videos/{cond}/videos/ori_{num}.mp4"
    out_paths: dict[str, Path] = {}
    missing = [cl for cl, mem in clip_to_member.items() if not (video_cache_dir / f"{cl}.mp4").is_file()]
    if missing:
        with zipfile.ZipFile(RemoteDirectory()) as zf:
            info_map = {e.filename: e for e in zf.infolist()}
            for cl in missing:
                mem = clip_to_member[cl]
                if mem in info_map:
                    blob = remote_member(info_map[mem])
                    (video_cache_dir / f"{cl}.mp4").write_bytes(blob)
    for cl in needed_clips:
        p = video_cache_dir / f"{cl}.mp4"
        if p.is_file():
            out_paths[cl] = p
    return out_paths


def _decode_clip_frames_and_median(mp4_path: Path, target_raw0: set[int]) -> tuple[dict[int, np.ndarray], np.ndarray | None]:
    """Decode requested 0-based raw frames plus a 21-frame temporal median background image."""
    cap = cv2.VideoCapture(str(mp4_path))
    n_tot = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    bg_ids = set(np.linspace(0, max(n_tot - 1, 0), min(21, n_tot), dtype=int).tolist())
    want = sorted(k for k in (target_raw0 | bg_ids) if 0 <= k < n_tot)
    frames_out: dict[int, np.ndarray] = {}
    bg_stack: list[np.ndarray] = []
    if want:
        idx_ptr = 0
        f_cur = 0
        while idx_ptr < len(want):
            ok, img = cap.read()
            if not ok:
                break
            if f_cur == want[idx_ptr]:
                if f_cur in target_raw0:
                    frames_out[f_cur] = img.copy()
                if f_cur in bg_ids:
                    bg_stack.append(img.copy())
                idx_ptr += 1
            f_cur += 1
    cap.release()
    med = np.median(np.stack(bg_stack, axis=0), axis=0).astype(np.uint8) if bg_stack else None
    return frames_out, med


def parse_raw_30hz_camera_and_world(lbl_path: Path) -> tuple[dict[int, list[dict]], int, dict[int, int], np.ndarray]:
    """Parse raw 30Hz objects with both camera/BEV pixel fields and calibrated ground (wx, wy) in metres."""
    by_raw: dict[int, list[dict]] = {}
    first_p: dict[int, int] = {}
    ped_chk: list[list[float]] = []
    min_raw = 10**9
    with lbl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ps = ln.split()
            if len(ps) < 11:
                continue
            rf = int(float(ps[0]))
            min_raw = min(min_raw, rf)
            kind, oid = ps[1].strip(), int(float(ps[2]))
            ox, oy, _, bw, bh = map(float, ps[3:8])
            wx, wy = float(ps[8]), float(ps[9])
            by_raw.setdefault(rf, []).append({
                "type": kind, "id": oid, "ox": ox, "oy": oy, "w": bw, "h": bh,
                "wx": wx, "wy": wy, "fixed": int(float(ps[10])),
            })
            if kind.lower().startswith("per"):
                first_p[oid] = min(rf, first_p.get(oid, rf))
                if (rf - 1) % 3 == 0:
                    ped_chk.append([ox, oy, bw, bh, wx, wy])
    arr_chk = np.asarray(ped_chk, dtype=np.float64) if ped_chk else np.zeros((0, 6), dtype=np.float64)
    return by_raw, (0 if min_raw == 10**9 else min_raw), first_p, arr_chk


def blind_pinns_homography_and_raw_video_audit(
    meta_dir: Path,
    pose_npz: Path,
    dp: dict,
    video_cache_dir: Path,
    sheets_zip_path: Path | None = None,
    visual_override_json: Path | None = None,
) -> dict:
    """Prediction-blind camera-homography vehicle check + raw ori_*.mp4 frame inspection on PINNS."""
    from scripts.benchmark_heldout_sites import load_site_windows

    groups, _, _ = load_site_windows(meta_dir)
    rows = [r for s in sorted(groups) for r in groups[s]]
    keys = [(str(r.clip), int(r.person_id), int(r.t0)) for r in rows]
    n = len(rows)
    with np.load(pose_npz, allow_pickle=False) as pz:
        ord_map = {k: i for i, k in enumerate(zip(pz["clip"].tolist(), pz["person_id"].tolist(), pz["t0"].tolist()))}
        ids = [ord_map[k] for k in keys]
        kpts = pz["hist_kpts"][ids].astype(np.float64)
        bbox = pz["hist_bbox"][ids].astype(np.float64)

    clip_info: dict[str, dict] = {}
    reproj_errs: list[float] = []
    for r in rows:
        if r.clip in clip_info:
            continue
        cond, _ = r.clip.rsplit("_", 1)
        cal = json.loads((meta_dir / "videos" / cond / "calib.json").read_text())
        vw = float(cal["u_size"]) if "u_size" in cal else float(cal["v_size"][0])
        vh = float(cal["v_size"]) if isinstance(cal["v_size"], (int, float)) else float(cal["v_size"][1])
        by_raw, min_raw, first_p, ped_chk = parse_raw_30hz_camera_and_world(meta_dir / "labels" / f"{r.clip}.txt")
        h_g2i, y_conj, med_px = fit_verified_ground_to_camera_homography(cal, ped_chk)
        reproj_errs.append(med_px)
        starts_xy = []
        for pid, rf0 in first_p.items():
            if rf0 <= min_raw + 6:
                continue
            for ob in by_raw.get(rf0, []):
                if ob["type"].lower().startswith("per") and ob["id"] == pid:
                    x1, y1, x2, y2 = ob["ox"] - 0.5 * ob["w"], ob["oy"] - ob["h"], ob["ox"] + 0.5 * ob["w"], ob["oy"]
                    if x1 > 45.0 and y1 > 35.0 and x2 < vw - 45.0 and y2 < vh - 35.0:
                        starts_xy.append((ob["ox"], ob["oy"] - 0.5 * ob["h"]))
                    break
        clip_info[r.clip] = {"by_raw": by_raw, "min_raw": min_raw, "wh": (vw, vh), "h_g2i": h_g2i, "y_conj": y_conj, "med_px": med_px, "starts": starts_xy}

    # Identify non-Frame0 first_contiguous candidate windows & decode their ori_*.mp4 frames
    cand_indices: list[int] = []
    clip_req_raw0: dict[str, set[int]] = {}
    prior_20 = np.zeros(n, dtype=bool)
    e_clip_start = np.zeros(n, dtype=bool)
    for i, r in enumerate(rows):
        ci = clip_info[r.clip]
        by_raw, min_raw = ci["by_raw"], ci["min_raw"]
        s_10 = r.t0 - HISTORY + 1
        r_start = 3 * s_10 + 1
        prev_cnt = 0
        while any(o["type"].lower().startswith("per") and o["id"] == r.person_id for o in by_raw.get(3 * (s_10 - prev_cnt - 1) + 1, [])):
            prev_cnt += 1
        prior_20[i] = prev_cnt >= 20
        if prev_cnt > 0:
            continue
        if r_start <= min_raw + 6:
            e_clip_start[i] = True
            continue
        cand_indices.append(i)
        s0 = max(0, r_start - 1)
        for off in (-12, -6, 0, 6, 12):
            clip_req_raw0.setdefault(r.clip, set()).add(max(0, s0 + off))

    overrides = json.loads(visual_override_json.read_text()) if (visual_override_json and visual_override_json.is_file()) else {}
    need_vid_decode = (cv2 is not None) and ((not overrides) or (sheets_zip_path is not None))
    mp4_map = _fetch_needed_ori_videos(sorted(clip_req_raw0), video_cache_dir) if need_vid_decode else {}
    decoded_clips: dict[str, tuple[dict[int, np.ndarray], np.ndarray | None]] = {}
    for cl, reqs in clip_req_raw0.items():
        if cl in mp4_map:
            decoded_clips[cl] = _decode_clip_frames_and_median(mp4_map[cl], reqs)
    e_buggy_bev_veh = np.zeros(n, dtype=bool)
    e_veh_cam = np.zeros(n, dtype=bool)
    e_pers_cam = np.zeros(n, dtype=bool)
    e_static_vid = np.zeros(n, dtype=bool)
    e_border = np.zeros(n, dtype=bool)
    e_open_late = np.zeros(n, dtype=bool)
    strip_rows: list[tuple[str, np.ndarray]] = []
    manifest_records: list[dict] = []

    for i in cand_indices:
        r = rows[i]
        ci = clip_info[r.clip]
        by_raw, min_raw, (vw, vh), h_g2i, y_conj = ci["by_raw"], ci["min_raw"], ci["wh"], ci["h_g2i"], ci["y_conj"]
        r_start = 3 * (r.t0 - HISTORY + 1) + 1
        cx, cy, bw, bh = bbox[i, 0]
        px1, py1, px2, py2 = cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh
        near_border = (px1 <= 45.0) or (py1 <= 30.0) or (px2 >= vw - 45.0) or (py2 >= vh - 30.0)

        bug_v, cam_v, cam_p = False, False, False
        cam_v_boxes: list[tuple[float, float, float, float]] = []
        for rf in range(max(min_raw, r_start - 15), r_start + 4):
            for ob in by_raw.get(rf, []):
                if ob["type"].lower().startswith("per") and ob["id"] == r.person_id:
                    continue
                if ob["type"].lower().startswith(VEH_PREFIXES):
                    if (abs(ob["ox"] - cx) - 0.55 * (ob["w"] + bw) <= 20.0) and (ob["oy"] >= py1 + 0.25 * bh):
                        bug_v = True
                    hdg = None
                    for rf_n in (rf + 3, rf - 3):
                        for o2 in by_raw.get(rf_n, []):
                            if o2["type"] == ob["type"] and o2["id"] == ob["id"]:
                                sgn = 1.0 if rf_n > rf else -1.0
                                hdg = (sgn * (o2["wx"] - ob["wx"]), sgn * (o2["wy"] - ob["wy"]))
                                break
                    vbox, _ = project_vehicle_3d_to_camera_box(ob["wx"], ob["wy"], ob["type"], h_g2i, y_conj, hdg)
                    if rf == r_start:
                        cam_v_boxes.append(vbox)
                    if camera_box_occludes_pedestrian(vbox, (px1, py1, px2, py2)):
                        cam_v = True
                elif ob["type"].lower().startswith("per"):
                    pbox = (ob["ox"] - 0.5 * ob["w"], ob["oy"] - ob["h"], ob["ox"] + 0.5 * ob["w"], ob["oy"])
                    if camera_box_occludes_pedestrian(pbox, (px1, py1, px2, py2)):
                        cam_p = True
        e_buggy_bev_veh[i] = bug_v
        arch_portal = (not near_border) and sum(np.hypot(sx - cx, sy - cy) <= max(65.0, 1.2 * bw) for sx, sy in ci["starts"]) >= 2

        # Raw video pixel check around pedestrian initial box [px1, py1, px2, py2]
        already_standing_open = False
        vid_has_fg_change = True
        frames_d, med_img = decoded_clips.get(r.clip, ({}, None))
        f0_idx = max(0, r_start - 1)
        if f0_idx in frames_d and max(0, f0_idx - 6) in frames_d:
            im0, im_pre = frames_d[f0_idx], frames_d[max(0, f0_idx - 6)]
            ih, iw = im0.shape[:2]
            rx1, ry1 = int(np.clip(px1, 0, iw - 2)), int(np.clip(py1, 0, ih - 2))
            rx2, ry2 = int(np.clip(px2, rx1 + 2, iw)), int(np.clip(py2, ry1 + 2, ih))
            pre_diff = float(np.abs(im0[ry1:ry2, rx1:rx2].astype(np.float32) - im_pre[ry1:ry2, rx1:rx2].astype(np.float32)).mean())
            bg_diff = float(np.abs(im_pre[ry1:ry2, rx1:rx2].astype(np.float32) - med_img[ry1:ry2, rx1:rx2].astype(np.float32)).mean()) if med_img is not None else 25.0
            already_standing_open = (pre_diff < 9.5) and (bg_diff > 18.0) and (not cam_v) and (not cam_p)
            vid_has_fg_change = pre_diff >= 9.5

            if sheets_zip_path is not None and cv2 is not None:
                tiles = []
                pad_x, pad_y = int(max(55.0, 1.3 * bw)), int(max(55.0, 0.9 * bh))
                cx1, cy1 = max(0, int(cx) - pad_x), max(0, int(cy) - pad_y)
                cx2, cy2 = min(iw, int(cx) + pad_x), min(ih, int(cy) + pad_y)
                for off_f in (-12, -6, 0, 6, 12):
                    im_t = frames_d.get(max(0, f0_idx + off_f), im0).copy()
                    for vx1, vy1, vx2, vy2 in cam_v_boxes:
                        cv2.rectangle(im_t, (int(vx1), int(vy1)), (int(vx2), int(vy2)), (255, 220, 0), 2)
                    col = (0, 255, 0) if off_f == 0 else (180, 180, 180)
                    cv2.rectangle(im_t, (int(px1), int(py1)), (int(px2), int(py2)), col, 2 if off_f == 0 else 1)
                    crop = cv2.resize(im_t[cy1:cy2, cx1:cx2], (112, 112))
                    cv2.putText(crop, f"{off_f/30:+.1f}s", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                    tiles.append(crop)
                row_img = np.concatenate(tiles, axis=1)
                strip_rows.append((f"idx={i} {r.clip} pid={r.person_id} f={r_start}", row_img))

        ov_cls = overrides.get(f"{r.clip}:{r.person_id}:{r.t0}", "")
        if ov_cls == "veh":
            e_veh_cam[i] = True; cls_tag = "veh_cam_occ"
        elif ov_cls == "pers":
            e_pers_cam[i] = True; cls_tag = "pers_cam_occ"
        elif ov_cls == "static":
            e_static_vid[i] = True; cls_tag = "static_vid_occ"
        elif ov_cls == "border":
            e_border[i] = True; cls_tag = "border_entry"
        elif ov_cls == "open":
            e_open_late[i] = True; cls_tag = "open_late_pickup"
        elif cam_v:
            e_veh_cam[i] = True; cls_tag = "veh_cam_occ"
        elif cam_p:
            e_pers_cam[i] = True; cls_tag = "pers_cam_occ"
        elif arch_portal and vid_has_fg_change and (not already_standing_open):
            e_static_vid[i] = True; cls_tag = "static_vid_occ"
        elif near_border:
            e_border[i] = True; cls_tag = "border_entry"
        else:
            e_open_late[i] = True; cls_tag = "open_late_pickup"
        manifest_records.append({
            "window_idx": int(i), "clip": str(r.clip), "person_id": int(r.person_id), "t0": int(r.t0),
            "r_start_30hz": int(r_start), "buggy_bev_veh": bool(bug_v), "homog_cam_veh": bool(cam_v),
            "cam_pers_occ": bool(cam_p), "assigned_class": cls_tag,
        })

    if sheets_zip_path is not None and strip_rows and cv2 is not None:
        sheets_zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(sheets_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as szf:
            per_page = 12
            for pg in range(0, len(strip_rows), per_page):
                chunk = strip_rows[pg : pg + per_page]
                canvas = np.zeros((len(chunk) * 134, 560, 3), dtype=np.uint8)
                for ri, (lbl_txt, row_im) in enumerate(chunk):
                    y0 = ri * 134
                    cv2.putText(canvas, lbl_txt, (6, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 245, 255), 1)
                    canvas[y0 + 22 : y0 + 134, :560] = row_im
                ok_enc, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
                if ok_enc:
                    szf.writestr(f"sheet_{pg // per_page:02d}.jpg", buf.tobytes())
            szf.writestr("blind_manifest.json", (json.dumps(manifest_records, indent=2) + "\n").encode("utf-8"))

    e_phys_all = e_veh_cam | e_pers_cam | e_static_vid
    bh_med = np.median(bbox[:, :, 3], axis=1)
    sh_hip_ok = ((kpts[:, :, [5, 6, 11, 12], 2] >= 0.25).all(axis=-1).sum(axis=1) >= 3) & (bh_med >= 50.0)
    feet_ok = ((kpts[:, :, [15, 16], 2] >= 0.25).all(axis=-1).sum(axis=1) >= 3) & (bh_med >= 50.0)
    disp5 = np.linalg.norm(np.diff(dp["history"].numpy(), axis=1), axis=-1).sum(axis=1)
    v0 = np.linalg.norm(dp["history"].numpy()[:, -1] - dp["history"].numpy()[:, -2], axis=-1) / 0.1

    ctrl = np.zeros(n, dtype=bool)
    site_arr, clip_arr = dp["site"], dp["clip"]
    for idx in np.flatnonzero(e_phys_all):
        cand = np.flatnonzero(prior_20 & (~ctrl) & (site_arr == site_arr[idx]))
        if len(cand) > 0:
            cost = np.abs(v0[cand] - v0[idx]) / 0.5 + np.abs(bh_med[cand] - bh_med[idx]) / 40.0
            ctrl[cand[int(np.argmin(cost))]] = True

    dev_clip_m = np.zeros(n, dtype=bool)
    for s in sorted(set(site_arr.tolist())):
        s_clips = sorted(set(clip_arr[site_arr == s].tolist()))
        dev_clip_m |= np.isin(clip_arr, s_clips[0::2])

    return {
        "e_buggy_bev_veh": e_buggy_bev_veh, "e_veh_cam": e_veh_cam, "e_pers_cam": e_pers_cam,
        "e_static_vid": e_static_vid, "e_phys_all": e_phys_all, "e_border": e_border,
        "e_clip_start": e_clip_start, "e_open_late": e_open_late, "vis_torso": sh_hip_ok,
        "vis_feet": feet_ok, "disp5": disp5, "matched_visible_ctrl": ctrl,
        "dev_clips": dev_clip_m, "locked_eval_clips": ~dev_clip_m,
        "median_ped_reproj_px_across_clips": float(np.median(reproj_errs)) if reproj_errs else 0.0,
        "n_videos_decoded": len(decoded_clips),
    }


def _fit_cond_precursor_head(
    seed: int,
    feat_all: torch.Tensor,
    base_act_tr: torch.Tensor,
    y_act_tr: torch.Tensor,
    v_act_tr: torch.Tensor,
    tr_ids: np.ndarray,
    device: torch.device,
    epochs: int = 45,
) -> ParityExactCondPrecursorNet:
    torch.manual_seed(seed + 911)
    head = ParityExactCondPrecursorNet().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3.5e-3, weight_decay=2e-3)
    x_tr = feat_all[torch.from_numpy(tr_ids).to(device)]
    pw = ((v_act_tr.float().sum(0) - y_act_tr.float().sum(0)) / y_act_tr.float().sum(0).clamp_min(1.0)).clamp(1.0, 14.0)
    head.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        lg = head(x_tr, base_act_tr)
        bce = F.binary_cross_entropy_with_logits(lg, y_act_tr.float(), pos_weight=pw, reduction="none")
        loss = (bce * v_act_tr.float()).sum() / v_act_tr.float().sum().clamp_min(1.0)
        loss.backward()
        opt.step()
    return head.eval()


def eval_cohort_wave23_5seed(
    tag: str,
    grp_arr: np.ndarray,
    unit_arr: np.ndarray,
    gd: dict,
    hist_np: np.ndarray,
    truth_np: np.ndarray,
    ya_np: np.ndarray,
    va_np: np.ndarray,
    onset_np: np.ndarray,
    strata: dict[str, np.ndarray],
    seeds: list[int],
    device: torch.device,
    dt: float = 0.1,
    bsz: int = 64,
    ep_slot: int = 28,
    lr_slot: float = 2.2e-3,
    is_pinns: bool = True,
) -> dict:
    n, tf = len(hist_np), truth_np.shape[1]
    groups = sorted(set(grp_arr.tolist()))
    t_s = np.arange(1, tf + 1, dtype=np.float64)[None, :, None] * dt
    cv_rel = ((hist_np[:, -1:] - hist_np[:, -2:-1]) / dt) * t_s
    err_cv = horizon_and_post_onset_errors(cv_rel + hist_np[:, -1:, :], truth_np, onset_np, dt=dt)

    slot_arms = (
        ("S6_M0_nopose_nonbr", False, False),
        ("S6_M1_pose_nonbr", True, False),
        ("S6_M2b_nopose_nbr", False, True),
        ("S6_Full_pose_nbr", True, True),
    )
    all_names = [a[0] for a in slot_arms] + ["S6_Full_CondMedoid", "S6_Full_CondF1Slot"]
    err_s = {nm: np.zeros((len(seeds), n, 6), dtype=np.float64) for nm in all_names}
    audits_per_seed: dict[str, list[dict]] = {nm: [] for nm in ("S6_M0_nopose_nonbr", "S6_Full_pose_nbr", "S6_Full_CondMedoid", "S6_Full_CondF1Slot")}
    act_prob_cond = np.zeros((len(seeds), n, 5), dtype=np.float64)
    act_bin_cond = np.zeros((len(seeds), n, 5), dtype=np.float64)

    with torch.inference_mode():
        feat20 = build_conditioned_precursor_features(
            gd["history"], gd["bio_even"], gd["pose_vec_world"], gd["signed_knee"],
            gd["quality"], gd["m_pose"], gd["edges"], gd["is_car"], gd["nbr_mask"], gd["obs_dt"],
        )
    cor_xy = gd["pose_vec_world"][:, -1, 0].detach().cpu().numpy()
    fut_s_np = gd["fut_s"].detach().cpu().numpy()
    ya_t = torch.from_numpy(ya_np).to(device)
    va_t = torch.from_numpy(va_np).to(device)
    gd["y_act"] = ya_t
    gd["valid_act"] = va_t
    truth_rel = truth_np - hist_np[:, -1:, :]
    diag_mask = strata.get("diag_emergence_union", np.ones(n, dtype=bool))

    for si, seed in enumerate(seeds):
        pk_seed = {nm: np.zeros((n, 6, tf, 2), dtype=np.float64) for nm in audits_per_seed}
        ix_seed = {nm: np.zeros(n, dtype=np.int64) for nm in audits_per_seed}
        for g in groups:
            tr, te = np.flatnonzero(grp_arr != g), np.flatnonzero(grp_arr == g)
            te_g = torch.from_numpy(te).to(device)
            src_g = sorted(set(grp_arr[tr].tolist()))
            val_sets = [src_g[k :: min(3, max(1, len(src_g)))] for k in range(min(3, max(1, len(src_g))))] if len(src_g) >= 2 else [src_g]

            for arm_nm, u_pose, u_nbr in slot_arms:
                oof_ml = np.zeros((len(tr), 6), dtype=np.float64)
                oof_al = np.zeros((len(tr), 5), dtype=np.float64)
                oof_pk = np.zeros((len(tr), 6, tf, 2), dtype=np.float64)
                oof_cal_al = np.zeros((len(tr), 5), dtype=np.float64)
                for v_locs in val_sets:
                    vm_l = np.isin(grp_arr[tr], v_locs) if len(val_sets) > 1 else (np.arange(len(tr)) % 2 == 0)
                    in_tr, in_va = tr[~vm_l], tr[vm_l]
                    m_in = fit_6slot_arm_fold(seed, u_pose, u_nbr, grp_arr, gd, in_tr, bsz, max(18, ep_slot - 8), lr_slot, device)
                    with torch.inference_mode():
                        oo = _fwd_6slot(m_in, gd, torch.from_numpy(in_va).to(device))
                        otr = _fwd_6slot(m_in, gd, torch.from_numpy(in_tr).to(device))
                    oof_ml[vm_l] = oo["logits"].cpu().numpy()
                    oof_al[vm_l] = oo["act_logits"].cpu().numpy()
                    oof_pk[vm_l] = oo["paths"].cpu().numpy()
                    if arm_nm == "S6_Full_pose_nbr":
                        ch_in = _fit_cond_precursor_head(seed, feat20, otr["act_logits"].detach(), ya_t[in_tr], va_t[in_tr], in_tr, device)
                        with torch.inference_mode():
                            oof_cal_al[vm_l] = ch_in(feat20[torch.from_numpy(in_va).to(device)], oo["act_logits"]).cpu().numpy()

                cal_s = fit_strict_source_platt_and_slot_selector(oof_ml, oof_al, oof_pk, truth_np[tr], ya_np[tr], va_np[tr])
                m_f = fit_6slot_arm_fold(seed, u_pose, u_nbr, grp_arr, gd, tr, bsz, ep_slot, lr_slot, device)
                with torch.inference_mode():
                    ot = _fwd_6slot(m_f, gd, te_g)
                    otr_f = _fwd_6slot(m_f, gd, torch.from_numpy(tr).to(device))
                    pwt, lgt, alt = ot["paths"].cpu().numpy(), ot["logits"].cpu().numpy(), ot["act_logits"].cpu().numpy()
                sel_i, pc_s, _ = apply_6slot_selector(lgt, alt, va_np[te], cal_s)
                err_s[arm_nm][si, te] = horizon_and_post_onset_errors(pwt[np.arange(len(te)), sel_i], truth_np[te], onset_np[te], dt=dt)
                if arm_nm in pk_seed:
                    pk_seed[arm_nm][te], ix_seed[arm_nm][te] = pwt - hist_np[te, -1:, None, :], sel_i
                if arm_nm == "S6_Full_pose_nbr":
                    # Upgrade Slots 1 & 2 with Delayed-Onset APA curves calibrated on source median onset step
                    med_st = float(np.median(onset_np[tr][ya_np[tr, 0], 0]) * dt) if ya_np[tr, 0].any() else 0.7
                    med_sp = float(np.median(onset_np[tr][ya_np[tr, 1], 1]) * dt) if ya_np[tr, 1].any() else 0.6
                    st_o, sp_o = build_delayed_onset_start_stop_candidates(hist_np[tr], cor_xy[tr], np.full(len(tr), med_st), fut_s_np[tr], dt_obs=dt)
                    oof_pk_apa = oof_pk.copy(); oof_pk_apa[:, 1] = st_o; oof_pk_apa[:, 2] = sp_o
                    cal_c = fit_strict_source_platt_and_slot_selector(oof_ml, oof_cal_al, oof_pk_apa, truth_np[tr], ya_np[tr], va_np[tr])
                    ch_f = _fit_cond_precursor_head(seed, feat20, otr_f["act_logits"].detach(), ya_t[tr], va_t[tr], tr, device)
                    with torch.inference_mode():
                        alt_c = ch_f(feat20[te_g], ot["act_logits"]).cpu().numpy()
                    _, pc_c, pb_c = apply_6slot_selector(lgt, alt_c, va_np[te], cal_c)
                    act_prob_cond[si, te], act_bin_cond[si, te] = pc_c, pb_c.astype(np.float64)
                    st_t, sp_t = build_delayed_onset_start_stop_candidates(hist_np[te], cor_xy[te], np.full(len(te), med_st), fut_s_np[te], dt_obs=dt)
                    pwt_apa = pwt.copy(); pwt_apa[:, 1] = st_t; pwt_apa[:, 2] = sp_t
                    idx_med = select_medoid_expected_ade(pwt_apa, lgt, pc_c, va_np[te], beta=max(float(cal_c["beta"]), 0.18), sw_thr=np.asarray(cal_c["sw_thr"]))
                    idx_f1 = select_medoid_expected_ade(pwt_apa, lgt, pc_c, va_np[te], beta=0.25, sw_thr=np.asarray(cal_c["thr_f1"])[:4])
                    err_s["S6_Full_CondMedoid"][si, te] = horizon_and_post_onset_errors(pwt_apa[np.arange(len(te)), idx_med], truth_np[te], onset_np[te], dt=dt)
                    err_s["S6_Full_CondF1Slot"][si, te] = horizon_and_post_onset_errors(pwt_apa[np.arange(len(te)), idx_f1], truth_np[te], onset_np[te], dt=dt)
                    pk_seed["S6_Full_CondMedoid"][te], ix_seed["S6_Full_CondMedoid"][te] = pwt_apa - hist_np[te, -1:, None, :], idx_med
                    pk_seed["S6_Full_CondF1Slot"][te], ix_seed["S6_Full_CondF1Slot"][te] = pwt_apa - hist_np[te, -1:, None, :], idx_f1
        for nm_a in audits_per_seed:
            audits_per_seed[nm_a].append(
                audit_6candidate_generation_vs_selection(
                    hist_np, pk_seed[nm_a], ix_seed[nm_a], truth_rel, cv_rel, ya_np, va_np, diag_mask, dt=dt,
                )
            )
        print(f"[W23 {tag}] Finished seed={seed}", flush=True)

    em = {nm: err_s[nm].mean(axis=0) for nm in all_names}

    def _ci2(diff2: np.ndarray, msk: np.ndarray) -> dict:
        if int(msk.sum()) < 4:
            return {"n": int(msk.sum())}
        if is_pinns:
            return safe_site_clip_ci(diff2[msk], grp_arr[msk], unit_arr[msk])
        if len(set(grp_arr[msk].tolist())) >= 2:
            return _loc_bootstrap_ci(diff2[msk], grp_arr[msk])
        return {"mean": {"ade_m": float(diff2[msk, 0].mean()), "fde_m": float(diff2[msk, 1].mean())}}

    def _agg_row(arr6: np.ndarray, sm: np.ndarray) -> dict[str, float]:
        gs = sorted(set(grp_arr[sm].tolist()))
        vals = np.mean([arr6[(grp_arr == g) & sm].mean(axis=0) for g in gs], axis=0)
        return {
            "ade1_full_m": float(vals[0]), "fde1_full_m": float(vals[1]),
            "ade1_trig_0p5s_m": float(vals[2]), "ade1_trig_1p0s_m": float(vals[3]),
            "ade1_post_onset_0p5s_m": float(vals[4]), "ade1_post_onset_1p0s_m": float(vals[5]),
        }

    strata_out = {}
    for s_nm, msk in strata.items():
        cnt = int(msk.sum())
        if cnt == 0:
            strata_out[s_nm] = {"n_windows": 0}
            continue
        mods_s = {"Last2_CV": _agg_row(err_cv, msk)}
        for nm in all_names:
            mods_s[nm] = _agg_row(em[nm], msk)
        strata_out[s_nm] = {
            "n_windows": cnt,
            "n_clips_or_units": len(set(unit_arr[msk].tolist())),
            "n_sites_or_locs": len(set(grp_arr[msk].tolist())),
            "maneuver_counts": {k: int(ya_np[msk, c].sum()) for c, k in enumerate(("Start", "Stop", "Turn_Left", "Turn_Right", "Maintain"))},
            "models": mods_s,
            "paired_ci": {
                "cond_medoid_minus_CV_3s": _ci2(em["S6_Full_CondMedoid"][:, :2] - err_cv[:, :2], msk),
                "full_minus_CV_3s": _ci2(em["S6_Full_pose_nbr"][:, :2] - err_cv[:, :2], msk),
                "pose_effect_no_nbr_3s": _ci2(em["S6_M1_pose_nonbr"][:, :2] - em["S6_M0_nopose_nonbr"][:, :2], msk),
                "pose_effect_no_nbr_post_onset": _ci2(em["S6_M1_pose_nonbr"][:, 4:6] - em["S6_M0_nopose_nonbr"][:, 4:6], msk),
                "pose_effect_with_nbr_3s": _ci2(em["S6_Full_pose_nbr"][:, :2] - em["S6_M2b_nopose_nbr"][:, :2], msk),
                "pose_effect_with_nbr_post_onset": _ci2(em["S6_Full_pose_nbr"][:, 4:6] - em["S6_M2b_nopose_nbr"][:, 4:6], msk),
                "cond_medoid_minus_map_full_3s": _ci2(em["S6_Full_CondMedoid"][:, :2] - em["S6_Full_pose_nbr"][:, :2], msk),
            },
        }

    return {
        "n_windows": n,
        "strata": strata_out,
        "candidate_generation_vs_selection_5seed": {nm: summarize_6candidate_5seed(audits_per_seed[nm]) for nm in audits_per_seed},
        "site_isolated_actions_cond_precursor": score_site_isolated_actions(ya_np, va_np, act_prob_cond, act_bin_cond, grp_arr),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--pose-npz", type=Path, required=True)
    ap.add_argument("--joint-zip", type=Path, required=True)
    ap.add_argument("--causal-map-bin", type=Path, required=True)
    ap.add_argument("--video-cache-dir", type=Path, required=True)
    ap.add_argument("--sheets-zip", type=Path, default=None)
    ap.add_argument("--visual-override-json", type=Path, default=None)
    ap.add_argument("--stage-sheets-only", action="store_true")
    ap.add_argument("--run-jrdb", action="store_true")
    ap.add_argument("--scratch-dir", type=Path, default=Path("/tmp/w23_jrdb"))
    ap.add_argument("--zip-path", type=Path, required=True)
    args = ap.parse_args()

    from scripts.run_jrdb_rc2_rc3_graph_and_actions_5seed import _stage_gpu_dict

    dp = load_ppt_cohort_with_causal_maps(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin)
    blind = blind_pinns_homography_and_raw_video_audit(
        args.meta_dir, args.pose_npz, dp, args.video_cache_dir, args.sheets_zip, args.visual_override_json,
    )
    f323_c = dp["strata"]["first_short_323"]
    census = {
        "median_ped_reproj_px": blind["median_ped_reproj_px_across_clips"],
        "n_ori_videos_decoded": blind["n_videos_decoded"],
        "visual_override_applied": bool(args.visual_override_json and args.visual_override_json.is_file()),
        "old_buggy_bev_vehicle_windows": int(blind["e_buggy_bev_veh"].sum()),
        "homography_or_visual_vehicle_windows": int(blind["e_veh_cam"].sum()),
        "old_buggy_bev_survived_homography": int((blind["e_buggy_bev_veh"] & blind["e_veh_cam"]).sum()),
        "cam_person_occluder_windows": int(blind["e_pers_cam"].sum()),
        "video_verified_static_portal_windows": int(blind["e_static_vid"].sum()),
        "total_verified_physical_emergence_windows": int(blind["e_phys_all"].sum()),
        "border_entry_windows": int(blind["e_border"].sum()),
        "clip_start_frame0_windows": int(blind["e_clip_start"].sum()),
        "open_field_or_late_detector_pickup_windows": int(blind["e_open_late"].sum()),
        "phys_emergence_clear_torso_ge50px": int((blind["e_phys_all"] & blind["vis_torso"]).sum()),
        "phys_emergence_clear_feet_ge50px": int((blind["e_phys_all"] & blind["vis_feet"]).sum()),
        "first_short_323_breakdown": {
            "n_windows": int(f323_c.sum()),
            "vehicle_occluder": int((f323_c & blind["e_veh_cam"]).sum()),
            "person_occluder": int((f323_c & blind["e_pers_cam"]).sum()),
            "static_portal_occluder": int((f323_c & blind["e_static_vid"]).sum()),
            "total_phys_emergence": int((f323_c & blind["e_phys_all"]).sum()),
            "border_entry": int((f323_c & blind["e_border"]).sum()),
            "clip_start_frame0": int((f323_c & blind["e_clip_start"]).sum()),
            "open_late_pickup": int((f323_c & blind["e_open_late"]).sum()),
        },
    }
    if args.stage_sheets_only:
        args.zip_path.write_text(json.dumps(census, indent=2) + "\n", encoding="utf-8")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hist_np, truth_np = dp["history"].numpy(), dp["truth"].numpy()
    n = len(hist_np)
    hm, fm = np.ones(5, dtype=np.float32), np.ones(30, dtype=np.float32)
    ya_np = np.zeros((n, 5), dtype=bool)
    va_np = np.zeros((n, 5), dtype=bool)
    onset_np = np.full((n, 5), -1, dtype=np.int64)
    for i in range(n):
        ya_np[i], va_np[i], onset_np[i] = compute_action_labels_with_onset(
            hist_np[i] - hist_np[i, -1:], hm, truth_np[i] - hist_np[i, -1:], fm, dt=0.1,
        )
    act_man = (ya_np[:, :4] & va_np[:, :4]).any(axis=1)
    turn_man = (ya_np[:, 2] & va_np[:, 2]) | (ya_np[:, 3] & va_np[:, 3])
    start_stop_man = (ya_np[:, 0] & va_np[:, 0]) | (ya_np[:, 1] & va_np[:, 1])
    e_phys = blind["e_phys_all"]
    f323 = dp["strata"]["first_short_323"]
    p_strata = {
        "all_1482": np.ones(n, dtype=bool),
        "verified_phys_emergence_all": e_phys,
        "verified_cam_vehicle_emergence": blind["e_veh_cam"],
        "verified_phys_emergence_locked_eval": e_phys & blind["locked_eval_clips"],
        "verified_phys_emergence_active_maneuvers": e_phys & act_man,
        "verified_phys_emergence_turn_only": e_phys & turn_man,
        "verified_phys_emergence_start_or_stop": e_phys & start_stop_man,
        "matched_continuously_visible_ctrl": blind["matched_visible_ctrl"],
        "late_detector_pickup_open_view": blind["e_open_late"],
        "clip_start_frame0": blind["e_clip_start"],
        "first_short_323": f323,
        "first_short_323_verified_phys_only": f323 & e_phys,
        "diag_emergence_union": e_phys | f323,
    }
    gd_p = {k: dp[k].to(device) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
    gd_p["obs_dt"] = torch.full((n, 4), 0.1, dtype=torch.float64, device=device)
    gd_p["fut_s"] = torch.arange(1, 31, dtype=torch.float64, device=device)[None].expand(n, -1) * 0.1
    pinns_res = eval_cohort_wave23_5seed(
        "PINNS", dp["site"], dp["clip"], gd_p, hist_np, truth_np, ya_np, va_np, onset_np,
        p_strata, list(SEEDS), device, dt=0.1, bsz=64, ep_slot=34, lr_slot=1.6e-3, is_pinns=True,
    )
    out = {"experiment": "wave23_pinns_video_audit_and_6slot_ranker", "pinns_blind_video_homography_census": census, "pinns_4site_loso": pinns_res}

    if args.run_jrdb:
        repo_root = Path(__file__).resolve().parents[1]
        dj, bj = load_jrdb_with_blind_emergence(repo_root)
        nj = len(dj["history"])
        hj, tj = dj["history"].numpy(), dj["truth"].numpy()
        gd_j = {k: dj[k].to(device) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
        gd_j["obs_dt"] = torch.full((nj, 1), 0.4, dtype=torch.float64, device=device)
        gd_j["fut_s"] = torch.arange(1, tj.shape[1] + 1, dtype=torch.float64, device=device)[None].expand(nj, -1) * 0.4
        yj, vj, oj = dj["y_act"], dj["valid_act"], dj["onset"]
        j_strata = {
            "all_5836": np.ones(nj, dtype=bool),
            "locked_4loc_eval": bj["locked_eval_mask"],
            "true_deocclusion_507": bj["e_true"],
            "true_deocclusion_active_maneuvers": bj["e_true"] & (yj[:, :4] & vj[:, :4]).any(axis=1),
            "diag_emergence_union": bj["e_true"],
        }
        out["jrdb_9loc_lolo"] = eval_cohort_wave23_5seed(
            "JRDB", dj["location"], dj["actors"], gd_j, hj, tj, yj, vj, oj,
            j_strata, list(SEEDS), device, dt=0.4, bsz=512, ep_slot=34, lr_slot=1.8e-3, is_pinns=False,
        )

    raw = (json.dumps(out, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = (json.dumps({"summary.json": hashlib.sha256(raw).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw)
        zf.writestr("sha256.json", sh)
    print("SAVED:", args.zip_path, args.zip_path.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
