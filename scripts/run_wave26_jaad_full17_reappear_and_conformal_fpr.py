"""Wave 26B: JAAD 2.0 Full-17 Head/Torso/Arm/Ankle Split AutoPose, Post-Reappearance Alarms, NP Conformal FPR, & Trajectory."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from scripts.run_wave18_conflict_gated_interaction_5seed import _loc_bootstrap_ci
from scripts.run_wave23_jaad_autopose_and_isofpr_leadtime import (
    JAAD_ZIP_URL,
    SEEDS,
    TAUS_10HZ,
    VEH_ACT_MAP,
    _fetch_url_bytes,
    _jaad_post_onset_err,
    _kpts5_to_autopose,
    extract_jaad_rtmpose_keypoints,
)
from scripts.run_wave25_jaad_reappearance_per_event_audit import (
    audit_per_event_timestamps_and_visible_alarms,
    exact_mcnemar_two_sided_p,
    extract_jaad_slide_reappearance_metadata,
)
from src.metrics.metrics_and_stats import _binary_average_precision
from src.models.emergence_kalman_residual_6slot import (
    calibrate_coherence_kalman_params,
    compute_coherence_kalman_velocity,
)
from src.models.jaad_full17_reappear_models import (
    JAADCoherenceSplitTrajectoryNet,
    JAADPostReappearPrecursorNet,
    calibrate_conformal_np_thresholds,
    kpts5_to_full17_split_autopose,
)


def build_jaad2_full17_cohorts(scratch_dir: Path) -> tuple[dict, dict]:
    """Build JAAD 2.0 trajectory (N=2363, test=953) & sliding cohorts with BOTH Old-12D and Full17-20D Split AutoPose."""
    jaad_zip = scratch_dir / "jaad2.zip"
    blob = _fetch_url_bytes(JAAD_ZIP_URL, jaad_zip)
    splits: dict[str, str] = {}
    parsed_tracks = []
    vid_frame_boxes: dict[str, dict[int, dict[str, tuple[float, float, float, float]]]] = {}

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        pfx = "JAAD-JAAD_2.0/"
        for sp in ("train", "val", "test"):
            for vid in zf.read(f"{pfx}split_ids/default/{sp}.txt").decode("utf-8").splitlines():
                if vid.strip():
                    splits[vid.strip()] = sp
        for af in sorted(n for n in zf.namelist() if n.startswith(f"{pfx}annotations/video_") and n.endswith(".xml")):
            vid = Path(af).stem
            sp = splits.get(vid, "train")
            root = ET.fromstring(zf.read(af))
            app_name = f"{pfx}annotations_appearance/{vid}_appearance.xml"
            pose_map: dict[tuple[str, int], np.ndarray] = {}
            if app_name in zf.namelist():
                for tr in ET.fromstring(zf.read(app_name)).findall(".//track"):
                    pa = tr.attrib.get("id", "")
                    for bx in tr.findall("box"):
                        pose_map[(pa, int(bx.attrib["frame"]))] = np.array([
                            float(bx.attrib.get("pose_front", "0")), float(bx.attrib.get("pose_back", "0")),
                            float(bx.attrib.get("pose_left", "0")), float(bx.attrib.get("pose_right", "0")),
                        ], dtype=np.float32)
            veh_name = f"{pfx}annotations_vehicle/{vid}_vehicle.xml"
            veh_map: dict[int, float] = {}
            if veh_name in zf.namelist():
                for vf in ET.fromstring(zf.read(veh_name)).findall(".//frame"):
                    veh_map[int(vf.attrib["id"])] = VEH_ACT_MAP.get(vf.attrib.get("action", "moving_slow"), 0.5)
            for tr_i, tr in enumerate(root.findall(".//track")):
                if tr.attrib.get("label") != "pedestrian":
                    continue
                boxes: dict[int, dict] = {}
                pid_t = ""
                for bx in tr.findall("box"):
                    if bx.attrib.get("outside", "0") == "1":
                        continue
                    fr = int(bx.attrib["frame"])
                    at = {a.attrib.get("name", ""): (a.text or "").strip() for a in bx.findall("attribute")}
                    pid_t = at.get("id", pid_t)
                    xtl, ytl, xbr, ybr = float(bx.attrib["xtl"]), float(bx.attrib["ytl"]), float(bx.attrib["xbr"]), float(bx.attrib["ybr"])
                    w, h = max(xbr - xtl, 4.0), max(ybr - ytl, 12.0)
                    boxes[fr] = {
                        "fcx": 0.5 * (xtl + xbr), "cy": 0.5 * (ytl + ybr), "fby": ybr, "w": w, "h": h,
                        "occ": at.get("occlusion", "none"), "walk": 1 if at.get("action", "standing") == "walking" else 0,
                        "look": 1.0 if at.get("look", "") == "looking" else 0.0,
                        "gest": 0.0 if at.get("hand_gesture", "__undefined__") == "__undefined__" else 1.0,
                        "nod": 1.0 if at.get("nod", "__undefined__") == "nodding" else 0.0,
                        "cross": 1.0 if at.get("cross", "") == "crossing" else 0.0,
                    }
                if len(boxes) >= 48:
                    parsed_tracks.append((vid, sp, f"{tr_i}:{pid_t}", pid_t, boxes, pose_map, veh_map))

    traj_specs, slide_specs = [], []
    ev_counter = 0

    def _reg(vid: str, trk_k: str, boxes: dict[int, dict], h_r: list[int]) -> None:
        vd = vid_frame_boxes.setdefault(vid, {})
        for rf in h_r:
            b = boxes[rf]
            vd.setdefault(rf, {})[trk_k] = (b["fcx"], b["cy"], b["w"], b["h"])

    for vid, sp, trk_k, pid_t, boxes, pose_map, veh_map in parsed_tracks:
        frs = sorted(boxes)
        f_min = frs[0]
        if len(boxes) >= 75:
            starts = list(range(f_min, frs[-1] - 72, 45))
            for k_i in range(4, len(frs) - 70):
                fn_c = frs[k_i]
                if boxes[fn_c]["occ"] == "none" and any(boxes.get(fn_c - d, {}).get("occ") in ("full", "part") for d in (3, 6, 9, 12)):
                    if all(abs(fn_c - 12 - ss) >= 18 for ss in starts) and fn_c - 12 >= f_min:
                        starts.append(fn_c - 12)
            for s_raw in sorted(starts):
                req = [s_raw + 3 * k for k in range(25)]
                if all(r_f in boxes for r_f in req):
                    _reg(vid, trk_k, boxes, req[:5])
                    traj_specs.append((vid, sp, trk_k, pid_t, boxes, pose_map, veh_map, f_min, s_raw, req))

        evs: list[tuple[int, int]] = []
        for k in range(15, len(frs) - 15):
            fc = frs[k]
            pr, po = [fc - 3 * d for d in range(1, 6)], [fc + 3 * d for d in range(0, 4)]
            if not all(rf in boxes for rf in pr + po):
                continue
            if all(boxes[rf]["walk"] == 0 for rf in pr) and all(boxes[rf]["walk"] == 1 for rf in po) and not any(c == 0 and abs(fc - fo) < 45 for fo, c in evs):
                evs.append((fc, 0))
            if all(boxes[rf]["walk"] == 1 for rf in pr) and all(boxes[rf]["walk"] == 0 for rf in po) and not any(c == 1 and abs(fc - fo) < 45 for fo, c in evs):
                evs.append((fc, 1))
            if all(boxes[rf]["cross"] == 0 for rf in pr) and all(boxes[rf]["cross"] == 1 for rf in po) and not any(c == 2 and abs(fc - fo) < 45 for fo, c in evs):
                evs.append((fc, 2))
        for f_on, c_id in evs:
            eid = ev_counter; ev_counter += 1
            for st in TAUS_10HZ:
                t0_r = f_on + 3 * st
                hr = [t0_r - 3 * (4 - j) for j in range(5)]
                if all(rf in boxes for rf in hr):
                    _reg(vid, trk_k, boxes, hr)
                    slide_specs.append((vid, sp, trk_k, pid_t, boxes, pose_map, veh_map, t0_r, hr, c_id, 0.1 * st, eid, False, f_on))
        if not evs:
            for f_neg in frs[18:-24:45]:
                wv = {boxes[rf]["walk"] for rf in frs if abs(rf - f_neg) <= 54}
                cv_s = {boxes[rf]["cross"] for rf in frs if abs(rf - f_neg) <= 54}
                hr = [f_neg - 3 * (4 - j) for j in range(5)]
                if len(wv) == 1 and len(cv_s) == 1 and all(rf in boxes for rf in hr):
                    _reg(vid, trk_k, boxes, hr)
                    eid = ev_counter; ev_counter += 1
                    slide_specs.append((vid, sp, trk_k, pid_t, boxes, pose_map, veh_map, f_neg, hr, -1, 0.0, eid, True, f_neg))

    vfb_list = {v: {f: list(d.items()) for f, d in fd.items()} for v, fd in vid_frame_boxes.items()}
    kpt_map = extract_jaad_rtmpose_keypoints(vfb_list, scratch_dir)
    z17 = np.zeros((17, 3), dtype=np.float32)

    hist_l, fut_l, ap12_l, aq2_l, ap20_l, aq4_l, hum_l, veh_l = ([] for _ in range(8))
    ya_l, va_l, on_l, vid_l, sp_l, m_strict, m_low84, m_broad = ([] for _ in range(8))
    for vid, sp, trk_k, pid_t, boxes, pose_map, veh_map, f_min, s_raw, req in traj_specs:
        h_r, f_r = req[:5], req[5:]
        h_ref = max(float(np.median([boxes[rf]["h"] for rf in h_r])), 25.0)
        pts = np.array([[(boxes[rf]["fcx"] - boxes[h_r[-1]]["fcx"]) / h_ref, (boxes[rf]["fby"] - boxes[h_r[-1]]["fby"]) / h_ref] for rf in req], dtype=np.float64)
        w_h = [boxes[rf]["walk"] for rf in h_r]
        w_f = np.array([boxes[rf]["walk"] for rf in f_r], dtype=int)
        c_f = np.array([boxes[rf]["cross"] > 0.5 for rf in f_r], dtype=bool)
        is_st, is_wk = sum(w_h[-3:]) <= 1, sum(w_h[-3:]) >= 2
        t_st = next((i for i in range(len(w_f) - 2) if is_st and w_f[i : i + 3].all()), -1)
        t_sp = next((i for i in range(len(w_f) - 2) if is_wk and (1 - w_f[i : i + 3]).all()), -1)
        t_cr = next((i for i in range(len(c_f) - 2) if (boxes[h_r[-1]]["cross"] < 0.5) and c_f[i : i + 3].all()), -1)
        k5 = np.stack([kpt_map.get((vid, trk_k, rf), z17) for rf in h_r], axis=0)
        p12, q2 = _kpts5_to_autopose(k5, h_ref)
        p20, q4 = kpts5_to_full17_split_autopose(k5, h_ref)
        hm5 = np.zeros((5, 8), dtype=np.float32)
        v5 = np.zeros((5, 3), dtype=np.float32)
        for ti, rf in enumerate(h_r):
            b = boxes[rf]
            hm5[ti, :4] = pose_map.get((pid_t, rf), np.zeros(4, dtype=np.float32))
            hm5[ti, 4:] = [b["look"], b["gest"], b["nod"], b["w"] / b["h"]]
            v5[ti] = [veh_map.get(rf, 0.5), (boxes[rf]["h"] - boxes[h_r[max(0, ti - 1)]]["h"]) / h_ref * 10.0, float(b["h"] >= 60.0)]
        occ_5 = [boxes[rf]["occ"] for rf in h_r]
        had_b = any(boxes.get(rf, {}).get("occ") in ("full", "part") for rf in range(max(f_min, h_r[0] - 12), h_r[-1])) or ((s_raw == f_min) and (f_min > 9))
        st5 = any(oc in ("full", "part") for oc in occ_5[:4]) and (occ_5[4] == "none")
        disp_bh = float(np.linalg.norm(np.diff(pts[:5], axis=0), axis=-1).sum())
        hist_l.append(pts[:5]); fut_l.append(pts[5:]); ap12_l.append(p12); aq2_l.append(q2); ap20_l.append(p20); aq4_l.append(q4)
        hum_l.append(hm5); veh_l.append(v5); vid_l.append(vid); sp_l.append(sp)
        ya_l.append([t_st >= 0, t_sp >= 0, t_cr >= 0, (t_st < 0 and t_sp < 0)])
        va_l.append([is_st, is_wk, boxes[h_r[-1]]["cross"] < 0.5, True])
        on_l.append([t_st, t_sp, t_cr, -1, 0])
        m_strict.append(st5); m_low84.append(bool(st5 and disp_bh < 0.25)); m_broad.append(bool(had_b))

    sk_l, sp12_l, sq2_l, sp20_l, sq4_l, sh_l, sqh_l, sv_l, sy_l, stau_l, seid_l, secls_l, sneg_l, sdeocc_b_l, sdeocc_s_l, svid_l, ssp_l = ([] for _ in range(17))
    for vid, sp, trk_k, pid_t, boxes, pose_map, veh_map, t0_r, hr, c_id, tau_v, eid, is_neg, f_on in slide_specs:
        href = max(float(np.median([boxes[rf]["h"] for rf in hr])), 25.0)
        xy = np.array([[(boxes[rf]["fcx"] - boxes[t0_r]["fcx"]) / href, (boxes[rf]["fby"] - boxes[t0_r]["fby"]) / href] for rf in hr], dtype=np.float32)
        vel = np.vstack((np.zeros((1, 2), dtype=np.float32), (xy[1:] - xy[:-1]) / 0.1))
        kin5 = np.concatenate((xy, vel, np.linalg.norm(vel, axis=-1, keepdims=True)), axis=-1)
        k5 = np.stack([kpt_map.get((vid, trk_k, rf), z17) for rf in hr], axis=0)
        p12, q2 = _kpts5_to_autopose(k5, href)
        p20, q4 = kpts5_to_full17_split_autopose(k5, href)
        hm5 = np.zeros((5, 8), dtype=np.float32)
        for j, rf in enumerate(hr):
            b = boxes[rf]
            p4 = pose_map.get((pid_t, rf), np.zeros(4, dtype=np.float32))
            hm5[j] = [p4[0], p4[1], p4[2], p4[3], b["look"], b["gest"], b["nod"], b["w"] / max(b["h"], 12.0)]
        oc_w = float(np.mean([0.25 if boxes[rf]["occ"] == "full" else (0.65 if boxes[rf]["occ"] == "part" else 1.0) for rf in hr]))
        qh2 = np.array([min(href / 80.0, 1.5), oc_w], dtype=np.float32)
        vs = np.array([veh_map.get(rf, 0.5) for rf in hr], dtype=np.float32)
        vh3 = np.array([vs[-1], vs[-1] - vs[0], float(vs.mean())], dtype=np.float32)
        yv = np.zeros(3, dtype=np.float32)
        if not is_neg:
            yv[c_id] = 1.0
        was_b = any(boxes.get(t0_r - 3 * d, {}).get("occ") in ("full", "part") for d in range(1, 16))
        was_s = any(boxes.get(f_on - 3 * d, {}).get("occ") in ("full", "part") for d in range(2, 16)) and (boxes.get(f_on, {}).get("occ") == "none")
        sk_l.append(kin5); sp12_l.append(p12); sq2_l.append(q2); sp20_l.append(p20); sq4_l.append(q4); sh_l.append(hm5); sqh_l.append(qh2)
        sv_l.append(vh3); sy_l.append(yv); stau_l.append(tau_v); seid_l.append(eid); secls_l.append(c_id)
        sneg_l.append(is_neg); sdeocc_b_l.append(was_b); sdeocc_s_l.append(was_s); svid_l.append(vid); ssp_l.append(sp)

    traj_ds = {
        "hist": np.asarray(hist_l, dtype=np.float64), "truth": np.asarray(fut_l, dtype=np.float64),
        "auto_pose12": np.asarray(ap12_l, dtype=np.float32), "auto_q2": np.asarray(aq2_l, dtype=np.float32),
        "auto_pose20": np.asarray(ap20_l, dtype=np.float32), "auto_q4": np.asarray(aq4_l, dtype=np.float32),
        "human_attr5": np.asarray(hum_l, dtype=np.float32), "veh5": np.asarray(veh_l, dtype=np.float32),
        "y_act": np.asarray(ya_l, dtype=bool), "v_act": np.asarray(va_l, dtype=bool), "onset": np.asarray(on_l, dtype=np.int64),
        "strict_5f_deocc": np.asarray(m_strict, dtype=bool), "strict_deocc_lowdisp_lt0p25_bh": np.asarray(m_low84, dtype=bool),
        "broad_occ_406": np.asarray(m_broad, dtype=bool), "vid": np.asarray(vid_l), "split": np.asarray(sp_l),
        "n_rtmpose_person_frames": len(kpt_map),
    }
    slide_ds = {
        "kin": np.asarray(sk_l, dtype=np.float32),
        "auto_pose12": np.asarray(sp12_l, dtype=np.float32), "auto_q2": np.asarray(sq2_l, dtype=np.float32),
        "auto_pose20": np.asarray(sp20_l, dtype=np.float32), "auto_q4": np.asarray(sq4_l, dtype=np.float32),
        "human_attr": np.asarray(sh_l, dtype=np.float32), "human_q": np.asarray(sqh_l, dtype=np.float32),
        "veh": np.asarray(sv_l, dtype=np.float32), "y": np.asarray(sy_l, dtype=np.float32),
        "tau_s": np.asarray(stau_l, dtype=np.float64), "event_id": np.asarray(seid_l, dtype=np.int64),
        "event_cls": np.asarray(secls_l, dtype=np.int64), "is_neg": np.asarray(sneg_l, dtype=bool),
        "is_deocc_broad": np.asarray(sdeocc_b_l, dtype=bool), "is_deocc_strict_recovered": np.asarray(sdeocc_s_l, dtype=bool),
        "vid": np.asarray(svid_l), "split": np.asarray(ssp_l),
    }
    return traj_ds, slide_ds


def fit_and_predict_wave26_lead_arms(
    kin: np.ndarray,
    arms_spec: list[tuple[str, np.ndarray, np.ndarray, bool, bool]],
    veh: np.ndarray,
    y: np.ndarray,
    tau_s: np.ndarray,
    is_neg: np.ndarray,
    split: np.ndarray,
    seeds: list[int],
    device: torch.device,
    epochs: int = 28,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    tr = np.flatnonzero(split == "train")
    va = np.flatnonzero(split == "val")
    te = np.flatnonzero(split == "test")
    kt = torch.from_numpy(kin).to(device)
    vt = torch.from_numpy(veh).to(device)
    yt = torch.from_numpy(y).to(device)
    dt_tgt = torch.from_numpy(np.clip(-tau_s, 0.0, 2.0).astype(np.float32)).to(device)
    nt = torch.from_numpy(is_neg).to(device)
    tr_t, va_t, te_t = torch.from_numpy(tr).to(device), torch.from_numpy(va).to(device), torch.from_numpy(te).to(device)
    pw = ((len(tr) - yt[tr_t].sum(0)) / yt[tr_t].sum(0).clamp_min(1.0)).clamp(1.0, 12.0)

    probs_tr = {nm: np.zeros((len(seeds), len(tr), 3), dtype=np.float64) for nm, *_ in arms_spec}
    probs_va = {nm: np.zeros((len(seeds), len(va), 3), dtype=np.float64) for nm, *_ in arms_spec}
    probs_te = {nm: np.zeros((len(seeds), len(te), 3), dtype=np.float64) for nm, *_ in arms_spec}

    for si, seed in enumerate(seeds):
        for nm, p_arr, q_arr, up, uv in arms_spec:
            pt = torch.from_numpy(p_arr).to(device)
            qt = torch.from_numpy(q_arr).to(device)
            q_obs = (qt[:, :2].mean(dim=1).clamp(0.0, 1.0)) if up else torch.ones(len(kin), device=device)
            w_vis = torch.where(nt, torch.ones_like(q_obs), (0.35 + 0.65 * torch.exp(-dt_tgt / 0.85)) * (0.45 + 0.55 * q_obs))

            torch.manual_seed(seed + 41)
            net = JAADPostReappearPrecursorNet(kin.shape[-1], p_arr.shape[-1], q_arr.shape[-1], veh.shape[-1], use_pose=up, use_veh=uv).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=2.2e-3, weight_decay=2.5e-4)
            gen = torch.Generator().manual_seed(seed + 233)
            best_sd, best_vl = None, 1e9
            for _ in range(epochs):
                net.train()
                for b in torch.randperm(len(tr_t), generator=gen).split(128):
                    ib = tr_t[b.to(device)]
                    opt.zero_grad(set_to_none=True)
                    lg, dl = net(kt[ib], pt[ib], qt[ib], vt[ib])
                    bce = F.binary_cross_entropy_with_logits(lg, yt[ib], pos_weight=pw, reduction="none").mean(dim=1)
                    l_cls = (bce * w_vis[ib]).mean()
                    pos_m = ~nt[ib]
                    l_dl = F.smooth_l1_loss((dl[pos_m] * yt[ib][pos_m]).sum(dim=1), dt_tgt[ib][pos_m]) if pos_m.any() else lg.new_tensor(0.0)
                    (l_cls + 0.15 * l_dl).backward()
                    opt.step()
                net.eval()
                with torch.inference_mode():
                    lv, _ = net(kt[va_t], pt[va_t], qt[va_t], vt[va_t])
                    vl = float(F.binary_cross_entropy_with_logits(lv, yt[va_t], pos_weight=pw).item())
                    if vl < best_vl:
                        best_vl = vl
                        best_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if best_sd is not None:
                net.load_state_dict(best_sd)
            net.eval()
            with torch.inference_mode():
                ltr, _ = net(kt[tr_t], pt[tr_t], qt[tr_t], vt[tr_t])
                lva, _ = net(kt[va_t], pt[va_t], qt[va_t], vt[va_t])
                lte, _ = net(kt[te_t], pt[te_t], qt[te_t], vt[te_t])
                probs_tr[nm][si] = torch.sigmoid(ltr).cpu().numpy()
                probs_va[nm][si] = torch.sigmoid(lva).cpu().numpy()
                probs_te[nm][si] = torch.sigmoid(lte).cpu().numpy()

    return (
        {k: v.mean(0) for k, v in probs_tr.items()},
        {k: v.mean(0) for k, v in probs_va.items()},
        {k: v.mean(0) for k, v in probs_te.items()},
    )


def audit_np_conformal_fpr_stability(
    p_tr: dict[str, np.ndarray],
    p_va: dict[str, np.ndarray],
    p_te: dict[str, np.ndarray],
    slide_ds: dict[str, np.ndarray],
) -> dict:
    """Evaluate split-conformal Neyman-Pearson validation calibration (finite-sample + speed-standardized) on test FPR."""
    tr = np.flatnonzero(slide_ds["split"] == "train")
    va = np.flatnonzero(slide_ds["split"] == "val")
    te = np.flatnonzero(slide_ds["split"] == "test")
    neg_tr, neg_va, neg_te = slide_ds["is_neg"][tr], slide_ds["is_neg"][va], slide_ds["is_neg"][te]
    tau_te, eid_te, cls_te = slide_ds["tau_s"][te], slide_ds["event_id"][te], slide_ds["event_cls"][te]
    strict_te = slide_ds["is_deocc_strict_recovered"][te] & (~neg_te)
    spd_all = np.median(slide_ds["kin"][:, 1:, 4], axis=1)
    slow_tr, slow_va, slow_te = spd_all[tr] < 0.45, spd_all[va] < 0.45, spd_all[te] < 0.45

    out = {"n_val_neg": int(neg_va.sum()), "n_test_neg": int(neg_te.sum()), "models": {}}
    for nm in p_te:
        # Speed-stratum logit standardization fitted strictly on train+val negatives? NO: strictly on TRAIN negatives only,
        # so val negatives remain 100% exchangeable with test negatives for exact finite-sample split-conformal calibration!
        lg_tr = np.log(np.clip(p_tr[nm], 1e-6, 1.0 - 1e-6) / np.clip(1.0 - p_tr[nm], 1e-6, 1.0 - 1e-6))
        lg_va = np.log(np.clip(p_va[nm], 1e-6, 1.0 - 1e-6) / np.clip(1.0 - p_va[nm], 1e-6, 1.0 - 1e-6))
        lg_te = np.log(np.clip(p_te[nm], 1e-6, 1.0 - 1e-6) / np.clip(1.0 - p_te[nm], 1e-6, 1.0 - 1e-6))
        z_va, z_te = np.zeros_like(lg_va), np.zeros_like(lg_te)
        for m_s_tr, m_s_va, m_s_te in ((slow_tr, slow_va, slow_te), (~slow_tr, ~slow_va, ~slow_te)):
            sub_tr = lg_tr[neg_tr & m_s_tr]
            mu = sub_tr.mean(axis=0) if len(sub_tr) >= 5 else lg_tr[neg_tr].mean(axis=0)
            sd = np.maximum(sub_tr.std(axis=0), 0.25) if len(sub_tr) >= 5 else np.maximum(lg_tr[neg_tr].std(axis=0), 0.25)
            z_va[m_s_va] = (lg_va[m_s_va] - mu) / sd
            z_te[m_s_te] = (lg_te[m_s_te] - mu) / sd

        thr_np = calibrate_conformal_np_thresholds(z_va[neg_va], target_fpr=0.05)
        cls_res = {}
        for ci, cn in enumerate(("Start", "Stop", "Cross")):
            fp_cnt = int((z_te[neg_te, ci] >= thr_np[ci]).sum())
            eids = sorted(set(eid_te[strict_te & (cls_te == ci)].tolist()))
            hits = sum(int(np.any(z_te[(eid_te == e) & (tau_te <= 1e-9), ci] >= thr_np[ci])) for e in eids)
            cls_res[cn] = {
                "conformal_np_test_fp_count": fp_cnt,
                "conformal_np_test_fpr": float(fp_cnt / max(int(neg_te.sum()), 1)),
                "strict_deocc_hits": int(hits),
                "strict_deocc_events": len(eids),
                "strict_deocc_recall": float(hits / max(len(eids), 1)),
            }
        out["models"][nm] = cls_res
    return out


def eval_jaad_wave26_trajectory_5seed(jd: dict, seeds: list[int], device: torch.device) -> dict:
    sp = jd["split"]
    tr, va, te = np.flatnonzero(sp == "train"), np.flatnonzero(sp == "val"), np.flatnonzero(sp == "test")
    trva = np.concatenate([tr, va])
    ht = torch.from_numpy(jd["hist"]).to(device)
    tt = torch.from_numpy(jd["truth"]).to(device)
    dt_all = torch.full((len(ht), 4), 0.1, dtype=torch.float64, device=device)
    ft_all = torch.arange(1, 21, dtype=torch.float64, device=device)[None].expand(len(ht), -1) * 0.1
    trva_g, tr_t, va_t, te_t = (torch.from_numpy(x).to(device) for x in (trva, tr, va, te))

    kcfg = calibrate_coherence_kalman_params(ht[trva_g], dt_all[trva_g], ft_all[trva_g], tt[trva_g], d_ref=0.25)
    t_s = np.arange(1, 21, dtype=np.float64)[None, :, None] * 0.1
    cv_te = ((jd["hist"][te, -1:] - jd["hist"][te, -2:-1]) / 0.1) * t_s
    err_cv = _jaad_post_onset_err(cv_te, jd["truth"][te], jd["onset"][te])
    with torch.inference_mode():
        vf_te = compute_coherence_kalman_velocity(ht[te_t], dt_all[te_t], **kcfg)["v_filt"].cpu().numpy()
    err_kf = _jaad_post_onset_err(vf_te[:, None, :] * t_s, jd["truth"][te], jd["onset"][te])

    vt = torch.from_numpy(jd["veh5"]).to(device)
    yat, vat = torch.from_numpy(jd["y_act"]).to(device), torch.from_numpy(jd["v_act"]).to(device)
    pw_c = ((vat[tr].float().sum(0) - yat[tr].float().sum(0)) / yat[tr].float().sum(0).clamp_min(1.0)).clamp(1.0, 12.0)

    arms = [
        ("J26_M0_Coh", jd["auto_pose20"], jd["auto_q4"], False, False),
        ("J26_AutoPose12_Old", jd["auto_pose12"], jd["auto_q2"], True, False),
        ("J26_AutoPose17_FullSplit", jd["auto_pose20"], jd["auto_q4"], True, False),
        ("J26_M2_Coh_Veh", jd["auto_pose20"], jd["auto_q4"], False, True),
        ("J26_AutoPose17_FullSplit_Veh", jd["auto_pose20"], jd["auto_q4"], True, True),
        ("J26_Full_Human_Veh", jd["human_attr5"], np.ones((len(sp), 2), dtype=np.float32), True, True),
    ]
    err_s = {nm: np.zeros((len(seeds), len(te), 4), dtype=np.float64) for nm, *_ in arms}
    ap_s = {nm: np.zeros((len(seeds), len(te), 4), dtype=np.float64) for nm, *_ in arms}

    for si, seed in enumerate(seeds):
        for nm, p_arr, q_arr, up, uv in arms:
            pt, qt = torch.from_numpy(p_arr).to(device), torch.from_numpy(q_arr).to(device)
            torch.manual_seed(seed + 23)
            net = JAADCoherenceSplitTrajectoryNet(p_arr.shape[-1], q_arr.shape[-1], up, uv, kalman_cfg=kcfg).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=2.2e-3, weight_decay=2.5e-4)
            gen = torch.Generator().manual_seed(seed + 137)
            best_sd, best_vl = None, 1e9
            for _ in range(30):
                net.train()
                for b in torch.randperm(len(tr_t), generator=gen).split(64):
                    ib = tr_t[b.to(device)]
                    opt.zero_grad(set_to_none=True)
                    o = net(ht[ib], pt[ib], qt[ib], vt[ib])
                    d_k = torch.linalg.vector_norm(o["paths"] - tt[ib][:, None], dim=-1)
                    l0 = (torch.sqrt(d_k[:, 0].square() + 0.0025) - 0.05).mean()
                    lm = d_k[:, 1:].mean(dim=-1).amin(dim=1).mean()
                    bce = F.binary_cross_entropy_with_logits(o["act_logits"], yat[ib].float(), pos_weight=pw_c, reduction="none")
                    lac = ((bce * vat[ib].float()).sum(1) / vat[ib].float().sum(1).clamp_min(1.0)).mean().double()
                    (l0 + 0.35 * lm + 0.25 * lac).backward()
                    opt.step()
                net.eval()
                with torch.inference_mode():
                    ov = net(ht[va_t], pt[va_t], qt[va_t], vt[va_t])
                    vl = float(torch.linalg.vector_norm(ov["paths"][:, 0] - tt[va_t], dim=-1).mean().item())
                    if vl < best_vl:
                        best_vl = vl
                        best_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if best_sd is not None:
                net.load_state_dict(best_sd)
            net.eval()
            with torch.inference_mode():
                ot = net(ht[te_t], pt[te_t], qt[te_t], vt[te_t])
                err_s[nm][si] = _jaad_post_onset_err(ot["paths"][:, 0].cpu().numpy(), jd["truth"][te], jd["onset"][te])
                ap_s[nm][si] = torch.sigmoid(ot["act_logits"]).cpu().numpy()

    em = {nm: err_s[nm].mean(axis=0) for nm, *_ in arms}
    v_te = jd["vid"][te]
    strata = {
        "all_locked_test_953": np.ones(len(te), dtype=bool),
        "strict_5frame_deoccluded_168": jd["strict_5f_deocc"][te],
        "strict_5f_deocc_lowdisp_lt0p25_bbox_h_84": jd["strict_deocc_lowdisp_lt0p25_bh"][te],
        "broad_occ_or_late_union_406": jd["broad_occ_406"][te],
    }

    def _r4(a4: np.ndarray, m: np.ndarray) -> dict[str, float]:
        v = a4[m].mean(0)
        return {"ade_2s": float(v[0]), "fde_2s": float(v[1]), "post_0p5s": float(v[2]), "post_1p0s": float(v[3])}

    st_out = {}
    for sk, sm in strata.items():
        md = {"Last2_CV": _r4(err_cv, sm), "J26_KinCohFilter_Only": _r4(err_kf, sm)}
        for nm, *_ in arms:
            md[nm] = _r4(em[nm], sm)
        st_out[sk] = {
            "n_windows": int(sm.sum()),
            "n_videos": len(set(v_te[sm].tolist())),
            "models": md,
            "paired_video_ci": {
                "kin_coh_minus_CV_2s": _loc_bootstrap_ci(err_kf[sm, 0:2] - err_cv[sm, 0:2], v_te[sm]),
                "autopose17_minus_m0_coh_2s": _loc_bootstrap_ci(em["J26_AutoPose17_FullSplit"][sm, 0:2] - em["J26_M0_Coh"][sm, 0:2], v_te[sm]),
                "autopose17_minus_autopose12_2s": _loc_bootstrap_ci(em["J26_AutoPose17_FullSplit"][sm, 0:2] - em["J26_AutoPose12_Old"][sm, 0:2], v_te[sm]),
                "autopose17_veh_minus_m2_veh_2s": _loc_bootstrap_ci(em["J26_AutoPose17_FullSplit_Veh"][sm, 0:2] - em["J26_M2_Coh_Veh"][sm, 0:2], v_te[sm]),
                "autopose17_veh_minus_CV_2s": _loc_bootstrap_ci(em["J26_AutoPose17_FullSplit_Veh"][sm, 0:2] - err_cv[sm, 0:2], v_te[sm]),
                "autopose17_minus_m0_coh_post_onset": _loc_bootstrap_ci(em["J26_AutoPose17_FullSplit"][sm, 2:4] - em["J26_M0_Coh"][sm, 2:4], v_te[sm]),
            },
        }

    ap_out = {}
    for sub_nm, sub_m in (("strict_5frame_deoccluded_168", jd["strict_5f_deocc"][te]), ("all_locked_test_953", np.ones(len(te), dtype=bool))):
        ap_out[sub_nm] = {}
        for nm, *_ in arms:
            pm = ap_s[nm].mean(0)
            ap_out[sub_nm][nm] = {
                cn: {"pos": int(jd["y_act"][te, ci][sub_m & jd["v_act"][te, ci]].sum()),
                     "ap": float(_binary_average_precision(jd["y_act"][te, ci][sub_m & jd["v_act"][te, ci]].astype(np.float64), pm[sub_m & jd["v_act"][te, ci], ci]))}
                for ci, cn in enumerate(("Start", "Stop", "Cross"))
            }
    return {"source_calibrated_kalman_params": kcfg, "test_strata": st_out, "maneuver_ap": ap_out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch-dir", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, default=Path("wave26_jaad_full17_reappear_and_conformal_fpr.zip"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    traj_ds, slide_ds = build_jaad2_full17_cohorts(args.scratch_dir)
    meta = extract_jaad_slide_reappearance_metadata((args.scratch_dir / "jaad2.zip").read_bytes())

    lead_arms = [
        ("Lead_M0_kin_only", slide_ds["auto_pose20"], slide_ds["auto_q4"], False, False),
        ("Lead_AutoPose_CovisSafe", slide_ds["auto_pose12"], slide_ds["auto_q2"], True, False),
        ("Lead_AutoPose_rtmpose17", slide_ds["auto_pose20"], slide_ds["auto_q4"], True, False),
        ("Lead_M2_kin_veh", slide_ds["auto_pose20"], slide_ds["auto_q4"], False, True),
        ("Lead_AutoPose_veh", slide_ds["auto_pose20"], slide_ds["auto_q4"], True, True),
        ("Lead_HumanAttr_orient_gaze", slide_ds["human_attr"], slide_ds["human_q"], True, False),
        ("Lead_Full_human_veh", slide_ds["human_attr"], slide_ds["human_q"], True, True),
    ]
    p_tr, p_va, p_te = fit_and_predict_wave26_lead_arms(
        slide_ds["kin"], lead_arms, slide_ds["veh"], slide_ds["y"], slide_ds["tau_s"],
        slide_ds["is_neg"], slide_ds["split"], list(SEEDS), device,
    )
    event_audit = audit_per_event_timestamps_and_visible_alarms(p_va, p_te, slide_ds, meta)
    np_conf = audit_np_conformal_fpr_stability(p_tr, p_va, p_te, slide_ds)
    traj_res = eval_jaad_wave26_trajectory_5seed(traj_ds, list(SEEDS), device)

    summary = {
        "wave": "wave26_jaad_full17_reappear_and_conformal_fpr_5seed",
        "lead_arm_legend": {
            "Lead_M0_kin_only": "KinCoh_BBox_Only",
            "Lead_AutoPose_CovisSafe": "AutoPose12_Legacy",
            "Lead_AutoPose_rtmpose17": "AutoPose17_FullSplit_HeadTorsoArmAnkle",
            "Lead_M2_kin_veh": "KinCoh_Veh",
            "Lead_AutoPose_veh": "AutoPose17_FullSplit_Veh",
            "Lead_HumanAttr_orient_gaze": "HumanAttr_OrientGaze",
            "Lead_Full_human_veh": "HumanAttr_OrientGaze_Veh",
        },
        "jaad2_per_event_reappearance_and_visible_alarm_audit": event_audit,
        "jaad2_conformal_np_fpr_stability_audit": np_conf,
        "jaad2_full17_coherence_trajectory_5seed": traj_res,
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
