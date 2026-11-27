"""Wave 22: Causal 5-Frame APA Precursor Lead-Time (ms before onset at matched FPR) & Delayed-Onset Start/Stop Selection."""
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

from scripts.run_ppt_mainline_o2_m0_m3_and_fixed_conformal import load_ppt_cohort_with_causal_maps
from scripts.run_wave18_conflict_gated_interaction_5seed import _loc_bootstrap_ci, safe_site_clip_ci
from scripts.run_wave21_jaad_deocclusion_and_eth_ucy_6dataset import JAAD_ZIP_URL, VEH_ACT_MAP, _fetch_url_bytes
from scripts.run_wave21_jrdb_and_jta_emergence_maneuvers import ST_RELEASE_URL, _download_verified_zip, load_jrdb_with_blind_emergence
from scripts.run_wave21_pinns_true_emergence_and_6slot_diag import SEEDS, _fwd_6slot, blind_pinns_physical_emergence_audit, fit_6slot_arm_fold
from src.models.apa_precursor_leadtime import (
    CausalAPAPrecursorLeadNet,
    build_delayed_onset_start_stop_candidates,
    calibrate_matched_fpr_thresholds,
    score_event_lead_times_at_matched_fpr,
)
from src.models.emergence_maneuver_6slot import (
    apply_6slot_selector,
    audit_6candidate_generation_vs_selection,
    compute_action_labels_with_onset,
    fit_strict_source_platt_and_slot_selector,
    horizon_and_post_onset_errors,
)

TAUS_10HZ = (-10, -8, -6, -4, -3, -2, -1, 0, 1, 2, 4)  # Steps of 0.1s around t_onset: [-1.0s .. +0.4s]


def extract_jaad2_sliding_leadtime_events(scratch_dir: Path) -> dict[str, np.ndarray]:
    """Extract dense causal 5-frame 10Hz windows around human-annotated JAAD 2.0 Start/Stop/Cross onsets + negative controls."""
    blob = _fetch_url_bytes(JAAD_ZIP_URL, scratch_dir / "jaad2.zip")
    splits: dict[str, str] = {}
    kin_l, pose_l, q_l, veh_l, y_l, tau_l, eid_l, ecls_l, neg_l, deocc_l, vid_l, sp_l = ([] for _ in range(12))
    ev_counter = 0

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        pfx = "JAAD-JAAD_2.0/"
        for sp in ("train", "val", "test"):
            for vid in zf.read(f"{pfx}split_ids/default/{sp}.txt").decode("utf-8").splitlines():
                if vid.strip():
                    splits[vid.strip()] = sp
        ann_files = sorted(n for n in zf.namelist() if n.startswith(f"{pfx}annotations/video_") and n.endswith(".xml"))

        for af in ann_files:
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
            veh_map = {}
            if veh_name in zf.namelist():
                for vf in ET.fromstring(zf.read(veh_name)).findall(".//frame"):
                    veh_map[int(vf.attrib["id"])] = VEH_ACT_MAP.get(vf.attrib.get("action", "moving_slow"), 0.5)

            for tr in root.findall(".//track"):
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
                    boxes[fr] = {
                        "fcx": 0.5 * (xtl + xbr), "fby": ybr, "w": max(xbr - xtl, 4.0), "h": max(ybr - ytl, 12.0),
                        "occ": at.get("occlusion", "none"), "walk": 1 if at.get("action", "standing") == "walking" else 0,
                        "look": 1.0 if at.get("look", "") == "looking" else 0.0,
                        "gest": 0.0 if at.get("hand_gesture", "__undefined__") == "__undefined__" else 1.0,
                        "nod": 1.0 if at.get("nod", "__undefined__") == "nodding" else 0.0,
                        "cross": 1 if at.get("cross", "") == "crossing" else 0,
                    }
                if len(boxes) < 48:
                    continue
                _append_jaad_track_events(boxes, pid_t, vid, sp, pose_map, veh_map, kin_l, pose_l, q_l, veh_l, y_l, tau_l, eid_l, ecls_l, neg_l, deocc_l, vid_l, sp_l, [ev_counter])
                ev_counter = eid_l[-1] + 1 if eid_l else ev_counter

    return {
        "kin": np.asarray(kin_l, dtype=np.float32), "pose": np.asarray(pose_l, dtype=np.float32),
        "q": np.asarray(q_l, dtype=np.float32), "veh": np.asarray(veh_l, dtype=np.float32),
        "y": np.asarray(y_l, dtype=np.float32), "tau_s": np.asarray(tau_l, dtype=np.float64),
        "event_id": np.asarray(eid_l, dtype=np.int64), "event_cls": np.asarray(ecls_l, dtype=np.int64),
        "is_neg": np.asarray(neg_l, dtype=bool), "is_deocc": np.asarray(deocc_l, dtype=bool),
        "vid": np.asarray(vid_l), "split": np.asarray(sp_l),
    }


def _append_jaad_track_events(
    boxes: dict[int, dict], pid_t: str, vid: str, sp: str, pose_map: dict, veh_map: dict,
    kin_l: list, pose_l: list, q_l: list, veh_l: list, y_l: list, tau_l: list,
    eid_l: list, ecls_l: list, neg_l: list, deocc_l: list, vid_l: list, sp_l: list, ev_ref: list[int],
) -> None:
    frs = sorted(boxes)
    events_found: list[tuple[int, int]] = []  # (fr_onset, cls_idx: 0=Start, 1=Stop, 2=Cross)
    for k in range(15, len(frs) - 15):
        f_cur = frs[k]
        # Require clean 0.5s prior steady state before human-annotated state transition
        prev_r = [f_cur - 3 * d for d in range(1, 6)]
        post_r = [f_cur + 3 * d for d in range(0, 4)]
        if not all(rf in boxes for rf in prev_r + post_r):
            continue
        if all(boxes[rf]["walk"] == 0 for rf in prev_r) and all(boxes[rf]["walk"] == 1 for rf in post_r):
            if not any(c == 0 and abs(f_cur - fo) < 45 for fo, c in events_found):
                events_found.append((f_cur, 0))
        if all(boxes[rf]["walk"] == 1 for rf in prev_r) and all(boxes[rf]["walk"] == 0 for rf in post_r):
            if not any(c == 1 and abs(f_cur - fo) < 45 for fo, c in events_found):
                events_found.append((f_cur, 1))
        if all(boxes[rf]["cross"] == 0 for rf in prev_r) and all(boxes[rf]["cross"] == 1 for rf in post_r):
            if not any(c == 2 and abs(f_cur - fo) < 45 for fo, c in events_found):
                events_found.append((f_cur, 2))

    def _feat5(t0_raw: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool] | None:
        h_r = [t0_raw - 3 * (4 - j) for j in range(5)]
        if not all(rf in boxes for rf in h_r):
            return None
        href = max(float(np.median([boxes[rf]["h"] for rf in h_r])), 25.0)
        xy = np.array([[(boxes[rf]["fcx"] - boxes[t0_raw]["fcx"]) / href, (boxes[rf]["fby"] - boxes[t0_raw]["fby"]) / href] for rf in h_r], dtype=np.float32)
        vel = np.vstack((np.zeros((1, 2), dtype=np.float32), (xy[1:] - xy[:-1]) / 0.1))
        kin5 = np.concatenate((xy, vel, np.linalg.norm(vel, axis=-1, keepdims=True)), axis=-1)
        ps5 = np.zeros((5, 8), dtype=np.float32)
        for j, rf in enumerate(h_r):
            b = boxes[rf]
            p4 = pose_map.get((pid_t, rf), np.zeros(4, dtype=np.float32))
            ps5[j] = [p4[0], p4[1], p4[2], p4[3], b["look"], b["gest"], b["nod"], b["w"] / max(b["h"], 12.0)]
        oc_w = np.mean([0.25 if boxes[rf]["occ"] == "full" else (0.65 if boxes[rf]["occ"] == "part" else 1.0) for rf in h_r])
        qp = np.array([min(href / 80.0, 1.5), oc_w], dtype=np.float32)
        vs = np.array([veh_map.get(rf, 0.5) for rf in h_r], dtype=np.float32)
        vh = np.array([vs[-1], vs[-1] - vs[0], float(vs.mean())], dtype=np.float32)
        was_occ = any(boxes.get(t0_raw - 3 * d, {}).get("occ") in ("full", "part") for d in range(1, 16))
        return kin5, ps5, qp, vh, was_occ

    for f_on, c_id in events_found:
        eid = ev_ref[0]; ev_ref[0] += 1
        for st in TAUS_10HZ:
            out = _feat5(f_on + 3 * st)
            if out is None:
                continue
            k5, p5, qp, vh, w_occ = out
            yv = np.zeros(3, dtype=np.float32); yv[c_id] = 1.0
            kin_l.append(k5); pose_l.append(p5); q_l.append(qp); veh_l.append(vh); y_l.append(yv)
            tau_l.append(0.1 * st); eid_l.append(eid); ecls_l.append(c_id); neg_l.append(False); deocc_l.append(w_occ)
            vid_l.append(vid); sp_l.append(sp)

    # Pure steady negative control windows (no Start/Stop/Cross anywhere within +-1.8s)
    if not events_found:
        for f_neg in frs[18:-24:45]:
            w_vals = {boxes[rf]["walk"] for rf in frs if abs(rf - f_neg) <= 54}
            c_vals = {boxes[rf]["cross"] for rf in frs if abs(rf - f_neg) <= 54}
            if len(w_vals) == 1 and len(c_vals) == 1:
                out = _feat5(f_neg)
                if out is not None:
                    k5, p5, qp, vh, w_occ = out
                    eid = ev_ref[0]; ev_ref[0] += 1
                    kin_l.append(k5); pose_l.append(p5); q_l.append(qp); veh_l.append(vh); y_l.append(np.zeros(3, dtype=np.float32))
                    tau_l.append(0.0); eid_l.append(eid); ecls_l.append(-1); neg_l.append(True); deocc_l.append(w_occ)
                    vid_l.append(vid); sp_l.append(sp)


def eval_sliding_leadtime_5seed(
    ds: dict[str, np.ndarray], seeds: list[int], device: torch.device, epochs: int = 26,
) -> dict:
    tr = np.flatnonzero(ds["split"] == "train")
    va = np.flatnonzero(ds["split"] == "val")
    te = np.flatnonzero(ds["split"] == "test")
    kt = torch.from_numpy(ds["kin"]).to(device)
    pt = torch.from_numpy(ds["pose"]).to(device)
    qt = torch.from_numpy(ds["q"]).to(device)
    vt = torch.from_numpy(ds["veh"]).to(device)
    yt = torch.from_numpy(ds["y"]).to(device)
    arms = (
        ("Lead_M0_kin_only", False, False),
        ("Lead_M1_kin_pose", True, False),
        ("Lead_M2_kin_nbr", False, True),
        ("Lead_Full_kin_pose_nbr", True, True),
    )
    va_probs = {a[0]: np.zeros((len(seeds), len(va), 3), dtype=np.float64) for a in arms}
    te_probs = {a[0]: np.zeros((len(seeds), len(te), 3), dtype=np.float64) for a in arms}
    pw = ((1.0 - yt[tr]).sum(0).clamp_min(1.0) / yt[tr].sum(0).clamp_min(1.0)).clamp(1.0, 12.0)

    for si, seed in enumerate(seeds):
        for arm_nm, u_p, u_n in arms:
            torch.manual_seed(seed)
            net = CausalAPAPrecursorLeadNet(kt.size(-1), pt.size(-1), vt.size(-1), use_pose=u_p, use_nbr=u_n).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=1.8e-3, weight_decay=2e-4)
            tr_g = torch.from_numpy(tr).to(device)
            net.train()
            for ep in range(epochs):
                perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed * 577 + ep)).to(device)
                for cg in perm.split(192):
                    ids = tr_g[cg]
                    lg, dl = net(kt[ids], pt[ids], qt[ids], vt[ids])
                    bce = F.binary_cross_entropy_with_logits(lg, yt[ids], pos_weight=pw)
                    pos_m = yt[ids].sum(1) > 0.5
                    tgt_d = torch.from_numpy(np.maximum(-ds["tau_s"][tr][cg.cpu().numpy()], 0.0)).float().to(device)
                    dl_l = F.smooth_l1_loss(dl[pos_m], tgt_d[pos_m]) if pos_m.any() else lg.new_tensor(0.0)
                    loss = bce + 0.25 * dl_l
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            net.eval()
            with torch.inference_mode():
                va_probs[arm_nm][si] = torch.sigmoid(net(kt[torch.from_numpy(va).to(device)], pt[torch.from_numpy(va).to(device)], qt[torch.from_numpy(va).to(device)], vt[torch.from_numpy(va).to(device)])[0]).cpu().numpy()
                te_probs[arm_nm][si] = torch.sigmoid(net(kt[torch.from_numpy(te).to(device)], pt[torch.from_numpy(te).to(device)], qt[torch.from_numpy(te).to(device)], vt[torch.from_numpy(te).to(device)])[0]).cpu().numpy()

    out_fpr: dict[str, dict] = {}
    va_neg = ds["is_neg"][va]
    te_neg = ds["is_neg"][te]
    for fpr_tag, fpr_v in (("matched_fpr_0p05", 0.05), ("matched_fpr_0p10", 0.10)):
        sub_res = {}
        for sub_nm, sub_m in (("locked_test_all_events", np.ones(len(te), dtype=bool)), ("locked_test_post_deocclusion", ds["is_deocc"][te] | te_neg)):
            if int((sub_m & (~te_neg)).sum()) < 5:
                continue
            arm_d = {}
            for arm_nm, _, _ in arms:
                thr = calibrate_matched_fpr_thresholds(va_probs[arm_nm].mean(0)[va_neg], target_fpr=fpr_v)
                sc = score_event_lead_times_at_matched_fpr(
                    te_probs[arm_nm].mean(0)[sub_m], ds["tau_s"][te][sub_m],
                    ds["event_id"][te][sub_m], ds["event_cls"][te][sub_m], te_neg[sub_m], thr,
                )
                arm_d[arm_nm] = {k: {kk: vv for kk, vv in v.items() if kk != "per_event_lead_s"} for k, v in sc.items()}
                arm_d[f"_{arm_nm}_raw"] = sc
            # Paired event-level lead-time gain (Pose - NoPose, in seconds; positive = earlier warning!)
            paired_lead = {}
            for cn in ("Start", "Stop", "Turn"):
                l_p = np.asarray(arm_d["_Lead_M1_kin_pose_raw"][cn]["per_event_lead_s"])
                l_0 = np.asarray(arm_d["_Lead_M0_kin_only_raw"][cn]["per_event_lead_s"])
                l_f = np.asarray(arm_d["_Lead_Full_kin_pose_nbr_raw"][cn]["per_event_lead_s"])
                l_n = np.asarray(arm_d["_Lead_M2_kin_nbr_raw"][cn]["per_event_lead_s"])
                if len(l_p) >= 4:
                    d_p0, d_fn = l_p - l_0, l_f - l_n
                    rng = np.random.default_rng(20261127)
                    bs_p0 = np.array([d_p0[rng.integers(0, len(d_p0), len(d_p0))].mean() for _ in range(2000)])
                    bs_fn = np.array([d_fn[rng.integers(0, len(d_fn), len(d_fn))].mean() for _ in range(2000)])
                    paired_lead[cn] = {
                        "n_events": int(len(l_p)),
                        "pose_minus_kin_lead_s_mean": float(d_p0.mean()),
                        "pose_minus_kin_lead_s_ci95": [float(np.quantile(bs_p0, 0.025)), float(np.quantile(bs_p0, 0.975))],
                        "full_minus_nbr_lead_s_mean": float(d_fn.mean()),
                        "full_minus_nbr_lead_s_ci95": [float(np.quantile(bs_fn, 0.025)), float(np.quantile(bs_fn, 0.975))],
                    }
            sub_res[sub_nm] = {"models": {k: v for k, v in arm_d.items() if not k.startswith("_")}, "paired_lead_gain_s": paired_lead}
        out_fpr[fpr_tag] = sub_res
    return out_fpr


def eval_delayed_onset_apa_upgrade_cohort(
    tag: str, grp_arr: np.ndarray, unit_arr: np.ndarray, is_pinns: bool,
    gd: dict, hist_np: np.ndarray, truth_np: np.ndarray, coronal_np: np.ndarray,
    ya_np: np.ndarray, va_np: np.ndarray, onset_np: np.ndarray, diag_mask: np.ndarray,
    dt: float, fut_s_np: np.ndarray, bsz: int, ep_slot: int, lr_slot: float,
    seeds: list[int], device: torch.device,
) -> dict:
    """Compare immediate-onset S6_Full vs Delayed-Onset APA S6_APA_Delayed on Start/Stop oracle & Top-1 selection."""
    n, tf, _ = truth_np.shape
    groups = sorted(set(grp_arr.tolist()))
    cv_rel = ((hist_np[:, -1:] - hist_np[:, -2:-1]) / dt) * fut_s_np[:, :, None]
    err_cv = horizon_and_post_onset_errors(cv_rel + hist_np[:, -1:], truth_np, onset_np, dt=dt)

    # Build compact kinematic + pose + neighbor features for APA onset-delay regressor
    t_pose = gd["bio_even"].size(1)
    xy_rel = (hist_np - hist_np[:, -1:, :]).astype(np.float32)
    vel = np.concatenate((np.zeros((n, 1, 2), dtype=np.float32), (xy_rel[:, 1:] - xy_rel[:, :-1]) / float(dt)), axis=1)
    kin_f = torch.from_numpy(np.concatenate((xy_rel, vel, np.linalg.norm(vel, axis=-1, keepdims=True)), axis=-1)).to(device)
    pose_f = torch.cat((gd["bio_even"], gd["pose_vec_world"].reshape(n, t_pose, 4).float(), gd["signed_knee"]), dim=-1)
    q_f = gd["quality"][:, :2]
    nbr_f = (gd["edges"][:, :, :6] * gd["nbr_mask"][:, :, None]).sum(1) / gd["nbr_mask"].sum(1, keepdim=True).clamp_min(1.0)
    ya3 = torch.from_numpy(np.column_stack((ya_np[:, 0], ya_np[:, 1], ya_np[:, 2] | ya_np[:, 3])).astype(np.float32)).to(device)

    err_base = np.zeros((len(seeds), n, 6), dtype=np.float64)
    err_apa = np.zeros((len(seeds), n, 6), dtype=np.float64)
    err_apa_np05 = np.zeros((len(seeds), n, 6), dtype=np.float64)
    pk_b0 = np.zeros((n, 6, tf, 2), dtype=np.float64)
    pk_a0 = np.zeros((n, 6, tf, 2), dtype=np.float64)
    idx_b0 = np.zeros(n, dtype=np.int64)
    idx_a0 = np.zeros(n, dtype=np.int64)
    idx_np0 = np.zeros(n, dtype=np.int64)
    pre_prob = {nm: np.zeros((len(seeds), n, 3), dtype=np.float64) for nm in ("NoPose", "WithPose")}

    for si, seed in enumerate(seeds):
        for g in groups:
            tr, te = np.flatnonzero(grp_arr != g), np.flatnonzero(grp_arr == g)
            tr_g, te_g = torch.from_numpy(tr).to(device), torch.from_numpy(te).to(device)
            src_g = sorted(set(grp_arr[tr].tolist()))
            val_sets = [src_g[k :: min(3, max(1, len(src_g)))] for k in range(min(3, max(1, len(src_g))))] if len(src_g) >= 2 else [src_g]

            # 1) Train APA delay + precursor detector (NoPose vs WithPose) strictly on tr
            for tag_p, up in (("NoPose", False), ("WithPose", True)):
                torch.manual_seed(seed + (13 if up else 7))
                apa_net = CausalAPAPrecursorLeadNet(kin_f.size(-1), pose_f.size(-1), nbr_f.size(-1), use_pose=up, use_nbr=True).to(device)
                opt_a = torch.optim.AdamW(apa_net.parameters(), lr=1.8e-3, weight_decay=2e-4)
                pw3 = ((1.0 - ya3[tr_g]).sum(0).clamp_min(1.0) / ya3[tr_g].sum(0).clamp_min(1.0)).clamp(1.0, 12.0)
                on_s_tr = np.where(onset_np[tr, :4].max(1) >= 0, (onset_np[tr, :4].max(1) + 1) * dt, 0.0)
                apa_net.train()
                for ep in range(28):
                    perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed * 431 + ep)).to(device)
                    for cg in perm.split(bsz):
                        ids = tr_g[cg]
                        lg3, dl_s = apa_net(kin_f[ids], pose_f[ids], q_f[ids], nbr_f[ids])
                        bce3 = F.binary_cross_entropy_with_logits(lg3, ya3[ids], pos_weight=pw3)
                        pm = ya3[ids].sum(1) > 0.5
                        td = torch.from_numpy(on_s_tr[cg.cpu().numpy()]).float().to(device)
                        dl_loss = F.smooth_l1_loss(dl_s[pm], td[pm]) if pm.any() else lg3.new_tensor(0.0)
                        (bce3 + 0.35 * dl_loss).backward(); opt_a.step(); opt_a.zero_grad(set_to_none=True)
                apa_net.eval()
                with torch.inference_mode():
                    lg_te, dl_te = apa_net(kin_f[te_g], pose_f[te_g], q_f[te_g], nbr_f[te_g])
                    pre_prob[tag_p][si, te] = torch.sigmoid(lg_te).cpu().numpy()
                    if up:
                        lg_tr, dl_tr = apa_net(kin_f[tr_g], pose_f[tr_g], q_f[tr_g], nbr_f[tr_g])
                        p_tr3 = torch.sigmoid(lg_tr).cpu().numpy()
                        d_tr_np, d_te_np = dl_tr.cpu().numpy(), dl_te.cpu().numpy()

            # 2) Train 6-slot OOF + full on tr, then compare immediate vs delayed-onset APA Slot 1/2
            oof_ml, oof_al, oof_pk = np.zeros((len(tr), 6)), np.zeros((len(tr), 5)), np.zeros((len(tr), 6, tf, 2))
            for v_locs in val_sets:
                vm_l = np.isin(grp_arr[tr], v_locs) if len(val_sets) > 1 else (np.arange(len(tr)) % 2 == 0)
                m_in = fit_6slot_arm_fold(seed, True, True, grp_arr, gd, tr[~vm_l], bsz, max(18, ep_slot - 8), lr_slot, device)
                with torch.inference_mode():
                    oo = _fwd_6slot(m_in, gd, torch.from_numpy(tr[vm_l]).to(device))
                    oof_ml[vm_l], oof_al[vm_l], oof_pk[vm_l] = oo["logits"].cpu().numpy(), oo["act_logits"].cpu().numpy(), oo["paths"].cpu().numpy()

            oof_pk_apa = oof_pk.copy()
            s_tr, b_tr = build_delayed_onset_start_stop_candidates(hist_np[tr], coronal_np[tr], d_tr_np, fut_s_np[tr], dt_obs=dt)
            oof_pk_apa[:, 1], oof_pk_apa[:, 2] = s_tr, b_tr
            cal_b = fit_strict_source_platt_and_slot_selector(oof_ml, oof_al, oof_pk, truth_np[tr], ya_np[tr], va_np[tr])
            cal_a = fit_strict_source_platt_and_slot_selector(oof_ml, oof_al, oof_pk_apa, truth_np[tr], ya_np[tr], va_np[tr])

            m_f = fit_6slot_arm_fold(seed, True, True, grp_arr, gd, tr, bsz, ep_slot, lr_slot, device)
            with torch.inference_mode():
                ot = _fwd_6slot(m_f, gd, te_g)
                pwt, lgt, alt = ot["paths"].cpu().numpy(), ot["logits"].cpu().numpy(), ot["act_logits"].cpu().numpy()
            pwt_apa = pwt.copy()
            s_te, b_te = build_delayed_onset_start_stop_candidates(hist_np[te], coronal_np[te], d_te_np, fut_s_np[te], dt_obs=dt)
            pwt_apa[:, 1], pwt_apa[:, 2] = s_te, b_te

            ib, _, _ = apply_6slot_selector(lgt, alt, va_np[te], cal_b)
            ia, _, _ = apply_6slot_selector(lgt, alt, va_np[te], cal_a)
            # Neyman-Pearson 5%-FPR source-calibrated Start/Stop override when stationary/moving eligibility holds
            neg_st = (~ya_np[tr, 0]) & va_np[tr, 0]
            neg_sp = (~ya_np[tr, 1]) & va_np[tr, 1]
            th_st = float(np.quantile(p_tr3[neg_st, 0], 0.95)) if neg_st.sum() >= 5 else 0.85
            th_sp = float(np.quantile(p_tr3[neg_sp, 1], 0.95)) if neg_sp.sum() >= 5 else 0.85
            inp = ia.copy()
            inp[(pre_prob["WithPose"][si, te, 0] >= th_st) & va_np[te, 0]] = 1
            inp[(pre_prob["WithPose"][si, te, 1] >= th_sp) & va_np[te, 1]] = 2

            err_base[si, te] = horizon_and_post_onset_errors(pwt[np.arange(len(te)), ib], truth_np[te], onset_np[te], dt=dt)
            err_apa[si, te] = horizon_and_post_onset_errors(pwt_apa[np.arange(len(te)), ia], truth_np[te], onset_np[te], dt=dt)
            err_apa_np05[si, te] = horizon_and_post_onset_errors(pwt_apa[np.arange(len(te)), inp], truth_np[te], onset_np[te], dt=dt)
            if si == 0:
                pk_b0[te], pk_a0[te] = pwt - hist_np[te, -1:, None, :], pwt_apa - hist_np[te, -1:, None, :]
                idx_b0[te], idx_a0[te], idx_np0[te] = ib, ia, inp

    truth_rel = truth_np - hist_np[:, -1:, :]
    eb, ea, enp = err_base.mean(0), err_apa.mean(0), err_apa_np05.mean(0)
    ci_fn = (lambda d: safe_site_clip_ci(d, grp_arr, unit_arr)) if is_pinns else (lambda d: _loc_bootstrap_ci(d, grp_arr))

    # Pre-onset recall stratified by onset lead horizon [0.1..0.8s] vs [0.9..2.8s] at matched 5% source FPR
    pre_recall = {}
    for ci, cn in enumerate(("Start", "Stop", "Turn")):
        y_c = ya_np[:, 2] | ya_np[:, 3] if ci == 2 else ya_np[:, ci]
        v_c = (va_np[:, 2] | va_np[:, 3]) if ci == 2 else va_np[:, ci]
        neg_m = (~y_c) & v_c
        sub_d = {}
        for tag_p in ("NoPose", "WithPose"):
            pm = pre_prob[tag_p].mean(0)[:, ci]
            # Per-group source-only 5% FPR threshold
            hits = np.zeros(n, dtype=bool)
            for g in groups:
                src_neg = neg_m & (grp_arr != g)
                th = float(np.quantile(pm[src_neg], 0.95)) if src_neg.sum() >= 5 else 0.85
                hits[grp_arr == g] = pm[grp_arr == g] >= th
            pos_m = y_c & v_c
            sub_d[tag_p] = {
                "n_pos": int(pos_m.sum()),
                "empirical_fpr": float(hits[neg_m].mean()) if neg_m.any() else 0.0,
                "pre_onset_recall_all": float(hits[pos_m].mean()) if pos_m.any() else 0.0,
            }
        pre_recall[cn] = sub_d

    return {
        "n_windows": n,
        "overall_ade1_fde1_post": {
            "Last2_CV": [float(x) for x in err_cv.mean(0)],
            "S6_Immediate_Base": [float(x) for x in eb.mean(0)],
            "S6_APA_Delayed_NetGain": [float(x) for x in ea.mean(0)],
            "S6_APA_Delayed_NP5pctFPR": [float(x) for x in enp.mean(0)],
        },
        "paired_apa_minus_immediate_3s": ci_fn(ea[:, 0:2] - eb[:, 0:2]),
        "paired_apa_minus_immediate_post_onset": ci_fn(ea[:, 4:6] - eb[:, 4:6]),
        "candidate_diag_seed0": {
            "S6_Immediate_Base": audit_6candidate_generation_vs_selection(hist_np, pk_b0, idx_b0, truth_rel, cv_rel, ya_np, va_np, diag_mask, dt=dt),
            "S6_APA_Delayed_NetGain": audit_6candidate_generation_vs_selection(hist_np, pk_a0, idx_a0, truth_rel, cv_rel, ya_np, va_np, diag_mask, dt=dt),
            "S6_APA_Delayed_NP5pctFPR": audit_6candidate_generation_vs_selection(hist_np, pk_a0, idx_np0, truth_rel, cv_rel, ya_np, va_np, diag_mask, dt=dt),
        },
        "matched_5pct_fpr_pre_onset_recall": pre_recall,
    }


def extract_jta_sliding_leadtime_events(st_zip: Path, max_train_files: int = 80) -> dict[str, np.ndarray]:
    """Extract causal 2.5Hz sliding windows around JTA urban Start/Stop/Turn onsets (tau in [-1.2s..+0.4s])."""
    from scripts.run_wave21_jrdb_and_jta_emergence_maneuvers import JTA22_TO_COCO17
    kin_l, pose_l, q_l, nbr_l, y_l, tau_l, eid_l, ecls_l, neg_l, deocc_l, seq_l, sp_l = ([] for _ in range(12))
    ev_c = 0
    hm2, fm8 = np.ones(2, dtype=np.float32), np.ones(8, dtype=np.float32)

    with zipfile.ZipFile(st_zip) as zf:
        tr_f = sorted(n for n in zf.namelist() if n.startswith("jta/data/train/") and n.endswith(".ndjson"))[:max_train_files]
        va_f = sorted(n for n in zf.namelist() if n.startswith("jta/data/val/") and n.endswith(".ndjson"))
        te_f = sorted(n for n in zf.namelist() if n.startswith("jta/data/test/") and n.endswith(".ndjson"))
        for sp, fn in [("train", x) for x in tr_f] + [("val", x) for x in va_f] + [("test", x) for x in te_f]:
            seq = Path(fn).stem
            by_f: dict[int, dict[int, dict]] = {}
            for ln in zf.read(fn).decode("utf-8", errors="ignore").splitlines():
                if ln.strip():
                    ob = json.loads(ln)
                    if "track" in ob:
                        t = ob["track"]; by_f.setdefault(int(t["f"]), {})[int(t["p"])] = t
            frs = sorted(by_f)
            if len(frs) < 16:
                continue
            f_step = int(np.median(np.diff(frs)))
            ptr: dict[int, list[int]] = {}
            for fi in frs:
                for p in by_f[fi]:
                    ptr.setdefault(p, []).append(fi)
            for pid, pfs in sorted(ptr.items()):
                fts = np.asarray(sorted(pfs))
                for run in np.split(fts, np.flatnonzero(np.diff(fts) != f_step) + 1):
                    if len(run) < 14:
                        continue
                    xy_all = np.array([[float(by_f[int(x)][pid]["x"]), float(by_f[int(x)][pid]["y"])] for x in run], dtype=np.float64)
                    yk, vk, ok = compute_action_labels_with_onset(xy_all[:2] - xy_all[1:2], hm2, xy_all[2:10] - xy_all[1:2], fm8, dt=0.4, min_run=2)
                    has_man = False
                    for ci, cls_out in ((0, 0), (1, 1), (2, 2), (3, 2)):
                        if yk[ci] and vk[ci] and ok[ci] >= 1:
                            has_man = True
                            idx_on = 2 + int(ok[ci])
                            eid = ev_c; ev_c += 1
                            for st in (-3, -2, -1, 0, 1):
                                ic = idx_on + st
                                if ic < 2 or ic >= len(run):
                                    continue
                                f_a, f_b, f_c = int(run[ic - 2]), int(run[ic - 1]), int(run[ic])
                                h3 = np.array([[float(by_f[ff][pid]["x"]), float(by_f[ff][pid]["y"])] for ff in (f_a, f_b, f_c)], dtype=np.float32)
                                hr = h3 - h3[-1:]
                                v3 = np.vstack((np.zeros((1, 2), dtype=np.float32), (hr[1:] - hr[:-1]) / 0.4))
                                k3 = np.concatenate((hr, v3, np.linalg.norm(v3, axis=-1, keepdims=True)), axis=-1)
                                p3 = np.zeros((3, 8), dtype=np.float32)
                                for ti, ff in enumerate((f_a, f_b, f_c)):
                                    rc = by_f[ff][pid]
                                    bw, bh = max(float(rc.get("bb_width", 35.0)), 8.0), max(float(rc.get("bb_height", 65.0)), 16.0)
                                    xx = [float(rc.get(f"xx{j}", 0.0)) for j in JTA22_TO_COCO17]
                                    yy = [float(rc.get(f"yy{j}", 0.0)) for j in JTA22_TO_COCO17]
                                    p3[ti] = [(xx[5] - xx[6]) / bw, (yy[5] - yy[6]) / bh, (xx[11] - xx[12]) / bw, (yy[11] - yy[12]) / bh,
                                              (0.5 * (xx[5] + xx[6]) - 0.5 * (xx[11] + xx[12])) / bw, (0.5 * (yy[5] + yy[6]) - 0.5 * (yy[11] + yy[12])) / bh,
                                              abs(xx[15] - xx[16]) / bw, bw / bh]
                                n_cnt = min(len(by_f[f_c]) - 1, 10) / 10.0
                                yv = np.zeros(3, dtype=np.float32); yv[cls_out] = 1.0
                                kin_l.append(k3); pose_l.append(p3); q_l.append([1.0, 1.0]); nbr_l.append([n_cnt, 0.0, 0.0])
                                y_l.append(yv); tau_l.append(0.4 * st); eid_l.append(eid); ecls_l.append(cls_out)
                                neg_l.append(False); deocc_l.append(ic <= 5); seq_l.append(seq); sp_l.append(sp)
                            break
                    if (not has_man) and yk[4] and len(run) >= 12:
                        eid = ev_c; ev_c += 1
                        f_a, f_b, f_c = int(run[3]), int(run[4]), int(run[5])
                        h3 = np.array([[float(by_f[ff][pid]["x"]), float(by_f[ff][pid]["y"])] for ff in (f_a, f_b, f_c)], dtype=np.float32)
                        hr = h3 - h3[-1:]; v3 = np.vstack((np.zeros((1, 2), dtype=np.float32), (hr[1:] - hr[:-1]) / 0.4))
                        k3 = np.concatenate((hr, v3, np.linalg.norm(v3, axis=-1, keepdims=True)), axis=-1)
                        p3 = np.zeros((3, 8), dtype=np.float32)
                        for ti, ff in enumerate((f_a, f_b, f_c)):
                            rc = by_f[ff][pid]
                            bw, bh = max(float(rc.get("bb_width", 35.0)), 8.0), max(float(rc.get("bb_height", 65.0)), 16.0)
                            xx = [float(rc.get(f"xx{j}", 0.0)) for j in JTA22_TO_COCO17]
                            yy = [float(rc.get(f"yy{j}", 0.0)) for j in JTA22_TO_COCO17]
                            p3[ti] = [(xx[5] - xx[6]) / bw, (yy[5] - yy[6]) / bh, (xx[11] - xx[12]) / bw, (yy[11] - yy[12]) / bh,
                                      (0.5 * (xx[5] + xx[6]) - 0.5 * (xx[11] + xx[12])) / bw, (0.5 * (yy[5] + yy[6]) - 0.5 * (yy[11] + yy[12])) / bh,
                                      abs(xx[15] - xx[16]) / bw, bw / bh]
                        kin_l.append(k3); pose_l.append(p3); q_l.append([1.0, 1.0]); nbr_l.append([min(len(by_f[f_c]) - 1, 10) / 10.0, 0.0, 0.0])
                        y_l.append(np.zeros(3, dtype=np.float32)); tau_l.append(0.0); eid_l.append(eid); ecls_l.append(-1)
                        neg_l.append(True); deocc_l.append(False); seq_l.append(seq); sp_l.append(sp)
    return {
        "kin": np.asarray(kin_l, dtype=np.float32), "pose": np.asarray(pose_l, dtype=np.float32),
        "q": np.asarray(q_l, dtype=np.float32), "veh": np.asarray(nbr_l, dtype=np.float32),
        "y": np.asarray(y_l, dtype=np.float32), "tau_s": np.asarray(tau_l, dtype=np.float64),
        "event_id": np.asarray(eid_l, dtype=np.int64), "event_cls": np.asarray(ecls_l, dtype=np.int64),
        "is_neg": np.asarray(neg_l, dtype=bool), "is_deocc": np.asarray(deocc_l, dtype=bool),
        "vid": np.asarray(seq_l), "split": np.asarray(sp_l),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=("gpu0_jaad_jta_pinns", "gpu1_jrdb"), required=True)
    ap.add_argument("--scratch-dir", type=Path, required=True)
    ap.add_argument("--meta-dir", type=Path, default=None)
    ap.add_argument("--pose-npz", type=Path, default=None)
    ap.add_argument("--joint-zip", type=Path, default=None)
    ap.add_argument("--causal-map-bin", type=Path, default=None)
    ap.add_argument("--zip-path", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)

    if args.part == "gpu0_jaad_jta_pinns":
        j_ev = extract_jaad2_sliding_leadtime_events(args.scratch_dir)
        jaad_lt = eval_sliding_leadtime_5seed(j_ev, seeds, dev, epochs=26)
        print(f"[W22-JAAD-LT] Completed n_samples={len(j_ev['kin'])}", flush=True)

        st_zip = args.scratch_dir / "st_releases.zip"
        if not st_zip.exists():
            _download_verified_zip(ST_RELEASE_URL, st_zip)
        t_ev = extract_jta_sliding_leadtime_events(st_zip, max_train_files=80)
        jta_lt = eval_sliding_leadtime_5seed(t_ev, seeds, dev, epochs=24)
        print(f"[W22-JTA-LT] Completed n_samples={len(t_ev['kin'])}", flush=True)

        dp = load_ppt_cohort_with_causal_maps(args.meta_dir, args.pose_npz, args.joint_zip, args.causal_map_bin)
        blind = blind_pinns_physical_emergence_audit(args.meta_dir, args.pose_npz, dp)
        hp_np, tp_np = dp["history"].numpy(), dp["truth"].numpy()
        n = len(hp_np)
        hm, fm = np.ones(5, dtype=np.float32), np.ones(30, dtype=np.float32)
        ya_np, va_np, on_np = np.zeros((n, 5), dtype=bool), np.zeros((n, 5), dtype=bool), np.full((n, 5), -1, dtype=np.int64)
        for i in range(n):
            ya_np[i], va_np[i], on_np[i] = compute_action_labels_with_onset(hp_np[i] - hp_np[i, -1:], hm, tp_np[i] - hp_np[i, -1:], fm, dt=0.1)
        gdp = {k: dp[k].to(dev) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
        gdp["y_act"], gdp["valid_act"] = torch.from_numpy(ya_np).to(dev), torch.from_numpy(va_np).to(dev)
        gdp["obs_dt"] = torch.full((n, 4), 0.1, dtype=torch.float64, device=dev)
        fut_s_np = np.broadcast_to(np.arange(1, 31, dtype=np.float64)[None, :] * 0.1, (n, 30)).copy()
        gdp["fut_s"] = torch.from_numpy(fut_s_np).to(dev)
        cor_np = dp["pose_vec_world"].numpy()[:, -1, 0, :]
        pinns_apa = eval_delayed_onset_apa_upgrade_cohort(
            "pinns_apa", dp["site"], dp["clip"], True, gdp, hp_np, tp_np, cor_np, ya_np, va_np, on_np,
            blind["e_true_all"] | dp["strata"]["first_short_323"], 0.1, fut_s_np, 64, 34, 1.6e-3, seeds, dev,
        )
        summary = {"experiment": "wave22_apa_leadtime_jaad_jta_and_pinns_delayed", "jaad2_sliding_leadtime": jaad_lt, "jta228_sliding_leadtime": jta_lt, "pinns_delayed_apa_upgrade": pinns_apa}
    else:
        repo_root = Path(__file__).resolve().parents[1]
        dj, bj = load_jrdb_with_blind_emergence(repo_root)
        nj = len(dj["history"])
        hj_np, tj_np = dj["history"].numpy(), dj["truth"].numpy()
        gdj = {k: dj[k].to(dev) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
        gdj["y_act"], gdj["valid_act"] = torch.from_numpy(dj["y_act"]).to(dev), torch.from_numpy(dj["valid_act"]).to(dev)
        gdj["obs_dt"] = torch.full((nj, 1), 0.4, dtype=torch.float64, device=dev)
        fut_j_np = np.broadcast_to(np.arange(1, 8, dtype=np.float64)[None, :] * 0.4, (nj, 7)).copy()
        gdj["fut_s"] = torch.from_numpy(fut_j_np).to(dev)
        cor_j = dj["pose_vec_world"].numpy()[:, -1, 0, :]
        jrdb_apa = eval_delayed_onset_apa_upgrade_cohort(
            "jrdb_apa", dj["location"], dj["actors"], False, gdj, hj_np, tj_np, cor_j, dj["y_act"], dj["valid_act"], dj["onset"],
            bj["e_true"], 0.4, fut_j_np, 512, 34, 1.8e-3, seeds, dev,
        )
        summary = {"experiment": "wave22_jrdb_delayed_apa_upgrade", "jrdb_9location_delayed_apa": jrdb_apa}

    raw = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = (json.dumps({"summary.json": hashlib.sha256(raw).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw); zf.writestr("sha256.json", sh)
    print("Saved:", args.zip_path, flush=True)


if __name__ == "__main__":
    main()
