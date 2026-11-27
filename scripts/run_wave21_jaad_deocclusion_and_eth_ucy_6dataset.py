"""Wave 21B-GPU1: Human-annotated JAAD 2.0 de-occlusion pose+vehicle precursor + ETH/UCY 5-scene + DUT/IMPTC."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.run_imptc_and_dair_o2_full_external import ensure_imptc_sample, load_imptc_with_neighbors
from scripts.run_wave18_action_calibration_and_rerank_5seed import fit_platt_and_thresholds
from scripts.run_wave18_conflict_gated_interaction_5seed import _loc_bootstrap_ci
from scripts.run_wave19_dut_imptc_conflict_and_conformal_5seed import adapt_imptc_cohort, load_dut_graph_cohort
from scripts.run_wave21_pinns_true_emergence_and_6slot_diag import (
    SEEDS,
    _fwd_6slot,
    fit_6slot_arm_fold,
    run_2x2_6slot_and_baseline_eval,
)
from src.data.scene_graph import build_neighbor_edges
from src.metrics.metrics_and_stats import _binary_average_precision
from src.models.emergence_maneuver_6slot import (
    apply_6slot_selector,
    audit_6candidate_generation_vs_selection,
    compute_action_labels_with_onset,
    fit_strict_source_platt_and_slot_selector,
    horizon_and_post_onset_errors,
)

JAAD_ZIP_URL = "https://codeload.github.com/ykotseruba/JAAD/zip/refs/heads/JAAD_2.0"
GRAPHTERN_ZIP_URL = "https://codeload.github.com/InhwanBae/GraphTERN/zip/refs/heads/main"
VEH_ACT_MAP = {"stopped": 0.0, "decelerating": 0.25, "moving_slow": 0.50, "moving_fast": 0.85, "accelerating": 1.0}


def _fetch_url_bytes(url: str, cache_file: Path) -> bytes:
    if cache_file.exists() and cache_file.stat().st_size > 100_000:
        return cache_file.read_bytes()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ShortPose-Wave21/1.0"})
    blob = urllib.request.urlopen(req, timeout=60).read()
    cache_file.write_bytes(blob)
    return blob


def load_jaad2_human_deocclusion_benchmark(scratch_dir: Path) -> dict:
    """Parse JAAD 2.0 346 videos: human-annotated full/part->none de-occlusion, body pose, look, & Start/Stop."""
    blob = _fetch_url_bytes(JAAD_ZIP_URL, scratch_dir / "jaad2.zip")
    splits: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        pfx = "JAAD-JAAD_2.0/"
        for sp in ("train", "val", "test"):
            for vid in zf.read(f"{pfx}split_ids/default/{sp}.txt").decode("utf-8").splitlines():
                if vid.strip():
                    splits[vid.strip()] = sp

        ann_files = sorted(n for n in zf.namelist() if n.startswith(f"{pfx}annotations/video_") and n.endswith(".xml"))
        hist_l, fut_l, pose_l, veh_l, ya_l, va_l, on_l = [], [], [], [], [], [], []
        deocc_l, lowd_l, ctrl_cand_l, vid_l, sp_l = [], [], [], [], []

        for af in ann_files:
            vid = Path(af).stem
            sp = splits.get(vid, "train")
            root = ET.fromstring(zf.read(af))
            # Appearance XML (pose_front/back/left/right)
            app_name = f"{pfx}annotations_appearance/{vid}_appearance.xml"
            pose_map: dict[tuple[str, int], np.ndarray] = {}
            if app_name in zf.namelist():
                aroot = ET.fromstring(zf.read(app_name))
                for tr in aroot.findall(".//track"):
                    pid_a = tr.attrib.get("id", "")
                    for bx in tr.findall("box"):
                        fr = int(bx.attrib["frame"])
                        pose_map[(pid_a, fr)] = np.array([
                            float(bx.attrib.get("pose_front", "0")),
                            float(bx.attrib.get("pose_back", "0")),
                            float(bx.attrib.get("pose_left", "0")),
                            float(bx.attrib.get("pose_right", "0")),
                        ], dtype=np.float32)

            veh_name = f"{pfx}annotations_vehicle/{vid}_vehicle.xml"
            veh_map: dict[int, float] = {}
            if veh_name in zf.namelist():
                vroot = ET.fromstring(zf.read(veh_name))
                for vf in vroot.findall(".//frame"):
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
                        "occ": at.get("occlusion", "none"),
                        "walk": 1 if at.get("action", "standing") == "walking" else 0,
                        "look": 1.0 if at.get("look", "") == "looking" else 0.0,
                        "gest": 0.0 if at.get("hand_gesture", "__undefined__") == "__undefined__" else 1.0,
                        "nod": 1.0 if at.get("nod", "__undefined__") == "nodding" else 0.0,
                        "cross": 1.0 if at.get("cross", "") == "crossing" else 0.0,
                    }
                if len(boxes) < 75:
                    continue
                frs = sorted(boxes)
                f_min = frs[0]
                # Sample non-overlapping 10Hz windows (stride 3 raw frames): 5 obs (0.4s) + 20 fut (2.0s)
                # Always sample the earliest valid emergence window + non-overlapping windows thereafter
                step_starts = list(range(f_min, frs[-1] - 72, 45))
                # Also explicitly add the first frame where occlusion transitions from full/part -> none!
                for k_i in range(4, len(frs) - 70):
                    fn_cur = frs[k_i]
                    if boxes[fn_cur]["occ"] == "none" and any(boxes.get(fn_cur - d, {}).get("occ") in ("full", "part") for d in (3, 6, 9, 12)):
                        if all(abs(fn_cur - 12 - ss) >= 18 for ss in step_starts) and fn_cur - 12 >= f_min:
                            step_starts.append(fn_cur - 12)
                for s_raw in sorted(step_starts):
                    req = [s_raw + 3 * k for k in range(25)]
                    if not all(r_f in boxes for r_f in req):
                        continue
                    h_r, f_r = req[:5], req[5:]
                    h_ref = max(float(np.median([boxes[r_f]["h"] for r_f in h_r])), 25.0)
                    # Body-height-normalized image-plane footpoint trajectory (scale-invariant across range)
                    pts = np.array([[(boxes[r_f]["fcx"] - boxes[h_r[-1]]["fcx"]) / h_ref, (boxes[r_f]["fby"] - boxes[h_r[-1]]["fby"]) / h_ref] for r_f in req], dtype=np.float64)
                    w_h = [boxes[r_f]["walk"] for r_f in h_r]
                    w_f = np.array([boxes[r_f]["walk"] for r_f in f_r], dtype=int)
                    c_f = np.array([boxes[r_f]["cross"] > 0.5 for r_f in f_r], dtype=bool)
                    # Human-annotated Start (standing -> >=3 steps walking), Stop (walking -> >=3 steps standing), Cross onset
                    is_stand = sum(w_h[-3:]) <= 1
                    is_walk = sum(w_h[-3:]) >= 2
                    t_st = next((i for i in range(len(w_f) - 2) if is_stand and w_f[i : i + 3].all()), -1)
                    t_sp = next((i for i in range(len(w_f) - 2) if is_walk and (1 - w_f[i : i + 3]).all()), -1)
                    t_cr = next((i for i in range(len(c_f) - 2) if (boxes[h_r[-1]]["cross"] < 0.5) and c_f[i : i + 3].all()), -1)
                    ya = np.array([t_st >= 0, t_sp >= 0, t_cr >= 0, (t_st < 0 and t_sp < 0)], dtype=bool)
                    va = np.array([is_stand, is_walk, boxes[h_r[-1]]["cross"] < 0.5, True], dtype=bool)
                    on = np.array([t_st, t_sp, t_cr, -1, 0], dtype=np.int64)

                    p5 = np.zeros((5, 8), dtype=np.float32)
                    v5 = np.zeros((5, 3), dtype=np.float32)
                    for ti, rf in enumerate(h_r):
                        b = boxes[rf]
                        po4 = pose_map.get((pid_t, rf), np.zeros(4, dtype=np.float32))
                        p5[ti, :4] = po4
                        p5[ti, 4:] = [b["look"], b["gest"], b["nod"], b["w"] / b["h"]]
                        dh_dt = (boxes[rf]["h"] - boxes[h_r[max(0, ti - 1)]]["h"]) / h_ref
                        v5[ti] = [veh_map.get(rf, 0.5), dh_dt * 10.0, float(b["h"] >= 60.0)]

                    had_occ = any(boxes.get(rf, {}).get("occ") in ("full", "part") for rf in range(max(f_min, h_r[0] - 12), h_r[-1]))
                    mid_emerge = (s_raw == f_min) and (f_min > 9)
                    unocc_long = (s_raw >= f_min + 45) and all(boxes.get(rf, {}).get("occ") == "none" for rf in range(s_raw - 30, h_r[-1] + 1) if rf in boxes)
                    disp_n = float(np.linalg.norm(np.diff(pts[:5], axis=0), axis=-1).sum())
                    hist_l.append(pts[:5]); fut_l.append(pts[5:])
                    pose_l.append(p5); veh_l.append(v5)
                    ya_l.append(ya); va_l.append(va); on_l.append(on)
                    deocc_l.append(bool(had_occ or mid_emerge))
                    lowd_l.append(bool((had_occ or mid_emerge) and disp_n < 0.25))
                    ctrl_cand_l.append(bool(unocc_long))
                    vid_l.append(vid); sp_l.append(sp)

    return {
        "hist": np.asarray(hist_l, dtype=np.float64), "truth": np.asarray(fut_l, dtype=np.float64),
        "pose5": np.asarray(pose_l, dtype=np.float32), "veh5": np.asarray(veh_l, dtype=np.float32),
        "y_act": np.asarray(ya_l, dtype=bool), "v_act": np.asarray(va_l, dtype=bool),
        "onset": np.asarray(on_l, dtype=np.int64), "deocc": np.asarray(deocc_l, dtype=bool),
        "deocc_lowdisp": np.asarray(lowd_l, dtype=bool), "ctrl": np.asarray(ctrl_cand_l, dtype=bool),
        "vid": np.asarray(vid_l), "split": np.asarray(sp_l),
    }



class JAADPrecursor2x2Net(nn.Module):
    """5-frame causal bbox-motion + human pose/gaze + vehicle context precursor net for JAAD 2.0."""

    def __init__(self, use_pose: bool, use_veh: bool) -> None:
        super().__init__()
        self.use_pose = use_pose
        self.use_veh = use_veh
        self.mot_gru = nn.GRU(4, 48, batch_first=True)
        self.pose_gru = nn.GRU(8, 24, batch_first=True)
        self.dpose_mlp = nn.Sequential(nn.Linear(8, 24), nn.SiLU(), nn.Linear(24, 24))
        self.veh_gru = nn.GRU(3, 16, batch_first=True)
        d_in = 48 + (24 if use_pose else 0) + (16 if use_veh else 0)
        self.act_head = nn.Sequential(nn.Linear(d_in, 48), nn.SiLU(), nn.Linear(48, 4))  # Start, Stop, Cross, Hold
        # 4 explicit trajectory slots: 0=CV, 1=Start/Cross-Left, 2=Start/Cross-Right, 3=Stop/Brake
        self.traj_res = nn.Linear(d_in, 4 * 20 * 2)
        self.mode_lg = nn.Linear(d_in, 4)
        nn.init.normal_(self.traj_res.weight, std=0.01)
        nn.init.zeros_(self.traj_res.bias)

    def forward(self, hist: torch.Tensor, pose5: torch.Tensor, veh5: torch.Tensor) -> dict[str, torch.Tensor]:
        vel = torch.cat((torch.zeros_like(hist[:, :1]), (hist[:, 1:] - hist[:, :-1]) / 0.1), dim=1)
        _, hm = self.mot_gru(torch.cat((hist, vel), dim=-1).float())
        feats = [hm[0]]
        if self.use_pose:
            _, hp = self.pose_gru(pose5)
            feats.append(hp[0] + self.dpose_mlp(pose5[:, -1] - pose5[:, 0]))
        if self.use_veh:
            _, hv = self.veh_gru(veh5)
            feats.append(hv[0])
        ctx = torch.cat(feats, dim=-1)
        t_s = torch.arange(1, 21, dtype=hist.dtype, device=hist.device)[None, :, None] * 0.1
        v_last = vel[:, -1:]
        cv = v_last * t_s
        stop_anc = v_last * (0.45 * (1.0 - torch.exp(-t_s / 0.45)))
        res = self.traj_res(ctx).view(-1, 4, 20, 2).double()
        # Slot 0 = exact CV; Slots 1,2 = Start/Cross residuals; Slot 3 = Stop braking anchor + residual
        p4 = torch.stack((cv, cv + res[:, 1], cv + res[:, 2], stop_anc + 0.3 * res[:, 3]), dim=1)
        return {"paths": p4, "logits": self.mode_lg(ctx), "act_logits": self.act_head(ctx)}


def eval_jaad2_5seed(jd: dict, seeds: list[int], device: torch.device) -> dict:
    sp = jd["split"]
    tr, va, te = np.flatnonzero(sp == "train"), np.flatnonzero(sp == "val"), np.flatnonzero(sp == "test")
    hist_t = torch.from_numpy(jd["hist"]).to(device)
    truth_t = torch.from_numpy(jd["truth"]).to(device)
    pose_t = torch.from_numpy(jd["pose5"]).to(device)
    veh_t = torch.from_numpy(jd["veh5"]).to(device)
    ya_t = torch.from_numpy(jd["y_act"]).to(device)
    va_t = torch.from_numpy(jd["v_act"]).to(device)

    arms = (("J_M0_bbox_only", False, False), ("J_M1_pose_gaze", True, False), ("J_M2_bbox_veh", False, True), ("J_Full_pose_veh", True, True))
    err_te = {a[0]: np.zeros((len(seeds), len(te), 6), dtype=np.float64) for a in arms}
    ap_te = {a[0]: np.zeros((len(seeds), len(te), 4), dtype=np.float64) for a in arms}
    bin_te = {a[0]: np.zeros((len(seeds), len(te), 4), dtype=np.float64) for a in arms}
    cv_te = ((jd["hist"][te, -1:] - jd["hist"][te, -2:-1]) / 0.1) * (np.arange(1, 21)[None, :, None] * 0.1)
    err_cv = horizon_and_post_onset_errors(cv_te, jd["truth"][te], jd["onset"][te], dt=0.1)

    pw = (((~ya_t[tr]) & va_t[tr]).sum(0).float().clamp_min(1.0) / (ya_t[tr] & va_t[tr]).sum(0).float().clamp_min(1.0)).clamp(1.0, 10.0)
    for si, seed in enumerate(seeds):
        for arm_nm, u_p, u_v in arms:
            torch.manual_seed(seed)
            net = JAADPrecursor2x2Net(u_p, u_v).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=1.8e-3, weight_decay=2e-4)
            tr_g = torch.from_numpy(tr).to(device)
            net.train()
            for ep in range(32):
                perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed * 613 + ep)).to(device)
                for cg in perm.split(128):
                    ids = tr_g[cg]
                    oo = net(hist_t[ids], pose_t[ids], veh_t[ids])
                    d = torch.linalg.vector_norm(oo["paths"] - truth_t[ids, None], dim=-1).mean(-1)
                    reg = d[:, 1:].amin(dim=1).mean()
                    soft = (-d.detach() / 0.12).softmax(dim=1).float()
                    cls_l = -(soft * oo["logits"].log_softmax(dim=1)).sum(dim=1).mean()
                    bce = F.binary_cross_entropy_with_logits(oo["act_logits"], ya_t[ids].float(), pos_weight=pw, reduction="none")
                    act_l = ((bce * va_t[ids].float()).sum(1) / va_t[ids].float().sum(1).clamp_min(1.0)).mean()
                    loss = reg + 0.25 * cls_l + 0.40 * act_l
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            net.eval()
            with torch.inference_mode():
                ova = net(hist_t[torch.from_numpy(va).to(device)], pose_t[torch.from_numpy(va).to(device)], veh_t[torch.from_numpy(va).to(device)])
                ote = net(hist_t[torch.from_numpy(te).to(device)], pose_t[torch.from_numpy(te).to(device)], veh_t[torch.from_numpy(te).to(device)])
            # Calibrate Platt & thresholds strictly on official val split (never touching test!)
            va_al = ova["act_logits"].cpu().numpy()
            te_al = ote["act_logits"].cpu().numpy()
            ya_va5 = np.column_stack((jd["y_act"][va], jd["y_act"][va, 3]))
            va_va5 = np.column_stack((jd["v_act"][va], np.ones(len(va), dtype=bool)))
            a_p, b_p, th_p = fit_platt_and_thresholds(np.column_stack((va_al, va_al[:, 3])), ya_va5, va_va5)
            p_te = 1.0 / (1.0 + np.exp(-np.clip(te_al * a_p[:4][None, :] + b_p[:4][None, :], -30.0, 30.0)))
            ap_te[arm_nm][si] = p_te
            bin_te[arm_nm][si] = ((p_te >= th_p[:4][None, :]) & jd["v_act"][te]).astype(np.float64)
            pw_te = ote["paths"].cpu().numpy()
            sel = ote["logits"].argmax(1).cpu().numpy()
            err_te[arm_nm][si] = horizon_and_post_onset_errors(pw_te[np.arange(len(te)), sel], jd["truth"][te], jd["onset"][te], dt=0.1)
    return {"te_ids": te, "err_cv": err_cv, "err_arms": {k: v.mean(0) for k, v in err_te.items()}, "ap_arms": ap_te, "bin_arms": bin_te}



def _summarize_jaad(jd: dict, res: dict) -> dict:
    from scripts.run_wave18_conflict_gated_interaction_5seed import safe_site_clip_ci

    te = res["te_ids"]
    vid_te = jd["vid"][te]
    grp_te = np.array([f"jaad_g{int(v.split('_')[1]) % 4}" for v in vid_te])
    ecv, ea = res["err_cv"], res["err_arms"]
    deocc_te = jd["deocc"][te]
    lowd_te = jd["deocc_lowdisp"][te]
    ctrl_te = jd["ctrl"][te]
    man_te = (jd["y_act"][te, :3] & jd["v_act"][te, :3]).any(axis=1)

    strata = {}
    for sk, sm in (
        ("locked_test_all", np.ones(len(te), dtype=bool)),
        ("human_annotated_deocclusion", deocc_te),
        ("human_deocclusion_lowdisp", lowd_te),
        ("human_deocclusion_start_stop_cross", deocc_te & man_te),
        ("matched_unoccluded_ctrl", ctrl_te),
    ):
        if int(sm.sum()) < 5:
            continue
        mods = {"Last2_CV": {"ade_full": float(ecv[sm, 0].mean()), "fde_full": float(ecv[sm, 1].mean()), "ade_post_onset_0p5s": float(ecv[sm, 4].mean()), "ade_post_onset_1p0s": float(ecv[sm, 5].mean())}}
        for nm, arr in ea.items():
            mods[nm] = {"ade_full": float(arr[sm, 0].mean()), "fde_full": float(arr[sm, 1].mean()), "ade_post_onset_0p5s": float(arr[sm, 4].mean()), "ade_post_onset_1p0s": float(arr[sm, 5].mean())}
        strata[sk] = {
            "n_windows": int(sm.sum()),
            "n_videos": int(len(set(vid_te[sm].tolist()))),
            "models": mods,
            "paired_video_ci": {
                "pose_effect_no_veh_post_onset": safe_site_clip_ci(ea["J_M1_pose_gaze"][sm, 4:6] - ea["J_M0_bbox_only"][sm, 4:6], grp_te[sm], vid_te[sm]),
                "pose_effect_with_veh_post_onset": safe_site_clip_ci(ea["J_Full_pose_veh"][sm, 4:6] - ea["J_M2_bbox_veh"][sm, 4:6], grp_te[sm], vid_te[sm]),
                "veh_effect_no_pose_post_onset": safe_site_clip_ci(ea["J_M2_bbox_veh"][sm, 4:6] - ea["J_M0_bbox_only"][sm, 4:6], grp_te[sm], vid_te[sm]),
                "full_minus_bbox_full_horizon": safe_site_clip_ci(ea["J_Full_pose_veh"][sm, 0:2] - ea["J_M0_bbox_only"][sm, 0:2], grp_te[sm], vid_te[sm]),
                "full_minus_CV_post_onset": safe_site_clip_ci(ea["J_Full_pose_veh"][sm, 4:6] - ecv[sm, 4:6], grp_te[sm], vid_te[sm]),
            },
        }

    act_out = {}
    for arm_nm in ea:
        pm = res["ap_arms"][arm_nm].mean(0)
        bm = res["bin_arms"][arm_nm].mean(0) >= 0.5
        cls_d = {}
        for ci, cn in enumerate(("Start", "Stop", "Cross")):
            vm = deocc_te & jd["v_act"][te, ci]
            yt, ps, pr = jd["y_act"][te, ci][vm], pm[vm, ci], bm[vm, ci]
            tp, fp, fn = int((pr & yt).sum()), int((pr & (~yt)).sum()), int(((~pr) & yt).sum())
            cls_d[cn] = {"n_pos": int(yt.sum()), "ap": float(_binary_average_precision(yt.astype(np.float64), ps)) if yt.sum() > 0 else 0.0, "f1": float((2 * tp) / max(2 * tp + fp + fn, 1)), "tp": tp, "fp": fp}
        act_out[arm_nm] = cls_d
    return {"n_total_windows": len(jd["hist"]), "n_test_windows": len(te), "strata": strata, "deocclusion_maneuver_detection": act_out}


def load_eth_ucy_5scene_emergence(scratch_dir: Path) -> tuple[dict, dict]:
    """Load all 5 ETH/UCY canonical scenes (eth, hotel, univ, zara1, zara2) at 2.5Hz (0.4s -> 3.2s)."""
    blob = _fetch_url_bytes(GRAPHTERN_ZIP_URL, scratch_dir / "graphtern_eth_ucy.zip")
    scene_files = {
        "eth": "datasets/eth/test/biwi_eth.txt",
        "hotel": "datasets/hotel/test/biwi_hotel.txt",
        "univ": "datasets/univ/test/students001.txt",
        "zara1": "datasets/zara1/test/crowds_zara01.txt",
        "zara2": "datasets/zara2/test/crowds_zara02.txt",
    }
    hist_l, truth_l, ed_l, nm_l, ya_l, va_l, on_l, sc_l, unit_l, emg_l, ctrl_l = [], [], [], [], [], [], [], [], [], [], []
    hm2, fm8 = np.ones(2, dtype=np.float32), np.ones(8, dtype=np.float32)

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        all_names = zf.namelist()
        for sc_name, suffix in scene_files.items():
            hits = [n for n in all_names if n.endswith(suffix)]
            if not hits:
                hits = [n for n in all_names if f"/{sc_name}/test/" in n and n.endswith(".txt")]
            by_f: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
            for fn in hits[:1]:
                for ln in zf.read(fn).decode("utf-8", errors="ignore").splitlines():
                    ps = ln.strip().split()
                    if len(ps) >= 4:
                        by_f[int(round(float(ps[0])))][int(round(float(ps[1])))] = np.array([float(ps[2]), float(ps[3])], dtype=np.float64)
            frs = sorted(by_f)
            if len(frs) < 12:
                continue
            f_step = int(np.median(np.diff(frs)))
            min_f = frs[0]
            p_trs: dict[int, list[int]] = defaultdict(list)
            for fi in frs:
                for pid in by_f[fi]:
                    p_trs[pid].append(fi)
            for pid, p_fs in sorted(p_trs.items()):
                fts = np.asarray(sorted(p_fs))
                for r_i, run in enumerate(np.split(fts, np.flatnonzero(np.diff(fts) != f_step) + 1)):
                    for off in range(0, len(run) - 9, 10):
                        f0, f1 = int(run[off]), int(run[off + 1])
                        futs = [int(x) for x in run[off + 2 : off + 10]]
                        h_xy = np.stack([by_f[f0][pid], by_f[f1][pid]], axis=0)
                        t_xy = np.stack([by_f[ff][pid] for ff in futs], axis=0)
                        yk, vk, ok = compute_action_labels_with_onset(h_xy - h_xy[-1:], hm2, t_xy - h_xy[-1:], fm8, dt=0.4, min_run=2)
                        nb_p, nb_v, nb_vv = [], [], []
                        for po, xyo in by_f[f1].items():
                            if po == pid:
                                continue
                            nb_p.append(xyo)
                            if po in by_f.get(f0, {}):
                                nb_v.append((xyo - by_f[f0][po]) / 0.4); nb_vv.append(1.0)
                            else:
                                nb_v.append(np.zeros(2)); nb_vv.append(0.0)
                        if nb_p:
                            nt = np.zeros(len(nb_p), dtype=np.int64)
                            ed, _, nmk = build_neighbor_edges(h_xy[1], (h_xy[1] - h_xy[0]) / 0.4, 1.0, 1.0, np.asarray(nb_p), np.asarray(nb_v), np.asarray(nb_vv), np.ones(len(nb_p)), nt)
                        else:
                            ed, nmk = np.zeros((12, 9), dtype=np.float32), np.zeros(12, dtype=np.float32)
                        is_emg = (off == 0) and (f0 > min_f + f_step)
                        hist_l.append(h_xy - h_xy[-1:]); truth_l.append(t_xy - h_xy[-1:])
                        ed_l.append(ed); nm_l.append(nmk)
                        ya_l.append(yk); va_l.append(vk); on_l.append(ok)
                        sc_l.append(sc_name); unit_l.append(f"{sc_name}:{pid}")
                        emg_l.append(bool(is_emg)); ctrl_l.append(bool(off >= 10))

    n = len(hist_l)
    return {
        "history": torch.from_numpy(np.asarray(hist_l, dtype=np.float64)),
        "truth": torch.from_numpy(np.asarray(truth_l, dtype=np.float64)),
        "bio_even": torch.zeros((n, 2, 11), dtype=torch.float32),
        "pose_vec_world": torch.zeros((n, 2, 2, 2), dtype=torch.float64),
        "signed_knee": torch.zeros((n, 2, 1), dtype=torch.float32),
        "quality": torch.zeros((n, 4), dtype=torch.float32),
        "m_pose": torch.zeros(n, dtype=torch.float32),
        "edges": torch.from_numpy(np.asarray(ed_l, dtype=np.float32)),
        "is_car": torch.zeros((n, 12), dtype=torch.float32),
        "nbr_mask": torch.from_numpy(np.asarray(nm_l, dtype=np.float32)),
        "y_act": np.asarray(ya_l, dtype=bool), "valid_act": np.asarray(va_l, dtype=bool),
        "onset": np.asarray(on_l, dtype=np.int64), "scene": np.asarray(sc_l), "actor": np.asarray(unit_l),
    }, {"e_true": np.asarray(emg_l, dtype=bool), "ctrl": np.asarray(ctrl_l, dtype=bool)}



def _eval_motion_interaction_dataset(tag: str, coh: dict, dt_nom: float, bsz: int, ep_s: int, seeds: list[int], device: torch.device) -> dict:
    """Evaluate 6-slot Motion vs 6-slot Conflict-Interaction + post-onset 0.5s/1.0s on DUT or IMPTC."""
    h_np, t_np = coh["history"], coh["truth"]
    n, tf, _ = t_np.shape
    hm = np.ones(h_np.shape[1], dtype=np.float32)
    fm = np.ones(tf, dtype=np.float32)
    ya, va, on = np.zeros((n, 5), dtype=bool), np.zeros((n, 5), dtype=bool), np.full((n, 5), -1, dtype=np.int64)
    for i in range(n):
        ya[i], va[i], on[i] = compute_action_labels_with_onset(h_np[i] - h_np[i, -1:], hm, t_np[i], fm, dt=dt_nom)
    gd = {
        "history": torch.from_numpy(h_np).to(device), "truth": torch.from_numpy(t_np).to(device),
        "bio_even": torch.zeros((n, h_np.shape[1], 11), device=device), "pose_vec_world": torch.zeros((n, h_np.shape[1], 2, 2), dtype=torch.float64, device=device),
        "signed_knee": torch.zeros((n, h_np.shape[1], 1), device=device), "quality": torch.zeros((n, 4), device=device),
        "m_pose": torch.zeros(n, device=device), "edges": torch.from_numpy(coh["edges"]).to(device),
        "is_car": torch.from_numpy(coh["is_car"]).to(device), "nbr_mask": torch.from_numpy(coh["nbr_mask"]).to(device),
        "y_act": torch.from_numpy(ya).to(device), "valid_act": torch.from_numpy(va).to(device),
        "obs_dt": torch.from_numpy(coh["obs_dt"]).to(device), "fut_s": torch.from_numpy(coh["fut_s"]).to(device),
    }
    v_l = (h_np[:, -1:] - h_np[:, -2:-1]) / np.maximum(coh["obs_dt"][:, -1:, None], 1e-4)
    cv_rel = v_l * coh["fut_s"][:, :, None]
    grp = coh["group"]
    # Blind close vehicle/agent conflict & low-displacement strata
    car_close = ((coh["nbr_mask"] > 0.5) & (coh["is_car"] > 0.5) & (coh["edges"][:, :, 5] > 0.25) & (coh["edges"][:, :, 4] <= 8.0)).any(axis=1)
    disp = np.linalg.norm(np.diff(h_np, axis=1), axis=-1).sum(axis=1)
    man = (ya[:, :4] & va[:, :4]).any(axis=1)
    strata = {
        "all_windows": np.ones(n, dtype=bool),
        "lowdisp_lt0p5m": disp < 0.50,
        "active_maneuver_start_stop_turn": man,
        "approaching_vehicle_conflict": car_close,
    }
    return run_2x2_6slot_and_baseline_eval(
        tag, grp, grp, False, gd, h_np, t_np + h_np[:, -1:, :], cv_rel, ya, va, on, strata,
        man | (disp < 0.50), dt_nom, bsz, (30, 20, 20, 22), ep_s, (2e-3, 1.4e-3, 1.5e-3, 1.4e-3), 1.6e-3, seeds, device, False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch-dir", type=Path, required=True)
    ap.add_argument("--dut-zip", type=Path, default=None)
    ap.add_argument("--imptc-tar", type=Path, default=None)
    ap.add_argument("--zip-path", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--only", choices=("all", "jaad_eth", "dut_imptc"), default="all")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    jaad_res, eth_res, dut_res, imptc_res = {}, {}, {}, {}

    if args.only in ("all", "jaad_eth"):
        # 1) JAAD 2.0 (346 videos: human-annotated physical de-occlusion + pose/gaze + vehicle context)
        jd = load_jaad2_human_deocclusion_benchmark(args.scratch_dir)
        jaad_res = _summarize_jaad(jd, eval_jaad2_5seed(jd, list(args.seeds), dev))
        print(f"[W21-JAAD2] Completed n_total={jaad_res['n_total_windows']} n_test={jaad_res['n_test_windows']}", flush=True)

        # 2) ETH/UCY 5-Scene Leave-One-Dataset-Out (eth, hotel, univ, zara1, zara2)
        de, be = load_eth_ucy_5scene_emergence(args.scratch_dir)
        ne = len(de["history"])
        he_np, te_np = de["history"].numpy(), de["truth"].numpy()
        gde = {k: de[k].to(dev) for k in ("history", "truth", "bio_even", "pose_vec_world", "signed_knee", "quality", "m_pose", "edges", "is_car", "nbr_mask")}
        gde["y_act"] = torch.from_numpy(de["y_act"]).to(dev)
        gde["valid_act"] = torch.from_numpy(de["valid_act"]).to(dev)
        gde["obs_dt"] = torch.full((ne, 1), 0.4, dtype=torch.float64, device=dev)
        gde["fut_s"] = torch.arange(1, 9, dtype=torch.float64, device=dev)[None].expand(ne, -1) * 0.4
        cv_e = ((he_np[:, 1:2] - he_np[:, 0:1]) / 0.4) * (np.arange(1, 9)[None, :, None] * 0.4)
        disp_e = np.linalg.norm(he_np[:, 1] - he_np[:, 0], axis=-1)
        man_e = (de["y_act"][:, :4] & de["valid_act"][:, :4]).any(axis=1)
        eth_strata = {
            "all_5scenes": np.ones(ne, dtype=bool),
            "true_emergence_midclip": be["e_true"],
            "true_emergence_lowdisp_lt0p5m": be["e_true"] & (disp_e < 0.50),
            "emergence_active_maneuver": be["e_true"] & man_e,
            "steady_visible_ctrl": be["ctrl"],
        }
        eth_res = run_2x2_6slot_and_baseline_eval(
            "eth_ucy_5scene", de["scene"], de["actor"], False, gde, he_np, te_np, cv_e, de["y_act"], de["valid_act"], de["onset"],
            eth_strata, be["e_true"] | man_e, 0.4, 512, (30, 20, 20, 22), 28, (2e-3, 1.4e-3, 1.5e-3, 1.4e-3), 1.6e-3, list(args.seeds), dev, False,
        )
        print(f"[W21-ETH-UCY] Completed n={ne}", flush=True)

    if args.only in ("all", "dut_imptc"):
        # 3) DUT (3,066 campus mixed-traffic windows) & IMPTC (564 intersection windows) if provided
        if args.dut_zip is not None and args.dut_zip.exists():
            dut_coh = load_dut_graph_cohort(args.dut_zip)
            dut_res = _eval_motion_interaction_dataset("dut_11clip", dut_coh, 0.1, 256, 28, list(args.seeds), dev)
        if args.imptc_tar is not None:
            imp_path = ensure_imptc_sample(args.imptc_tar)
            imp_coh = adapt_imptc_cohort(load_imptc_with_neighbors(imp_path))
            imptc_res = _eval_motion_interaction_dataset("imptc_4seq", imp_coh, 0.1, 128, 28, list(args.seeds), dev)

    summary = {
        "experiment": "wave21_jaad_deocclusion_and_eth_ucy_6dataset",
        "jaad2_human_deocclusion_346vid": jaad_res,
        "eth_ucy_5scene_lodo": eth_res,
        "dut_3066_maneuver_onset": dut_res,
        "imptc_564_maneuver_onset": imptc_res,
    }
    raw = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode("utf-8")
    sh = (json.dumps({"summary.json": hashlib.sha256(raw).hexdigest()}, indent=2) + "\n").encode("utf-8")
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", raw); zf.writestr("sha256.json", sh)
    print("Saved:", args.zip_path, flush=True)


if __name__ == "__main__":
    main()
