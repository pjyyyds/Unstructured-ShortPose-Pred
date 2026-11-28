"""Wave 25B: JAAD 2.0 per-event [t_reappear, t_first_alarm, t_onset] audit, visible-only alarms, & <0.25 bbox-H."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.run_wave18_conflict_gated_interaction_5seed import _loc_bootstrap_ci
from scripts.run_wave23_jaad_autopose_and_isofpr_leadtime import (
    JAAD_ZIP_URL,
    SEEDS,
    TAUS_10HZ,
    VEH_ACT_MAP,
    _fetch_url_bytes,
    _fit_and_predict_lead_arms,
    _jaad_post_onset_err,
    build_jaad2_strict_and_sliding_cohorts,
)
from src.models.apa_precursor_leadtime import (
    calibrate_matched_fpr_thresholds,
    evaluate_iso_test_fpr_and_unconflated_leadtime,
    paired_iso_fpr_comparison,
)


def exact_mcnemar_two_sided_p(hit_base: np.ndarray, hit_alt: np.ndarray) -> dict[str, int | float]:
    """Exact two-sided binomial (McNemar) p-value on paired event hits."""
    hb = np.asarray(hit_base, dtype=bool)
    ha = np.asarray(hit_alt, dtype=bool)
    both = int((hb & ha).sum())
    alt_only = int(((~hb) & ha).sum())
    base_only = int((hb & (~ha)).sum())
    neither = int(((~hb) & (~ha)).sum())
    n_disc = alt_only + base_only
    if n_disc == 0:
        p_val = 1.0
    else:
        k_min = min(alt_only, base_only)
        tail = sum(math.comb(n_disc, j) for j in range(k_min + 1)) / (2.0 ** n_disc)
        p_val = float(min(1.0, 2.0 * tail))
    return {
        "n_both_hit": both,
        "n_alt_only_hit": alt_only,
        "n_base_only_hit": base_only,
        "n_both_miss": neither,
        "n_discordant": n_disc,
        "exact_two_sided_p": p_val,
    }


def extract_jaad_slide_reappearance_metadata(jaad_zip_bytes: bytes) -> dict[str, np.ndarray]:
    """Deterministically mirror slide_specs to attach exact (vid, pid, t0_r, f_on, f_last_occ, f_reappear, occ_t0)."""
    splits: dict[str, str] = {}
    parsed_tracks = []
    with zipfile.ZipFile(io.BytesIO(jaad_zip_bytes)) as zf:
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
                    boxes[fr] = {
                        "occ": at.get("occlusion", "none"),
                        "walk": 1 if at.get("action", "standing") == "walking" else 0,
                        "cross": 1.0 if at.get("cross", "") == "crossing" else 0.0,
                    }
                if len(boxes) >= 48:
                    parsed_tracks.append((vid, sp, f"{tr_i}:{pid_t}", pid_t, boxes))

    vid_l, pid_l, eid_l, cls_l, t0_l, fon_l, floc_l, freap_l, occt0_l, all5none_l = ([] for _ in range(10))
    ev_counter = 0
    for vid, sp, trk_k, pid_t, boxes in parsed_tracks:
        frs = sorted(boxes)
        evs: list[tuple[int, int]] = []
        for k in range(15, len(frs) - 15):
            fc = frs[k]
            pr = [fc - 3 * d for d in range(1, 6)]
            po = [fc + 3 * d for d in range(0, 4)]
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
            # Compute exact f_last_occ in [f_on - 45, f_on] and f_reappear (earliest frame after f_last_occ continuously 'none' to f_on)
            occ_steps = [f_on - 3 * d for d in range(15, -1, -1) if (f_on - 3 * d) in boxes]
            occ_frames = [rf for rf in occ_steps if boxes[rf]["occ"] in ("full", "part")]
            f_last_occ = max(occ_frames) if occ_frames else -1
            if f_last_occ < 0:
                f_reappear = occ_steps[0] if occ_steps else f_on
            elif boxes.get(f_on, {}).get("occ") != "none":
                f_reappear = 10**9  # still occluded at onset
            else:
                after = [rf for rf in occ_steps if rf > f_last_occ]
                f_reappear = after[0] if after else f_on
            for st in TAUS_10HZ:
                t0_r = f_on + 3 * st
                hr = [t0_r - 3 * (4 - j) for j in range(5)]
                if all(rf in boxes for rf in hr):
                    vid_l.append(vid); pid_l.append(pid_t); eid_l.append(eid); cls_l.append(c_id)
                    t0_l.append(t0_r); fon_l.append(f_on); floc_l.append(f_last_occ); freap_l.append(f_reappear)
                    occt0_l.append(boxes[t0_r]["occ"])
                    all5none_l.append(all(boxes[rf]["occ"] == "none" for rf in hr))
        if not evs:
            for f_neg in frs[18:-24:45]:
                wv = {boxes[rf]["walk"] for rf in frs if abs(rf - f_neg) <= 54}
                cv_s = {boxes[rf]["cross"] for rf in frs if abs(rf - f_neg) <= 54}
                hr = [f_neg - 3 * (4 - j) for j in range(5)]
                if len(wv) == 1 and len(cv_s) == 1 and all(rf in boxes for rf in hr):
                    eid = ev_counter; ev_counter += 1
                    vid_l.append(vid); pid_l.append(pid_t); eid_l.append(eid); cls_l.append(-1)
                    t0_l.append(f_neg); fon_l.append(f_neg); floc_l.append(-1); freap_l.append(f_neg)
                    occt0_l.append(boxes[f_neg]["occ"])
                    all5none_l.append(all(boxes[rf]["occ"] == "none" for rf in hr))
    return {
        "vid": np.asarray(vid_l), "pid": np.asarray(pid_l), "event_id": np.asarray(eid_l, dtype=np.int64),
        "event_cls": np.asarray(cls_l, dtype=np.int64), "t0_frame_30hz": np.asarray(t0_l, dtype=np.int64),
        "f_onset_30hz": np.asarray(fon_l, dtype=np.int64), "f_last_occ_30hz": np.asarray(floc_l, dtype=np.int64),
        "f_reappear_30hz": np.asarray(freap_l, dtype=np.int64), "occ_at_t0": np.asarray(occt0_l),
        "all_5f_none": np.asarray(all5none_l, dtype=bool),
    }


def audit_per_event_timestamps_and_visible_alarms(
    p_va: dict[str, np.ndarray],
    p_te: dict[str, np.ndarray],
    slide_ds: dict[str, np.ndarray],
    meta: dict[str, np.ndarray],
) -> dict:
    """Save exact [t_reappear, t_first_alarm, t_onset] per event + visible-only post-reappearance protocol."""
    va = np.flatnonzero(slide_ds["split"] == "val")
    te = np.flatnonzero(slide_ds["split"] == "test")
    if not np.array_equal(slide_ds["event_id"], meta["event_id"]):
        raise RuntimeError("Event ID alignment mismatch between slide_ds and XML reappearance metadata!")

    neg_va = slide_ds["is_neg"][va]
    neg_te = slide_ds["is_neg"][te]
    tau_te = slide_ds["tau_s"][te]
    eid_te = slide_ds["event_id"][te]
    cls_te = slide_ds["event_cls"][te]
    strict_te = slide_ds["is_deocc_strict_recovered"][te] & (~neg_te)
    occ_te = meta["occ_at_t0"][te]
    t0_te = meta["t0_frame_30hz"][te]
    fon_te = meta["f_onset_30hz"][te]
    floc_te = meta["f_last_occ_30hz"][te]
    freap_te = meta["f_reappear_30hz"][te]
    vid_te, pid_te = meta["vid"][te], meta["pid"][te]

    n_neg_all = int(neg_te.sum())
    vis_te = (occ_te == "none") & (neg_te | (t0_te >= freap_te))
    n_neg_vis = int((neg_te & vis_te).sum())

    # Thresholds under (A) Iso-Test-FPR=5% on all 431 test negatives, (B) Iso-Test-FPR=5% on visible test negatives, (C) Val-calibrated 5%
    thr_iso_all = {nm: calibrate_matched_fpr_thresholds(p_te[nm][neg_te], target_fpr=0.05) for nm in p_te}
    thr_iso_vis = {nm: calibrate_matched_fpr_thresholds(p_te[nm][neg_te & vis_te], target_fpr=0.05) for nm in p_te}
    thr_val_cal = {nm: calibrate_matched_fpr_thresholds(p_va[nm][neg_va], target_fpr=0.05) for nm in p_te}

    cls_names = ("Start", "Stop", "Cross")
    per_event_strict_log = {cn: [] for cn in cls_names}
    reappear_summary = {}

    for ci, cn in enumerate(cls_names):
        eids_c = sorted(set(eid_te[strict_te & (cls_te == ci)].tolist()))
        n_any_occ_pre = 0
        hit_vec_iso = {nm: [] for nm in p_te}
        hit_vec_vis_at_iso = {nm: [] for nm in p_te}
        hit_vec_prot2 = {nm: [] for nm in p_te}
        hit_vec_val = {nm: [] for nm in p_te}

        for eid in eids_c:
            em = (eid_te == eid) & (tau_te <= 1e-9)
            idx_ord = np.flatnonzero(em)[np.argsort(t0_te[em])]
            has_occ_in_pre = bool(np.any(occ_te[idx_ord] != "none"))
            if has_occ_in_pre:
                n_any_occ_pre += 1
            f_on_v = int(fon_te[idx_ord[0]])
            f_lo_v = int(floc_te[idx_ord[0]])
            f_re_v = int(freap_te[idx_ord[0]])
            vis_ord = [ix for ix in idx_ord if (occ_te[ix] == "none") and (t0_te[ix] >= f_re_v)]

            mod_rec = {}
            for nm in p_te:
                th_a = float(thr_iso_all[nm][ci])
                th_v = float(thr_iso_vis[nm][ci])
                th_c = float(thr_val_cal[nm][ci])
                fired_all = [ix for ix in idx_ord if p_te[nm][ix, ci] >= th_a]
                fired_vis_iso = [ix for ix in vis_ord if p_te[nm][ix, ci] >= th_a]
                fired_p2 = [ix for ix in vis_ord if p_te[nm][ix, ci] >= th_v]
                fired_val = [ix for ix in idx_ord if p_te[nm][ix, ci] >= th_c]

                hit_a = len(fired_all) > 0
                f_alm = int(t0_te[fired_all[0]]) if hit_a else None
                occ_alm = str(occ_te[fired_all[0]]) if hit_a else None
                after_reap = bool(hit_a and (f_alm >= f_re_v) and (occ_alm == "none"))
                has_vis_alarm_iso = len(fired_vis_iso) > 0
                f_alm_vis_iso = int(t0_te[fired_vis_iso[0]]) if has_vis_alarm_iso else None

                hit_vec_iso[nm].append(hit_a)
                hit_vec_vis_at_iso[nm].append(has_vis_alarm_iso)
                hit_vec_prot2[nm].append(len(fired_p2) > 0)
                hit_vec_val[nm].append(len(fired_val) > 0)

                mod_rec[nm] = {
                    "hit_pre_onset_iso_5p1": hit_a,
                    "f_first_alarm_30hz": f_alm,
                    "lead_ms_from_first_alarm": float((f_on_v - f_alm) / 30.0 * 1000.0) if f_alm is not None else None,
                    "occ_at_first_alarm": occ_alm,
                    "first_alarm_strictly_after_reappearance": after_reap,
                    "has_post_reappear_visible_alarm_iso_5p1": has_vis_alarm_iso,
                    "f_first_visible_alarm_30hz": f_alm_vis_iso,
                    "lead_ms_from_first_visible_alarm": float((f_on_v - f_alm_vis_iso) / 30.0 * 1000.0) if f_alm_vis_iso is not None else None,
                    "hit_protocol2_visible_only_iso_fpr": len(fired_p2) > 0,
                    "f_first_protocol2_alarm_30hz": int(t0_te[fired_p2[0]]) if fired_p2 else None,
                    "hit_val_calibrated": len(fired_val) > 0,
                }
            per_event_strict_log[cn].append({
                "event_id": int(eid),
                "video_id": str(vid_te[idx_ord[0]]),
                "ped_id": str(pid_te[idx_ord[0]]),
                "f_last_occ_30hz": f_lo_v,
                "f_reappear_30hz": f_re_v,
                "f_onset_30hz": f_on_v,
                "visible_window_span_ms_before_onset": float(max(0, f_on_v - f_re_v) / 30.0 * 1000.0),
                "n_scored_pre_onset_windows": len(idx_ord),
                "n_visible_post_reappear_pre_onset_windows": len(vis_ord),
                "has_any_occluded_decision_frame_in_pre_window": has_occ_in_pre,
                "per_model": mod_rec,
            })

        hb_iso = np.asarray(hit_vec_iso["Lead_M0_kin_only"], dtype=bool)
        ha_iso = np.asarray(hit_vec_iso["Lead_AutoPose_rtmpose17"], dtype=bool)
        new_auto_idx = [j for j in range(len(eids_c)) if (not hb_iso[j]) and ha_iso[j]]
        new_first_after_reap = sum(
            1 for j in new_auto_idx
            if per_event_strict_log[cn][j]["per_model"]["Lead_AutoPose_rtmpose17"]["first_alarm_strictly_after_reappearance"]
        )
        new_has_vis_alarm = sum(
            1 for j in new_auto_idx
            if per_event_strict_log[cn][j]["per_model"]["Lead_AutoPose_rtmpose17"]["has_post_reappear_visible_alarm_iso_5p1"]
        )

        mod_stats = {}
        for nm in p_te:
            fp_iso = int((p_te[nm][neg_te, ci] >= float(thr_iso_all[nm][ci])).sum())
            fp_vis = int((p_te[nm][neg_te & vis_te, ci] >= float(thr_iso_vis[nm][ci])).sum())
            fp_val = int((p_te[nm][neg_te, ci] >= float(thr_val_cal[nm][ci])).sum())
            vis_leads_p2 = [
                (per_event_strict_log[cn][j]["f_onset_30hz"] - per_event_strict_log[cn][j]["per_model"][nm]["f_first_protocol2_alarm_30hz"]) / 30.0 * 1000.0
                for j in range(len(eids_c))
                if per_event_strict_log[cn][j]["per_model"][nm]["hit_protocol2_visible_only_iso_fpr"]
            ]
            mod_stats[nm] = {
                "iso_all_neg_fp_count": fp_iso,
                "iso_all_neg_denominator": n_neg_all,
                "iso_all_neg_fpr": float(fp_iso / max(n_neg_all, 1)),
                "iso_hits_unrestricted": int(sum(hit_vec_iso[nm])),
                "iso_hits_with_visible_post_reappear_alarm": int(sum(hit_vec_vis_at_iso[nm])),
                "protocol2_visible_neg_fp_count": fp_vis,
                "protocol2_visible_neg_denominator": n_neg_vis,
                "protocol2_visible_neg_fpr": float(fp_vis / max(n_neg_vis, 1)),
                "protocol2_visible_only_hits": int(sum(hit_vec_prot2[nm])),
                "protocol2_mean_lead_ms_given_hit": float(np.mean(vis_leads_p2)) if vis_leads_p2 else 0.0,
                "val_calibrated_threshold": float(thr_val_cal[nm][ci]),
                "val_calibrated_test_fp_count": fp_val,
                "val_calibrated_test_neg_denominator": n_neg_all,
                "val_calibrated_test_fpr": float(fp_val / max(n_neg_all, 1)),
                "val_calibrated_hits": int(sum(hit_vec_val[nm])),
            }

        reappear_summary[cn] = {
            "n_strict_recovered_events": len(eids_c),
            "n_videos": len({r["video_id"] for r in per_event_strict_log[cn]}),
            "n_events_with_any_occluded_decision_frame_in_pre_window": n_any_occ_pre,
            "n_events_with_at_least_one_visible_post_reappear_window": sum(
                1 for r in per_event_strict_log[cn] if r["n_visible_post_reappear_pre_onset_windows"] > 0
            ),
            "newly_hit_by_autopose_over_kin_count": len(new_auto_idx),
            "newly_hit_first_alarm_strictly_after_reappearance": new_first_after_reap,
            "newly_hit_has_visible_post_reappear_alarm": new_has_vis_alarm,
            "mcnemar_iso_unrestricted": {
                "autopose_vs_kin": exact_mcnemar_two_sided_p(hit_vec_iso["Lead_M0_kin_only"], hit_vec_iso["Lead_AutoPose_rtmpose17"]),
                "autopose_veh_vs_veh": exact_mcnemar_two_sided_p(hit_vec_iso["Lead_M2_kin_veh"], hit_vec_iso["Lead_AutoPose_veh"]),
                "human_vs_kin": exact_mcnemar_two_sided_p(hit_vec_iso["Lead_M0_kin_only"], hit_vec_iso["Lead_HumanAttr_orient_gaze"]),
            },
            "mcnemar_protocol2_visible_only": {
                "autopose_vs_kin": exact_mcnemar_two_sided_p(hit_vec_prot2["Lead_M0_kin_only"], hit_vec_prot2["Lead_AutoPose_rtmpose17"]),
                "autopose_veh_vs_veh": exact_mcnemar_two_sided_p(hit_vec_prot2["Lead_M2_kin_veh"], hit_vec_prot2["Lead_AutoPose_veh"]),
                "human_vs_kin": exact_mcnemar_two_sided_p(hit_vec_prot2["Lead_M0_kin_only"], hit_vec_prot2["Lead_HumanAttr_orient_gaze"]),
            },
            "models": mod_stats,
        }

    # Also compute full Protocol II (visible-only t0 >= f_reappear & occ=='none') paired lead-time comparisons
    prot2_eval_strict = (neg_te & vis_te) | (strict_te & vis_te)
    raw_p2 = {
        nm: evaluate_iso_test_fpr_and_unconflated_leadtime(
            p_te[nm][prot2_eval_strict], tau_te[prot2_eval_strict], eid_te[prot2_eval_strict],
            cls_te[prot2_eval_strict], (neg_te & vis_te)[prot2_eval_strict], target_test_fpr=0.05,
        )
        for nm in p_te
    }
    prot2_all_eval = (neg_te & vis_te) | ((~neg_te) & vis_te)
    raw_p2_all = {
        nm: evaluate_iso_test_fpr_and_unconflated_leadtime(
            p_te[nm][prot2_all_eval], tau_te[prot2_all_eval], eid_te[prot2_all_eval],
            cls_te[prot2_all_eval], (neg_te & vis_te)[prot2_all_eval], target_test_fpr=0.05,
        )
        for nm in p_te
    }
    prot2_pairs = {
        "autopose_vs_kin_only": paired_iso_fpr_comparison(raw_p2["Lead_AutoPose_rtmpose17"], raw_p2["Lead_M0_kin_only"]),
        "autopose_covis_vs_kin_only": paired_iso_fpr_comparison(raw_p2["Lead_AutoPose_CovisSafe"], raw_p2["Lead_M0_kin_only"]),
        "autopose_veh_vs_veh_only": paired_iso_fpr_comparison(raw_p2["Lead_AutoPose_veh"], raw_p2["Lead_M2_kin_veh"]),
        "human_attr_vs_kin_only": paired_iso_fpr_comparison(raw_p2["Lead_HumanAttr_orient_gaze"], raw_p2["Lead_M0_kin_only"]),
    }
    prot2_all_pairs = {
        "autopose_vs_kin_only": paired_iso_fpr_comparison(raw_p2_all["Lead_AutoPose_rtmpose17"], raw_p2_all["Lead_M0_kin_only"]),
        "autopose_covis_vs_kin_only": paired_iso_fpr_comparison(raw_p2_all["Lead_AutoPose_CovisSafe"], raw_p2_all["Lead_M0_kin_only"]),
        "autopose_veh_vs_veh_only": paired_iso_fpr_comparison(raw_p2_all["Lead_AutoPose_veh"], raw_p2_all["Lead_M2_kin_veh"]),
        "human_attr_vs_kin_only": paired_iso_fpr_comparison(raw_p2_all["Lead_HumanAttr_orient_gaze"], raw_p2_all["Lead_M0_kin_only"]),
    }
    return {
        "n_all_test_neg_windows": n_neg_all,
        "n_visible_unoccluded_test_neg_windows": n_neg_vis,
        "strict_recovered_summary_by_class": reappear_summary,
        "protocol2_visible_only_strict_paired": prot2_pairs,
        "protocol2_visible_only_all_events_paired": prot2_all_pairs,
        "per_event_strict_recovered_log": per_event_strict_log,
    }


class JAADCovisSafeTrajectoryNet(nn.Module):
    """5-frame JAAD trajectory net with co-visible delta-pose masking & val-calibrated low-disp shrinkage."""

    def __init__(self, use_auto_pose: bool, use_veh: bool, low_shrink: float = 1.0) -> None:
        super().__init__()
        self.use_auto_pose = use_auto_pose
        self.use_veh = use_veh
        self.low_shrink = float(low_shrink)
        self.mot_gru = nn.GRU(4, 48, batch_first=True)
        self.auto_gru = nn.GRU(12, 24, batch_first=True)
        self.dauto_mlp = nn.Sequential(nn.Linear(24, 24), nn.SiLU(), nn.Linear(24, 24))
        self.q_w_raw = nn.Parameter(torch.tensor([[0.8, 0.8]]))
        self.q_b = nn.Parameter(torch.tensor([-0.2]))
        self.veh_gru = nn.GRU(3, 16, batch_first=True)
        d_in = 48 + (24 if use_auto_pose else 0) + (16 if use_veh else 0)
        self.act_head = nn.Sequential(nn.Linear(d_in, 48), nn.SiLU(), nn.Linear(48, 4))
        self.traj_res = nn.Linear(d_in, 4 * 20 * 2)
        self.mode_lg = nn.Linear(d_in, 4)
        nn.init.normal_(self.traj_res.weight, std=0.01)
        nn.init.zeros_(self.traj_res.bias)

    def forward(self, hist: torch.Tensor, auto_p: torch.Tensor, auto_q: torch.Tensor, veh5: torch.Tensor) -> dict[str, torch.Tensor]:
        vel = torch.cat((torch.zeros_like(hist[:, :1]), (hist[:, 1:] - hist[:, :-1]) / 0.1), dim=1)
        _, hm = self.mot_gru(torch.cat((hist, vel), dim=-1).float())
        feats = [hm[0]]
        if self.use_auto_pose:
            c_t = auto_p[:, :, -1:].clamp(0.0, 1.0)
            _, ha = self.auto_gru(auto_p * (c_t >= 0.35).float())
            w04 = (c_t[:, 0] >= 0.45).float() * (c_t[:, -1] >= 0.45).float()
            w34 = (c_t[:, -2] >= 0.45).float() * (c_t[:, -1] >= 0.45).float()
            d_cov = torch.cat(((auto_p[:, -1] - auto_p[:, 0]) * w04, (auto_p[:, -1] - auto_p[:, -2]) * w34), dim=-1)
            gq = torch.sigmoid(F.linear(auto_q, F.softplus(self.q_w_raw), self.q_b)) * (c_t[:, -1] >= 0.40).float()
            feats.append(gq * (ha[0] + self.dauto_mlp(d_cov)))
        if self.use_veh:
            _, hv = self.veh_gru(veh5)
            feats.append(hv[0])
        ctx = torch.cat(feats, dim=-1)
        disp = torch.linalg.vector_norm(hist[:, 1:] - hist[:, :-1], dim=-1).sum(dim=-1)
        low_env = torch.exp(-((disp / 0.18) ** 2))
        sh = (1.0 - (1.0 - self.low_shrink) * low_env)[:, None, None]
        t_s = torch.arange(1, 21, dtype=hist.dtype, device=hist.device)[None, :, None] * 0.1
        v_last = vel[:, -1:]
        cv_s = sh * v_last * t_s
        mob = torch.tanh((disp / 0.15) ** 2)[:, None, None]
        stop_anc = v_last * (0.45 * (1.0 - torch.exp(-t_s / 0.45)))
        res = self.traj_res(ctx).view(-1, 4, 20, 2).double()
        p4 = torch.stack((cv_s, cv_s + res[:, 1], cv_s + mob * res[:, 2], stop_anc + 0.3 * mob * res[:, 3]), dim=1)
        return {"paths": p4, "logits": self.mode_lg(ctx), "act_logits": self.act_head(ctx)}


def eval_jaad_covis_lowdisp_5seed(jd: dict, seeds: list[int], device: torch.device) -> dict:
    sp = jd["split"]
    tr, va, te = np.flatnonzero(sp == "train"), np.flatnonzero(sp == "val"), np.flatnonzero(sp == "test")
    # Calibrate low_shrink strictly on val < 0.25 bbox-H
    h_va, y_va = jd["hist"][va], jd["truth"][va]
    disp_va = np.linalg.norm(np.diff(h_va, axis=1), axis=-1).sum(axis=1)
    low_m = disp_va < 0.25
    t_s = np.arange(1, 21, dtype=np.float64)[None, :, None] * 0.1
    v_va = (h_va[:, -1:] - h_va[:, -2:-1]) / 0.1
    best_sh, best_ve = 1.0, 1e9
    if int(low_m.sum()) >= 5:
        for a in (0.55, 0.70, 0.82, 0.92, 1.00):
            sh = (1.0 - (1.0 - a) * np.exp(-((disp_va[low_m] / 0.18) ** 2)))[:, None, None]
            e = float(np.linalg.norm(sh * v_va[low_m] * t_s - y_va[low_m], axis=-1).mean())
            if e < best_ve:
                best_ve, best_sh = e, float(a)

    ht, tt = torch.from_numpy(jd["hist"]).to(device), torch.from_numpy(jd["truth"]).to(device)
    apt, aqt = torch.from_numpy(jd["auto_pose5"]).to(device), torch.from_numpy(jd["auto_q"]).to(device)
    vt = torch.from_numpy(jd["veh5"]).to(device)
    yat, vat = torch.from_numpy(jd["y_act"]).to(device), torch.from_numpy(jd["v_act"]).to(device)
    pw_c = ((vat[tr].float().sum(0) - yat[tr].float().sum(0)) / yat[tr].float().sum(0).clamp_min(1.0)).clamp(1.0, 12.0)
    tr_t, va_t, te_t = torch.from_numpy(tr).to(device), torch.from_numpy(va).to(device), torch.from_numpy(te).to(device)

    arms = (
        ("J_M0_CovisSafe", False, False),
        ("J_AutoPose_CovisSafe", True, False),
        ("J_M2_Veh_CovisSafe", False, True),
        ("J_AutoPose_Veh_CovisSafe", True, True),
    )
    cv_te = ((jd["hist"][te, -1:] - jd["hist"][te, -2:-1]) / 0.1) * t_s
    err_cv = _jaad_post_onset_err(cv_te, jd["truth"][te], jd["onset"][te])
    err_s = {nm: np.zeros((len(seeds), len(te), 4), dtype=np.float64) for nm, *_ in arms}

    for si, seed in enumerate(seeds):
        for nm, ua, uv in arms:
            torch.manual_seed(seed + 17)
            net = JAADCovisSafeTrajectoryNet(ua, uv, low_shrink=best_sh).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=2e-4)
            gen = torch.Generator().manual_seed(seed + 101)
            best_st, best_vl = None, 1e9
            for _ in range(28):
                net.train()
                for b in torch.randperm(len(tr_t), generator=gen).split(64):
                    ib = tr_t[b.to(device)]
                    opt.zero_grad(set_to_none=True)
                    o = net(ht[ib], apt[ib], aqt[ib], vt[ib])
                    d_k = torch.linalg.vector_norm(o["paths"] - tt[ib][:, None], dim=-1).mean(dim=-1)
                    s_w = (-d_k / 0.15).softmax(dim=1).float()
                    l_tr = d_k.min(dim=1).values.mean() + 0.25 * (-(s_w * o["logits"].log_softmax(dim=1)).sum(dim=1)).mean().double()
                    bce = F.binary_cross_entropy_with_logits(o["act_logits"], yat[ib].float(), pos_weight=pw_c, reduction="none")
                    l_ac = ((bce * vat[ib].float()).sum(1) / vat[ib].float().sum(1).clamp_min(1.0)).mean().double()
                    (l_tr + 0.30 * l_ac).backward()
                    opt.step()
                net.eval()
                with torch.inference_mode():
                    ov = net(ht[va_t], apt[va_t], aqt[va_t], vt[va_t])
                    ix_v = ov["logits"].argmax(dim=1)
                    vl = float(torch.linalg.vector_norm(ov["paths"][torch.arange(len(va_t), device=device), ix_v] - tt[va_t], dim=-1).mean())
                    if vl < best_vl:
                        best_vl = vl
                        best_st = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if best_st is not None:
                net.load_state_dict(best_st)
            net.eval()
            with torch.inference_mode():
                ot = net(ht[te_t], apt[te_t], aqt[te_t], vt[te_t])
                pw = ot["paths"].cpu().numpy()
                ix = ot["logits"].argmax(dim=1).cpu().numpy()
                err_s[nm][si] = _jaad_post_onset_err(pw[np.arange(len(te)), ix], jd["truth"][te], jd["onset"][te])
    em = {nm: err_s[nm].mean(axis=0) for nm, *_ in arms}
    v_te = jd["vid"][te]
    strata = {
        "all_locked_test_953": np.ones(len(te), dtype=bool),
        "strict_5frame_deoccluded_168": jd["strict_5f_deocc"][te],
        "strict_5f_deocc_lowdisp_lt0p25_bbox_h_84": jd["strict_deocc_lowdisp_lt0p25_bh"][te],
    }
    out = {"val_calibrated_low_disp_shrink": best_sh, "test_strata": {}}
    for sk, sm in strata.items():
        md = {"Last2_CV": {k: float(v) for k, v in zip(("ade_2s", "fde_2s", "post_0p5s", "post_1p0s"), err_cv[sm].mean(0))}}
        for nm, *_ in arms:
            md[nm] = {k: float(v) for k, v in zip(("ade_2s", "fde_2s", "post_0p5s", "post_1p0s"), em[nm][sm].mean(0))}
        out["test_strata"][sk] = {
            "n_windows": int(sm.sum()),
            "n_videos": len(set(v_te[sm].tolist())),
            "models": md,
            "paired_video_ci": {
                "autopose_covis_minus_CV_2s": _loc_bootstrap_ci(em["J_AutoPose_CovisSafe"][sm, 0:2] - err_cv[sm, 0:2], v_te[sm]),
                "autopose_covis_minus_m0_covis_2s": _loc_bootstrap_ci(em["J_AutoPose_CovisSafe"][sm, 0:2] - em["J_M0_CovisSafe"][sm, 0:2], v_te[sm]),
                "autopose_veh_covis_minus_CV_2s": _loc_bootstrap_ci(em["J_AutoPose_Veh_CovisSafe"][sm, 0:2] - err_cv[sm, 0:2], v_te[sm]),
            },
        }
    return out


def _build_covis_masked_slide_autopose(ap5: np.ndarray, aq2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Suppress occluded early-frame spikes in 5-frame sliding AutoPose so frame-4-vs-0 delta reflects visible pose."""
    out_p = ap5.copy()
    c = ap5[:, :, -1]
    # Replace low-confidence early frames (<0.42) with last visible frame so (ap[-1] - ap[0]) inside LeadNet has 0 truncation jump
    for i in range(len(out_p)):
        vis_idx = np.flatnonzero(c[i] >= 0.42)
        if len(vis_idx) == 0:
            out_p[i] = 0.0
        else:
            first_good = int(vis_idx[0])
            for t in range(5):
                if c[i, t] < 0.42:
                    out_p[i, t] = out_p[i, first_good]
    q_cov = aq2.copy()
    q_cov[:, 0] = q_cov[:, 0] * (c[:, -1] >= 0.42).astype(np.float32)
    return out_p, q_cov


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch-dir", type=Path, required=True)
    ap.add_argument("--zip-path", type=Path, default=Path("wave25_jaad_reappearance_per_event_audit.zip"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    traj_ds, slide_ds = build_jaad2_strict_and_sliding_cohorts(args.scratch_dir)
    jaad_blob = (args.scratch_dir / "jaad2.zip").read_bytes()
    meta = extract_jaad_slide_reappearance_metadata(jaad_blob)

    ap_cov, aq_cov = _build_covis_masked_slide_autopose(slide_ds["auto_pose"], slide_ds["auto_q"])
    j_arms = [
        ("Lead_M0_kin_only", slide_ds["auto_pose"], slide_ds["auto_q"], False, False),
        ("Lead_AutoPose_rtmpose17", slide_ds["auto_pose"], slide_ds["auto_q"], True, False),
        ("Lead_AutoPose_CovisSafe", ap_cov, aq_cov, True, False),
        ("Lead_AutoPose_veh", slide_ds["auto_pose"], slide_ds["auto_q"], True, True),
        ("Lead_HumanAttr_orient_gaze", slide_ds["human_attr"], slide_ds["human_q"], True, False),
        ("Lead_M2_kin_veh", slide_ds["auto_pose"], slide_ds["auto_q"], False, True),
        ("Lead_Full_human_veh", slide_ds["human_attr"], slide_ds["human_q"], True, True),
    ]
    p_va_j, p_te_j = _fit_and_predict_lead_arms(
        slide_ds["kin"], j_arms, slide_ds["veh"], slide_ds["y"], slide_ds["tau_s"],
        slide_ds["is_neg"], slide_ds["split"], list(SEEDS), device,
    )
    event_audit = audit_per_event_timestamps_and_visible_alarms(p_va_j, p_te_j, slide_ds, meta)
    covis_traj = eval_jaad_covis_lowdisp_5seed(traj_ds, list(SEEDS), device)

    summary = {
        "wave": "wave25_jaad_reappearance_per_event_audit_5seed",
        "jaad2_per_event_reappearance_and_visible_alarm_audit": event_audit,
        "jaad2_covis_safe_lowdisp_trajectory_5seed": covis_traj,
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
