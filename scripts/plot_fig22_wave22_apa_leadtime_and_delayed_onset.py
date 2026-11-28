"""Render 300-DPI Figure 22: Iso-Test-FPR Unconflated Recall & Common-Hit Alarm Lead Time (Auto RTMPose-17 vs Human Attr)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_fig22(results_dir: Path, out_png: Path) -> Path:
    j23 = json.loads((results_dir / "wave23_jaad_autopose_and_isofpr_leadtime_5seed_verified.json").read_text())
    w24_p = results_dir / "wave24_pinns_raw_video_annotated_emergence_5seed_verified.json"
    w23_p = results_dir / "wave23_pinns_video_audit_and_6slot_ranker_5seed_verified.json"
    p_w23 = json.loads((w24_p if w24_p.is_file() else w23_p).read_text())
    p_jr = json.loads(w23_p.read_text())["jrdb_9loc_lolo"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.9,
        "axes.titlesize": 9.7,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.8,
    })
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), dpi=300)
    fig.patch.set_facecolor("white")

    # Panel (a): JAAD 2.0 Strict De-Occlusion Pre-Onset Recall (%) at Exact Iso-Test-FPR = 5.10% (No Miss Penalty)
    ax = axes[0, 0]
    sub_s = j23["jaad2_sliding_isofpr_leadtime_5seed"]["iso_test_fpr"]["fpr_0p050"]["locked_test_post_deocclusion_strict_recovered"]["models"]
    cls3 = ["Start", "Stop", "Turn"]
    x3 = np.arange(3)
    m0_r = [100.0 * sub_s["Lead_M0_kin_only"][c]["pre_onset_recall"] for c in cls3]
    ap_r = [100.0 * sub_s["Lead_AutoPose_rtmpose17"][c]["pre_onset_recall"] for c in cls3]
    hm_r = [100.0 * sub_s["Lead_HumanAttr_orient_gaze"][c]["pre_onset_recall"] for c in cls3]
    fl_r = [100.0 * sub_s["Lead_Full_human_veh"][c]["pre_onset_recall"] for c in cls3]
    b_m0 = ax.bar(x3 - 0.27, m0_r, width=0.17, color="#AEB6BF", label="BBox Motion Only (5.1% FPR)")
    b_ap = ax.bar(x3 - 0.09, ap_r, width=0.17, color="#1F618D", label="Auto RTMPose-17 Video Pose (5.1% FPR)")
    b_hm = ax.bar(x3 + 0.09, hm_r, width=0.17, color="#2E86C1", label="Human Orient+Gaze Attr (5.1% FPR)")
    b_fl = ax.bar(x3 + 0.27, fl_r, width=0.17, color="#117A65", label="Full Human Attr + Veh (5.1% FPR)")
    for bars, mk in ((b_m0, "Lead_M0_kin_only"), (b_ap, "Lead_AutoPose_rtmpose17"), (b_hm, "Lead_HumanAttr_orient_gaze"), (b_fl, "Lead_Full_human_veh")):
        for b, c in zip(bars, cls3):
            ht = sub_s[mk][c]["pre_onset_hit_count"]
            nt = sub_s[mk][c]["n_events"]
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{ht}/{nt}", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(x3, ["Start (N=12)\nStand->Walk", "Stop (N=10)\nWalk->Stand", "Cross (N=29)\n->Crossing"])
    ax.set_ylabel("Strict Pre-Onset Recall (tau <= 0, %)")
    ax.set_ylim(0, 74)
    ax.set_title("(a) JAAD Strict De-Occl Recall at Identical 5.10% Test FPR")
    ax.legend(frameon=False, fontsize=7.4, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    # Panel (b): Unconflated Common-Hit Alarm Timing Delta (ms earlier on BOTH-HIT events at 5.10% Test FPR)
    ax = axes[0, 1]
    all_p = j23["jaad2_sliding_isofpr_leadtime_5seed"]["iso_test_fpr"]["fpr_0p050"]["locked_test_all_events"]["apple_to_apple_pairs"]
    pairs_plot = [
        ("AutoPose vs.\nBBox Only", "autopose_vs_kin_only"),
        ("AutoPose+Veh\nvs. Veh Only", "autopose_veh_vs_veh_only"),
        ("HumanAttr vs.\nBBox Only", "human_attr_vs_kin_only"),
        ("HumanAttr+Veh\nvs. Veh Only", "human_veh_vs_veh_only"),
    ]
    xp = np.arange(len(pairs_plot))
    st_ms = [all_p[pk]["Start"]["common_hit_lead_diff_ms_mean"] for _, pk in pairs_plot]
    cr_ms = [all_p[pk]["Turn"]["common_hit_lead_diff_ms_mean"] for _, pk in pairs_plot]
    ax.axhline(0.0, color="#2C3E50", linewidth=1.0, linestyle=":")
    bs1 = ax.bar(xp - 0.18, st_ms, width=0.34, color="#2874A6", alpha=0.90, label="Start Common-Hit Lead Gain (ms)")
    bs2 = ax.bar(xp + 0.18, cr_ms, width=0.34, color="#D35400", alpha=0.88, label="Cross Common-Hit Lead Gain (ms)")
    for b, (_, pk) in zip(bs1, pairs_plot):
        nb = all_p[pk]["Start"]["n_both_pre_hit"]
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 12, f"+{b.get_height():.0f}ms\n(n={nb})", ha="center", va="bottom", fontsize=7.0)
    for b, (_, pk) in zip(bs2, pairs_plot):
        nb = all_p[pk]["Turn"]["n_both_pre_hit"]
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 12, f"+{b.get_height():.0f}ms\n(n={nb})", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(xp, [lbl for lbl, _ in pairs_plot], fontsize=7.8)
    ax.set_ylabel("Earlier Alarm on Common Hits (ms, >0 Earlier)")
    ax.set_ylim(0, max(st_ms + cr_ms) + 145)
    ax.set_title("(b) Hit-Only Alarm Timing Gain at Exact 5.10% Test FPR (No Miss Penalty)")
    ax.legend(frameon=False, fontsize=7.6, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    # Panel (c): JAAD 2.0 Strict 5-Frame De-Occluded (N=168) Maneuver AP — Auto RTMPose-17 vs Human Attr
    ax = axes[1, 0]
    ap_168 = j23["jaad2_trajectory_autopose_vs_human_5seed"]["maneuver_ap_by_stratum"]["strict_5frame_deoccluded_168"]
    c_ap = ["Start", "Stop", "Cross"]
    xa = np.arange(3)
    v_b0 = [100.0 * ap_168["J_M0_bbox_motion_only"][c]["ap"] for c in c_ap]
    v_bv = [100.0 * ap_168["J_M2_bbox_veh"][c]["ap"] for c in c_ap]
    v_av = [100.0 * ap_168["J_AutoPose_veh"][c]["ap"] for c in c_ap]
    v_hv = [100.0 * ap_168["J_Full_human_veh"][c]["ap"] for c in c_ap]
    ax.bar(xa - 0.27, v_b0, width=0.17, color="#BDC3C7", label="BBox Motion Only")
    ax.bar(xa - 0.09, v_bv, width=0.17, color="#7F8C8D", label="BBox + Vehicle")
    ax.bar(xa + 0.09, v_av, width=0.17, color="#1F618D", label="Auto RTMPose-17 + Vehicle")
    ax.bar(xa + 0.27, v_hv, width=0.17, color="#117A65", label="Human Orient+Gaze + Vehicle")
    ax.set_xticks(xa, [f"Start (pos={ap_168['J_M0_bbox_motion_only']['Start']['pos']})", f"Stop (pos={ap_168['J_M0_bbox_motion_only']['Stop']['pos']})", f"Cross (pos={ap_168['J_M0_bbox_motion_only']['Cross']['pos']})"])
    ax.set_ylabel("Average Precision AP (%)")
    ax.set_ylim(0, 106)
    ax.set_title("(c) JAAD Strict 5-Frame De-Occluded (N=168) Action AP")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    # Panel (d): 5-Seed PINNS & JRDB Candidate Generation (100% Start/Stop) & CondMedoid ADE1 Improvement
    ax = axes[1, 1]
    p_all = p_w23["pinns_4site_loso"]["strata"]["all_1482"]["models"]
    p_cd = p_w23["pinns_4site_loso"]["candidate_generation_vs_selection_5seed"]
    r_cd = p_jr["candidate_generation_vs_selection_5seed"]
    lbls4 = ["PINNS Start\n5-Seed Gen", "PINNS Stop\n5-Seed Gen", "JRDB Start\n5-Seed Gen", "JRDB Stop\n5-Seed Gen"]
    x4 = np.arange(4)
    g_m0 = [
        100.0 * p_cd["S6_M0_nopose_nonbr"]["Start"]["any6_generated_hits_5seed_mean"] / 9.0,
        100.0 * p_cd["S6_M0_nopose_nonbr"]["Stop"]["any6_generated_hits_5seed_mean"] / 18.0,
        100.0 * r_cd["S6_M0_nopose_nonbr"]["Start"]["any6_generated_hits_5seed_mean"] / 11.0,
        100.0 * r_cd["S6_M0_nopose_nonbr"]["Stop"]["any6_generated_hits_5seed_mean"] / 2.0,
    ]
    g_cm = [
        100.0 * p_cd["S6_Full_CondMedoid"]["Start"]["any6_generated_hits_5seed_mean"] / 9.0,
        100.0 * p_cd["S6_Full_CondMedoid"]["Stop"]["any6_generated_hits_5seed_mean"] / 18.0,
        100.0 * r_cd["S6_Full_CondMedoid"]["Start"]["any6_generated_hits_5seed_mean"] / 11.0,
        100.0 * r_cd["S6_Full_CondMedoid"]["Stop"]["any6_generated_hits_5seed_mean"] / 2.0,
    ]
    ax.bar(x4 - 0.18, g_m0, width=0.34, color="#95A5A6", label="No-Pose S6_M0 Any-6 Gen (5-Seed Mean)")
    ax.bar(x4 + 0.18, g_cm, width=0.34, color="#6C3483", label="CondMedoid + Delayed APA Any-6 Gen (5-Seed)")
    ax.text(0.03, 0.87, f"PINNS All-1,482 Top-1 ADE1/FDE1:\nCV: {p_all['Last2_CV']['ade1_full_m']:.4f}/{p_all['Last2_CV']['fde1_full_m']:.4f} m -> CondMedoid: {p_all['S6_Full_CondMedoid']['ade1_full_m']:.4f}/{p_all['S6_Full_CondMedoid']['fde1_full_m']:.4f} m",
            transform=ax.transAxes, fontsize=7.6, bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F4ECF7", "edgecolor": "#8E44AD"})
    ax.set_xticks(x4, lbls4, fontsize=8.0)
    ax.set_ylabel("5-Seed Mean Candidate Generation (%)")
    ax.set_ylim(0, 128)
    ax.set_title("(d) 5-Seed 100% Start/Stop Generation & CondMedoid Best (0.4382m)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    fig.suptitle("Waves 23-24: Automatic RTMPose-17 Video Pose, Iso-Test-FPR Lead Time, & 5-Seed CondMedoid", fontsize=11.4, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.960))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-png", type=Path, default=Path("docs/figures/fig22_wave22_apa_leadtime_and_delayed_onset.png"))
    args = ap.parse_args()
    render_fig22(args.results_dir, args.out_png)
    print("Rendered:", args.out_png)


if __name__ == "__main__":
    main()
