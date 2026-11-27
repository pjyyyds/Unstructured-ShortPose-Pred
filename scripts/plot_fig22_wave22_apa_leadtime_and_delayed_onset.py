"""Render 300-DPI Figure 22: Causal APA Millisecond Pre-Onset Lead-Time & Delayed-Onset Start/Stop Selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_fig22(results_dir: Path, out_png: Path) -> Path:
    w_jtp = json.loads((results_dir / "wave22_apa_leadtime_jaad_jta_and_pinns_delayed_5seed_verified.json").read_text())
    w_jr = json.loads((results_dir / "wave22_jrdb_delayed_apa_upgrade_5seed_verified.json").read_text())

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.titlesize": 10.1,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.2,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), dpi=300)
    fig.patch.set_facecolor("white")

    # Panel (a): JAAD 2.0 Human-Annotated Maneuver Lead-Time Gain (Pose - KinOnly, ms at Matched 5% FPR)
    ax = axes[0, 0]
    j_sub = w_jtp["jaad2_sliding_leadtime"]["matched_fpr_0p05"]
    j_key = "locked_test_post_deocclusion" if "locked_test_post_deocclusion" in j_sub else "locked_test_all_events"
    j_mod = j_sub[j_key]["models"]
    cls3 = ["Start", "Stop", "Turn"]
    x3 = np.arange(3)
    m0_l = [1000.0 * j_mod["Lead_M0_kin_only"][c]["mean_lead_s_censored_all"] for c in cls3]
    m1_l = [1000.0 * j_mod["Lead_M1_kin_pose"][c]["mean_lead_s_censored_all"] for c in cls3]
    mf_l = [1000.0 * j_mod["Lead_Full_kin_pose_nbr"][c]["mean_lead_s_censored_all"] for c in cls3]
    ax.axhline(0.0, color="#2C3E50", linewidth=1.0, linestyle=":")
    ax.bar(x3 - 0.25, m0_l, width=0.23, color="#AEB6BF", label="Position/BBox Only")
    ax.bar(x3, m1_l, width=0.23, color="#2E86C1", label="+ 5-Frame Pose & Gaze")
    ax.bar(x3 + 0.25, mf_l, width=0.23, color="#1B4F72", label="Full Pose + Vehicle")
    ax.set_xticks(x3, ["Start\n(Stand->Walk)", "Stop\n(Walk->Stand)", "Cross\n(->Crossing)"])
    ax.set_ylabel("Earliest Detection Lead Time (ms before onset)")
    ax.set_title("(a) JAAD 2.0 Pre-Onset Lead Time at Matched 5% Test FPR")
    ax.legend(frameon=False, fontsize=8.0, loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    # Panel (b): JTA-228 Urban Sequence Pre-Onset Recall (%) at Matched 5% Test FPR
    ax = axes[0, 1]
    t_sub = w_jtp["jta228_sliding_leadtime"]["matched_fpr_0p05"]["locked_test_all_events"]["models"]
    t_m0 = [100.0 * t_sub["Lead_M0_kin_only"][c]["pre_onset_recall"] for c in cls3]
    t_m1 = [100.0 * t_sub["Lead_M1_kin_pose"][c]["pre_onset_recall"] for c in cls3]
    t_mf = [100.0 * t_sub["Lead_Full_kin_pose_nbr"][c]["pre_onset_recall"] for c in cls3]
    ax.bar(x3 - 0.25, t_m0, width=0.23, color="#AEB6BF", label="Kinematics Only")
    ax.bar(x3, t_m1, width=0.23, color="#28B463", label="+ 3D Skeleton Pose")
    ax.bar(x3 + 0.25, t_mf, width=0.23, color="#1D8348", label="Full Pose + Crowd")
    ax.set_xticks(x3, ["Start", "Stop", "Turn"])
    ax.set_ylabel("Strict Pre-Onset Recall (tau <= 0, %)")
    ax.set_ylim(0, max(t_m0 + t_m1 + t_mf + [25.0]) * 1.25)
    ax.set_title("(b) JTA-228 Pre-Onset Detection Recall at Matched 5% FPR")
    ax.legend(frameon=False, fontsize=8.0, loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    # Panel (c): PINNS & JRDB Pre-Onset Recall at Matched 5% Source FPR (NoPose vs WithPose)
    ax = axes[1, 0]
    p_rec = w_jtp["pinns_delayed_apa_upgrade"]["matched_5pct_fpr_pre_onset_recall"]
    r_rec = w_jr["jrdb_9location_delayed_apa"]["matched_5pct_fpr_pre_onset_recall"]
    cats6 = ["PINNS\nStart", "PINNS\nStop", "PINNS\nTurn", "JRDB\nStart", "JRDB\nStop", "JRDB\nTurn"]
    x6 = np.arange(6)
    v_np = [100.0 * p_rec[c]["NoPose"]["pre_onset_recall_all"] for c in cls3] + [100.0 * r_rec[c]["NoPose"]["pre_onset_recall_all"] for c in cls3]
    v_wp = [100.0 * p_rec[c]["WithPose"]["pre_onset_recall_all"] for c in cls3] + [100.0 * r_rec[c]["WithPose"]["pre_onset_recall_all"] for c in cls3]
    ax.bar(x6 - 0.18, v_np, width=0.34, color="#BDC3C7", label="Position + Interaction Only")
    ax.bar(x6 + 0.18, v_wp, width=0.34, color="#6C3483", label="+ 5-Frame Ground Pose (APA)")
    ax.set_xticks(x6, cats6)
    ax.set_ylabel("Pre-Onset Detection Recall at 5% FPR (%)")
    ax.set_ylim(0, max(v_np + v_wp + [25.0]) * 1.25)
    ax.set_title("(c) PINNS & JRDB Pre-Onset Recall at Matched 5% FPR")
    ax.legend(frameon=False, fontsize=8.0, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    # Panel (d): Delayed-Onset APA Start/Stop Candidate Upgrade (Oracle-6 Beats CV & Top-1 Hits)
    ax = axes[1, 1]
    cd_p = w_jtp["pinns_delayed_apa_upgrade"]["candidate_diag_seed0"]
    cd_r = w_jr["jrdb_9location_delayed_apa"]["candidate_diag_seed0"]
    lbls4 = ["PINNS Start\nOracle>CV", "PINNS Start\nTop-1 Hit", "JRDB Start\nOracle>CV", "JRDB Start\nTop-1 Hit"]
    x4 = np.arange(4)
    n_ps = max(cd_p["S6_Immediate_Base"]["Start"]["n_true"], 1)
    n_rs = max(cd_r["S6_Immediate_Base"]["Start"]["n_true"], 1)
    b_imm = [
        100.0 * cd_p["S6_Immediate_Base"]["Start"]["oracle6_beats_cv_by_2cm"] / n_ps,
        100.0 * cd_p["S6_Immediate_Base"]["Start"]["top1_selected_hits"] / n_ps,
        100.0 * cd_r["S6_Immediate_Base"]["Start"]["oracle6_beats_cv_by_2cm"] / n_rs,
        100.0 * cd_r["S6_Immediate_Base"]["Start"]["top1_selected_hits"] / n_rs,
    ]
    b_apa = [
        100.0 * cd_p["S6_APA_Delayed_NP5pctFPR"]["Start"]["oracle6_beats_cv_by_2cm"] / n_ps,
        100.0 * cd_p["S6_APA_Delayed_NP5pctFPR"]["Start"]["top1_selected_hits"] / n_ps,
        100.0 * cd_r["S6_APA_Delayed_NP5pctFPR"]["Start"]["oracle6_beats_cv_by_2cm"] / n_rs,
        100.0 * cd_r["S6_APA_Delayed_NP5pctFPR"]["Start"]["top1_selected_hits"] / n_rs,
    ]
    ax.bar(x4 - 0.18, b_imm, width=0.34, color="#95A5A6", label="Immediate t=0 Linear Start")
    ax.bar(x4 + 0.18, b_apa, width=0.34, color="#D35400", label="Delayed-Onset APA + 5%-FPR Selector")
    ax.set_xticks(x4, lbls4, fontsize=8.3)
    ax.set_ylabel("Rate Among True Start Events (%)")
    ax.set_ylim(0, 112)
    ax.set_title("(d) Fixing 'Generated But Not Selected' via Delayed-Onset APA")
    ax.legend(frameon=False, fontsize=8.0, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    fig.suptitle("Wave 22: Biomechanical APA Pre-Onset Lead Time & Delayed-Onset Start/Stop Selection", fontsize=11.8, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.962))
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
