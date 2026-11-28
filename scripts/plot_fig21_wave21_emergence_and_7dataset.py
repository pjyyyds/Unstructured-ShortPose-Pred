"""Render 300-DPI Figure 21: Zero-Default Explicit-241 Census, Core <0.5m Emergence Per-Site/Pooled, Benchmarks, & Single-Factor Ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_fig21(results_dir: Path, out_png: Path) -> Path:
    w25 = json.loads((results_dir / "wave25_pinns_explicit_audit_and_emergence_safe_5seed_verified.json").read_text())
    w23_p = json.loads((results_dir / "wave23_pinns_video_audit_and_6slot_ranker_5seed_verified.json").read_text())
    p_jr = w23_p["jrdb_9loc_lolo"]
    p_j23 = json.loads((results_dir / "wave23_jaad_autopose_and_isofpr_leadtime_5seed_verified.json").read_text())
    p_jj = json.loads((results_dir / "wave21_jrdb_and_jta_emergence_maneuvers_5seed_verified.json").read_text())
    p_j6 = json.loads((results_dir / "wave21_jaad_deocclusion_and_eth_ucy_6dataset_5seed_verified.json").read_text())

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.7,
    })
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.2), dpi=300)
    fig.patch.set_facecolor("white")

    # Panel (a): Zero-Default Explicit-241 Raw-Video Visual Audit vs Homography Rule (N=449)
    ax = axes[0, 0]
    c25 = w25["explicit_241_audit_census"]["class_counts_241"]
    cats = ["Vehicle\nOccluder", "Static Pillar/\nPole/Portal", "Foreground\nPerson", "Border\nEntry", "Open-View\nLate Start", "Frame-0\nClip Start"]
    w24_v = [32, 63, 45, 27, 74, 208]
    exp_v = [c25["veh"], c25["static"], c25["pers"], c25["border"], c25["open"], 208]
    x = np.arange(len(cats))
    b1 = ax.bar(x - 0.18, w24_v, width=0.34, color="#A9CCE3", alpha=0.90, label="Wave 24 Audit (45 Default Pers, 140 Pos)")
    b2 = ax.bar(x + 0.18, exp_v, width=0.34, color="#5B2C6F", alpha=0.92, label="Wave 25 Explicit-241 (0 Default, 118 Pos)")
    for bars in (b1, b2):
        for b in bars:
            h = int(round(b.get_height()))
            ax.text(b.get_x() + b.get_width() / 2, h + 3, str(h), ha="center", va="bottom", fontsize=7.5)
    ax.annotate("Old 34 BEV-car rule:\n1/34 homog, 3/34 visual", xy=(0.18, exp_v[0]), xytext=(0.45, 165),
                arrowprops={"arrowstyle": "->", "color": "#C0392B", "lw": 1.1}, fontsize=7.4, color="#922B21", fontweight="bold")
    ax.set_xticks(x, cats, fontsize=7.8)
    ax.set_ylabel("Window Count (First-Contiguous N=449)")
    ax.set_title("(a) PINNS Explicit-241 Raw-Video Audit (118 Confirmed: US 93, JP 23, IT 1, TH 1)")
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    ax.set_ylim(0, 248)

    # Panel (b): Core < 0.5 m Physical Emergence (Explicit N=74 & Wave-24 N=86) — Pooled vs Site-Equal & Per-Site
    ax = axes[0, 1]
    c74 = w25["pinns_4site_loso"]["strata"]["core_explicit_short_phys_lt0p5m"]
    ps = c74["per_site_breakdown"]
    grp_lbls = [
        "Pooled\n(N=74)",
        "4-Site Eq\n(IT=1,TH=1)",
        "America\n(N=49)",
        "Japan\n(N=23)",
        "Italy/Thai\n(N=1+1)",
    ]
    xg = np.arange(len(grp_lbls))
    it_th = lambda m: 0.5 * (ps["italy_park"]["models"][m]["ade1_full_m"] + ps["thailand_street"]["models"][m]["ade1_full_m"])
    v_cv = [c74["models_pooled_window_weighted"]["Last2_CV"]["ade1_full_m"], c74["models_site_equal"]["Last2_CV"]["ade1_full_m"], ps["america_crossroad"]["models"]["Last2_CV"]["ade1_full_m"], ps["japan_crossroad"]["models"]["Last2_CV"]["ade1_full_m"], it_th("Last2_CV")]
    v_s1 = [c74["models_pooled_window_weighted"]["S6_M1_pose_nonbr"]["ade1_full_m"], c74["models_site_equal"]["S6_M1_pose_nonbr"]["ade1_full_m"], ps["america_crossroad"]["models"]["S6_M1_pose_nonbr"]["ade1_full_m"], ps["japan_crossroad"]["models"]["S6_M1_pose_nonbr"]["ade1_full_m"], it_th("S6_M1_pose_nonbr")]
    v_cm = [c74["models_pooled_window_weighted"]["S6_Full_CondMedoid"]["ade1_full_m"], c74["models_site_equal"]["S6_Full_CondMedoid"]["ade1_full_m"], ps["america_crossroad"]["models"]["S6_Full_CondMedoid"]["ade1_full_m"], ps["japan_crossroad"]["models"]["S6_Full_CondMedoid"]["ade1_full_m"], it_th("S6_Full_CondMedoid")]
    v_es = [c74["models_pooled_window_weighted"]["ES6_Full_vispose_nbr"]["ade1_full_m"], c74["models_site_equal"]["ES6_Full_vispose_nbr"]["ade1_full_m"], ps["america_crossroad"]["models"]["ES6_Full_vispose_nbr"]["ade1_full_m"], ps["japan_crossroad"]["models"]["ES6_Full_vispose_nbr"]["ade1_full_m"], it_th("ES6_Full_vispose_nbr")]

    ax.bar(xg - 0.27, v_cv, width=0.18, color="#2C3E50", label="Last-2 CV (Best Pooled 0.5249m)")
    ax.bar(xg - 0.09, v_s1, width=0.18, color="#E67E22", label="S6_M1 (+Pose, IT/TH Drift)")
    ax.bar(xg + 0.09, v_cm, width=0.18, color="#2E86C1", label="S6_Full_CondMedoid")
    ax.bar(xg + 0.27, v_es, width=0.18, color="#117A65", label="ES6_Full (Stationary-Safe)")
    ax.set_xticks(xg, grp_lbls, fontsize=7.7)
    ax.set_ylabel("3.0s Top-1 ADE_1 (m)")
    ax.set_ylim(0, 0.88)
    ax.set_title("(b) Core <0.5m Emergence (N=74, 1 Start): CV Still Leads Pooled")
    ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    # Panel (c): Partitioned Pose-Bearing vs Zero-Pose Control Benchmarks (% ADE Change vs Last-2 CV)
    ax = axes[1, 0]
    j_t = p_j23["jaad2_trajectory_autopose_vs_human_5seed"]["test_strata"]
    a1482 = w25["pinns_4site_loso"]["strata"]["all_1482"]["models_site_equal"]
    m61 = w25["pinns_4site_loso"]["strata"]["explicit_phys_emergence_active_maneuvers"]["models_site_equal"]
    ds7 = [
        ("PINNS\n(2D+Nbr)", a1482["S6_Full_CondMedoid"]["ade1_full_m"], a1482["Last2_CV"]["ade1_full_m"], m61["ES6_Full_vispose_nbr"]["ade1_full_m"], m61["Last2_CV"]["ade1_full_m"]),
        ("JRDB\n(M2b/3D)", p_jr["strata"]["all_5836"]["models"]["S6_M2b_nopose_nbr"]["ade1_full_m"], p_jr["strata"]["all_5836"]["models"]["Last2_CV"]["ade1_full_m"], p_jr["strata"]["true_deocclusion_active_maneuvers"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"], p_jr["strata"]["true_deocclusion_active_maneuvers"]["models"]["Last2_CV"]["ade1_full_m"]),
        ("JAAD 2.0\n(AutoPose)", j_t["all_locked_test_953"]["models"]["J_AutoPose_veh"]["ade_2p0s_bbox_h"], j_t["all_locked_test_953"]["models"]["Last2_CV"]["ade_2p0s_bbox_h"], j_t["strict_5frame_deoccluded_168"]["models"]["J_AutoPose_veh"]["ade_2p0s_bbox_h"], j_t["strict_5frame_deoccluded_168"]["models"]["Last2_CV"]["ade_2p0s_bbox_h"]),
        ("JTA-228*\n(2D+Jac)", p_jj["jta_native_urban_emergence"]["strata"]["official_locked_test_all"]["models"]["S6_M1_pose_nonbr"]["ade1_full_m"], p_jj["jta_native_urban_emergence"]["strata"]["official_locked_test_all"]["models"]["Last2_CV"]["ade1_full_m"], p_jj["jta_native_urban_emergence"]["strata"]["official_locked_test_all"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"], p_jj["jta_native_urban_emergence"]["strata"]["official_locked_test_all"]["models"]["Last2_CV"]["ade1_full_m"]),
        ("ETH/UCY*\n(0-Pose)", p_j6["eth_ucy_5scene_lodo"]["strata"]["all_5scenes"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"], p_j6["eth_ucy_5scene_lodo"]["strata"]["all_5scenes"]["models"]["Last2_CV"]["ade1_full_m"], p_j6["eth_ucy_5scene_lodo"]["strata"]["emergence_active_maneuver"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"], p_j6["eth_ucy_5scene_lodo"]["strata"]["emergence_active_maneuver"]["models"]["Last2_CV"]["ade1_full_m"]),
        ("DUT*\n(0-Pose)", p_j6["dut_3066_maneuver_onset"]["strata"]["all_windows"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"], p_j6["dut_3066_maneuver_onset"]["strata"]["all_windows"]["models"]["Last2_CV"]["ade1_full_m"], p_j6["dut_3066_maneuver_onset"]["strata"]["active_maneuver_start_stop_turn"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"], p_j6["dut_3066_maneuver_onset"]["strata"]["active_maneuver_start_stop_turn"]["models"]["Last2_CV"]["ade1_full_m"]),
        ("IMPTC*\n(0-Pose)", p_j6["imptc_564_maneuver_onset"]["strata"]["all_windows"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"], p_j6["imptc_564_maneuver_onset"]["strata"]["all_windows"]["models"]["Last2_CV"]["ade1_full_m"], p_j6["imptc_564_maneuver_onset"]["strata"]["active_maneuver_start_stop_turn"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"], p_j6["imptc_564_maneuver_onset"]["strata"]["active_maneuver_start_stop_turn"]["models"]["Last2_CV"]["ade1_full_m"]),
    ]
    xd = np.arange(len(ds7))
    rel_all = [100.0 * (a - c) / c for _, a, c, _, _ in ds7]
    rel_sub = [100.0 * (sa - sc) / sc for _, _, _, sa, sc in ds7]
    ax.axhline(0.0, color="#2C3E50", lw=1.0, linestyle=":")
    ax.axvline(3.5, color="#922B21", lw=1.1, linestyle="--", alpha=0.75)
    ax.bar(xd - 0.18, rel_all, width=0.34, color="#2E86C1", alpha=0.88, label="Overall Benchmark vs. Last-2 CV (%)")
    ax.bar(xd + 0.18, rel_sub, width=0.34, color="#117A65", alpha=0.90, label="De-Occl / Maneuver Subcohort vs. CV (%)")
    ax.text(1.4, -32.6, "Pose-Bearing (JTA*: 2D+Clip Jac)", fontsize=7.3, color="#1B4F72", fontweight="bold", ha="center")
    ax.text(5.15, -32.6, "Zero-Pose Controls*", fontsize=7.3, color="#922B21", fontweight="bold", ha="center")
    ax.set_xticks(xd, [d[0] for d in ds7], fontsize=7.5)
    ax.set_ylabel("Relative Top-1 ADE_1 Change vs. CV (%)")
    ax.set_ylim(-35.0, 5.5)
    ax.set_title("(c) Partitioned Benchmarks: Pose-Bearing vs. Zero-Pose Controls")
    ax.legend(frameon=False, fontsize=7.4, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    # Panel (d): Single-Factor Isolation of S6_Full -> S6_Full_CondMedoid on PINNS all_1482 (5 Seeds)
    ax = axes[1, 1]
    facs = [
        ("S6_Full\n(Base MAP)", a1482["S6_Full_pose_nbr"]["ade1_full_m"], "#7F8C8D"),
        ("+Only\nDelayedAPA", a1482["S6_Full_OnlyDelayedAPA"]["ade1_full_m"], "#5DADE2"),
        ("+Only\nCondScorer", a1482["S6_Full_OnlyCondScorer"]["ade1_full_m"], "#2E86C1"),
        ("+Only\nMedoid", a1482["S6_Full_OnlyMedoid"]["ade1_full_m"], "#48C9B0"),
        ("APA +\nCondScorer", a1482["S6_Full_APA_CondScorer"]["ade1_full_m"], "#1F618D"),
        ("Full\nCondMedoid", a1482["S6_Full_CondMedoid"]["ade1_full_m"], "#5B2C6F"),
    ]
    xf = np.arange(len(facs))
    vf = [v for _, v, _ in facs]
    cf = [c for _, _, c in facs]
    bf = ax.bar(xf, vf, width=0.52, color=cf, alpha=0.92)
    for b, v in zip(bf, vf):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0007, f"{v:.4f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.set_xticks(xf, [lb for lb, _, _ in facs], fontsize=7.6)
    ax.set_ylabel("PINNS 4-Site Equal ADE_1 (m, 5 Seeds)")
    ax.set_ylim(0.428, 0.451)
    ax.set_title("(d) Single-Factor Isolation of CondMedoid Gain on PINNS (N=1,482)")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    fig.tight_layout(pad=1.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Rendered: {out_png}")
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-png", type=Path, default=Path("docs/figures/fig21_wave21_true_emergence_6slot_and_7dataset_precursors.png"))
    args = ap.parse_args()
    render_fig21(args.results_dir, args.out_png)


if __name__ == "__main__":
    main()
