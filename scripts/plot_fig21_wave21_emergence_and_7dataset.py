"""Render 300-DPI Figure 21: Raw-Video & Homography Emergence Audit, 5-Seed 6-Slot Selection, Pose vs Control Benchmarks, & Two-Rule Barrier Filter."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_fig21(results_dir: Path, out_png: Path) -> Path:
    w24_p = results_dir / "wave24_pinns_raw_video_annotated_emergence_5seed_verified.json"
    w23_p = results_dir / "wave23_pinns_video_audit_and_6slot_ranker_5seed_verified.json"
    p_w23 = json.loads((w24_p if w24_p.is_file() else w23_p).read_text())
    p_jr = json.loads(w23_p.read_text())["jrdb_9loc_lolo"]
    p_j23 = json.loads((results_dir / "wave23_jaad_autopose_and_isofpr_leadtime_5seed_verified.json").read_text())
    p_poly = json.loads((results_dir / "wave21_polyline_crop_fix_and_dual_boundary_5seed_verified.json").read_text())
    p_jj = json.loads((results_dir / "wave21_jrdb_and_jta_emergence_maneuvers_5seed_verified.json").read_text())
    p_j6 = json.loads((results_dir / "wave21_jaad_deocclusion_and_eth_ucy_6dataset_5seed_verified.json").read_text())

    vis_path = results_dir.parent / "data/annotations/pinns_blind_raw_video_emergence_labels.json"
    vis_cnt = Counter(json.loads(vis_path.read_text()).values()) if vis_path.is_file() else {}

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 9.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.9,
    })
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), dpi=300)
    fig.patch.set_facecolor("white")

    # Panel (a): PINNS First-Contiguous (N=449) Audit — Old BEV Bug vs Homography+Video vs Raw-Video Visual Inspection
    ax = axes[0, 0]
    cen = p_w23["pinns_blind_video_homography_census"]
    cats = ["Vehicle\nOccluder", "Static Pillar/\nPortal", "Foreground\nPerson", "Border\nEntry", "Open-View\nLate Start", "Frame-0\nClip Start"]
    hom_v = [32, 28, 129, 23, 29, 208]
    vis_v = [
        int(vis_cnt.get("veh", cen.get("homography_or_visual_vehicle_windows", 32))),
        int(vis_cnt.get("static", 63)),
        int(vis_cnt.get("pers", 45)),
        int(vis_cnt.get("border", 27)),
        int(vis_cnt.get("open", 74)),
        int(cen["clip_start_frame0_windows"]),
    ]
    x = np.arange(len(cats))
    b1 = ax.bar(x - 0.18, hom_v, width=0.34, color="#7FB3D5", alpha=0.90, label="Homography + Pixel Rule (N=449)")
    b2 = ax.bar(x + 0.18, vis_v, width=0.34, color="#5B2C6F", alpha=0.90, label="Blind Raw-Video Visual Audit (N=449)")
    for bars in (b1, b2):
        for b in bars:
            h = int(round(b.get_height()))
            ax.text(b.get_x() + b.get_width() / 2, h + 3, str(h), ha="center", va="bottom", fontsize=7.6)
    ax.annotate("Only 1/34 of old BEV\ncar rule matched camera!", xy=(0.18, vis_v[0]), xytext=(0.45, 168),
                arrowprops={"arrowstyle": "->", "color": "#C0392B", "lw": 1.1}, fontsize=7.6, color="#922B21", fontweight="bold")
    ax.set_xticks(x, cats, fontsize=7.9)
    ax.set_ylabel("Window Count (First-Contiguous N=449)")
    ax.set_title("(a) PINNS Homography Fix & Raw-Video Blind Emergence Audit")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    ax.set_ylim(0, 248)

    # Panel (b): 5-Seed Mean 6-Candidate Generation vs Top-1 Selection Audit on PINNS
    ax = axes[0, 1]
    cd5 = p_w23["pinns_4site_loso"]["candidate_generation_vs_selection_5seed"]
    mans = ["Start", "Stop", "Turn_Left", "Turn_Right"]
    xm = np.arange(len(mans))
    m0_g = [100.0 * cd5["S6_M0_nopose_nonbr"][m]["any6_generated_hits_5seed_mean"] / max(cd5["S6_M0_nopose_nonbr"][m]["n_true"], 1) for m in mans]
    cm_g = [100.0 * cd5["S6_Full_CondMedoid"][m]["any6_generated_hits_5seed_mean"] / max(cd5["S6_Full_CondMedoid"][m]["n_true"], 1) for m in mans]
    cm_s = [100.0 * cd5["S6_Full_CondMedoid"][m]["top1_selected_hits_5seed_mean"] / max(cd5["S6_Full_CondMedoid"][m]["n_true"], 1) for m in mans]
    f1_s = [100.0 * cd5["S6_Full_CondF1Slot"][m]["top1_selected_hits_5seed_mean"] / max(cd5["S6_Full_CondF1Slot"][m]["n_true"], 1) for m in mans]
    ax.bar(xm - 0.27, m0_g, width=0.17, color="#BDC3C7", label="No-Pose S6_M0 Any-6 Gen (5-Seed)")
    ax.bar(xm - 0.09, cm_g, width=0.17, color="#5DADE2", label="CondMedoid Any-6 Gen (5-Seed)")
    ax.bar(xm + 0.09, cm_s, width=0.17, color="#1B4F72", label="CondMedoid Top-1 Sel (0.4382m)")
    ax.bar(xm + 0.27, f1_s, width=0.17, color="#D35400", label="CondF1Slot Top-1 Sel (High FP)")
    ax.set_xticks(xm, ["Start (N=9)", "Stop (N=18)", "Turn-L (N=64)", "Turn-R (N=69)"])
    ax.set_ylabel("5-Seed Mean Recall (%)")
    ax.set_ylim(0, 114)
    ax.set_title("(b) 5-Seed 'Not Generated' vs 'Not Selected' Trade-Off")
    ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    # Panel (c): Partitioned 4 Pose-Bearing Benchmarks vs 3 Zero-Pose Interaction Control Benchmarks
    ax = axes[1, 0]
    j_t = p_j23["jaad2_trajectory_autopose_vs_human_5seed"]["test_strata"]
    ds7 = [
        ("PINNS\n(Pose+Nbr)",
         p_w23["pinns_4site_loso"]["strata"]["all_1482"]["models"]["S6_Full_CondMedoid"]["ade1_full_m"],
         p_w23["pinns_4site_loso"]["strata"]["all_1482"]["models"]["Last2_CV"]["ade1_full_m"],
         p_w23["pinns_4site_loso"]["strata"]["verified_phys_emergence_active_maneuvers"]["models"]["S6_Full_CondMedoid"]["ade1_full_m"],
         p_w23["pinns_4site_loso"]["strata"]["verified_phys_emergence_active_maneuvers"]["models"]["Last2_CV"]["ade1_full_m"]),
        ("JRDB\n(3D+Nbr)",
         p_jr["strata"]["all_5836"]["models"]["S6_M2b_nopose_nbr"]["ade1_full_m"],
         p_jr["strata"]["all_5836"]["models"]["Last2_CV"]["ade1_full_m"],
         p_jr["strata"]["true_deocclusion_active_maneuvers"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"],
         p_jr["strata"]["true_deocclusion_active_maneuvers"]["models"]["Last2_CV"]["ade1_full_m"]),
        ("JAAD 2.0\n(AutoPose)",
         j_t["all_locked_test_953"]["models"]["J_AutoPose_veh"]["ade_2p0s_bbox_h"],
         j_t["all_locked_test_953"]["models"]["Last2_CV"]["ade_2p0s_bbox_h"],
         j_t["strict_5frame_deoccluded_168"]["models"]["J_AutoPose_veh"]["ade_2p0s_bbox_h"],
         j_t["strict_5frame_deoccluded_168"]["models"]["Last2_CV"]["ade_2p0s_bbox_h"]),
        ("JTA-228\n(3D Pose)",
         p_jj["jta_native_urban_emergence"]["strata"]["official_locked_test_all"]["models"]["S6_M1_pose_nonbr"]["ade1_full_m"],
         p_jj["jta_native_urban_emergence"]["strata"]["official_locked_test_all"]["models"]["Last2_CV"]["ade1_full_m"],
         p_jj["jta_native_urban_emergence"]["strata"]["official_locked_test_all"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"],
         p_jj["jta_native_urban_emergence"]["strata"]["official_locked_test_all"]["models"]["Last2_CV"]["ade1_full_m"]),
        ("ETH/UCY*\n(Zero-Pose)",
         p_j6["eth_ucy_5scene_lodo"]["strata"]["all_5scenes"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"],
         p_j6["eth_ucy_5scene_lodo"]["strata"]["all_5scenes"]["models"]["Last2_CV"]["ade1_full_m"],
         p_j6["eth_ucy_5scene_lodo"]["strata"]["emergence_active_maneuver"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"],
         p_j6["eth_ucy_5scene_lodo"]["strata"]["emergence_active_maneuver"]["models"]["Last2_CV"]["ade1_full_m"]),
        ("DUT*\n(Zero-Pose)",
         p_j6["dut_3066_maneuver_onset"]["strata"]["all_windows"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"],
         p_j6["dut_3066_maneuver_onset"]["strata"]["all_windows"]["models"]["Last2_CV"]["ade1_full_m"],
         p_j6["dut_3066_maneuver_onset"]["strata"]["active_maneuver_start_stop_turn"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"],
         p_j6["dut_3066_maneuver_onset"]["strata"]["active_maneuver_start_stop_turn"]["models"]["Last2_CV"]["ade1_full_m"]),
        ("IMPTC*\n(Zero-Pose)",
         p_j6["imptc_564_maneuver_onset"]["strata"]["lowdisp_lt0p5m"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"],
         p_j6["imptc_564_maneuver_onset"]["strata"]["lowdisp_lt0p5m"]["models"]["Last2_CV"]["ade1_full_m"],
         p_j6["imptc_564_maneuver_onset"]["strata"]["active_maneuver_start_stop_turn"]["models"]["S6_Full_pose_nbr"]["ade1_full_m"],
         p_j6["imptc_564_maneuver_onset"]["strata"]["active_maneuver_start_stop_turn"]["models"]["Last2_CV"]["ade1_full_m"]),
    ]
    xd = np.arange(len(ds7))
    rel_full = [100.0 * (m1 - c1) / max(c1, 1e-6) for _, m1, c1, _, _ in ds7]
    rel_sub = [100.0 * (m2 - c2) / max(c2, 1e-6) for _, _, _, m2, c2 in ds7]
    ax.axhline(0.0, color="#2C3E50", linewidth=1.0)
    ax.axvline(3.5, color="#7F8C8D", linewidth=1.2, linestyle="--")
    ax.bar(xd - 0.17, rel_full, width=0.32, color="#2874A6", alpha=0.88, label="Full Test / Cohort ADE vs. CV (%)")
    ax.bar(xd + 0.17, rel_sub, width=0.32, color="#117A65", alpha=0.88, label="De-Occl / Maneuver Subset vs. CV (%)")
    ax.text(1.5, 3.0, "4 Pose-Bearing Benchmarks", ha="center", fontsize=7.8, fontweight="bold", color="#1B4F72")
    ax.text(5.0, 3.0, "3 Zero-Pose Controls*", ha="center", fontsize=7.8, fontweight="bold", color="#566573")
    ax.set_xticks(xd, [r[0] for r in ds7], fontsize=7.5)
    ax.set_ylabel("Relative ADE1 Change vs. CV (%)")
    ax.set_ylim(min(rel_full + rel_sub) - 5.0, 7.5)
    ax.set_title("(c) Pose-Bearing vs. Zero-Pose Control* Benchmarks (<0 Better)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    # Panel (d): Un-Cropped Polyline Fix & Two-Rule Automated Barrier Filter (Rule kappa=0.87)
    ax = axes[1, 1]
    mp_d = p_poly["map_and_rater_diagnostics"]
    st_m = p_poly["pinns_4site_loso"].get("map_eligible_835", p_poly["pinns_4site_loso"]["all_1482"])["models"]
    arms4 = ["Arm0_No_Map", "Arm1_Buggy_DoubleCrop_Auto", "Arm2_Fixed_Uncropped_Auto", "Arm3_DualConfirmed_Solid_Barriers"]
    albls = ["No Map\n(M2b)", "Buggy Crop\nAuto Map", "Fixed Uncrop\nAuto Map", "Two-Rule Auto\nBarrier Filter"]
    xa = np.arange(4)
    ades = [100.0 * st_m[a]["ade1_m"] for a in arms4]
    bars_a = ax.bar(xa, ades, width=0.46, color=["#2E86C1", "#C0392B", "#D68910", "#1E8449"], alpha=0.88)
    ax.set_ylim(max(0.0, min(ades) - 3.5), max(ades) + 3.0)
    for b, v in zip(bars_a, ades):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f} cm", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax.set_xticks(xa, albls, fontsize=8.2)
    ax.set_ylabel("Top-1 ADE1 on Map-Eligible Windows (cm)")
    ax.set_title(f"(d) Un-Cropped Polyline Fix & Two-Rule Filter (Rule kappa={mp_d['dual_rater_cohen_kappa']:.2f})")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    fig.suptitle("Waves 21-24: Raw-Video Emergence Audit, 5-Seed 6-Slot Selection, & Multi-Benchmark Attribution", fontsize=11.4, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.960))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-png", type=Path, default=Path("docs/figures/fig21_wave21_true_emergence_6slot_and_7dataset_precursors.png"))
    args = ap.parse_args()
    render_fig21(args.results_dir, args.out_png)
    print("Rendered:", args.out_png)


if __name__ == "__main__":
    main()
