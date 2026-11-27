"""Render 300-DPI Figure 21: Blind Emergence Census, 6-Candidate Diagnosis, 7-Dataset Precursor Effect, & Solid Barrier Audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_fig21(results_dir: Path, out_png: Path) -> Path:
    p_pinns = json.loads((results_dir / "wave21_pinns_true_emergence_and_6slot_diag_5seed_verified.json").read_text())
    p_poly = json.loads((results_dir / "wave21_polyline_crop_fix_and_dual_boundary_5seed_verified.json").read_text())
    p_jj = json.loads((results_dir / "wave21_jrdb_and_jta_emergence_maneuvers_5seed_verified.json").read_text())
    p_j6 = json.loads((results_dir / "wave21_jaad_deocclusion_and_eth_ucy_6dataset_5seed_verified.json").read_text())

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.titlesize": 10.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.2,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), dpi=300)
    fig.patch.set_facecolor("white")

    # Panel (a): Blind physical emergence decomposition on PINNS (first_449 vs first_323)
    ax = axes[0, 0]
    cen = p_pinns["blind_emergence_census"]
    cats = ["Vehicle\nOccluder", "Person/Arch\nOccluder", "Image\nBorder", "Clip Start\n(Frame 0)", "Open-Field\nRe-ID"]
    k_keys = ["true_vehicle_occluder", "true_person_or_arch_occluder", "border_entry", "clip_start_frame0_artifact", "open_field_reid"]
    x = np.arange(len(cats))
    v449 = [cen["first_contiguous_449"][k] for k in k_keys]
    v323 = [cen["first_short_323"][k] for k in k_keys]
    b1 = ax.bar(x - 0.18, v449, width=0.34, color="#5B2C6F", alpha=0.88, label=f"First Contiguous (N={cen['first_contiguous_449']['n_windows']})")
    b2 = ax.bar(x + 0.18, v323, width=0.34, color="#1F618D", alpha=0.88, label=f"First <0.5m Short (N={cen['first_short_323']['n_windows']})")
    for bars in (b1, b2):
        for b in bars:
            h = int(round(b.get_height()))
            ax.text(b.get_x() + b.get_width() / 2, h + 3, str(h), ha="center", va="bottom", fontsize=8.0)
    ax.set_xticks(x, cats)
    ax.set_ylabel("Window Count")
    ax.set_title("(a) PINNS Blind Prediction-Free Emergence Decomposition")
    ax.legend(frameon=False, fontsize=8.2, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    ax.set_ylim(0, max(v449 + [10]) * 1.20)

    # Panel (b): 6-Candidate Generation vs Top-1 Selection Rates across Maneuvers (Seed 0)
    ax = axes[0, 1]
    cd = p_pinns["pinns_4site_loso"]["candidate_generation_vs_selection_seed0"]
    mans = ["Start", "Stop", "Turn_Left", "Turn_Right"]
    xm = np.arange(len(mans))
    u_gen = [100.0 * cd["U_M3_decoupled_site_thr"][m]["any6_gen_hit_rate"] for m in mans]
    u_sel = [100.0 * cd["U_M3_decoupled_site_thr"][m]["top1_selected_hit_rate"] for m in mans]
    s_gen = [100.0 * cd["S6_Full_pose_nbr"][m]["any6_gen_hit_rate"] for m in mans]
    s_sel = [100.0 * cd["S6_Full_pose_nbr"][m]["top1_selected_hit_rate"] for m in mans]
    ax.bar(xm - 0.27, u_gen, width=0.17, color="#AEB6BF", label="Legacy U_M3 Any-6 Gen (%)")
    ax.bar(xm - 0.09, u_sel, width=0.17, color="#566573", label="Legacy U_M3 Top-1 Sel (%)")
    ax.bar(xm + 0.09, s_gen, width=0.17, color="#85C1E9", label="6-Slot S6_Full Any-6 Gen (%)")
    ax.bar(xm + 0.27, s_sel, width=0.17, color="#1B4F72", label="6-Slot S6_Full Top-1 Sel (%)")
    ax.set_xticks(xm, ["Start", "Stop", "Turn-L", "Turn-R"])
    ax.set_ylabel("Recall Among True Maneuvers (%)")
    ax.set_ylim(0, 112)
    ax.set_title("(b) 'Not Generated' vs 'Not Selected' 6-Candidate Audit")
    ax.legend(frameon=False, fontsize=7.8, ncol=2, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    # Panel (c): 7-Dataset Emergence / Maneuver-Onset Gain vs Constant Velocity (cm, negative = better)
    ax = axes[1, 0]
    ds_rows = [
        ("PINNS\n(True Emg)", p_pinns["pinns_4site_loso"]["strata"]["true_occluder_emergence_all"]["models"], "S6_Full_pose_nbr", "ade1_full_m", "ade1_post_onset_1p0s_m"),
        ("JRDB\n(9-Loc Emg)", p_jj["jrdb_9location_lolo"]["strata"]["true_emergence_all"]["models"], "S6_Full_pose_nbr", "ade1_full_m", "ade1_post_onset_1p0s_m"),
        ("JTA-228\n(Test Emg)", p_jj["jta_native_urban_emergence"]["strata"]["official_locked_test_true_emergence"]["models"], "S6_Full_pose_nbr", "ade1_full_m", "ade1_post_onset_1p0s_m"),
        ("JAAD 2.0\n(De-Occl*)", p_j6["jaad2_human_deocclusion_346vid"]["strata"]["human_annotated_deocclusion"]["models"], "J_Full_pose_veh", "ade_full", "ade_post_onset_1p0s"),
        ("ETH/UCY\n(5-Scene)", p_j6["eth_ucy_5scene_lodo"]["strata"]["true_emergence_midclip"]["models"], "S6_Full_pose_nbr", "ade1_full_m", "ade1_post_onset_1p0s_m"),
    ]
    if p_j6.get("dut_3066_maneuver_onset", {}).get("strata"):
        ds_st = p_j6["dut_3066_maneuver_onset"]["strata"]
        dk = "true_emergence_or_entry" if "true_emergence_or_entry" in ds_st else next(iter(ds_st))
        ds_rows.append(("DUT\n(11-Clip)", ds_st[dk]["models"], "S6_Full_pose_nbr", "ade1_full_m", "ade1_post_onset_1p0s_m"))
    if p_j6.get("imptc_564_maneuver_onset", {}).get("strata"):
        im_st = p_j6["imptc_564_maneuver_onset"]["strata"]
        ik = "true_emergence_or_entry" if "true_emergence_or_entry" in im_st else next(iter(im_st))
        ds_rows.append(("IMPTC\n(4-Seq)", im_st[ik]["models"], "S6_Full_pose_nbr", "ade1_full_m", "ade1_post_onset_1p0s_m"))

    xd = np.arange(len(ds_rows))
    g_full, g_post = [], []
    for lbl, md, m_key, kf, kp in ds_rows:
        scale = 1.0 if "JAAD" in lbl else 100.0  # JAAD normalized/px scale vs metric meters -> cm
        g_full.append(scale * (md[m_key][kf] - md["Last2_CV"][kf]))
        g_post.append(scale * (md[m_key][kp] - md["Last2_CV"][kp]))
    ax.axhline(0.0, color="#2C3E50", linewidth=1.0)
    ax.bar(xd - 0.17, g_full, width=0.32, color="#2874A6", alpha=0.88, label="Full-Horizon ADE1 - CV (cm; *JAAD norm)")
    ax.bar(xd + 0.17, g_post, width=0.32, color="#117A65", alpha=0.88, label="Post-Onset +1.0s ADE1 - CV (cm; *JAAD norm)")
    ax.set_xticks(xd, [r[0] for r in ds_rows], fontsize=8.0)
    ax.set_ylabel("Paired Error Delta vs. CV (Negative = Better)")
    ax.set_title("(c) Cross-Dataset Emergence & Post-Onset Gain Across 7 Benchmarks")
    ax.legend(frameon=False, fontsize=7.8, loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    # Panel (d): Un-Cropped Polyline Bug Fix & Dual-Confirmed Solid Barrier Audit
    ax = axes[1, 1]
    mp_d = p_poly["map_and_rater_diagnostics"]
    st_m = p_poly["pinns_4site_loso"].get("map_eligible_835", p_poly["pinns_4site_loso"]["all_1482"])["models"]
    arms4 = ["Arm0_No_Map", "Arm1_Buggy_DoubleCrop_Auto", "Arm2_Fixed_Uncropped_Auto", "Arm3_DualConfirmed_Solid_Barriers"]
    albls = ["No Map\n(M2b)", "Buggy Crop\nAuto Map", "Fixed Uncrop\nAuto Map", "Dual-Confirmed\nSolid Barriers"]
    xa = np.arange(4)
    ades = [100.0 * st_m[a]["ade1_m"] for a in arms4]
    bars_a = ax.bar(xa, ades, width=0.46, color=["#2E86C1", "#C0392B", "#D68910", "#1E8449"], alpha=0.88)
    ylo = max(0.0, min(ades) - 3.5)
    ax.set_ylim(ylo, max(ades) + 3.0)
    for b, v in zip(bars_a, ades):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f} cm", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax.set_xticks(xa, albls)
    ax.set_ylabel("Top-1 ADE1 on Map-Eligible Windows (cm)")
    ax.set_title(f"(d) Polyline Crop Fix & Solid Barrier Audit (kappa={mp_d['dual_rater_cohen_kappa']:.2f})")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    fig.suptitle("Wave 21: True Physical Occluder Emergence, 6-Slot Precursor Diagnosis & 7-Dataset Validation", fontsize=11.8, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.962))
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
