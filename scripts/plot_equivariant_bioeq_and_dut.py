"""Plot 300-DPI Figure 10: PINNS 4-site LOSO progression & DUT 4-angle rotation test."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build_figure(output: Path) -> None:
    st = json.loads((ROOT / "reviews/social_transmotion_adaptation/scores.json").read_text())["four_site_equal"]
    geo = json.loads((ROOT / "reviews/pinns_ordered_geometry/geometry_scores.json").read_text())["four_site_macro_m"]
    v1 = json.loads((ROOT / "results/pinns_bioeq_v1_5seed_verified.json").read_text())["five_seed_site_equal"]
    v2 = json.loads((ROOT / "results/pinns_bioeq_v2_5seed_verified.json").read_text())["five_seed_site_equal"]
    dut_m = json.loads((ROOT / "reviews/dut_shared_space/motion_aggregate.json").read_text())["axis_and_timing"]
    dut_f = json.loads((ROOT / "reviews/dut_shared_space/fusion_aggregate.json").read_text())["by_axis_and_availability"]
    dut_o2 = json.loads((ROOT / "reviews/dut_shared_space/dut_o2_site_balanced_transfer.json").read_text())["angles"]

    rows_a = [
        ("Social-Transmotion (traj + 2D box, 1-seed)", st["traj_box"]["ade_m"], "#c96a4b"),
        ("Social-Transmotion (traj only, 1-seed)", st["traj"]["ade_m"], "#d89278"),
        ("Global Pose M1 (5-seed)", 0.537524, "#9aaab0"),
        ("Global Motion M0 (5-seed)", 0.512562, "#7f969c"),
        ("Last-2 Constant Velocity (CV)", v2["last_two_cv"]["ade_m"], "#485e60"),
        ("Local Heading Ridge (motion)", geo["motion"]["ade_m"], "#669e98"),
        ("SO(2) BioEq v1 (bio_pose, 5-seed)", v1["bio_pose"]["ade_m"], "#2a8c82"),
        ("SO(2) Motion Backbone v2 (5-seed)", v2["motion"]["ade_m"], "#12786e"),
        ("O(2) BioEq v2 (bio_pose, 5-seed)", v2["bio_pose"]["ade_m"], "#065f56"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6), gridspec_kw={"width_ratios": [1.18, 1.0]})
    fig.subplots_adjust(left=0.25, right=0.97, top=0.80, bottom=0.14, wspace=0.28)
    fig.text(0.03, 0.95, "Cross-site coordinate equivariance and camera-invariant biomechanical gating",
             fontsize=15.5, weight="bold", color="#152f32")
    fig.text(0.03, 0.895, "Left: PINNS 4-site leave-one-site-out (1,482 windows). Right: DUT campus shared space (3,066 windows) across 90° map rotations.",
             fontsize=10.0, color="#485e60")

    y = np.arange(len(rows_a))
    ax1.set_axisbelow(True)
    ax1.grid(axis="x", color="#e4e9e8", linewidth=0.7)
    cv_val = v2["last_two_cv"]["ade_m"]
    ax1.axvline(cv_val, color="#485e60", linestyle="--", linewidth=1.0, zorder=1)
    for i, (label, val, col) in enumerate(rows_a):
        ax1.barh(i, val, height=0.58, color=col, zorder=2)
        ax1.text(val + 0.006, i, f"{val:.4f} m", va="center", ha="left", fontsize=8.5,
                 weight="bold" if i >= 7 else "regular", color="#152f32",
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.88}, zorder=3)
    ax1.set_yticks(y, [r[0] for r in rows_a], fontsize=8.8)
    ax1.get_yticklabels()[-1].set_weight("bold")
    ax1.set_xlim(0.38, 0.715)
    ax1.set_ylim(len(rows_a) - 0.45, -0.55)
    ax1.set_xlabel("Four-site equal-weighted Top-1 ADE (m, lower is better)", fontsize=9.5, color="#152f32")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("(a) PINNS 4-site LOSO trajectory error", loc="left", fontsize=12.0, weight="bold", color="#152f32", pad=12)

    angs = ["0", "90", "180", "270"]
    x = np.arange(4)
    m0_fde = [dut_m[a]["all"]["M0"]["window_equal"]["fde_m"] for a in angs]
    m1_fde = [dut_f[a]["all"]["M1"]["window_equal"]["fde_m"] for a in angs]
    o2_fde = [dut_o2[a]["o2_ridge_clock_aware"]["window_equal"]["fde_m"] for a in angs]
    cv_fde = [dut_o2[a]["cv_measured"]["window_equal"]["fde_m"] for a in angs]

    ax2.set_axisbelow(True)
    ax2.grid(axis="y", color="#e4e9e8", linewidth=0.7)
    ax2.plot(x, m0_fde, "o-", color="#c96a4b", lw=2.1, ms=6, label="Non-equivariant M0 (23.4 cm swing)")
    ax2.plot(x, m1_fde, "s--", color="#d89278", lw=1.8, ms=5.5, label="Quality-gated M1 (97.2% pose missing)")
    ax2.plot(x, o2_fde, "D-", color="#08766e", lw=2.4, ms=6, label="Clock-aware O(2) Equivariant (0.0 cm swing)")
    ax2.plot(x, cv_fde, "^:", color="#485e60", lw=1.8, ms=5.5, label="Measured-clock CV baseline")
    ax2.set_xticks(x, ["0°", "90°", "180°", "270°"], fontsize=9.5)
    top_y = max(max(m0_fde), max(m1_fde)) + 0.22
    ax2.set_ylim(0.88, top_y)
    ax2.set_ylabel("Window-equal 3.0 s FDE on DUT (m)", fontsize=9.5, color="#152f32")
    ax2.set_xlabel("External campus coordinate axis rotation", fontsize=9.5, color="#152f32")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#dfe5e5", fontsize=8.2, loc="upper right")
    ax2.set_title("(b) DUT 3,066-window zero-shot rotation invariance", loc="left", fontsize=12.0, weight="bold", color="#152f32", pad=12)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


if __name__ == "__main__":
    build_figure(ROOT / "docs/figures/fig10_equivariant_bioeq_and_dut.png")
