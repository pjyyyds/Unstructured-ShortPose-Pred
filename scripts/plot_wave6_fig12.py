"""Render 300-DPI Figure 12: Occlusion degradation, speed-gated biomechanics, split-conformal tubes, and O(2) DUT graph."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    res = root / "results"
    occ = json.loads((res / "pinns_o2_occlusion_stress_5seed_verified.json").read_text())["report"]
    conf = json.loads((res / "pinns_conformal_and_events_5seed_verified.json").read_text())["report"]
    gr = json.loads((res / "o2_hetero_graph_pinns_and_dut_5seed_verified.json").read_text())["report"]
    hor = json.loads((res / "pinns_horizon_and_speed_regimes_5seed_verified.json").read_text())["report"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.9,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4), dpi=300)
    fig.patch.set_facecolor("white")
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Panel (a): Pose degradation & occlusion stress (3.0 s FDE across 5 regimes)
    r_keys = ["clean", "lower_limb_occluded", "frame_dropout_50pct", "heavy_jitter_025", "complete_blackout"]
    r_lbls = ["Clean\nRTMPose", "Lower-Limb\nOccluded", "50% Frame\nDropout", "Heavy Jitter\n(0.25 H)", "Complete\nBlackout"]
    x1 = np.arange(len(r_keys))
    y_mot = [occ["regimes"][k]["o2_motion"]["fde_m"] for k in r_keys]
    y_raw = [occ["regimes"][k]["o2_raw_pose"]["fde_m"] for k in r_keys]
    y_bio = [occ["regimes"][k]["o2_bioeq_v2"]["fde_m"] for k in r_keys]
    y_raw_clipped = [min(v, 1.225) for v in y_raw]
    ax1.plot(x1, y_raw_clipped, "s--", color="#c26a4a", lw=2.0, ms=6.2, label="O(2) Raw-Pose (34-D kpts)")
    ax1.plot(x1, y_mot, "o:", color="#64748b", lw=1.8, ms=5.5, label="O(2) Motion Backbone")
    ax1.plot(x1, y_bio, "*-", color="#0b4f4a", lw=2.5, ms=8.8, label="O(2) BioEq v2 (Gated 11-D)")
    ax1.annotate(f"Off-scale: {y_raw[-1]:.2f} m\n(+{y_raw[-1]-y_bio[-1]:.2f} m explosion)",
                 (x1[-1], 1.225), xytext=(-108, -28), textcoords="offset points",
                 fontsize=8.0, fontweight="bold", color="#b91c1c",
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.0),
                 bbox=dict(boxstyle="round,pad=0.2", fc="#fef2f2", ec="#fecaca", lw=0.6))
    ax1.set_ylim(1.015, 1.245)
    ax1.set_xticks(x1, r_lbls, fontsize=8.4)
    ax1.set_ylabel("Four-site equal 3.0 s FDE (m)", fontsize=9.4)
    ax1.set_title("(a) Graceful degradation under pose occlusion & noise (1,482 windows)", fontsize=10.4, fontweight="bold", loc="left", pad=7)
    ax1.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.legend(loc="best", fontsize=8.0, framealpha=0.95)

    # Panel (b): Multi-horizon step error crossover (0.5s..3.0s) & speed-regime gate decay
    h_keys = ["0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
    t_sec = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    cv_step = [hor["horizons"][k]["cv_last2"]["step_err_m"] for k in h_keys]
    o2_step = [hor["horizons"][k]["o2_bioeq_v2_bayes"]["step_err_m"] for k in h_keys]
    ax2.plot(t_sec, cv_step, "o--", color="#64748b", lw=1.9, ms=5.8, label="Last-2 Constant Velocity")
    ax2.plot(t_sec, o2_step, "*-", color="#0b4f4a", lw=2.5, ms=8.8, label="O(2) BioEq-Act v2 (Bayesian)")
    ax2.fill_between(t_sec, o2_step, cv_step, where=np.array(o2_step) <= np.array(cv_step), color="#2a9d8f", alpha=0.16)
    ax2.annotate(f"-{(cv_step[-1]-o2_step[-1])*100:.1f} cm at 3.0s", (3.0, o2_step[-1]), xytext=(-76, 12),
                 textcoords="offset points", fontsize=8.3, fontweight="bold", color="#0b4f4a",
                 arrowprops=dict(arrowstyle="->", color="#0b4f4a", lw=0.9))
    s_keys = ["quasi_static_lt_0.25", "slow_transition_0.25_0.80", "normal_walk_0.80_1.40", "brisk_ge_1.40"]
    g_x = np.array([0.6, 1.3, 2.1, 2.85])
    gate_w = [hor["causal_speed_strata"][k]["mean_bio_gate_weight"] for k in s_keys]
    ax2_t = ax2.twinx()
    ax2_t.plot(g_x, gate_w, "D:", color="#c26a4a", lw=1.8, ms=5.6, label="Biomech. gate g(v0, q) across speed bins")
    ax2.set_xticks(t_sec, h_keys, fontsize=8.8)
    ax2.set_ylabel("Four-site equal horizon error (m)", fontsize=9.2, color="#0b4f4a")
    ax2_t.set_ylabel("Mean biomech. gate weight g(v0, q)", fontsize=8.8, color="#c26a4a")
    ax2_t.set_ylim(0.0, 0.21)
    ax2.set_title("(b) Horizon crossover (1.0s-3.0s) & speed-regime gate decay", fontsize=10.4, fontweight="bold", loc="left", pad=7)
    ax2.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax2.spines["top"].set_visible(False); ax2_t.spines["top"].set_visible(False)
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2_t.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.8, framealpha=0.95)

    # Panel (c): LOSO Split-Conformal 90% & 95% trajectory safety tube radii & empirical coverage
    c_keys = ["cv_single", "o2_top1_tube", "o2_union6_tube", "o2_union6_fde"]
    c_lbls = ["CV Single\n3.0s Tube", "O(2) Top-1\n3.0s Tube", "O(2) K=6\nUnion Tube", "O(2) K=6\nEndpoint"]
    x3 = np.arange(len(c_keys))
    r90 = [conf["split_conformal_loso"]["90"][k]["mean_radius_m"] for k in c_keys]
    r95 = [conf["split_conformal_loso"]["95"][k]["mean_radius_m"] for k in c_keys]
    c90 = [conf["split_conformal_loso"]["90"][k]["empirical_site_equal_coverage"] * 100.0 for k in c_keys]
    ax3.bar(x3 - 0.18, r90, width=0.34, color="#2a9d8f", label="90% LOSO Conformal Radius (m)")
    ax3.bar(x3 + 0.18, r95, width=0.34, color="#0b4f4a", label="95% LOSO Conformal Radius (m)")
    for xi, r_v, cov_v in zip(x3 - 0.18, r90, c90):
        ax3.text(xi, r_v + 0.06, f"{r_v:.2f}m\n({cov_v:.0f}%)", ha="center", va="bottom", fontsize=7.3, color="#0b4f4a")
    ax3.set_xticks(x3, c_lbls, fontsize=8.4)
    ax3.set_ylim(0, max(r95) * 1.22)
    ax3.set_ylabel("Calibrated safety radius (m, lower is tighter)", fontsize=9.2)
    ax3.set_title("(c) Leave-one-site-out split-conformal safety tubes", fontsize=10.4, fontweight="bold", loc="left", pad=7)
    ax3.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    ax3.legend(loc="upper right", fontsize=8.0, framealpha=0.95)

    # Panel (d): Zero-shot 5-axis O(2) Null-Sink Ped-Vehicle Graph on DUT (3,066 shared-space windows)
    ax_keys = ["0", "90", "180", "270", "mirror_y"]
    ax_lbls = ["0 deg", "90 deg", "180 deg", "270 deg", "Mirror-Y"]
    x4 = np.arange(len(ax_keys))
    m2b_arch = [0.5625, 0.7289, 0.7131, 0.7518, 0.6251]
    cv_dut = [gr["dut_zero_shot_5seed_transfer"]["axes"][k]["cv_measured"]["ade_m"] for k in ax_keys]
    o2_gr = [gr["dut_zero_shot_5seed_transfer"]["axes"][k]["o2_nullsink_real_bayes"]["ade_m"] for k in ax_keys]
    ax4.plot(x4, m2b_arch, "s--", color="#c26a4a", lw=2.0, ms=6.2, label="Archived Global M2b Graph (18.9 cm swing)")
    ax4.plot(x4, cv_dut, "k:", lw=1.7, label=f"Measured-Clock CV ({cv_dut[0]:.4f} m)")
    ax4.plot(x4, o2_gr, "*-", color="#0b4f4a", lw=2.6, ms=9.0, label=f"O(2) Null-Sink Graph ({o2_gr[0]:.4f} m, 0.0 cm swing)")
    ax4.set_xticks(x4, ax_lbls, fontsize=8.8)
    ax4.set_ylim(min(o2_gr) - 0.04, 0.79)
    ax4.set_ylabel("DUT 11-clip equal ADE (m, 3,066 windows)", fontsize=9.2)
    ax4.set_title("(d) Zero-shot 5-axis O(2) ped-vehicle graph transfer on DUT", fontsize=10.4, fontweight="bold", loc="left", pad=7)
    ax4.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)
    ax4.legend(loc="upper left", fontsize=7.9, framealpha=0.95)

    fig.tight_layout(pad=1.6)
    out = root / "docs/figures/fig12_occlusion_conformal_and_o2_graph.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Rendered:", out)


if __name__ == "__main__":
    main()
