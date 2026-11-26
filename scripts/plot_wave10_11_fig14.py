"""Render 300-DPI Figure 14: Apple-to-apple Table 1 benchmark (497 windows) & matched T=5,10,20 Ours_O2_Full scaling."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    res = root / "results"
    t1 = json.loads((res / "table1_apple_to_apple_ours_o2_full_5seed_verified.json").read_text())["report"]
    md = json.loads((res / "matched_multiduration_unified_o2_5seed_verified.json").read_text())["report"]
    ref_full = json.loads((root / "reviews/pinns_map_axis_comparison/full_suite/corrected_map_full_suite_summary.json").read_text())["summaries"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.9,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.1), dpi=300)
    fig.patch.set_facecolor("white")

    # Panel (a): Apple-to-apple Table 1 PINNS T=5 across Overall, Clear, Partial, Heavy, Transition
    buckets = ["Overall", "Clear", "Partial", "Heavy", "Transition"]
    b_lbls = ["Overall\n(497)", "Clear\n(111)", "Partial\n(252)", "Heavy\n(134)", "Transition\n(99)"]
    x1 = np.arange(len(buckets))
    mcenet = [ref_full["PINNS_T5_Ext1_MCENET"]["slices"][b]["ADE_1_mean"] for b in buckets]
    st_mod = [ref_full["PINNS_T5_Ext3_SocialTransmotion"]["slices"][b]["ADE_1_mean"] for b in buckets]
    old_full = [ref_full["PINNS_T5_Ours_Full"]["slices"][b]["ADE_1_mean"] for b in buckets]
    m1p_ref = [ref_full["PINNS_T5_M1p"]["slices"][b]["ADE_1_mean"] for b in buckets]
    o2_top = [t1["durations"]["T5"]["slices"][b]["ADE_1"] for b in buckets]

    ax1.bar(x1 - 0.30, mcenet, width=0.14, color="#cbd5e1", label="MCENET (Top-1)")
    ax1.bar(x1 - 0.15, st_mod, width=0.14, color="#c26a4a", label="Social-Transmotion (Top-1)")
    ax1.bar(x1 + 0.00, old_full, width=0.14, color="#94a3b8", label="Archived Ours_Full (Top-1)")
    ax1.bar(x1 + 0.15, m1p_ref, width=0.14, color="#64748b", label="Global M1p (Top-1)")
    ax1.bar(x1 + 0.30, o2_top, width=0.14, color="#0b4f4a", label="Ours_O2_Full (Top-1, 20.7k)")
    for xi, tv in zip(x1 + 0.30, o2_top):
        ax1.text(xi, tv + 0.012, f"{tv:.3f}", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color="#0b4f4a")
    ax1.set_xticks(x1, b_lbls, fontsize=8.5)
    ax1.set_ylim(0.32, max(max(old_full), max(mcenet)) * 1.13)
    ax1.set_ylabel("Table 1 test split Top-1 ADE (m, lower better)", fontsize=9.3)
    ax1.set_title("(a) Same-decoder Top-1 Table 1 PINNS benchmark (5 seeds)", fontsize=10.3, fontweight="bold", loc="left", pad=7)
    ax1.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", fontsize=7.7, framealpha=0.95)

    # Panel (b): 623 matched multi-duration (T=5, 10, 20) trajectory & 5-action scaling of Ours_O2_Full
    durs = ["5", "10", "20"]
    d_lbls = ["T=5 (0.4 s)", "T=10 (0.9 s)", "T=20 (1.9 s)"]
    x2 = np.arange(3)
    bay_fde = [md["durations"][d]["traj_site_equal"]["bayes_fde_m"] for d in durs]
    mfde6 = [md["durations"][d]["traj_site_equal"]["minfde6_m"] for d in durs]
    act_ap = [md["durations"][d]["action_eligible_site_equal"]["macro_ap"] for d in durs]

    ax2.plot(x2, [1.1035, 1.1035, 1.1035], "k:", lw=1.6, label="Last-2 CV 3.0s FDE (1.1035 m)")
    ax2.plot(x2, bay_fde, "*-", color="#0b4f4a", lw=2.5, ms=9.0, label="Ours_O2_Full Bayesian FDE (m)")
    ax2.plot(x2, mfde6, "D--", color="#2a9d8f", lw=1.9, ms=6.2, label="Ours_O2_Full minFDE_6 (m)")
    bbox_w = {"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.88}
    for xi, fv in zip(x2, bay_fde):
        ax2.annotate(f"{fv:.4f} m", (xi, fv), xytext=(0, -15), textcoords="offset points",
                     ha="center", fontsize=7.8, fontweight="bold", color="#0b4f4a", bbox=bbox_w)
    for xi, mv in zip(x2, mfde6):
        ax2.annotate(f"{mv:.4f} m", (xi, mv), xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=7.6, fontweight="bold", color="#2a9d8f", bbox=bbox_w)

    ax2_t = ax2.twinx()
    ax2_t.plot(x2, act_ap, "s-", color="#c26a4a", lw=2.1, ms=6.4, label="5-Class Action Macro-AP")
    for xi, av in zip(x2, act_ap):
        ax2_t.annotate(f"{av:.4f} AP", (xi, av), xytext=(0, 8), textcoords="offset points",
                       ha="center", fontsize=7.8, fontweight="bold", color="#c26a4a", bbox=bbox_w)

    ax2.set_xlim(-0.26, 2.26)
    ax2.set_xticks(x2, d_lbls, fontsize=9.0)
    ax2.set_ylim(0.64, 1.155)
    ax2_t.set_ylim(0.375, 0.485)
    ax2.set_ylabel("Four-site equal 3.0 s FDE / minFDE_6 (m)", fontsize=9.2, color="#0b4f4a")
    ax2_t.set_ylabel("Five-class eligible Macro-AP", fontsize=9.2, color="#c26a4a")
    ax2.set_title("(b) 623 matched windows: Ours_O2_Full T=5,10,20 scaling", fontsize=10.5, fontweight="bold", loc="left", pad=7)
    ax2.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax2.spines["top"].set_visible(False); ax2_t.spines["top"].set_visible(False)
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2_t.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0.01, 0.76), fontsize=7.5, framealpha=0.95)

    fig.tight_layout(pad=1.6)
    out = root / "docs/figures/fig14_table1_apple_to_apple_and_multiduration_full.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Rendered:", out)


if __name__ == "__main__":
    main()
