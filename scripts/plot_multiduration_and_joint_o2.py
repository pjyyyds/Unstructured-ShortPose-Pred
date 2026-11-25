"""Render 300-DPI Figure 11: Multi-duration O(2) scaling (T=5,10,20) and Joint Action-Trajectory Pareto."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    md = json.loads((root / "results/pinns_matched_o2_multiduration_5seed_verified.json").read_text())
    ja1 = json.loads((root / "results/pinns_o2_joint_actions_5seed_verified.json").read_text())
    ja2 = json.loads((root / "results/pinns_o2_joint_v2_5seed_verified.json").read_text())

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.9,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9), dpi=300)
    fig.patch.set_facecolor("white")

    # Panel (a): Matched 623-window duration scaling (3.0 s FDE across T=5, 10, 20)
    x = np.array([0, 1, 2])
    labels_x = ["T=5 (0.4 s)", "T=10 (0.9 s)", "T=20 (1.9 s)"]
    m1_orig = [1.3277, 1.2894, 1.3091]
    m0_orig = [1.3418, 1.2901, 1.2472]
    ols_cv = [md["five_seed_summary"][t]["ols_cv"]["site_equal"]["fde_m"] for t in ("5", "10", "20")]
    last2_cv = [1.1035, 1.1035, 1.1035]
    o2_top1 = [md["five_seed_summary"][t]["o2_bioeq_5seed"]["site_equal"]["fde_m"] for t in ("5", "10", "20")]
    o2_bayes = [md["five_seed_summary"][t]["o2_bioeq_5seed"]["site_equal"]["bayes_fde_m"] for t in ("5", "10", "20")]

    ax1.plot(x, m1_orig, "s--", color="#c26a4a", lw=1.8, ms=6, label="Archived Global M1 (5-seed)")
    ax1.plot(x, m0_orig, "o--", color="#7c8b99", lw=1.8, ms=6, label="Archived Global M0 (5-seed)")
    ax1.plot(x, ols_cv, "^:", color="#64748b", lw=1.6, ms=6, label="Full-window OLS CV")
    ax1.plot(x, last2_cv, "k:", lw=1.6, label="Last-2 Constant Velocity (1.1035 m)")
    ax1.plot(x, o2_top1, "D-", color="#1b7f79", lw=2.2, ms=6.5, label="O(2) BioEq Top-1 (5-seed)")
    ax1.plot(x, o2_bayes, "*-", color="#0b4f4a", lw=2.5, ms=9, label="O(2) BioEq Bayesian (5-seed)")
    for xi, yi in zip(x, o2_bayes):
        ax1.annotate(f"{yi:.4f} m", (xi, yi), xytext=(0, -14), textcoords="offset points",
                     ha="center", fontsize=8.5, fontweight="bold", color="#0b4f4a")
    ax1.set_xticks(x, labels_x, fontsize=9.5)
    ax1.set_ylim(1.045, 1.38)
    ax1.set_ylabel("Four-site equal 3.0 s FDE (m, lower is better)", fontsize=9.8)
    ax1.set_title("(a) 623 matched windows across T = 5, 10, 20", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax1.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc=(0.03, 0.34), fontsize=8.0, framealpha=0.95)

    # Panel (b): Joint Action & Trajectory Pareto (1,482 LOSO windows: Macro-AP vs Top-1 3.0s FDE)
    pts = [
        ("Global Aux (1-seed)", 0.3260, 1.0997, "#64748b", "o", (10, -6)),
        ("Global End-to-End (1-seed)", 0.3056, 1.0846, "#c26a4a", "s", (4, 28)),
        ("Global Stop-Grad (1-seed)", 0.3212, 1.0855, "#d97757", "^", (10, 10)),
        ("O(2) BioEq-Act 1-stage (5-seed)", ja1["five_seed_arms"]["bio_stop_grad"]["action_eligible"]["site_equal"]["macro_ap"],
         ja1["five_seed_arms"]["bio_stop_grad"]["traj_site_equal"]["fde_m"], "#2a9d8f", "D", (-158, -22)),
        ("O(2) Motion Stop-Grad (5-seed)", ja2["motion_stop_grad"]["action_eligible"]["site_equal"]["macro_ap"],
         ja2["motion_stop_grad"]["traj_site_equal"]["fde_m"], "#1b7f79", "P", (-165, 6)),
        ("O(2) BioEq-Act v2 (5-seed)", ja2["bio_v2_stop_grad"]["action_eligible"]["site_equal"]["macro_ap"],
         ja2["bio_v2_stop_grad"]["traj_site_equal"]["fde_m"], "#093b37", "*", (-165, -16)),
    ]
    for label, ap, fde, col, mk, off in pts:
        sz = 150 if mk == "*" else 72
        ax2.scatter([ap], [fde], c=col, marker=mk, s=sz, zorder=4, edgecolors="white", linewidths=0.7)
        fw = "bold" if "v2" in label else "normal"
        ax2.annotate(f"{label} ({ap:.3f} AP, {fde:.3f} m)", (ap, fde), xytext=off, textcoords="offset points",
                     fontsize=7.8, fontweight=fw, color=col,
                     arrowprops=dict(arrowstyle="-", color="#cbd5e1", lw=0.7),
                     bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#e2e8f0", lw=0.5, alpha=0.94))
    ax2.axhline(1.1046, color="#475569", ls=":", lw=1.3, label="Last-2 CV FDE (1.1046 m)")
    ax2.set_xlim(0.300, 0.362)
    ax2.set_ylim(1.012, 1.118)
    ax2.invert_yaxis()
    ax2.set_xlabel("Five-class observable action Macro-AP (higher is better →)", fontsize=9.8)
    ax2.set_ylabel("Top-1 3.0 s FDE (m, upper is better ↑)", fontsize=9.8)
    ax2.set_title("(b) 1,482-window joint action-trajectory Pareto", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax2.grid(True, color="#e2e8f0", lw=0.7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower left", fontsize=8.2, framealpha=0.94)

    fig.suptitle("O(2) parity-equivariant multi-duration scaling and joint action-trajectory prediction",
                 fontsize=13, fontweight="bold", x=0.02, ha="left", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = root / "docs/figures/fig11_multiduration_and_joint_o2.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    main()
