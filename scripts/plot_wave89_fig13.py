"""Render 300-DPI Figure 13: Unified Ours_O2_Full ablation, JTA/cross-dataset matrix, hybrid actions, and T4 latency."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    res = root / "results"
    jta = json.loads((res / "jta_o2_bioeq_5seed_verified.json").read_text())["report"]
    bi = json.loads((res / "bidirectional_pinns_jta_o2_5seed_verified.json").read_text())["report"]
    hy = json.loads((res / "pinns_hybrid_action_events_5seed_verified.json").read_text())["report"]
    un = json.loads((res / "unified_o2_full_5seed_and_t4_latency_verified.json").read_text())["report"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.9,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.5), dpi=300)
    fig.patch.set_facecolor("white")
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Panel (a): Unified Ours_O2_Full 5-seed LOSO progressive ablation (1,482 windows, 3.0s FDE)
    arms = ["U0_O2_Motion", "U1_O2_BioEq", "U2_O2_BioEq_Graph", "Ours_O2_Full"]
    lbls_a = ["Last-2 CV\n(0 params)", "U0: O(2)\nMotion", "U1: +Gated\nBiomech.", "U2: +Null-Sink\nGraph", "Ours_O2_Full\n(20.7k params)"]
    top_fde = [1.1045585] + [un["progressive_ablation_5seed_site_equal"][a]["fde_m"] for a in arms]
    bay_fde = [1.1045585] + [un["progressive_ablation_5seed_site_equal"][a]["bayes_fde_m"] for a in arms]
    x1 = np.arange(len(lbls_a))
    ax1.bar(x1 - 0.17, top_fde, width=0.33, color="#2a9d8f", label="Top-1 3.0s FDE (m)")
    ax1.bar(x1 + 0.17, bay_fde, width=0.33, color="#0b4f4a", label="Bayesian 3.0s FDE (m)")
    for xi, bv in zip(x1 + 0.17, bay_fde):
        ax1.text(xi, bv + 0.004, f"{bv:.4f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#0b4f4a")
    ax1.set_xticks(x1, lbls_a, fontsize=8.3)
    ax1.set_ylim(min(bay_fde) - 0.035, 1.135)
    ax1.set_ylabel("Four-site equal 3.0 s FDE (m, lower better)", fontsize=9.2)
    ax1.set_title("(a) Progressive 5-seed LOSO ablation of Ours_O2_Full", fontsize=10.4, fontweight="bold", loc="left", pad=7)
    ax1.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.95)

    # Panel (b): JTA 2.5Hz urban benchmark (960 windows, 4.8s ADE) across T=2 (0.4s) & T=8 (2.8s)
    lbls_b = ["Social-\nTransmotion", "HST\n(3D/2D)", "Archived\nOurs_Full", "Archived\nM0", "O(2) BioEq\nT=2 (0.4s)", "O(2) Graph\nT=2 (0.4s)", "O(2) Graph\nT=8 (2.8s)"]
    vals_b = [
        1.9963,
        1.9432,
        1.8529,
        1.6973,
        jta["durations"]["T2"]["o2_bioeq_v2"]["slices"]["Overall"]["ADE_1"],
        jta["durations"]["T2"]["o2_nullsink_graph"]["slices"]["Overall"]["ADE_1"],
        jta["durations"]["T8"]["o2_nullsink_graph"]["slices"]["Overall"]["ADE_1"],
    ]
    cols_b = ["#c26a4a", "#d97757", "#94a3b8", "#64748b", "#2a9d8f", "#1b7f79", "#0b4f4a"]
    x2 = np.arange(len(lbls_b))
    ax2.bar(x2, vals_b, width=0.56, color=cols_b)
    ax2.axhline(1.3166, color="#0f172a", ls=":", lw=1.5, label="2.5Hz Last-2 CV (1.3166 m)")
    for xi, vv in zip(x2, vals_b):
        ax2.text(xi, vv + 0.018, f"{vv:.3f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#1e293b")
    ax2.set_xticks(x2, lbls_b, fontsize=7.9)
    ax2.set_ylim(1.10, 2.16)
    ax2.set_ylabel("JTA 960-window 4.8 s ADE (m, lower better)", fontsize=9.2)
    ax2.set_title("(b) JTA 2.5 Hz urban benchmark (-33.4% vs Social-Transmotion)", fontsize=10.4, fontweight="bold", loc="left", pad=7)
    ax2.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper right", fontsize=8.0, framealpha=0.95)

    # Panel (c): Five-class & rare Start/Stop action F1 / AP calibration (1,482 windows, 5 seeds)
    cls_names = ["Start\n(49)", "Stop\n(53)", "Turn-L\n(214)", "Turn-R\n(208)", "Keep\n(975)", "Site-Eq\nMacro"]
    f1_head = [hy["parity_head_eligible_t05"]["pooled_classes"][c]["f1"] for c in ("start", "stop", "left", "right", "maintain")] + [hy["parity_head_eligible_t05"]["site_equal"]["macro_f1"]]
    f1_imp = [hy["implied_kinematic_only_t05"]["pooled_classes"][c]["f1"] for c in ("start", "stop", "left", "right", "maintain")] + [hy["implied_kinematic_only_t05"]["site_equal"]["macro_f1"]]
    f1_cal = [hy["hybrid_inner_loso_calibrated_f1"]["pooled_classes"][c]["f1"] for c in ("start", "stop", "left", "right", "maintain")] + [hy["hybrid_inner_loso_calibrated_f1"]["site_equal_macro_f1"]]
    x3 = np.arange(len(cls_names))
    ax3.bar(x3 - 0.24, f1_head, width=0.23, color="#94a3b8", label="Parity Head (t=0.50)")
    ax3.bar(x3, f1_imp, width=0.23, color="#2a9d8f", label="Implied-Kinematic (t=0.50)")
    ax3.bar(x3 + 0.24, f1_cal, width=0.23, color="#0b4f4a", label="Hybrid Inner-LOSO Calib.")
    ax3.set_xticks(x3, cls_names, fontsize=8.4)
    ax3.set_ylim(0, max(max(f1_cal), 0.80) * 1.15)
    ax3.set_ylabel("Out-of-site event F1 score (higher better)", fontsize=9.2)
    ax3.set_title("(c) Hybrid kinematic + parity-head rare-event calibration", fontsize=10.4, fontweight="bold", loc="left", pad=7)
    ax3.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    ax3.legend(loc="upper left", fontsize=7.8, framealpha=0.95)

    # Panel (d): Tesla T4 N=12 multi-pedestrian forward latency (ms) across T=5, 10, 20
    t_lbls = ["T=5 (0.4 s)\nN=12", "T=10 (0.9 s)\nN=12", "T=20 (1.9 s)\nN=12"]
    v1_lat = [18.308, 19.456, 20.960]
    v2_lat = [14.762, 13.998, 15.063]
    o2_lat = [un["hardware_and_complexity"]["cells"][k]["mean_ms"] for k in ("T5_N12", "T10_N12", "T20_N12")]
    o2_p95 = [un["hardware_and_complexity"]["cells"][k]["p95_ms"] for k in ("T5_N12", "T10_N12", "T20_N12")]
    x4 = np.arange(3)
    ax4.bar(x4 - 0.25, v1_lat, width=0.24, color="#cbd5e1", label="V1 Full (471k)")
    ax4.bar(x4, v2_lat, width=0.24, color="#64748b", label="V2 Ours_Full (471k, 791 MFLOPs)")
    ax4.bar(x4 + 0.25, o2_lat, width=0.24, color="#0b4f4a", label="Ours_O2_Full (20.7k, 10.5 MFLOPs)")
    ax4.axhline(25.0, color="#b91c1c", ls="--", lw=1.2, label="25 ms real-time budget")
    for xi, mv, pv in zip(x4 + 0.25, o2_lat, o2_p95):
        ax4.text(xi, mv + 0.35, f"{mv:.2f} ms\n(P95 {pv:.2f})", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#0b4f4a")
    ax4.set_xticks(x4, t_lbls, fontsize=8.6)
    ax4.set_ylim(0, 27.5)
    ax4.set_ylabel("Synchronized Tesla T4 wall-clock (ms)", fontsize=9.2)
    ax4.set_title("(d) Real-hardware Tesla T4 N=12 latency & 75x FLOP reduction", fontsize=10.4, fontweight="bold", loc="left", pad=7)
    ax4.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)
    ax4.legend(loc="upper left", fontsize=7.7, framealpha=0.95)

    fig.tight_layout(pad=1.6)
    out = root / "docs/figures/fig13_unified_o2_ablation_and_cross_dataset.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Rendered:", out)


if __name__ == "__main__":
    main()
