"""Render 300-DPI Figure 15: Emergence strata, action threshold sensitivity, IMPTC/DAIR unified, & +1 frame retrain."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _panel_ab(ax_a, ax_b, emg: dict) -> None:
    st = emg["emergence_strata"]
    keys = ["first_contiguous_449", "first_raw_261", "first_short_323", "prior_35_1033"]
    lbls = ["First Sampled\n(n=449)", "First Raw Label\n(n=261)", "First+<0.5m\n(n=323)", "Prior >=35\n(n=1033)"]
    xa = np.arange(len(keys))
    cv_v = [st[k]["models"]["cv"]["site_equal_ade_m"] for k in keys]
    pad_v = [st[k]["models"]["naive_pad_t20_bayes"]["site_equal_ade_m"] for k in keys]
    u0_v = [st[k]["models"]["u0_bayes"]["site_equal_ade_m"] for k in keys]
    fl_v = [st[k]["models"]["full_bayes"]["site_equal_ade_m"] for k in keys]

    ax_a.bar(xa - 0.27, cv_v, width=0.18, color="#cbd5e1", label="Last-2 CV")
    ax_a.bar(xa - 0.09, pad_v, width=0.18, color="#c26a4a", label="First-Frame Pad T=20")
    ax_a.bar(xa + 0.09, u0_v, width=0.18, color="#64748b", label="U0 Motion (No Pose)")
    ax_a.bar(xa + 0.27, fl_v, width=0.18, color="#0b4f4a", label="T=5 Ours_O2_Full")
    for xi, cv_i, fl_i in zip(xa, cv_v, fl_v):
        ax_a.text(xi - 0.27, cv_i + 0.008, f"{cv_i:.3f}", ha="center", va="bottom", fontsize=6.6, color="#475569")
        ax_a.text(xi + 0.27, fl_i + 0.008, f"{fl_i:.3f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#0b4f4a")
    ax_a.set_xticks(xa, lbls, fontsize=8.2)
    ax_a.set_ylim(0.28, max(pad_v) * 1.25)
    ax_a.set_ylabel("Four-site equal 3.0 s ADE (m, lower better)", fontsize=9.0)
    ax_a.set_title("(a) First-annotation & <0.5m subgroups (CV wins <0.5m)", fontsize=10.0, fontweight="bold", loc="left", pad=7)
    ax_a.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax_a.spines["top"].set_visible(False); ax_a.spines["right"].set_visible(False)
    ax_a.legend(loc="upper right", ncol=2, fontsize=7.2, framealpha=0.95)

    sn = emg["action_threshold_sensitivity"]
    sc_keys = ["scale_0.50x", "scale_0.75x", "scale_1.00x", "scale_1.25x", "scale_1.50x"]
    xb = np.array([0.50, 0.75, 1.00, 1.25, 1.50])
    ap_n = [sn[k]["neural_head_fixed05"]["site_equal"]["macro_ap"] for k in sc_keys]
    f1_n = [sn[k]["neural_head_fixed05"]["site_equal"]["macro_f1"] for k in sc_keys]
    ap_h = [sn[k]["hybrid_inner_calibrated"]["site_equal"]["macro_ap"] for k in sc_keys]
    f1_h = [sn[k]["hybrid_inner_calibrated"]["site_equal"]["macro_f1"] for k in sc_keys]
    ax_b.plot(xb, ap_n, "o--", color="#64748b", lw=1.7, ms=5.5, label="Neural Head Macro-AP")
    ax_b.plot(xb, ap_h, "s-", color="#0b4f4a", lw=2.2, ms=6.2, label="Hybrid Kinematic Macro-AP")
    ax_b.plot(xb, f1_n, "^:", color="#94a3b8", lw=1.6, ms=5.5, label="Neural Fixed-0.5 Macro-F1")
    ax_b.plot(xb, f1_h, "D-", color="#c26a4a", lw=2.2, ms=6.0, label="Hybrid Calibrated Macro-F1")
    ax_b.set_xticks(xb, [f"{v:.2f}x" for v in xb], fontsize=8.5)
    ax_b.set_ylim(min(min(f1_n), min(ap_n)) - 0.04, max(max(ap_h), max(f1_h)) + 0.055)
    ax_b.set_xlabel("Action speed & turn threshold multiplier", fontsize=8.8)
    ax_b.set_ylabel("Four-site equal Macro-AP / Macro-F1", fontsize=9.0)
    ax_b.set_title("(b) 5-class action threshold sensitivity (0.50x-1.50x)", fontsize=10.2, fontweight="bold", loc="left", pad=7)
    ax_b.grid(True, color="#e2e8f0", lw=0.7)
    ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
    ax_b.legend(loc="best", fontsize=7.4, framealpha=0.95)


def _panel_cd(ax_c, ax_d, ext: dict, off: dict) -> None:
    im = ext["imptc_564"]["by_transform"]["0deg"]
    da = ext["dair_v2x_54"]["by_transform"]["0deg"]
    cats = ["IMPTC\n(564, 4-seq)", "DAIR All\n(54, 4-junc)", "DAIR Strict\n(31, ±1ms)"]
    xc = np.arange(3)
    arch = [0.3944, da["all_54"]["archived_M1"]["ade_m"], da["strict_31"]["archived_M1"]["ade_m"]]
    cv_c = [im["actual_time_cv"]["seq_equal_ade_m"], da["all_54"]["cv_measured"]["ade_m"], da["strict_31"]["cv_measured"]["ade_m"]]
    o2_c = [im["full_graph_clk_bayes"]["seq_equal_ade_m"], da["all_54"]["ours_o2_full_t2_clk_bayes"]["ade_m"], da["strict_31"]["ours_o2_full_t2_clk_bayes"]["ade_m"]]
    ax_c.bar(xc - 0.24, arch, width=0.22, color="#c26a4a", label="Archived Non-Eq (M0 / M1)")
    ax_c.bar(xc + 0.00, cv_c, width=0.22, color="#cbd5e1", label="Measured-Clock CV")
    ax_c.bar(xc + 0.24, o2_c, width=0.22, color="#0b4f4a", label="Ours_O2_Full (0.0 cm swing)")
    for xi, v in zip(xc + 0.24, o2_c):
        ax_c.text(xi, v + 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#0b4f4a")
    ax_c.set_xticks(xc, cats, fontsize=8.4)
    ax_c.set_ylim(0.0, max(max(arch), max(cv_c)) * 1.15)
    ax_c.set_ylabel("Zero-shot 3.0 s ADE (m, lower better)", fontsize=9.0)
    ax_c.set_title("(c) External zero-shot: IMPTC (DE) & DAIR-V2X (CN)", fontsize=10.2, fontweight="bold", loc="left", pad=7)
    ax_c.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)
    ax_c.legend(loc="upper left", fontsize=7.5, framealpha=0.95)

    mx = off["train_test_2x2_matrix"]
    m_keys = ["train0_test0", "train0_test1", "train1_test1", "train1_test0"]
    m_lbls = ["Tr(0)->Te(0)\nBaseline", "Tr(0)->Te(+1)\nFrozen Replay", "Tr(+1)->Te(+1)\nFull Retrain", "Tr(+1)->Te(0)\nCross Replay"]
    xd = np.arange(4)
    s_fde = [mx[k]["site_equal_fde_m"] for k in m_keys]
    a_fde = [mx[k]["affected_597_fde_m"] for k in m_keys]
    ax_d.plot(xd, s_fde, "o-", color="#0b4f4a", lw=2.3, ms=7.0, label="4-Site Equal FDE (1,482, left)")
    for xi, sv in zip(xd, s_fde):
        ax_d.annotate(f"{sv:.4f} m", (xi, sv), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold", color="#0b4f4a")
    ax_dr = ax_d.twinx()
    ax_dr.plot(xd, a_fde, "s--", color="#c26a4a", lw=2.0, ms=6.2, label="3 Zero-Origin US Clips FDE (597, right)")
    for xi, av in zip(xd, a_fde):
        ax_dr.annotate(f"{av:.4f} m", (xi, av), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold", color="#c26a4a")
    ax_d.set_xlim(-0.35, 3.35)
    ax_d.set_xticks(xd, m_lbls, fontsize=8.1)
    ms, ma = float(np.mean(s_fde)), float(np.mean(a_fde))
    ax_d.set_ylim(ms - 0.008, ms + 0.012)
    ax_dr.set_ylim(ma - 0.012, ma + 0.008)
    ax_d.set_ylabel("4-Site Equal 3.0 s FDE (m)", fontsize=8.9, color="#0b4f4a")
    ax_dr.set_ylabel("597 Affected US Windows FDE (m)", fontsize=8.9, color="#c26a4a")
    ax_d.set_title("(d) +1 video-frame 2x2 retraining (<0.8 mm shift)", fontsize=10.2, fontweight="bold", loc="left", pad=7)
    ax_d.grid(True, color="#e2e8f0", lw=0.7)
    ax_d.spines["top"].set_visible(False); ax_dr.spines["top"].set_visible(False)
    h1, l1 = ax_d.get_legend_handles_labels(); h2, l2 = ax_dr.get_legend_handles_labels()
    ax_d.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.3, framealpha=0.95)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    res = root / "results"
    emg = json.loads((res / "emergence_and_action_sensitivity_5seed_verified.json").read_text())["report"]
    ext = json.loads((res / "imptc_and_dair_o2_full_5seed_verified.json").read_text())["report"]
    off = json.loads((res / "pinns_frame_offset_retrain_5seed_verified.json").read_text())["report"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#334155", "axes.linewidth": 0.9})
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6), dpi=300)
    fig.patch.set_facecolor("white")
    _panel_ab(axes[0, 0], axes[0, 1], emg)
    _panel_cd(axes[1, 0], axes[1, 1], ext, off)
    fig.tight_layout(pad=1.7)
    out = root / "docs/figures/fig15_emergence_imptc_dair_and_frame_offset.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Rendered:", out)


if __name__ == "__main__":
    main()
