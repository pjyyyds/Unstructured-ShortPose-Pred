"""Render 300-DPI Figure 17: PPT mainline M0-M3 O(2) chain, fixed-predictor conformal, 3x3 RC1, & <0.5m anchor."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _panel_ab(ax_a, ax_b, ppt: dict) -> None:
    st = ppt["strata"]["all_1482"]
    arms = ["CV", "M0", "M1a", "M1b", "M1", "M2a", "M2b_no_vrad", "M2b", "M2", "M3a", "M3b", "M3"]
    lbls = ["CV", "M0", "M1a\n(Direct)", "M1b\n(No-q)", "M1\n(+q)", "M2a\n(Poly)", "M2b\n(-vrad)", "M2b\n(+vrad)", "M2\n(Both)", "M3a\n(Aux)", "M3b\n(E2E)", "M3\n(StopG)"]
    xa = np.arange(len(arms))
    ade1 = [st["models"][a]["site_equal"]["ade1_m"] for a in arms]
    mfde6 = [st["models"][a]["site_equal"]["min_fde6_m"] for a in arms]
    cols = ["#cbd5e1", "#0b4f4a", "#c26a4a", "#94a3b8", "#0f766e", "#d97706", "#475569", "#334155", "#64748b", "#b45309", "#dc2626", "#115e59"]
    ax_a.bar(xa, ade1, width=0.62, color=cols, alpha=0.92, label="Same-decoder Top-1 ADE_1 (m)")
    for xi, v in zip(xa, ade1):
        ax_a.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#1e293b")
    ax_ar = ax_a.twinx()
    ax_ar.plot(xa[1:], mfde6[1:], "D--", color="#7c2d12", lw=1.6, ms=4.5, label="6-Mode minFDE_6 (m, right)")
    ax_a.set_xticks(xa, lbls, fontsize=7.0)
    ax_a.set_ylim(0.42, 0.538)
    ax_ar.set_ylim(0.645, 0.700)
    ax_a.set_ylabel("4-Site LOSO Top-1 ADE_1 (m)", fontsize=8.8)
    ax_ar.set_ylabel("6-Mode minFDE_6 (m)", fontsize=8.6, color="#7c2d12")
    ax_a.set_title("(a) PPT Slide 13 O(2) M0-M3 chain (M3 beats M3b: -0.35 cm, CI<0)", fontsize=9.8, fontweight="bold", loc="left", pad=7)
    ax_a.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax_a.spines["top"].set_visible(False); ax_ar.spines["top"].set_visible(False)
    h1, l1 = ax_a.get_legend_handles_labels(); h2, l2 = ax_ar.get_legend_handles_labels()
    ax_a.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.2, framealpha=0.95)

    cf = ppt["fixed_predictor_split_conformal"]
    w90, w95 = cf["within_site_clip_split_fixed_3site_predictor"]["90"], cf["within_site_clip_split_fixed_3site_predictor"]["95"]
    c90, c95 = cf["cross_site_12pair_fixed_2site_predictor_no_retrain"]["90"], cf["cross_site_12pair_fixed_2site_predictor_no_retrain"]["95"]
    xb = np.arange(4)
    xlb = ["Within-Site\n90% Target", "Within-Site\n95% Target", "Cross-Site\n90% (No Retrain)", "Cross-Site\n95% (No Retrain)"]
    cv_c = [w90["cv_cov"] * 100, w95["cv_cov"] * 100, c90["cv_cov"] * 100, c95["cv_cov"] * 100]
    t1_c = [w90["top1_cov"] * 100, w95["top1_cov"] * 100, c90["top1_cov"] * 100, c95["top1_cov"] * 100]
    u6_c = [w90["u6_cov"] * 100, w95["u6_cov"] * 100, c90["u6_cov"] * 100, c95["u6_cov"] * 100]
    ax_b.bar(xb - 0.24, cv_c, width=0.22, color="#cbd5e1", label="Last-2 CV Tube")
    ax_b.bar(xb + 0.00, t1_c, width=0.22, color="#64748b", label="O(2) M3 Top-1 Tube")
    ax_b.bar(xb + 0.24, u6_c, width=0.22, color="#0b4f4a", label="O(2) M3 Union-6 (-37~40% rad)")
    ax_b.hlines([90.0, 95.0, 90.0, 95.0], xb - 0.36, xb + 0.36, colors="#dc2626", linestyles="--", lw=1.3, label="Nominal 1-alpha")
    for xi, v in zip(xb + 0.24, u6_c):
        ax_b.text(xi, v + 0.45, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#0b4f4a")
    ax_b.set_xticks(xb, xlb, fontsize=8.0)
    ax_b.set_ylim(80.0, 100.5)
    ax_b.set_ylabel("Fixed-predictor empirical coverage (%)", fontsize=8.8)
    ax_b.set_title("(b) Fixed-predictor split-conformal (no retraining after calibration)", fontsize=9.8, fontweight="bold", loc="left", pad=7)
    ax_b.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
    ax_b.legend(loc="lower left", ncol=2, fontsize=7.1, framealpha=0.95)


def _panel_cd(ax_c, ax_d, rc1_3x3: dict, sd_anc: dict) -> None:
    t5 = rc1_3x3["durations"]["T5"]
    sl_keys = ["Clear", "Partial", "Heavy", "Overall_623"]
    sl_lbls = ["T=5 Clear\n(n=188)", "T=5 Partial\n(n=283)", "T=5 Heavy\n(n=152)", "T=5 Overall\n(n=623)"]
    xc = np.arange(len(sl_keys))
    m0_v = [t5[k]["models"]["M0"]["site_equal"]["ade1_m"] for k in sl_keys]
    m1a_v = [t5[k]["models"]["M1a"]["site_equal"]["ade1_m"] for k in sl_keys]
    m1b_v = [t5[k]["models"]["M1b"]["site_equal"]["ade1_m"] for k in sl_keys]
    m1_v = [t5[k]["models"]["M1"]["site_equal"]["ade1_m"] for k in sl_keys]
    ax_c.bar(xc - 0.27, m0_v, width=0.18, color="#64748b", label="M0 (Pure Motion)")
    ax_c.bar(xc - 0.09, m1a_v, width=0.18, color="#c26a4a", label="M1a (Direct, g=1.0)")
    ax_c.bar(xc + 0.09, m1b_v, width=0.18, color="#94a3b8", label="M1b (No-q, g~0.91)")
    ax_c.bar(xc + 0.27, m1_v, width=0.18, color="#0b4f4a", label="M1 (+q Gate, g~0.39)")
    for xi, v0, v1 in zip(xc, m0_v, m1_v):
        ax_c.text(xi - 0.27, v0 + 0.008, f"{v0:.3f}", ha="center", va="bottom", fontsize=6.5, color="#475569")
        ax_c.text(xi + 0.27, v1 + 0.008, f"{v1:.3f}", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color="#0b4f4a")
    ax_c.set_xticks(xc, sl_lbls, fontsize=8.2)
    ax_c.set_ylim(0.24, 0.83)
    ax_c.set_ylabel("Matched 623 Top-1 ADE_1 (m, lower better)", fontsize=8.8)
    ax_c.set_title("(c) RC1 quality gating: M1 beats M1b on T=5 Clear (-1.06 cm, CI<0)", fontsize=9.8, fontweight="bold", loc="left", pad=7)
    ax_c.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)
    ax_c.legend(loc="upper left", ncol=2, fontsize=7.1, framealpha=0.95)

    st_d = sd_anc["strata"]
    d_keys = ["first_short_323", "first_contiguous_449", "heavy_pose", "all_1482"]
    d_lbls = ["First+<0.5m\n(n=323)", "First Sampled\n(n=449)", "Heavy Pose\n(n=382)", "All 4-Site LOSO\n(n=1,482)"]
    xd = np.arange(len(d_keys))
    cv_d = [st_d[k]["models"]["CV"]["site_equal"]["ade1_m"] for k in d_keys]
    u_d = [st_d[k]["models"]["M0_unanchored"]["site_equal"]["ade1_m"] for k in d_keys]
    a_d = [st_d[k]["models"]["M0_anchor"]["site_equal"]["ade1_m"] for k in d_keys]
    hq_d = [st_d[k]["models"]["M1_hardq_clear_partial"]["site_equal"]["ade1_m"] for k in d_keys]
    ax_d.bar(xd - 0.27, cv_d, width=0.18, color="#cbd5e1", label="Last-2 CV")
    ax_d.bar(xd - 0.09, u_d, width=0.18, color="#c26a4a", label="M0 Unanchored")
    ax_d.bar(xd + 0.09, a_d, width=0.18, color="#64748b", label="M0 Low-Disp Anchor")
    ax_d.bar(xd + 0.27, hq_d, width=0.18, color="#0b4f4a", label="M1 Hard-q Anchor")
    for xi, cv_i, hq_i in zip(xd, cv_d, hq_d):
        ax_d.text(xi - 0.27, cv_i + 0.008, f"{cv_i:.3f}", ha="center", va="bottom", fontsize=6.5, color="#475569")
        ax_d.text(xi + 0.27, hq_i + 0.008, f"{hq_i:.3f}", ha="center", va="bottom", fontsize=6.7, fontweight="bold", color="#0b4f4a")
    ax_d.set_xticks(xd, d_lbls, fontsize=8.2)
    ax_d.set_ylim(0.28, 0.84)
    ax_d.set_ylabel("4-Site LOSO Top-1 ADE_1 (m, lower better)", fontsize=8.8)
    ax_d.set_title("(d) Low-disp anchor closes <0.5m gap to CV (+0.19 cm, CI crosses 0)", fontsize=9.7, fontweight="bold", loc="left", pad=7)
    ax_d.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax_d.spines["top"].set_visible(False); ax_d.spines["right"].set_visible(False)
    ax_d.legend(loc="upper left", ncol=2, fontsize=7.1, framealpha=0.95)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    res = root / "results"
    ppt = json.loads((res / "ppt_mainline_o2_m0_m3_and_fixed_conformal_5seed_verified.json").read_text())["report"]
    rc1 = json.loads((res / "ppt_rc1_3x3_duration_quality_matrix_5seed_verified.json").read_text())["report"]
    sda = json.loads((res / "pinns_short_disp_hardq_anchor_5seed_verified.json").read_text())["report"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#334155", "axes.linewidth": 0.9})
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.8), dpi=300)
    fig.patch.set_facecolor("white")
    _panel_ab(axes[0, 0], axes[0, 1], ppt)
    _panel_cd(axes[1, 0], axes[1, 1], rc1, sda)
    fig.tight_layout(pad=1.8)
    out = root / "docs/figures/fig17_ppt_mainline_conformal_and_short_disp.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Rendered:", out)


if __name__ == "__main__":
    main()
