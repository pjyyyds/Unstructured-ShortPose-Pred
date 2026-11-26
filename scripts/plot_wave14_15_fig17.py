"""Render 300-DPI Figure 17: Verified 3x3 homography & ego-world M0-M3 chain, fixed conformal, <0.5m ground pose, & JRDB LOLO."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _panel_ab(ax_a, ax_b, ppt: dict) -> None:
    st = ppt["strata"]["all_1482"]
    arms = ["CV", "M0", "M1a", "M1b", "M1", "M2a", "M2b_no_vrad", "M2b", "M2", "M3a", "M3b", "M3"]
    lbls = ["CV", "M0", "M1a\n(g=1)", "M1b\n(No-q)", "M1\n(+q)", "M2a\n(20m)", "M2b\n(-vr)", "M2b\n(+vr)", "M2\n(Both)", "M3a\n(Aux)", "M3b\n(E2E)", "M3\n(StopG)"]
    xa = np.arange(len(arms))
    ade1 = [st["models"][a]["site_equal"]["ade1_m"] for a in arms]
    mfde6 = [st["models"][a]["site_equal"]["min_fde6_m"] for a in arms]
    cols = ["#cbd5e1", "#0b4f4a", "#c26a4a", "#64748b", "#0f766e", "#d97706", "#475569", "#1e293b", "#94a3b8", "#b45309", "#dc2626", "#115e59"]
    ax_a.bar(xa, ade1, width=0.62, color=cols, alpha=0.92, label="Same-decoder Top-1 ADE_1 (m)")
    for xi, v in zip(xa, ade1):
        ax_a.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=6.3, fontweight="bold", color="#1e293b")
    ax_ar = ax_a.twinx()
    ax_ar.plot(xa[1:], mfde6[1:], "D--", color="#7c2d12", lw=1.6, ms=4.5, label="6-Mode minFDE_6 (m, right)")
    ax_a.set_xticks(xa, lbls, fontsize=6.9)
    ax_a.set_ylim(0.42, 0.545)
    ax_ar.set_ylim(0.640, 0.700)
    ax_a.set_ylabel("4-Site LOSO Top-1 ADE_1 (m)", fontsize=8.6)
    ax_ar.set_ylabel("6-Mode minFDE_6 (m)", fontsize=8.4, color="#7c2d12")
    ax_a.set_title("(a) Corrected PINNS M0-M3 (3x3 H & ego 20m; M2b minFDE6=0.654m)", fontsize=9.2, fontweight="bold", loc="left", pad=7)
    ax_a.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax_a.spines["top"].set_visible(False); ax_ar.spines["top"].set_visible(False)
    h1, l1 = ax_a.get_legend_handles_labels(); h2, l2 = ax_ar.get_legend_handles_labels()
    ax_a.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.0, framealpha=0.95)

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
    ax_b.bar(xb + 0.24, u6_c, width=0.22, color="#0b4f4a", label="O(2) M3 Union-6 (-36~40% rad)")
    ax_b.hlines([90.0, 95.0, 90.0, 95.0], xb - 0.36, xb + 0.36, colors="#dc2626", linestyles="--", lw=1.3, label="Nominal 1-alpha")
    for xi, v in zip(xb + 0.24, u6_c):
        ax_b.text(xi, v + 0.45, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#0b4f4a")
    ax_b.set_xticks(xb, xlb, fontsize=8.0)
    ax_b.set_ylim(80.0, 100.5)
    ax_b.set_ylabel("Fixed-predictor empirical coverage (%)", fontsize=8.6)
    ax_b.set_title("(b) Fixed-predictor split-conformal (zero retraining after calibration)", fontsize=9.2, fontweight="bold", loc="left", pad=7)
    ax_b.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
    ax_b.legend(loc="lower left", ncol=2, fontsize=7.0, framealpha=0.95)


def _panel_cd(ax_c, ax_d, sda: dict, jrdb: dict) -> None:
    st323 = sda["strata"]["first_short_323"]["models"]
    st1482 = sda["strata"]["all_1482"]["models"]
    models_c = ["CV", "M0_unanchored", "M0_anchor", "M1_ground_only", "M1_hardq_clear"]
    lbls_c = ["Last-2 CV\n(Ref)", "M0\nUnanchored", "M0\nAnchor", "M1 Ground\nOrient Only", "M1 2D+Gnd\n(Clear)"]
    xc = np.arange(len(models_c))
    ade323 = [st323[m]["site_equal"]["ade1_m"] for m in models_c]
    fde323 = [st323[m]["site_equal"]["fde1_m"] for m in models_c]
    ade1482 = [st1482[m]["site_equal"]["ade1_m"] for m in models_c]
    ax_c.bar(xc - 0.25, ade323, width=0.23, color="#0b4f4a", label="<0.5m ADE_1 (n=323)")
    ax_c.bar(xc + 0.00, ade1482, width=0.23, color="#64748b", label="All LOSO ADE_1 (n=1,482)")
    ax_c.bar(xc + 0.25, fde323, width=0.23, color="#c26a4a", alpha=0.88, label="<0.5m 3.0s FDE_1 (n=323)")
    for xi, v1, v3 in zip(xc, ade323, fde323):
        ax_c.text(xi - 0.25, v1 + 0.01, f"{v1:.3f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#0b4f4a")
        ax_c.text(xi + 0.25, v3 + 0.01, f"{v3:.3f}", ha="center", va="bottom", fontsize=6.4, color="#9a3412")
    ax_c.set_xticks(xc, lbls_c, fontsize=7.8)
    ax_c.set_ylim(0.26, 1.06)
    ax_c.set_ylabel("4-Site Equal Error (m, lower better)", fontsize=8.6)
    ax_c.set_title("(c) <0.5m verified 3x3 H: CV keeps lowest 3.0s FDE_1 (+0.64~+0.69 cm)", fontsize=9.1, fontweight="bold", loc="left", pad=7)
    ax_c.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)
    ax_c.legend(loc="upper left", fontsize=7.0, framealpha=0.95)

    st_j = jrdb["strata"]["all_10scenes"]["models"]
    j_arms = ["CV", "M0", "M1a", "M1b", "M1", "M2b_no_vrad", "M2b", "M3a", "M3b", "M3"]
    j_lbls = ["CV", "M0", "M1a", "M1b", "M1", "M2b\n(-vr)", "M2b\n(+vr)", "M3a", "M3b", "M3"]
    xd = np.arange(len(j_arms))
    j_ade = [st_j[a]["location_equal"]["ade1_m"] for a in j_arms]
    j_mf6 = [st_j[a]["location_equal"]["min_fde6_m"] for a in j_arms]
    j_cols = ["#cbd5e1", "#475569", "#c26a4a", "#94a3b8", "#64748b", "#334155", "#1e293b", "#0f766e", "#0b4f4a", "#115e59"]
    ax_d.bar(xd, j_ade, width=0.60, color=j_cols, alpha=0.92, label="9-Location Equal ADE_1 (m)")
    for xi, v in zip(xd, j_ade):
        ax_d.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#1e293b")
    ax_dr = ax_d.twinx()
    ax_dr.plot(xd[1:], j_mf6[1:], "o--", color="#7c2d12", lw=1.6, ms=4.5, label="6-Mode minFDE_6 (m, right)")
    ax_d.set_xticks(xd, j_lbls, fontsize=7.4)
    ax_d.set_ylim(0.24, 0.335)
    ax_dr.set_ylim(0.295, 0.355)
    ax_d.set_ylabel("JRDB 9-Loc LOLO ADE_1 (m)", fontsize=8.6)
    ax_dr.set_ylabel("JRDB 6-Mode minFDE_6 (m)", fontsize=8.4, color="#7c2d12")
    ax_d.set_title("(d) JRDB 5,836w LOLO: M3a/b beat M2b (-0.80 cm); CV lowest Top-1", fontsize=9.1, fontweight="bold", loc="left", pad=7)
    ax_d.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax_d.spines["top"].set_visible(False); ax_dr.spines["top"].set_visible(False)
    hd1, ld1 = ax_d.get_legend_handles_labels(); hd2, ld2 = ax_dr.get_legend_handles_labels()
    ax_d.legend(hd1 + hd2, ld1 + ld2, loc="upper left", fontsize=7.0, framealpha=0.95)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    res = root / "results"
    ppt = json.loads((res / "ppt_mainline_o2_m0_m3_corrected_v3_5seed_verified.json").read_text())["report"]
    sda = json.loads((res / "pinns_short_disp_hardq_anchor_corrected_v2_5seed_verified.json").read_text())["report"]
    jrdb = json.loads((res / "jrdb_m0_m3_unconfounded_5seed_verified.json").read_text())["report"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#334155", "axes.linewidth": 0.9})
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.8), dpi=300)
    fig.patch.set_facecolor("white")
    _panel_ab(axes[0, 0], axes[0, 1], ppt)
    _panel_cd(axes[1, 0], axes[1, 1], sda, jrdb)
    fig.tight_layout(pad=1.8)
    out = root / "docs/figures/fig17_ppt_mainline_conformal_and_short_disp.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Rendered:", out)


if __name__ == "__main__":
    main()
