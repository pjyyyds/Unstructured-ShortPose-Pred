"""Render 300-DPI Figure 17: Pure-motion O(2) low-displacement inner-LOSO anchoring vs Last-2 CV (ADE_1 & FDE_1)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _panel_ab(ax_a, ax_b, st: dict) -> None:
    keys = ["first_short_323", "first_raw_261", "first_contiguous_449", "all_1482"]
    lbls = ["First+<0.5m\n(n=323)", "First Raw\n(n=261)", "First Sampled\n(n=449)", "All 4-Site LOSO\n(n=1,482)"]
    x = np.arange(len(keys))
    for ax, metric, title, ylim in (
        (ax_a, "ade1_m", "(a) Top-1 ADE_1 (m): <0.5m +0.19 cm vs CV (CI [-0.04, +0.32] cm)", (0.26, 0.66)),
        (ax_b, "fde1_m", "(b) 3.0 s FDE_1 (m): <0.5m +0.69 cm vs CV (CI [+0.23, +0.89] cm)", (0.68, 1.38)),
    ):
        cv = [st[k]["models"]["CV"]["site_equal"][metric] for k in keys]
        un = [st[k]["models"]["M0_unanchored"]["site_equal"][metric] for k in keys]
        an = [st[k]["models"]["M0_anchor"]["site_equal"][metric] for k in keys]
        ax.bar(x - 0.24, cv, width=0.22, color="#cbd5e1", label="Last-2 CV (Reference)")
        ax.bar(x + 0.00, un, width=0.22, color="#c26a4a", label="M0 Unanchored (Pure Motion)")
        ax.bar(x + 0.24, an, width=0.22, color="#0b4f4a", label="M0 Inner-LOSO Anchor (Pure Motion)")
        for xi, v_cv, v_an in zip(x, cv, an):
            ax.text(xi - 0.24, v_cv + 0.01, f"{v_cv:.3f}", ha="center", va="bottom", fontsize=6.8, color="#334155")
            ax.text(xi + 0.24, v_an + 0.01, f"{v_an:.3f}", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color="#0b4f4a")
        ax.set_xticks(x, lbls, fontsize=8.3)
        ax.set_ylim(*ylim)
        ax.set_ylabel("4-Site Equal Error (m, lower better)", fontsize=8.8)
        ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left", pad=7)
        ax.grid(axis="y", color="#e2e8f0", lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", fontsize=7.4, framealpha=0.95)


def _panel_cd(ax_c, ax_d, sda: dict) -> None:
    st = sda["strata"]
    c_keys = ["all_1482", "first_contiguous_449", "first_raw_261", "prior_35_1033", "first_short_323"]
    c_lbls = ["All LOSO\n(1,482)", "First Sampled\n(449)", "First Raw\n(261)", "Long Prior\n(1,033)", "First+<0.5m\n(323)"]
    yc = np.arange(len(c_keys))
    for off, mk, col, lbl in ((-0.16, "ade_m", "#0b4f4a", "ADE_1 diff (M0_anchor - CV)"), (0.16, "fde_m", "#b45309", "3.0 s FDE_1 diff (M0_anchor - CV)")):
        means = [st[k]["paired_top1_clip_ci"]["M0_anchor_minus_CV"]["mean"][mk] * 100.0 for k in c_keys]
        lo = [st[k]["paired_top1_clip_ci"]["M0_anchor_minus_CV"]["ci95"][mk][0] * 100.0 for k in c_keys]
        hi = [st[k]["paired_top1_clip_ci"]["M0_anchor_minus_CV"]["ci95"][mk][1] * 100.0 for k in c_keys]
        err = np.vstack((np.array(means) - np.array(lo), np.array(hi) - np.array(means)))
        ax_c.errorbar(means, yc + off, xerr=err, fmt="o", color=col, lw=1.7, capsize=3.5, ms=5.2, label=lbl)
    ax_c.axvline(0.0, color="#dc2626", ls="--", lw=1.3, label="Zero diff vs Last-2 CV")
    ax_c.set_yticks(yc, c_lbls, fontsize=8.2)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Paired 47-clip difference vs Last-2 CV (cm, <0 favors M0_anchor)", fontsize=8.6)
    ax_c.set_title("(c) Paired 47-clip 95% CIs: <0.5m FDE_1 remains +0.69 cm > CV", fontsize=9.5, fontweight="bold", loc="left", pad=7)
    ax_c.grid(axis="x", color="#e2e8f0", lw=0.7)
    ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)
    ax_c.legend(loc="lower left", fontsize=7.3, framealpha=0.95)

    d_refs = [float(x["d_ref"]) for x in sda["inner_loso_selected"]]
    vals, cnts = np.unique(d_refs, return_counts=True)
    ax_d.bar([f"{v:.2f} m" for v in vals], cnts, width=0.45, color="#334155", alpha=0.9)
    for idx, c in enumerate(cnts):
        ax_d.text(idx, c + 0.3, f"{int(c)}/20 folds", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#1e293b")
    ax_d.set_ylim(0, max(cnts) + 3.5)
    ax_d.set_xlabel("Inner-LOSO selected low-displacement scale d_ref (3 source sites only)", fontsize=8.6)
    ax_d.set_ylabel("Count across 5 seeds x 4 folds (20 runs)", fontsize=8.6)
    ax_d.set_title("(d) Zero-leakage inner-LOSO d_ref selection (pure motion only)", fontsize=9.5, fontweight="bold", loc="left", pad=7)
    ax_d.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax_d.spines["top"].set_visible(False); ax_d.spines["right"].set_visible(False)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sda = json.loads((root / "results/pinns_short_disp_hardq_anchor_5seed_verified.json").read_text())["report"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#334155", "axes.linewidth": 0.9})
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.6), dpi=300)
    fig.patch.set_facecolor("white")
    _panel_ab(axes[0, 0], axes[0, 1], sda["strata"])
    _panel_cd(axes[1, 0], axes[1, 1], sda)
    fig.tight_layout(pad=1.8)
    out = root / "docs/figures/fig17_ppt_mainline_conformal_and_short_disp.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Rendered:", out)


if __name__ == "__main__":
    main()
