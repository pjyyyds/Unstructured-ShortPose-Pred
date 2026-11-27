"""Render 300-DPI Figure 20: Decoupled unified chain, DUT/IMPTC zero-shot transfer, usable pose boundary, & adaptive conformal."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _panel_a_unified_and_decoupled(ax: plt.Axes, w19u: dict, w20d: dict) -> None:
    pu = w19u["pinns_4site_loso_1482"]["strata"]["all_1482"]["models"]
    pd = w20d["pinns_4site_loso_1482"]["strata"]["all_1482"]["models"]
    ju = w19u["jrdb_9location_lolo_5836"]["strata"]["all_10scenes"]["models"]
    jd = w20d["jrdb_9location_lolo_5836"]["strata"]["all_10scenes"]["models"]
    cats = ["Last-2\nCV", "U_M0\n(Motion)", "U_M1_mono\n(Mono Gate)", "U_M2b_conf\n(Conflict)", "U_M3_coad\n(+Act Joint)", "U_M3_dec\n(Decoupled+Rk)"]
    p_vals = [
        pu["Last2_CV"]["group_equal"]["ade1_m"] * 100,
        pu["U_M0"]["group_equal"]["ade1_m"] * 100,
        pu["U_M1_mono"]["group_equal"]["ade1_m"] * 100,
        pu["U_M2b_mono_conflict"]["group_equal"]["ade1_m"] * 100,
        pd["U_M3_coadapt_knots"]["group_equal_ade1_m"] * 100,
        pd["U_M3_decoupled_platt_rerank"]["group_equal_ade1_m"] * 100,
    ]
    j_vals = [
        ju["Last2_CV"]["group_equal"]["ade1_m"] * 100,
        ju["U_M0"]["group_equal"]["ade1_m"] * 100,
        ju["U_M1_mono"]["group_equal"]["ade1_m"] * 100,
        ju["U_M2b_mono_conflict"]["group_equal"]["ade1_m"] * 100,
        ju["U_M3_mono_conflict_rerank"]["group_equal"]["ade1_m"] * 100,
        jd["U_M3_decoupled_platt_rerank"]["group_equal_ade1_m"] * 100,
    ]
    xs = np.arange(len(cats))
    w = 0.36
    b1 = ax.bar(xs - w / 2, p_vals, width=w, color="#2b5c8f", alpha=0.88, label="PINNS 4-Site LOSO (n=1,482)")
    b2 = ax.bar(xs + w / 2, j_vals, width=w, color="#e07b39", alpha=0.88, label="JRDB 9-Loc LOLO (n=5,836)")
    for rect in (*b1, *b2):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.6, f"{h:.2f}", ha="center", va="bottom", fontsize=7.1)
    ax.set_xticks(xs, cats, fontsize=8.1)
    ax.set_ylim(22, 54.5)
    ax.set_ylabel("Cross-Site Top-1 ADE_1 (cm)")
    ax.set_title("(a) End-to-End Remediated Chain & Decoupled M3 Synergy", fontweight="bold", fontsize=10.0)
    ax.grid(axis="y", alpha=0.28)
    ax.legend(loc="upper right", fontsize=8.2, frameon=True)


def _panel_b_dut_imptc_zeroshot(ax: plt.Axes, w19e: dict) -> None:
    d_st = w19e["dut_3066_zero_shot"]["strata"]
    i_st = w19e["imptc_564_zero_shot"]["strata"]
    items = [
        ("DUT All (3,066)\nConflict - Frozen", d_st["all"]["paired_top1_ci"]["M2b_conflict_gate_minus_M2b_frozen"], "#1f77b4", "o"),
        ("DUT TTC<=4s (1,365)\nConflict - Frozen", d_st["S3_ttc_lt4s"]["paired_top1_ci"]["M2b_conflict_gate_minus_M2b_frozen"], "#1f77b4", "o"),
        ("DUT Car-App. (207)\nConflict - Frozen", d_st["S4_approaching_car"]["paired_top1_ci"]["M2b_conflict_gate_minus_M2b_frozen"], "#d62728", "D"),
        ("IMPTC All (564)\nConflict - Frozen", i_st["all"]["paired_top1_ci"]["M2b_conflict_gate_minus_M2b_frozen"], "#2ca02c", "s"),
        ("IMPTC Close (120)\nConflict - Frozen", i_st["S2_close_approaching"]["paired_top1_ci"]["M2b_conflict_gate_minus_M2b_frozen"], "#2ca02c", "s"),
    ]
    ys = np.arange(len(items))
    for y, (_, ci_d, col, mk) in zip(ys, items):
        m = ci_d["mean"]["ade_m"] * 100.0
        lo = m - ci_d["ci95"]["ade_m"][0] * 100.0
        hi = ci_d["ci95"]["ade_m"][1] * 100.0 - m
        ax.errorbar([m], [y], xerr=[[lo], [hi]], fmt=mk, color=col, capsize=4, lw=1.6, ms=6.2)
    ax.axvline(0.0, color="#333333", ls="--", lw=1.1)
    ax.set_yticks(ys, [lbl for lbl, _, _, _ in items], fontsize=8.3)
    ax.set_ylim(-0.55, 4.95)
    ax.invert_yaxis()
    dc = w19e["dut_3066_zero_shot"]["zero_retrain_split_conformal"]["cross_group_ordered_pairs_no_retrain"]["M2b_ttc_conflict_gate"]
    ic = w19e["imptc_564_zero_shot"]["zero_retrain_split_conformal"]["cross_group_ordered_pairs_no_retrain"]["M2b_ttc_conflict_gate"]
    txt = (
        f"Zero-Retrain Cross-Pair Union-6 Conformal (0.0m 5-Axis Swing):\n"
        f"  DUT (72 pairs): {dc['90']['cov_u6']*100:.2f}% / {dc['95']['cov_u6']*100:.2f}% (rad -31.2%/-29.0%)\n"
        f"  IMPTC (12 pairs): {ic['90']['cov_u6']*100:.2f}% / {ic['95']['cov_u6']*100:.2f}% (rad -38.7%/-37.7%)"
    )
    ax.text(0.02, 0.05, txt, transform=ax.transAxes, fontsize=7.6,
            bbox={"boxstyle": "round,pad=0.32", "fc": "#eef5fc", "ec": "#2b5c8f"})
    ax.set_xlabel("Zero-Shot Paired Top-1 ADE_1 Difference (cm, < 0 favors conflict gate)")
    ax.set_title("(b) Zero-Shot Transfer on DUT (3,066) & IMPTC (564)", fontweight="bold", fontsize=10.0)
    ax.grid(axis="x", alpha=0.28)


def _panel_c_usable_pose_boundary(ax: plt.Axes, w19b: dict, w20m: dict) -> None:
    ps = w19b["pinns_4site_loso_1482"]["strata"]
    bins = [
        ("B0_tiny_lt50px", "< 50 px\n(n=98)"),
        ("B1_small_50_80px", "50-80 px\n(n=746)"),
        ("B2_medium_80_130px", "80-130 px\n(n=491)"),
        ("B3_large_ge130px", ">= 130 px\n(n=147)"),
    ]
    xs = np.arange(len(bins))
    for off, key, lbl, col, mk in (
        (-0.16, "M1a_minus_M0", "Direct Fusion (M1a - M0)", "#d62728", "^"),
        (0.00, "M1_unconstrained_minus_M0", "Unconstr. Gate (M1 - M0)", "#ff7f0e", "s"),
        (0.16, "M1_mono_q_minus_M0", "Monotonic Gate (M1_mono_q - M0)", "#2ca02c", "o"),
    ):
        ms, lo, hi = [], [], []
        for bk, _ in bins:
            d = ps[bk]["paired_top1_ci"][key]
            m = d["mean"]["ade_m"] * 100.0
            ms.append(m)
            lo.append(m - d["ci95"]["ade_m"][0] * 100.0)
            hi.append(d["ci95"]["ade_m"][1] * 100.0 - m)
        ax.errorbar(xs + off, ms, yerr=[lo, hi], fmt=mk, color=col, capsize=3.5, lw=1.5, ms=5.8, label=lbl)
    ax.axhline(0.0, color="#333333", ls="--", lw=1.1)
    ax.set_xticks(xs, [lbl for _, lbl in bins])
    ax.set_ylim(-1.5, 6.8)
    r5 = w20m["durations"]["T5"]["gate_audit"]["M1_mono_q"]["spearman_gate_vs_q1"]
    r10 = w20m["durations"]["T10"]["gate_audit"]["M1_mono_q"]["spearman_gate_vs_q1"]
    r20 = w20m["durations"]["T20"]["gate_audit"]["M1_mono_q"]["spearman_gate_vs_q1"]
    ax.text(0.28, 0.72, f"True 20f 3x3 M1_mono_q rho(g, q1):\nT=5: +{r5:.2f} | T=10: +{r10:.2f} | T=20: +{r20:.2f}",
            transform=ax.transAxes, fontsize=7.8, bbox={"boxstyle": "round,pad=0.32", "fc": "#eef7ee", "ec": "#2ca02c"})
    ax.set_ylabel("Paired Top-1 ADE_1 vs. Pure Motion M0 (cm)")
    ax.set_title("(c) Minimum Usable BBox-Height Boundary (< 50 px Degrades)", fontweight="bold", fontsize=9.8)
    ax.grid(axis="y", alpha=0.28)
    ax.legend(loc="upper right", fontsize=7.9, frameon=True)


def _panel_d_adaptive_conformal(ax: plt.Axes, w19c: dict) -> None:
    jc = w19c["jrdb_72pair_adaptive_conformal"]["cross_group_ordered_pairs_fixed_predictor_no_retrain"]
    cats = ["90% Marginal\nUnion-6 Cov.", "90% Fast (>=1m/s)\nUnion-6 Cov.", "95% Marginal\nUnion-6 Cov.", "95% Fast (>=1m/s)\nUnion-6 Cov."]
    c_const = [
        jc["90"]["M3_constant"]["u6_cov"] * 100,
        jc["90"]["M3_constant"]["fast_u6_cov"] * 100,
        jc["95"]["M3_constant"]["u6_cov"] * 100,
        jc["95"]["M3_constant"]["fast_u6_cov"] * 100,
    ]
    c_hyb = [
        jc["90"]["M3_speed_spread_scaled"]["u6_cov"] * 100,
        jc["90"]["M3_speed_spread_scaled"]["fast_u6_cov"] * 100,
        jc["95"]["M3_speed_spread_scaled"]["u6_cov"] * 100,
        jc["95"]["M3_speed_spread_scaled"]["fast_u6_cov"] * 100,
    ]
    xs = np.arange(len(cats))
    w = 0.36
    b1 = ax.bar(xs - w / 2, c_const, width=w, color="#8c564b", alpha=0.86, label="Constant Radius (0.749 / 1.069 m)")
    b2 = ax.bar(xs + w / 2, c_hyb, width=w, color="#2ca02c", alpha=0.88, label="Causal Speed+Spread (0.630 / 0.915 m)")
    ax.axhline(90.0, color="#555555", ls=":", lw=1.1)
    ax.axhline(95.0, color="#222222", ls="--", lw=1.1)
    for rect in (*b1, *b2):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.6)
    ax.set_xticks(xs, cats, fontsize=8.2)
    ax.set_ylim(68.0, 102.5)
    ax.set_ylabel("72-Pair Zero-Retrain Coverage (%)")
    ax.set_title("(d) JRDB 72-Pair Locally-Weighted Split-Conformal (No Retrain)", fontweight="bold", fontsize=9.9)
    ax.grid(axis="y", alpha=0.28)
    ax.legend(loc="lower right", fontsize=8.0, frameon=True)


def render_fig20(repo_root: Path) -> Path:
    res = repo_root / "results"
    w19e = json.loads((res / "wave19_dut_imptc_conflict_and_conformal_5seed_verified.json").read_text())
    w19u = json.loads((res / "wave19_unified_clean_m0_to_m3_5seed_verified.json").read_text())
    w19b = json.loads((res / "wave19_usable_pose_boundary_audit_5seed_verified.json").read_text())
    w19c = json.loads((res / "wave19_adaptive_conformal_12_and_72pair_5seed_verified.json").read_text())
    w20d = json.loads((res / "wave20_decoupled_m2b_m3_synergy_5seed_verified.json").read_text())
    w20m = json.loads((res / "wave20_true20f_mono_3x3_matrix_5seed_verified.json").read_text())

    plt.rcParams.update({"font.size": 9.3, "axes.titlesize": 10.0, "axes.labelsize": 9.3})
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.8), dpi=300)
    _panel_a_unified_and_decoupled(axes[0, 0], w19u, w20d)
    _panel_b_dut_imptc_zeroshot(axes[0, 1], w19e)
    _panel_c_usable_pose_boundary(axes[1, 0], w19b, w20m)
    _panel_d_adaptive_conformal(axes[1, 1], w19c)
    fig.suptitle("Waves 19-20 Synthesis: Decoupled Synergy, DUT/IMPTC Transfer, Usable Pose Boundary, & Adaptive Conformal",
                 fontsize=11.3, fontweight="bold", y=0.992)
    fig.tight_layout(rect=(0, 0, 1, 0.978))
    out = repo_root / "docs" / "figures" / "fig20_wave19_20_unified_dut_boundary_and_adaptive_conformal.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = render_fig20(Path(__file__).resolve().parents[1])
    print("Saved:", p, p.stat().st_size)
