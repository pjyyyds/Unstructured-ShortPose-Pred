"""Render 300-DPI Figure 18: 2x2 factorial, <0.5m regime bound, Slide-5 gate audit, & JRDB 72-pair conformal."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_fig18(repo_root: Path) -> Path:
    res = repo_root / "results"
    f1 = json.loads((res / "wave17_rc1_2x2_factorial_and_lowdisp_regime_5seed_verified.json").read_text())
    f3 = json.loads((res / "wave17_rc3_clean_chain_and_jrdb_conformal_5seed_verified.json").read_text())
    s5 = json.loads((res / "wave17_slide5_pose_noise_and_gate_response_5seed_verified.json").read_text())

    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5})
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.4), dpi=300)
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel (a): 2x2 (q x speed-damping) factorial paired effects (cm)
    keys = [
        ("q_at_no_v_M1q_minus_M1b", "+q (no speed damp)\nM1_q - M1b"),
        ("q_at_v_M1qv_minus_M1v", "+q (with speed damp)\nM1_qv - M1_v"),
        ("v_at_no_q_M1v_minus_M1b", "+speed damp (no q)\nM1_v - M1b"),
        ("v_at_q_M1qv_minus_M1q", "+speed damp (with q)\nM1_qv - M1_q"),
    ]
    y_pos = np.arange(len(keys))
    p_ci = f1["pinns_2x2_factorial"]["all_1482"]["paired_clip_ci"]
    j_ci = f1["jrdb_2x2_factorial"]["all_10scenes"]["paired_location_ci"]
    for off, (tag, src, col, mk) in ((-0.16, ("PINNS (1,482, 4-site clip CI)", p_ci, "#1f77b4", "o")),
                                     (0.16, ("JRDB (5,836, 9-loc CI)", j_ci, "#d62728", "s"))):
        means = [src[k]["mean"]["ade_m"] * 100.0 for k, _ in keys]
        los = [means[i] - src[k]["ci95"]["ade_m"][0] * 100.0 for i, (k, _) in enumerate(keys)]
        his = [src[k]["ci95"]["ade_m"][1] * 100.0 - means[i] for i, (k, _) in enumerate(keys)]
        ax_a.errorbar(means, y_pos + off, xerr=[los, his], fmt=mk, color=col, capsize=4, lw=1.6, ms=6, label=tag)
    ax_a.axvline(0.0, color="#333333", ls="--", lw=1.1)
    ax_a.set_yticks(y_pos, [lbl for _, lbl in keys])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Paired Top-1 ADE_1 Effect (cm, < 0 favors treatment)")
    ax_a.set_title("(a) Orthogonal 2x2 Factorial: Explicit q vs. Speed Damping", fontweight="bold")
    ax_a.grid(axis="x", alpha=0.28)
    ax_a.legend(loc="upper left", fontsize=8.4, frameon=True)

    # Panel (b): <0.5m first_short_323 emergence regime comparison (cm)
    r_sub = f1["pinns_lowdisp_regime_loso"]["first_short_323"]["models"]
    m_specs = [
        ("OLS5_CV", "OLS-5 CV", "#8c564b"),
        ("Last2_CV", "Last-2 CV\n(Ref)", "#2ca02c"),
        ("Regime_M0", "Regime M0\n(<0.5m train)", "#7f7f7f"),
        ("Regime_M1_ground", "Regime M1_gnd\n(3x3 facing)", "#1f77b4"),
        ("Regime_M1_full", "Regime M1_full\n(2D+ground)", "#9467bd"),
    ]
    xs = np.arange(len(m_specs))
    ades = [r_sub[k]["site_equal"]["ade1_m"] * 100.0 for k, _, _ in m_specs]
    fdes = [r_sub[k]["site_equal"]["fde1_m"] * 100.0 for k, _, _ in m_specs]
    w = 0.36
    b1 = ax_b.bar(xs - w / 2, ades, width=w, color="#2b5c8f", alpha=0.88, label="Top-1 ADE_1 (cm)")
    b2 = ax_b.bar(xs + w / 2, fdes, width=w, color="#e07b39", alpha=0.88, label="3.0 s Top-1 FDE_1 (cm)")
    ax_b.axhline(ades[1], color="#2b5c8f", ls=":", lw=1.1)
    ax_b.axhline(fdes[1], color="#e07b39", ls=":", lw=1.1)
    for rect in (*b1, *b2):
        h = rect.get_height()
        ax_b.text(rect.get_x() + rect.get_width() / 2, h + 1.0, f"{h:.1f}", ha="center", va="bottom", fontsize=7.6)
    ax_b.set_xticks(xs, [lbl for _, lbl, _ in m_specs])
    ax_b.set_ylim(0, 103)
    ax_b.set_ylabel("4-Site Equal Error on first_short_323 (cm)")
    ax_b.set_title("(b) < 0.5 m Emergence (323 Win.; M1_gnd vs. CV CI [-0.20, +0.26] cm)", fontweight="bold", fontsize=9.8)
    ax_b.grid(axis="y", alpha=0.28)
    ax_b.legend(loc="upper right", fontsize=8.5, frameon=True)

    # Panel (c): Slide-5 controlled degradation & m_pose=0 shutoff (PINNS & JRDB)
    ps5 = s5["pinns_4site_loso_1482"]
    js5 = s5["jrdb_9location_lolo_5836"]
    c_labels = [
        "VarQ - Clean\n(Test_Natural)",
        "VarQ - Clean\n(Test_Jitter020)",
        "m_pose=0 Shutoff\nvs. No-Shutoff",
    ]
    p_vals = [
        ps5["slide5_social_pose_comparisons"]["VarQTrain_minus_CleanTrain_M1_Test_Natural"],
        ps5["slide5_social_pose_comparisons"]["VarQTrain_minus_CleanTrain_M1_Test_Jitter020"],
        ps5["train_test_matrix"]["VarQTrain"]["Test_FullDrop"]["M1_shutoff_minus_M1a_no_shutoff"],
    ]
    j_vals = [
        js5["slide5_social_pose_comparisons"]["VarQTrain_minus_CleanTrain_M1_Test_Natural"]["person_equal"],
        js5["slide5_social_pose_comparisons"]["VarQTrain_minus_CleanTrain_M1_Test_Jitter020"]["person_equal"],
        js5["train_test_matrix"]["VarQTrain"]["Test_FullDrop"]["M1_shutoff_minus_M1a_no_shutoff"]["person_equal"],
    ]
    yc = np.arange(len(c_labels))
    for off, tag, arrs, col, mk in ((-0.16, "PINNS (47-clip CI)", p_vals, "#1f77b4", "o"),
                                    (0.16, "JRDB (800-person CI)", j_vals, "#2ca02c", "D")):
        ms = [x["mean"]["ade_m"] * 100.0 for x in arrs]
        lo = [ms[i] - x["ci95"]["ade_m"][0] * 100.0 for i, x in enumerate(arrs)]
        hi = [x["ci95"]["ade_m"][1] * 100.0 - ms[i] for i, x in enumerate(arrs)]
        ax_c.errorbar(ms, yc + off, xerr=[lo, hi], fmt=mk, color=col, capsize=4, lw=1.6, ms=6, label=tag)
    ax_c.axvline(0.0, color="#333333", ls="--", lw=1.1)
    rp_rho = ps5["gate_response_audit"]["VarQTrain"]["natural_strata_M1"]["spearman_gate_vs_q1_valid_joint_ratio"]
    rj_rho = js5["gate_response_audit"]["VarQTrain"]["natural_strata_M1"]["spearman_gate_vs_q1_valid_joint_ratio"]
    ax_c.set_ylim(-0.45, 2.55)
    ax_c.invert_yaxis()
    ax_c.text(0.45, 0.06, f"Unconstrained M1 Gate Rank Inversion:\nSpearman rho(g, q1) = {rp_rho:.3f} (PINNS)\nSpearman rho(g, q1) = {rj_rho:.3f} (JRDB)",
              transform=ax_c.transAxes, fontsize=8.1, bbox={"boxstyle": "round,pad=0.35", "fc": "#fff6e6", "ec": "#d99b26"})
    ax_c.set_yticks(yc, c_labels)
    ax_c.set_xlabel("Paired Top-1 ADE_1 Difference (cm, < 0 favors VarQ / Shutoff)")
    ax_c.set_title("(c) Slide-5 Pose Stress, m_pose=0 Shutoff, & Gate Inversion", fontweight="bold", fontsize=9.9)
    ax_c.grid(axis="x", alpha=0.28)
    ax_c.legend(loc="upper right", fontsize=8.5, frameon=True)

    # Panel (d): JRDB Fixed-Predictor Split-Conformal (Within-Scene & 72-Pair Cross-Location)
    jc = f3["jrdb_fixed_predictor_split_conformal"]
    ws = jc["within_scene_temporal_split_9loc_lolo"]["M2b"]
    cl = jc["cross_location_72pair_fixed_7loc_predictor_no_retrain"]
    cats = ["Within-Scene\n90% (Fixed 8-Loc)", "Within-Scene\n95% (Fixed 8-Loc)", "72-Pair Cross-Loc\n90% (Fixed 7-Loc)", "72-Pair Cross-Loc\n95% (Fixed 7-Loc)"]
    cov_cv = [ws["90"]["cov_cv"] * 100, ws["95"]["cov_cv"] * 100, cl["90"]["cov_cv"] * 100, cl["95"]["cov_cv"] * 100]
    cov_u6 = [ws["90"]["cov_u6"] * 100, ws["95"]["cov_u6"] * 100, cl["90"]["cov_u6"] * 100, cl["95"]["cov_u6"] * 100]
    rad_red = [(1.0 - ws["90"]["rad_u6"] / ws["90"]["rad_cv"]) * 100, (1.0 - ws["95"]["rad_u6"] / ws["95"]["rad_cv"]) * 100,
               (1.0 - cl["90"]["rad_u6"] / cl["90"]["rad_cv"]) * 100, (1.0 - cl["95"]["rad_u6"] / cl["95"]["rad_cv"]) * 100]
    xd = np.arange(len(cats))
    b_cv = ax_d.bar(xd - 0.18, cov_cv, width=0.34, color="#9aa5b1", label="Last-2 CV Coverage (%)")
    b_u6 = ax_d.bar(xd + 0.18, cov_u6, width=0.34, color="#1b9e77", label="O2 Union-6 Coverage (%, No Retrain)")
    ax_d.axhline(90.0, color="#555555", ls="--", lw=0.9)
    ax_d.axhline(95.0, color="#222222", ls=":", lw=0.9)
    for i, rect in enumerate(b_u6):
        h = rect.get_height()
        ax_d.text(rect.get_x() + rect.get_width() / 2, h + 0.6, f"{h:.1f}%\n(-{rad_red[i]:.1f}% r)", ha="center", va="bottom", fontsize=7.5, color="#0f5c44", fontweight="bold")
    ax_d.set_xticks(xd, cats, fontsize=8.4)
    ax_d.set_ylim(75, 102)
    ax_d.set_ylabel("Empirical Coverage (%)")
    ax_d.set_title("(d) JRDB Fixed-Predictor Split-Conformal Coverage & Radius", fontweight="bold", fontsize=9.8)
    ax_d.grid(axis="y", alpha=0.28)
    ax_d.legend(loc="lower right", fontsize=8.3, frameon=True)

    fig.tight_layout(pad=1.6)
    out = repo_root / "docs/figures/fig18_wave17_factorial_slide5_and_jrdb_conformal.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = render_fig18(Path(__file__).resolve().parents[1])
    print("Saved 300-DPI Figure 18:", p)
