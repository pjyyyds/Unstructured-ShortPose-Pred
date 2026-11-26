"""Render 300-DPI Figure 19: conflict gate, external baselines (Raw vs Frenet), monotonic gate, & T4 Pareto."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _unwrap_ci(item: dict) -> dict:
    return item["location_equal"] if "location_equal" in item else item


def _panel_a_conflict_gate(ax: plt.Axes, cg: dict) -> None:
    strata = [
        ("all", "All Windows"),
        ("S1_non_approaching", "Non-Approaching\nNeighbors (S1)"),
        ("S2_close_approaching", "Close Approaching\nd<=5m, v_rad<0 (S2)"),
        ("S3_ttc_lt4s", "High-Risk\nTTC <= 4s (S3)"),
        ("S4_approaching_car", "Approaching\nVehicle (S4)"),
    ]
    y = np.arange(len(strata))
    ps = cg["pinns_4site_loso_1482"]["strata"]
    js = cg["jrdb_9location_lolo_5836"]["strata"]
    specs = [
        (-0.22, "PINNS: Un-gated M2b - M1", ps, "M2b_frozen_minus_M1", "#d62728", "x"),
        (0.00, "PINNS: Conflict-Gated M2b - M1", ps, "M2b_conflict_gate_minus_M1", "#1f77b4", "o"),
        (0.22, "JRDB: Conflict-Gated - Un-gated M2b", js, "M2b_conflict_gate_minus_M2b_frozen", "#2ca02c", "s"),
    ]
    for off, lbl, src, comp_key, col, mk in specs:
        ms, lo, hi, ys = [], [], [], []
        for idx, (sk, _) in enumerate(strata):
            if sk not in src or comp_key not in src[sk]["paired_top1_ci"]:
                continue
            item = _unwrap_ci(src[sk]["paired_top1_ci"][comp_key])
            if item.get("ci95") is None:
                continue
            m = item["mean"]["ade_m"] * 100.0
            ms.append(m)
            lo.append(m - item["ci95"]["ade_m"][0] * 100.0)
            hi.append(item["ci95"]["ade_m"][1] * 100.0 - m)
            ys.append(y[idx] + off)
        ax.errorbar(ms, ys, xerr=[lo, hi], fmt=mk, color=col, capsize=3.5, lw=1.5, ms=5.8, label=lbl)
    ax.axvline(0.0, color="#333333", ls="--", lw=1.1)
    ax.set_yticks(y, [lbl for _, lbl in strata])
    ax.set_ylim(-0.45, 5.15)
    ax.invert_yaxis()
    ax.set_xlabel("Paired Top-1 ADE_1 Difference (cm, < 0 favors gated interaction)")
    ax.set_title("(a) Causal Radial-Conflict Gate (M2b_ttc_conflict_gate, 5 Seeds)", fontweight="bold", fontsize=10.0)
    ax.grid(axis="x", alpha=0.28)
    ax.legend(loc="lower right", fontsize=8.0, frameon=True)


def _panel_b_external_baselines(ax: plt.Axes, ext: dict, cg: dict) -> None:
    p_mod = ext["pinns_4site_loso_1482"]["strata"]["all_1482"]
    j_mod = ext["jrdb_9location_lolo_5836"]["strata"]["all_10scenes"]
    p_cg = cg["pinns_4site_loso_1482"]["strata"]["all"]["models"]["M2b_ttc_conflict_gate"]["group_equal"]["ade1_m"]
    cats = ["MCENET\n(Raw)", "MCENET\n(+Frenet)", "HST\n(Raw)", "HST\n(+Frenet)", "Soc.-Trans.\n(Raw)", "Soc.-Trans.\n(+Frenet)", "Last-2\nCV", "Ours\nO2 Best"]
    p_vals = [
        p_mod["Ext1_MCENET"]["group_equal"]["ade1_m"] * 100,
        p_mod["Ext1_MCENET_Frenet"]["group_equal"]["ade1_m"] * 100,
        p_mod["Ext2_HST"]["group_equal"]["ade1_m"] * 100,
        p_mod["Ext2_HST_Frenet"]["group_equal"]["ade1_m"] * 100,
        p_mod["Ext3_SocialTransmotion"]["group_equal"]["ade1_m"] * 100,
        p_mod["Ext3_SocialTransmotion_Frenet"]["group_equal"]["ade1_m"] * 100,
        p_mod["Last2_CV"]["group_equal"]["ade1_m"] * 100,
        p_cg * 100,
    ]
    j_vals = [
        j_mod["Ext1_MCENET"]["group_equal"]["ade1_m"] * 100,
        j_mod["Ext1_MCENET_Frenet"]["group_equal"]["ade1_m"] * 100,
        j_mod["Ext2_HST"]["group_equal"]["ade1_m"] * 100,
        j_mod["Ext2_HST_Frenet"]["group_equal"]["ade1_m"] * 100,
        j_mod["Ext3_SocialTransmotion"]["group_equal"]["ade1_m"] * 100,
        j_mod["Ext3_SocialTransmotion_Frenet"]["group_equal"]["ade1_m"] * 100,
        j_mod["Last2_CV"]["group_equal"]["ade1_m"] * 100,
        j_mod["O2_M3a_ref"]["group_equal"]["ade1_m"] * 100,
    ]
    xs = np.arange(len(cats))
    w = 0.37
    b1 = ax.bar(xs - w / 2, p_vals, width=w, color="#2b5c8f", alpha=0.88, label="PINNS 4-Site LOSO (3.0 s)")
    b2 = ax.bar(xs + w / 2, j_vals, width=w, color="#e07b39", alpha=0.88, label="JRDB 9-Loc LOLO (2.8 s)")
    for rect in (*b1, *b2):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.0, f"{h:.1f}", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(xs, cats, fontsize=8.0)
    ax.set_ylim(0, 102)
    ax.set_ylabel("Cross-Site Top-1 ADE_1 (cm)")
    ax.set_title("(b) External Baselines: Raw Scene-Axis vs. SO(2) Frenet vs. Ours", fontweight="bold", fontsize=10.0)
    ax.grid(axis="y", alpha=0.28)
    ax.legend(loc="upper right", fontsize=8.3, frameon=True)


def _panel_c_monotonic_and_rerank(ax: plt.Axes, t4: dict, ar: dict) -> None:
    pm = t4["pinns_monotonic_gate_1482"]
    jm = t4["jrdb_monotonic_gate_5836"]
    conds = [
        ("Test_Natural", "Natural\nTest"),
        ("Test_Jitter020", "20 px Keypoint\nJitter"),
        ("Test_LowerOcc", "Lower-Body\nOcclusion"),
    ]
    yc = np.arange(len(conds))
    for off, lbl, src, col, mk in (
        (-0.16, "PINNS: M1_mono_q - M1_unconstr.", pm["test_conditions"], "#1f77b4", "o"),
        (0.16, "JRDB: M1_mono_q - M1_unconstr.", jm["test_conditions"], "#2ca02c", "D"),
    ):
        ms, lo, hi = [], [], []
        for ck, _ in conds:
            d = _unwrap_ci(src[ck]["M1_mono_minus_M1_unconstrained"])
            m = d["mean"]["ade_m"] * 100.0
            ms.append(m)
            lo.append(m - d["ci95"]["ade_m"][0] * 100.0)
            hi.append(d["ci95"]["ade_m"][1] * 100.0 - m)
        ax.errorbar(ms, yc + off, xerr=[lo, hi], fmt=mk, color=col, capsize=4, lw=1.6, ms=6.0, label=lbl)
    ax.axvline(0.0, color="#333333", ls="--", lw=1.1)
    ax.set_yticks(yc, [lbl for _, lbl in conds])
    ax.set_ylim(-0.48, 2.65)
    ax.invert_yaxis()
    pr0 = pm["gate_audit"]["M1_unconstrained"]["spearman_gate_vs_q1_valid_joint_ratio"]
    pr1 = pm["gate_audit"]["M1_mono_q"]["spearman_gate_vs_q1_valid_joint_ratio"]
    jr0 = jm["gate_audit"]["M1_unconstrained"]["spearman_gate_vs_q1_valid_joint_ratio"]
    jr1 = jm["gate_audit"]["M1_mono_q"]["spearman_gate_vs_q1_valid_joint_ratio"]
    pe0 = ar["pinns_4site_loso_1482"]["action_metrics_raw_thr05"]["group_equal"]["mean_ece"]
    pe1 = ar["pinns_4site_loso_1482"]["action_metrics_platt_thr05"]["group_equal"]["mean_ece"]
    je0 = ar["jrdb_9location_lolo_5836"]["action_metrics_raw_thr05"]["group_equal"]["mean_ece"]
    je1 = ar["jrdb_9location_lolo_5836"]["action_metrics_platt_thr05"]["group_equal"]["mean_ece"]
    box_txt = (
        f"Spearman rho(g, q1) Restored:\n"
        f"  PINNS: {pr0:+.2f} -> {pr1:+.2f} | JRDB: {jr0:+.2f} -> {jr1:+.2f}\n"
        f"5-Action Platt ECE Calibration:\n"
        f"  PINNS: {pe0:.3f} -> {pe1:.3f} | JRDB: {je0:.3f} -> {je1:.4f}"
    )
    ax.text(0.03, 0.06, box_txt, transform=ax.transAxes, fontsize=7.9,
            bbox={"boxstyle": "round,pad=0.35", "fc": "#eef7ee", "ec": "#2ca02c"})
    ax.set_xlabel("Paired Top-1 ADE_1 Difference (cm, < 0 favors M1_mono_q)")
    ax.set_title("(c) Monotonic Gate Remedy (M1_mono_q) & Action Calibration", fontweight="bold", fontsize=10.0)
    ax.grid(axis="x", alpha=0.28)
    ax.legend(loc="upper left", fontsize=8.2, frameon=True)


def _panel_d_t4_pareto(ax: plt.Axes, t4: dict, ext: dict, cg: dict) -> None:
    b12 = t4["hardware_profile"]["batch_sizes"]["B12"]
    pm = ext["pinns_4site_loso_1482"]["strata"]["all_1482"]
    cg_m = cg["pinns_4site_loso_1482"]["strata"]["all"]["models"]
    pts = [
        ("O2_M0", t4["pinns_monotonic_gate_1482"]["test_conditions"]["Test_Natural"]["M0"]["group_equal_ade1_m"], "#1f77b4", "o", (-8, -14), "O2-M0 (28k, 0.17M)"),
        ("O2_M1_mono_q", t4["pinns_monotonic_gate_1482"]["test_conditions"]["Test_Natural"]["M1_mono_q"]["group_equal_ade1_m"], "#2ca02c", "D", (-10, 9), "O2-M1_mono_q (28k, 0.32M)"),
        ("O2_M2b_conflict_gate", cg_m["M2b_ttc_conflict_gate"]["group_equal"]["ade1_m"], "#d62728", "*", (-12, -16), "O2-M2b_conflict (28k, 1.56M)"),
        ("O2_M3", pm["O2_M3_ref"]["group_equal"]["ade1_m"], "#9467bd", "s", (-42, 8), "O2-M3 (28k, 9.47M)"),
        ("Ext1_MCENET", pm["Ext1_MCENET_Frenet"]["group_equal"]["ade1_m"], "#8c564b", "^", (-54, 7), "MCENET+Frenet (216k, 299M)"),
        ("Ext2_HST", pm["Ext2_HST_Frenet"]["group_equal"]["ade1_m"], "#e377c2", "v", (-36, -15), "HST+Frenet (508k, 618M)"),
        ("Ext3_SocialTransmotion", pm["Ext3_SocialTransmotion_Frenet"]["group_equal"]["ade1_m"], "#7f7f7f", "P", (-52, 7), "Soc.-Trans.+Frenet (517k, 384M)"),
    ]
    for key, ade_m, col, mk, xyoff, lbl in pts:
        p50 = b12[key]["p50_ms"]
        p95 = b12[key]["p95_ms"]
        ade_cm = ade_m * 100.0
        ax.errorbar([p50], [ade_cm], xerr=[[0.0], [p95 - p50]], fmt=mk, color=col, ms=8.5 if mk == "*" else 6.8, capsize=3.5, lw=1.4)
        ax.annotate(lbl, (p50, ade_cm), xytext=xyoff, textcoords="offset points", fontsize=7.6,
                    fontweight="bold" if key == "O2_M2b_conflict_gate" else "normal")
    ax.axhline(pm["Last2_CV"]["group_equal"]["ade1_m"] * 100.0, color="#444444", ls=":", lw=1.1, label="Last-2 CV (48.39 cm)")
    ax.set_xlim(2.3, 10.6)
    ax.set_ylim(43.8, 65.0)
    ax.set_xlabel("Real Tesla T4 Batch-12 Forward Latency: P50 -> P95 Whisker (ms)")
    ax.set_ylabel("PINNS Cross-Site Top-1 ADE_1 (cm)")
    ax.set_title("(d) Tesla T4 Accuracy-Latency-FLOPs Pareto (B = 12 Agents)", fontweight="bold", fontsize=10.0)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left", fontsize=8.2, frameon=True)


def render_fig19(repo_root: Path) -> Path:
    res = repo_root / "results"
    cg = json.loads((res / "wave18_conflict_gated_interaction_5seed_verified.json").read_text())
    ar = json.loads((res / "wave18_action_calibration_and_rerank_5seed_verified.json").read_text())
    ext = json.loads((res / "wave17_ppt_ext_baselines_loso_5seed_verified.json").read_text())
    t4 = json.loads((res / "wave18_t4_latency_and_monotonic_gate_5seed_verified.json").read_text())

    plt.rcParams.update({"font.size": 9.3, "axes.titlesize": 10.0, "axes.labelsize": 9.3})
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.8), dpi=300)
    _panel_a_conflict_gate(axes[0, 0], cg)
    _panel_b_external_baselines(axes[0, 1], ext, cg)
    _panel_c_monotonic_and_rerank(axes[1, 0], t4, ar)
    _panel_d_t4_pareto(axes[1, 1], t4, ext, cg)
    fig.suptitle("Wave 18 PPT Mainline Synthesis: Causal Conflict Gate, External Baselines, Monotonic Gate, & Tesla T4 Pareto",
                 fontsize=11.4, fontweight="bold", y=0.992)
    fig.tight_layout(rect=(0, 0, 1, 0.978))
    out = repo_root / "docs" / "figures" / "fig19_wave18_conflict_rerank_ext_and_t4_pareto.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = render_fig19(Path(__file__).resolve().parents[1])
    print("Saved:", p, p.stat().st_size)
