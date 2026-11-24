"""Generate 5 publication-grade 300-DPI figures from `master_experiment_report.json`."""

from pathlib import Path
from typing import Dict, List
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

PALETTE = {
    "M0": "#7F8C8D",
    "M1a": "#E74C3C",
    "M1b": "#F39C12",
    "M1": "#2980B9",
    "M1p": "#16A085",
    "M3": "#2C3E50",
    "Ours_Full": "#6C3483",
    "Ext1_MCENET": "#95A5A6",
    "Ext2_HST": "#D35400",
    "Ext3_SocialTransmotion": "#27AE60",
}


def _get_m(summaries: Dict, tag: str, sl: str = "Overall", metric: str = "ADE_1") -> tuple:
    d = summaries.get(tag, {}).get("slices", {}).get(sl, {})
    return float(d.get(f"{metric}_mean", np.nan)), float(d.get(f"{metric}_std", 0.0))


def plot_fig1_window_bucket_matrix(summaries: Dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharey=False)
    models = [("M0", "M0 (Motion)"), ("M1a", "M1a (Concat)"), ("M1b", "M1b (No-q)"), ("M1", "M1 (q-Gate)"), ("M1p", "M1p (BPD-Gate)")]
    buckets = ["Clear", "Partial", "Heavy"]
    x = np.arange(len(buckets))
    w = 0.15

    for ax_i, t_win in enumerate((5, 10, 20)):
        ax = axes[ax_i]
        for m_i, (m_code, m_lbl) in enumerate(models):
            means, stds = [], []
            for bk in buckets:
                mu, sd = _get_m(summaries, f"PINNS_T{t_win}_{m_code}", sl=bk, metric="ADE_1")
                means.append(mu); stds.append(sd)
            ax.bar(
                x + (m_i - 2) * w,
                means,
                width=w,
                yerr=stds,
                capsize=2.2,
                label=m_lbl if ax_i == 0 else None,
                color=PALETTE[m_code],
                alpha=0.92,
            )
        tau_s = (t_win - 1) * 0.1
        ax.set_title(f"Window T={t_win} ({tau_s:.1f}s Obs)")
        ax.set_xticks(x)
        ax.set_xticklabels(["Clear\nBucket", "Partial\nOccl.", "Heavy\nOccl."])
        ax.set_ylim(0.34, 0.61)
        if ax_i == 0:
            ax.set_ylabel("Top-1 ADE_1 (m) [Lower is Better]")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.legend(loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    dst = out_dir / "fig1_window_quality_3x3_matrix.png"
    fig.savefig(dst, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return dst


def plot_fig2_staircase_and_gates(summaries: Dict, out_dir: Path, v1_sums: Dict = None) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.3), gridspec_kw={"width_ratios": [1.55, 1.0]})
    order = ["M0", "M1a", "M1b", "M1", "M1p", "M2a", "M2b_norad", "M2b", "M2", "M3a", "M3b", "M3", "M3p", "Ours_Full"]
    ade_m, ade_s = [], []
    for m in order:
        mu, sd = _get_m(summaries, f"PINNS_T5_{m}", "Overall", "ADE_1")
        ade_m.append(mu); ade_s.append(sd)

    xs = np.arange(len(order))
    if v1_sums:
        v1_m = [_get_m(v1_sums, f"PINNS_T5_{m}", "Overall", "ADE_1")[0] for m in order]
        ax1.bar(xs - 0.18, v1_m, width=0.35, color="#BDC3C7", alpha=0.75, label="V1 Initial (Un-gated / Hard CE)")
        colors = ["#6C3483" if m == "Ours_Full" else ("#E74C3C" if m == "M1a" else "#2980B9") for m in order]
        ax1.bar(xs + 0.18, ade_m, width=0.35, yerr=ade_s, capsize=2.0, color=colors, alpha=0.92, label="V2 Refined (Null-Sink + Soft-Rank)")
    else:
        colors = ["#6C3483" if m == "Ours_Full" else ("#E74C3C" if m == "M1a" else "#2980B9") for m in order]
        ax1.bar(xs, ade_m, yerr=ade_s, capsize=2.5, color=colors, alpha=0.9)
    ax1.axhline(ade_m[0], color="#2C3E50", linestyle="--", linewidth=1.1, label="V2 M0 Motion Baseline (0.486m)")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(order, rotation=38, ha="right")
    ax1.set_ylim(0.42, 0.66)
    ax1.set_ylabel("PINNS T=5 Overall ADE_1 (m)")
    ax1.set_title("(a) V1 -> V2 Diagnostic Iteration & 5-Seed Ablation")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    buckets = ["Clear", "Partial", "Heavy"]
    bx = np.arange(len(buckets))
    g_m1 = [summaries.get("PINNS_T5_M1", {}).get("gate_by_bucket", {}).get(b, [0.0])[0] for b in buckets]
    g_p = summaries.get("PINNS_T5_M1p", {}).get("gate_by_bucket", {})
    gh = [g_p.get(b, [0, 0, 0])[0] for b in buckets]
    gt = [g_p.get(b, [0, 0, 0])[1] for b in buckets]
    gl = [g_p.get(b, [0, 0, 0])[2] for b in buckets]

    bw = 0.18
    ax2.bar(bx - 1.5 * bw, g_m1, width=bw, label="M1 Explicit q-Gate g", color="#2980B9")
    ax2.bar(bx - 0.5 * bw, gh, width=bw, label="M1p Head-Shoulder g_hs", color="#16A085")
    ax2.bar(bx + 0.5 * bw, gt, width=bw, label="M1p Torso g_tr", color="#F39C12")
    ax2.bar(bx + 1.5 * bw, gl, width=bw, label="M1p Lower-Limb g_lg", color="#8E44AD")
    ax2.set_xticks(bx)
    ax2.set_xticklabels(buckets)
    ax2.set_ylim(0.0, 0.175)
    ax2.set_ylabel("Mean Effective Gate Weight")
    ax2.set_title("(b) Monotonic Quality-Regulated Gate Response")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    dst = out_dir / "fig2_staircase_and_gates.png"
    fig.savefig(dst, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return dst


def plot_fig3_scene_and_action_bridges(summaries: Dict, out_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.0))

    # Left: Research 2 Map & Graph on ADE_1 and Boundary Violation %
    r2_mods = ["M1", "M2a", "M2b_norad", "M2b", "M2"]
    rx = np.arange(len(r2_mods))
    ade_vals = [_get_m(summaries, f"PINNS_T5_{m}", "Overall", "ADE_1")[0] for m in r2_mods]
    viol_vals = [100.0 * _get_m(summaries, f"PINNS_T5_{m}", "Overall", "BoundaryViol")[0] for m in r2_mods]

    ax1_twin = ax1.twinx()
    ax1_twin.spines["top"].set_visible(False)
    b1 = ax1.bar(rx - 0.17, ade_vals, width=0.34, color="#2980B9", label="ADE_1 (m)")
    b2 = ax1_twin.bar(rx + 0.17, viol_vals, width=0.34, color="#E67E22", alpha=0.88, label="Boundary Viol. (%)")
    ax1.set_xticks(rx)
    ax1.set_xticklabels(["M1\n(Indiv.)", "M2a\n(+Map)", "M2b\n(w/o v_rad)", "M2b\n(+v_rad)", "M2\n(Map+GAT)"])
    ax1.set_ylim(0.43, 0.56)
    ax1_twin.set_ylim(0.9, 2.05)
    ax1.set_ylabel("ADE_1 (m)", color="#2980B9")
    ax1_twin.set_ylabel("Boundary Violation Rate (%)", color="#E67E22")
    ax1.set_title("(a) Non-Lane Polylines & Null-Sink GATv2")
    lines = [b1, b2]
    ax1.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper left")

    # Right: Research 3 Action Bridges on Transition ADE_1 and Macro-F1
    r3_mods = ["M2", "M3a", "M3b", "M3", "M3p", "Ours_Full"]
    tx = np.arange(len(r3_mods))
    tr_ade = [_get_m(summaries, f"PINNS_T5_{m}", "Transition", "ADE_1")[0] for m in r3_mods]
    act_f1 = [100.0 * float(summaries.get(f"PINNS_T5_{m}", {}).get("action_macro_f1_mean", 0.0)) for m in r3_mods]

    ax2_twin = ax2.twinx()
    ax2_twin.spines["top"].set_visible(False)
    c1 = ax2.bar(tx - 0.17, tr_ade, width=0.34, color="#6C3483", alpha=0.88, label="Transition ADE_1 (m)")
    c2 = ax2_twin.plot(tx[1:], act_f1[1:], marker="o", linewidth=2.2, color="#16A085", label="Action Macro-F1 (%)")
    ax2.set_xticks(tx)
    ax2.set_xticklabels(["M2", "M3a\n(Aux)", "M3b\n(E2E)", "M3\n(StopG)", "M3p\n(+ACMC)", "Ours\nFull"])
    ax2.set_ylim(0.66, 0.81)
    ax2_twin.set_ylim(25.0, 35.5)
    ax2.set_ylabel("State-Transition Subset ADE_1 (m)", color="#6C3483")
    ax2_twin.set_ylabel("5-Class Action Macro-F1 (%)", color="#16A085")
    ax2.set_title("(b) Stop-Grad Bridge Preserves Action Macro-F1")
    h_l = [c1, c2[0]]
    ax2.legend(h_l, [h.get_label() for h in h_l], frameon=False, loc="upper left", fontsize=8)

    fig.tight_layout()
    dst = out_dir / "fig3_scene_and_action_bridges.png"
    fig.savefig(dst, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return dst


def plot_fig4_sota_and_jta(summaries: Dict, out_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.2))

    # Left: PINNS Multi-window comparison vs External SOTA (including Proposed M1 & M3)
    wins = [5, 10, 20]
    wx = np.array([0.4, 0.9, 1.9])
    for m_code, lbl, mk, lw in [
        ("Ext1_MCENET", "MCENET (ITSC'20)", "^-", 1.8),
        ("Ext3_SocialTransmotion", "Social-Transmotion (ICLR'24)", "v-", 1.8),
        ("Ours_Full", "Ours Full (Map+GAT+ACMC)", "*-", 1.9),
        ("M3", "Proposed M3 (Stop-Grad)", "P-", 1.9),
        ("Ext2_HST", "HST (IEEE RA-L'23)", "d-", 1.8),
        ("M0", "M0 (Masked Motion GRU)", "s--", 1.8),
        ("M1", "Proposed M1 (Explicit q-Gate)", "o-", 2.6),
    ]:
        ys = [_get_m(summaries, f"PINNS_T{t}_{m_code}", "Overall", "ADE_1")[0] for t in wins]
        ax1.plot(wx, ys, mk, linewidth=lw, markersize=6.5, color=PALETTE[m_code], label=lbl)
    ax1.set_xticks(wx)
    ax1.set_xticklabels(["0.4s (T=5)", "0.9s (T=10)", "1.9s (T=20)"])
    ax1.set_ylim(0.455, 0.565)
    ax1.set_ylabel("PINNS Test ADE_1 (m) [Lower is Better]")
    ax1.set_title("(a) Multi-Window SOTA Comparison on PINNS")
    ax1.legend(frameon=False, fontsize=7.6, ncol=2, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Right: JTA 0.4s Clean / Partial / Heavy (sigma=0.20)
    j_mods = ["M0", "M1a", "Ext2_HST", "Ext3_SocialTransmotion", "M1", "M1p", "Ours_Full"]
    jx = np.arange(len(j_mods))
    jw = 0.25
    for bi, (bk, col) in enumerate([("Clear", "#2ECC71"), ("Partial", "#F39C12"), ("Heavy", "#E74C3C")]):
        vals = [_get_m(summaries, f"JTA_T2_{m}", bk, "ADE_1")[0] for m in j_mods]
        ax2.bar(jx + (bi - 1) * jw, vals, width=jw, label=f"{bk} Bucket", color=col, alpha=0.9)
    ax2.set_xticks(jx)
    ax2.set_xticklabels(["M0", "M1a\nConcat", "HST", "Social\nTransm.", "M1", "M1p\nBPD", "Ours\nFull"])
    ax2.set_ylim(1.15, 2.72)
    ax2.set_ylabel("JTA (0.4s Obs) ADE_1 (m)")
    ax2.set_title("(b) Cross-Dataset Generalization on JTA (0.4s)")
    ax2.legend(frameon=False, fontsize=8.2)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    dst = out_dir / "fig4_sota_and_jta_generalization.png"
    fig.savefig(dst, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return dst


def plot_fig5_latency_and_sensitivity(report: Dict, out_dir: Path, v1_rep: Dict = None) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.0))
    lat = report.get("latency_cuda", {})
    agents = [1, 4, 8, 12]
    if v1_rep:
        v1_lat = v1_rep.get("latency_cuda", {})
        v1_t5 = [float(v1_lat.get("T=5", {}).get(f"N={n}", {}).get("mean_ms", 0.0)) for n in agents]
        ax1.plot(agents, v1_t5, "^:", color="#95A5A6", linewidth=1.6, label="V1 Unfused T=5 Mean")
    for t_win, col in [(5, "#6C3483"), (10, "#2980B9"), (20, "#16A085")]:
        means = [float(lat.get(f"T={t_win}", {}).get(f"N={n}", {}).get("mean_ms", 0.0)) for n in agents]
        p95s = [float(lat.get(f"T={t_win}", {}).get(f"N={n}", {}).get("p95_ms", 0.0)) for n in agents]
        ax1.plot(agents, means, "o-", color=col, linewidth=2.1, label=f"V2 JIT T={t_win} Mean")
        ax1.plot(agents, p95s, "s--", color=col, alpha=0.50, linewidth=1.1, label=f"V2 JIT T={t_win} P95")
    ax1.axhline(25.0, color="#E74C3C", linestyle="--", linewidth=1.4, label="25 ms Edge Real-Time Target")
    ax1.set_xticks(agents)
    ax1.set_ylim(12.5, 26.8)
    ax1.set_xlabel("Concurrent Target Pedestrians N per Scene")
    ax1.set_ylabel("Tesla T4 Inference Latency (ms)")
    ax1.set_title("(a) Multi-Target Real-Time Scaling on Tesla T4")
    ax1.legend(frameon=False, ncol=2, fontsize=7.5, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.35)

    summaries = report.get("summaries", {})
    scales = [(50, "0.50x\n(-50%)"), (75, "0.75x\n(-25%)"), (100, "1.00x\n(Nominal)"), (125, "1.25x\n(+25%)"), (150, "1.50x\n(+50%)")]
    sx = np.arange(len(scales))
    s_means, s_stds = [], []
    for sc, _ in scales:
        tag = "PINNS_T5_Ours_Full" if sc == 100 else f"SENS_{sc}_Ours_Full"
        mu, sd = _get_m(summaries, tag, "Overall", "ADE_1")
        s_means.append(mu); s_stds.append(sd)
    ax2.errorbar(sx, s_means, yerr=s_stds, fmt="o-", color="#6C3483", capsize=4, linewidth=2.0, label="Ours_Full (V2)")
    ax2.axhline(_get_m(summaries, "PINNS_T5_Ext1_MCENET", "Overall", "ADE_1")[0], color="#95A5A6", linestyle="--", label="MCENET Baseline (0.552m)")
    ax2.set_xticks(sx)
    ax2.set_xticklabels([lbl for _, lbl in scales])
    ax2.set_ylim(0.495, 0.578)
    ax2.set_ylabel("PINNS T=5 ADE_1 (m)")
    ax2.set_title("(b) Action Event Threshold Sensitivity (+-25% / +-50%)")
    ax2.legend(frameon=False, fontsize=8.2, loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    dst = out_dir / "fig5_latency_and_sensitivity.png"
    fig.savefig(dst, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return dst


def generate_all_figures(report_json: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = json.loads(report_json.read_text(encoding="utf-8"))
    v1_path = report_json.parent / "v1_initial_diagnosis" / "master_experiment_report.json"
    v1_rep = json.loads(v1_path.read_text(encoding="utf-8")) if v1_path.exists() else None
    v1_sums = v1_rep.get("summaries") if v1_rep else None
    sums = rep["summaries"]
    return [
        plot_fig1_window_bucket_matrix(sums, out_dir),
        plot_fig2_staircase_and_gates(sums, out_dir, v1_sums=v1_sums),
        plot_fig3_scene_and_action_bridges(sums, out_dir),
        plot_fig4_sota_and_jta(sums, out_dir),
        plot_fig5_latency_and_sensitivity(rep, out_dir, v1_rep=v1_rep),
    ]


if __name__ == "__main__":
    import sys
    r_path = Path(sys.argv[1])
    o_dir = Path(sys.argv[2])
    paths = generate_all_figures(r_path, o_dir)
    print("Generated figures:", [str(p) for p in paths])
