"""Render the five figures accompanying the archived short-pose benchmark."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "savefig.dpi": 300,
})

MOTION = "#343a40"
FULL = "#7c3768"
BLUE = "#216e92"
GREEN = "#267669"
ORANGE = "#b05c29"
BUCKETS = ("Clear", "Partial", "Heavy")


def _metric(sums, tag, subset="Overall", metric="ADE_1"):
    values = sums[tag]["slices"][subset]
    return values[f"{metric}_mean"], values[f"{metric}_std"]


def _dots(ax, x, values, color, label=None, marker="o"):
    means, deviations = zip(*values)
    ax.errorbar(x, means, yerr=deviations, fmt=marker, color=color, ms=4.5,
                elinewidth=1, capsize=2.5, label=label, zorder=3)


def _export(fig, path, note):
    for ax in fig.axes:
        ax.grid(axis="y", color="#d7dadc", linewidth=0.65)
        ax.tick_params(labelsize=9)
    fig.text(0.5, 0.016, note, ha="center", va="bottom", fontsize=8.5, color="#404448")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.13)
    plt.close(fig)
    return path


def plot_fig1_window_bucket_matrix(summaries: Dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.85), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.78, bottom=0.19, wspace=0.12)
    models = [("M0", "M0  motion only", MOTION), ("M1a", "M1a  equal-weight sum", ORANGE),
              ("M1b", "M1b  implicit gate", "#8d7035"), ("M1", "M1  quality gate", BLUE),
              ("M1p", "M1p  part gates", GREEN)]
    x = np.arange(3)
    for ax, window in zip(axes, (5, 10, 20)):
        for i, (model, label, color) in enumerate(models):
            values = [_metric(summaries, f"PINNS_T{window}_{model}", bucket) for bucket in BUCKETS]
            _dots(ax, x + (i - 2) * 0.14, values, color, label)
        ax.set_xticks(x, BUCKETS)
        ax.set_title(f"{window} frames · {(window - 1) / 10:.1f} s")
    axes[0].set_ylabel("Top-mode ADE (m)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=5, frameon=False)
    return _export(fig, out_dir / "fig1_window_quality_3x3_matrix.png",
                   "PINNS test windows grouped by observed pose quality. Points show means ± 1 SD across five training seeds.")


def plot_fig2_staircase_and_gates(summaries: Dict, out_dir: Path, v1_sums: Optional[Dict] = None) -> Path:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.7, 1]})
    fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.28, wspace=0.27)
    order = ["M0", "M1a", "M1b", "M1", "M1p", "M2a", "M2b_norad", "M2b", "M2", "M3a", "M3b", "M3", "M3p", "Ours_Full"]
    x = np.arange(len(order))
    if v1_sums:
        _dots(left, x - 0.16, [_metric(v1_sums, f"PINNS_T5_{m}") for m in order], "#777f87", "V1", "s")
    _dots(left, x + 0.16, [_metric(summaries, f"PINNS_T5_{m}") for m in order], BLUE, "V2")
    left.axhline(_metric(summaries, "PINNS_T5_M0")[0], ls=":", color=MOTION, label="V2 motion-only mean")
    left.set_xticks(x, [m.replace("Ours_Full", "Full").replace("M2b_norad", "M2b no radial") for m in order],
                    rotation=40, ha="right")
    left.set_ylabel("Top-mode ADE (m)")
    left.set_title("(a) PINNS five-frame model comparisons")
    left.legend(frameon=False, loc="upper left", ncol=2)
    for model, index, label, color in [("M1", 0, "M1", BLUE), ("M1p", 0, "M1p head/shoulders", GREEN),
                                       ("M1p", 1, "M1p torso", ORANGE), ("M1p", 2, "M1p legs", FULL)]:
        values = [summaries[f"PINNS_T5_{model}"]["gate_by_bucket"][b][index] for b in BUCKETS]
        right.plot(BUCKETS, values, marker="o", color=color, label=label)
    right.set_ylim(bottom=0)
    right.set_ylabel("Mean effective gate weight")
    right.set_title("(b) V2 gate values by quality group")
    right.legend(frameon=False, loc="best", fontsize=8)
    return _export(fig, out_dir / "fig2_staircase_and_gates.png",
                   "Panel a shows five-seed mean ± SD on the same split. Panel b includes the built-in quality envelope.")


def plot_fig3_scene_and_action_bridges(summaries: Dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.1, 5.9), sharex="col")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.17, wspace=0.28, hspace=0.23)
    scene = ["M1", "M2a", "M2b_norad", "M2b", "M2"]
    action = ["M2", "M3a", "M3b", "M3", "M3p", "Ours_Full"]
    for ax, models, subset, metric, color, scale in [
        (axes[0, 0], scene, "Overall", "ADE_1", BLUE, 1),
        (axes[1, 0], scene, "Overall", "BoundaryViol", ORANGE, 100),
        (axes[0, 1], action, "Transition", "ADE_1", FULL, 1),
    ]:
        values = [np.array(_metric(summaries, f"PINNS_T5_{m}", subset, metric)) * scale for m in models]
        _dots(ax, np.arange(len(models)), values, color)
    f1 = [(100 * summaries[f"PINNS_T5_{m}"]["action_macro_f1_mean"],
           100 * summaries[f"PINNS_T5_{m}"]["action_macro_f1_std"]) for m in action[1:]]
    _dots(axes[1, 1], np.arange(1, len(action)), f1, GREEN)
    axes[0, 0].set_title("(a) Local geometry and interaction variants")
    axes[0, 1].set_title("(b) Action conditioning variants")
    for ax, label in zip(axes.flat, ("Overall ADE (m)", "Transition ADE (m)", "Boundary violation (%)", "Action Macro-F1 (%)")):
        ax.set_ylabel(label)
    axes[1, 0].set_xticks(range(len(scene)), ["M1", "M2a", "M2b\nno radial", "M2b", "M2"])
    axes[1, 1].set_xticks(range(len(action)), ["M2", "M3a", "M3b", "M3", "M3p", "Full"])
    return _export(fig, out_dir / "fig3_scene_and_action_bridges.png",
                   "PINNS, five observed frames. Points show means ± 1 SD across five seeds. M2 has no action classifier.")


def plot_fig4_sota_and_jta(summaries: Dict, out_dir: Path) -> Path:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.65), gridspec_kw={"width_ratios": [1.12, 1]})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.38, wspace=0.28)
    models = [("M0", "M0 motion", MOTION, "s", 2.4), ("Ours_Full", "Full", FULL, "D", 2.4),
              ("M1", "M1", BLUE, "o", 1.4), ("M3", "M3", "#8e7537", "p", 1.0),
              ("Ext1_MCENET", "MCENET adapted", "#87949d", "^", 1),
              ("Ext2_HST", "HST adapted", ORANGE, "v", 1), ("Ext3_SocialTransmotion", "Social-Transmotion adapted", GREEN, "x", 1)]
    x = np.arange(3)
    for model, label, color, marker, lw in models:
        means = [_metric(summaries, f"PINNS_T{t}_{model}")[0] for t in (5, 10, 20)]
        left.plot(x, means, marker=marker, lw=lw, color=color, label=label)
    left.set_xticks(x, ["5 frames\n0.4 s", "10 frames\n0.9 s", "20 frames\n1.9 s"])
    left.set_ylabel("Mean top-mode ADE (m)")
    left.set_title("(a) PINNS by observation window")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False, fontsize=7.7)
    j_models = ["M0", "M1a", "M1", "M1p", "Ours_Full", "Ext2_HST", "Ext3_SocialTransmotion"]
    for i, (bucket, color) in enumerate(zip(BUCKETS, (BLUE, ORANGE, FULL))):
        values = [_metric(summaries, f"JTA_T2_{m}", bucket) for m in j_models]
        _dots(right, np.arange(len(j_models)) + (i - 1) * 0.16, values, color, bucket)
    right.set_xticks(range(len(j_models)), ["M0\nmotion", "M1a\nsum", "M1", "M1p", "Full", "HST\nadapted", "Social-T.\nadapted"])
    right.set_title("(b) JTA-trained models under synthetic corruption")
    right.set_ylabel("Top-mode ADE (m)")
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False)
    return _export(fig, out_dir / "fig4_sota_and_jta_generalization.png",
                   "Means across five seeds. JTA error bars show ± 1 SD. M0 and Full are evaluated at every displayed condition.")


def plot_fig5_latency_and_sensitivity(report: Dict, out_dir: Path, v1_rep: Optional[Dict] = None) -> Path:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.8, 4.1))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=0.24, wspace=0.25)
    latency = report["latency_cuda"]
    agents = (1, 4, 8, 12)
    if v1_rep:
        first_times = v1_rep["latency_cuda"]["T=5"]
        left.plot(agents, [first_times[f"N={n}"]["mean_ms"] for n in agents], "^:",
                  color=MOTION, lw=1.3, ms=4, label="V1 T=5 mean")
    for t, color in ((5, FULL), (10, BLUE), (20, GREEN)):
        measurements = [latency[f"T={t}"][f"N={n}"] for n in agents]
        for key, style, metric in (("mean_ms", "o-", "mean"), ("p95_ms", "s--", "p95")):
            left.plot(agents, [d[key] for d in measurements], style, color=color, lw=1.3, ms=4,
                      label=f"V2 T={t} {metric}")
    left.set_xticks(agents)
    left.set_xlabel("Batch size (targets)")
    left.set_ylabel("Forward time (ms)")
    left.set_title("(a) Full predictor on one Tesla T4")
    left.legend(frameon=False, ncol=3, fontsize=7.8, loc="upper center")
    left.margins(y=0.42)
    scales = (50, 75, 100, 125, 150)
    rows = []
    for scale in scales:
        tag = "PINNS_T5_Ours_Full" if scale == 100 else f"SENS_{scale}_Ours_Full"
        # Trainer sorts by numeric seed. The four sensitivity runs use FIXED_SEEDS[:3].
        seed_values = report["summaries"][tag]["slices"]["Overall"]["ADE_1_seeds"][:3]
        if len(seed_values) != 3:
            raise ValueError(f"{tag} requires all three sensitivity seeds")
        rows.append(seed_values)
    means = np.mean(rows, axis=1)
    sds = np.std(rows, axis=1, ddof=1)
    right.errorbar(scales, means, yerr=sds, fmt="o", color=FULL, capsize=3, label="Mean ± 1 SD")
    right.set_xticks(scales, ["−50%", "−25%", "Default", "+25%", "+50%"])
    right.set_xlabel("Change in speed and angle thresholds")
    right.set_ylabel("PINNS five-frame ADE (m)")
    right.set_title("(b) Action-label thresholds, same three seeds")
    right.legend(frameon=False, loc="best")
    return _export(fig, out_dir / "fig5_latency_and_sensitivity.png",
                   "Panel a starts from GPU-resident input tensors. Panel b uses seeds 42, 1024 and 2025 at every threshold.")


def generate_all_figures(report_json: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_json.read_text(encoding="utf-8"))
    v1_path = report_json.parent / "v1_initial_diagnosis" / "master_experiment_report.json"
    v1 = json.loads(v1_path.read_text(encoding="utf-8")) if v1_path.exists() else {}
    sums = report["summaries"]
    return [plot_fig1_window_bucket_matrix(sums, out_dir),
            plot_fig2_staircase_and_gates(sums, out_dir, v1.get("summaries")),
            plot_fig3_scene_and_action_bridges(sums, out_dir),
            plot_fig4_sota_and_jta(sums, out_dir),
            plot_fig5_latency_and_sensitivity(report, out_dir, v1)]


if __name__ == "__main__":
    for path in generate_all_figures(Path(sys.argv[1]), Path(sys.argv[2])):
        print(path)
