"""Compile the archived empirical metrics into Markdown."""

from pathlib import Path
from typing import Dict, List
import json
from statistics import mean, stdev


def _fmt_ms(d: Dict, metric: str = "ADE_1") -> str:
    mu = d.get(f"{metric}_mean")
    sd = d.get(f"{metric}_std", 0.0)
    if mu is None:
        return "—"
    return f"{mu:.4f} ± {sd:.4f}"


def build_pinns_main_ablation_table(sums: Dict) -> str:
    rows = [
        ("M0", "Motion Only (Masked GRU)"),
        ("M1a", "Equal-weight sum of motion and available pose, then LayerNorm"),
        ("M1b", "+ Implicit Gate (No q-vector, Param-Matched)"),
        ("M1", "+ Explicit quality gate (q-vector)"),
        ("M1p", "+ Biomechanical Part-Decomposed Gate (BPD-Gate)"),
        ("M2a", "M1 + Non-Lane Local Polylines Only"),
        ("M2b_norad", "M1 + Directed GATv2 (w/o v_rad)"),
        ("M2b", "M1 + Directed GATv2 (w/ v_rad)"),
        ("M2", "M1 + Polylines + v_rad Directed GATv2"),
        ("M3a", "M2 + 5-Class Action Aux-Loss Only"),
        ("M3b", "M2 + Action Cond. Decoder (End-to-End)"),
        ("M3", "M2 + Action Cond. Decoder (Stop-Grad)"),
        ("M3p", "M3 + Kinematic-Action Mode Calibration (ACMC)"),
        ("Ours_Full", "Ours Full (BPD-Gate + Map&GATv2 + ACMC)"),
        ("Ext1_MCENET", "MCENET local adaptation"),
        ("Ext2_HST", "HST local adaptation"),
        ("Ext3_SocialTransmotion", "Social-Transmotion local adaptation"),
    ]
    lines = [
        "| Model Variant | Description | Params (M) | Overall ADE_1 (m) ↓ | Overall FDE_1 (m) ↓ | minADE_6 (m) ↓ | Clear ADE_1 ↓ | Partial ADE_1 ↓ | Heavy ADE_1 ↓ | Transition ADE_1 ↓ | Mode NLL ↓ | Action Macro-F1 ↑ |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for code, desc in rows:
        tag = f"PINNS_T5_{code}"
        if tag not in sums:
            continue
        rec = sums[tag]
        sl = rec["slices"]
        p_m = rec["params"] / 1e6
        act_f1 = rec.get("action_macro_f1_mean", 0.0)
        act_str = f"{100.0 * act_f1:.2f}%" if code in ("M3a", "M3b", "M3", "M3p", "Ours_Full") else "—"
        lines.append(
            f"| `{code}` | {desc} | {p_m:.3f} | {_fmt_ms(sl.get('Overall', {}), 'ADE_1')} | "
            f"{_fmt_ms(sl.get('Overall', {}), 'FDE_1')} | {_fmt_ms(sl.get('Overall', {}), 'minADE_6')} | "
            f"{_fmt_ms(sl.get('Clear', {}), 'ADE_1')} | {_fmt_ms(sl.get('Partial', {}), 'ADE_1')} | "
            f"{_fmt_ms(sl.get('Heavy', {}), 'ADE_1')} | {_fmt_ms(sl.get('Transition', {}), 'ADE_1')} | "
            f"{_fmt_ms(sl.get('Overall', {}), 'NLL')} | {act_str} |"
        )
    return "\n".join(lines)


def build_multi_window_table(sums: Dict) -> str:
    models = ["M0", "M1a", "M1b", "M1", "M1p", "Ext1_MCENET", "Ext2_HST", "Ext3_SocialTransmotion", "M3", "Ours_Full"]
    lines = [
        "| Model | T=5 (0.4s) ADE_1 ↓ | T=5 FDE_1 ↓ | T=10 (0.9s) ADE_1 ↓ | T=10 FDE_1 ↓ | T=20 (1.9s) ADE_1 ↓ | T=20 FDE_1 ↓ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m in models:
        if f"PINNS_T5_{m}" not in sums:
            continue
        s5 = sums.get(f"PINNS_T5_{m}", {}).get("slices", {}).get("Overall", {})
        s10 = sums.get(f"PINNS_T10_{m}", {}).get("slices", {}).get("Overall", {})
        s20 = sums.get(f"PINNS_T20_{m}", {}).get("slices", {}).get("Overall", {})
        lines.append(
            f"| `{m}` | {_fmt_ms(s5, 'ADE_1')} | {_fmt_ms(s5, 'FDE_1')} | "
            f"{_fmt_ms(s10, 'ADE_1')} | {_fmt_ms(s10, 'FDE_1')} | "
            f"{_fmt_ms(s20, 'ADE_1')} | {_fmt_ms(s20, 'FDE_1')} |"
        )
    return "\n".join(lines)


def build_jta_cross_dataset_table(sums: Dict) -> str:
    models = ["M0", "M1a", "M1b", "M1", "M1p", "Ext2_HST", "Ext3_SocialTransmotion", "M3", "Ours_Full"]
    lines = [
        "| Model | Overall ADE_1 (m) ↓ | Overall FDE_1 (m) ↓ | Clear ADE_1 ↓ | Partial (simulated lower-body occlusion) ADE_1 ↓ | Heavy (injected noise, sigma=0.20) ADE_1 ↓ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m in models:
        tag = f"JTA_T2_{m}"
        if tag not in sums:
            continue
        sl = sums[tag]["slices"]
        lines.append(
            f"| `{m}` | {_fmt_ms(sl.get('Overall', {}), 'ADE_1')} | {_fmt_ms(sl.get('Overall', {}), 'FDE_1')} | "
            f"{_fmt_ms(sl.get('Clear', {}), 'ADE_1')} | {_fmt_ms(sl.get('Partial', {}), 'ADE_1')} | "
            f"{_fmt_ms(sl.get('Heavy', {}), 'ADE_1')} |"
        )
    return "\n".join(lines)


def build_holm_tables(stat_dict: Dict[str, List[Dict]]) -> str:
    blocks = []
    for fam, rows in stat_dict.items():
        if not rows:
            continue
        family_label = "Within-dataset JTA" if fam == "CrossDataset_JTA" else fam
        blocks.append(f"#### {family_label}\n")
        blocks.append(
            "| Hypothesis Comparison | N | Model ADE_1 | Ref ADE_1 | Model − reference (m) | Rel Change (%) | Test | Raw p | Holm-Adjusted p | Sig (<0.05) | MPSES |"
        )
        blocks.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in rows:
            sig = "YES" if r.get("significant_005") else "no"
            mp = "YES" if r.get("mpses_met") else "no"
            blocks.append(
                f"| {r['comparison']} | {r['n']} | {r['mean_model']:.4f} | {r['mean_ref']:.4f} | "
                f"{r['abs_delta']:+.4f} | {r['rel_pct']:+.2f}% | `{r['test_used']}` | "
                f"{r['raw_p']:.3e} | {r.get('holm_p', r['raw_p']):.3e} | **{sig}** | {mp} |"
            )
        blocks.append("")
    return "\n".join(blocks)


def build_latency_markdown(report: Dict) -> str:
    lat = report.get("latency_cuda", {})
    gpu = report.get("gpu_name", "GPU")
    lines = [
        f"| Window | Batch size N | Mean Latency (ms) on {gpu} | P50 (ms) | P95 (ms) | Peak allocated memory (MiB) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for tw in ("T=5", "T=10", "T=20"):
        for na in ("N=1", "N=4", "N=8", "N=12"):
            d = lat.get(tw, {}).get(na, {})
            if d:
                lines.append(
                    f"| `{tw}` | `{na}` | {d['mean_ms']:.3f} | {d['p50_ms']:.3f} | "
                    f"{d['p95_ms']:.3f} | {d['peak_vram_mb']:.2f} |"
                )
    return "\n".join(lines)


def build_sensitivity_table(sums: Dict) -> str:
    """Use the three seeds shared by the nominal and shifted event-label thresholds."""
    lines = [
        "| Threshold scale | Seed 42 | Seed 1024 | Seed 2025 | Mean ADE (m) | SD (m) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scale in (50, 75, 100, 125, 150):
        tag = "PINNS_T5_Ours_Full" if scale == 100 else f"SENS_{scale}_Ours_Full"
        # The archive stores seed values in ascending seed order, rounded to six decimals.
        values = sums[tag]["slices"]["Overall"]["ADE_1_seeds"][:3]
        if len(values) != 3:
            raise ValueError(f"{tag} requires all three sensitivity seeds")
        cells = " | ".join(f"{v:.4f}" for v in values)
        lines.append(f"| {scale / 100:.2f}× | {cells} | {mean(values):.4f} | {stdev(values):.4f} |")
    return "\n".join(lines)


def compile_empirical_tables_markdown(report_json: Path, out_md: Path) -> None:
    rep = json.loads(report_json.read_text(encoding="utf-8"))
    sums = rep["summaries"]
    diagnostics = rep["diagnostics"]
    pinns = diagnostics["pinns"]["split_counts"]
    jta = diagnostics["jta"]["split_counts"]
    buckets = diagnostics["pinns_test_buckets"]
    full_tag = "PINNS_T5_Ours_Full"
    motion_tag = "PINNS_T5_M0"
    full = sums[full_tag]["slices"]["Overall"]["ADE_1_mean"]
    motion = sums[motion_tag]["slices"]["Overall"]["ADE_1_mean"]
    full_jta = sums["JTA_T2_Ours_Full"]["slices"]["Overall"]["ADE_1_mean"]
    motion_jta = sums["JTA_T2_M0"]["slices"]["Overall"]["ADE_1_mean"]
    runs = sum(record["num_seeds"] for record in sums.values())
    original_v1 = report_json.parent.name == "v1_initial_diagnosis"
    version_note = ("This archive records the first model version." if original_v1 else
                    "V2 was developed after inspection of V1 test results and reused the split. "
                    "The archived V1 table uses older labels. In both versions, M1a adds equal-weight "
                    "motion and available pose features, external baselines are local adaptations, "
                    "and JTA models are trained on JTA.")
    md = "\n\n".join(
        [
            "# Pedestrian prediction benchmark",
            f"**Full model and motion-only reference.** On PINNS at five frames their respective "
            f"top-mode ADEs are **{full:.4f} m** and **{motion:.4f} m**. On JTA they are "
            f"**{full_jta:.4f} m** and **{motion_jta:.4f} m**. Lower ADE is better.",
            f"{len(sums)} configurations, {runs} training runs. "
            f"{sum(v['num_seeds'] == 5 for v in sums.values())} configurations used five seeds and "
            f"{sum(v['num_seeds'] == 3 for v in sums.values())} used three. "
            f"Training hardware was {rep['gpu_count']} × {rep['gpu_name']}. "
            "Trajectory entries with ± show the mean and sample standard deviation across training seeds.",
            version_note,
            f"PINNS has {pinns['train']:,}/{pinns['val']:,}/{pinns['test']:,} training/validation/test windows "
            f"from four sites. JTA has {jta['train']:,}/{jta['val']:,}/{jta['test']:,} windows. "
            "JRDB and independent external validation remain open.",
            f"The PINNS five-frame test groups contain {buckets['Clear']} clear, {buckets['Partial']} partial and "
            f"{buckets['Heavy']} heavy windows. {diagnostics['pinns_test_valid_actions']} windows have valid action labels. "
            f"Positive labels for start, stop, left, right and maintain are {diagnostics['pinns_test_action_pos_counts']}.",
            "## 1. PINNS five-frame model comparisons",
            "The five observed frames span 0.4 s. The prediction horizon is 3.0 s. "
            "The quality groups are derived from observation features. Landmark localization accuracy requires separate ground truth.",
            build_pinns_main_ablation_table(sums),
            "## 2. PINNS across observation windows",
            build_multi_window_table(sums),
            "## 3. Within-dataset JTA evaluation with simulated pose corruption",
            "Each model was trained on the JTA training split. Two observed frames span 0.4 s and the prediction horizon is 4.8 s. "
            "The tests apply synthetic lower-body occlusion or coordinate noise to the pose data.",
            build_jta_cross_dataset_table(sums),
            "## 4. Paired window-level tests and Holm-adjusted p values",
            "Differences use model minus reference. Negative ADE differences favor the model. "
            "Holm adjustment is within each family. Overlapping windows from one track or clip may be correlated. "
            "The reported p values do not account for that clustering. MPSES is an ADE reduction of at least 5% or 0.04 m.",
            build_holm_tables(rep.get("statistical_tests", {})),
            "## 5. Full predictor forward time and memory on one GPU",
            "The timer starts with synthetic feature tensors already on the GPU. "
            "Each setting uses 25 warmups and 100 timed forwards. Video decoding, detection, pose extraction, "
            "tracking and feature preparation fall outside this measurement. Baseline latency was not timed.",
            build_latency_markdown(rep),
            "## 6. Action-label threshold sensitivity",
            "PINNS five-frame overall ADE for the full model. Speed and angle thresholds were scaled together. "
            "The default is recalculated from the same three seeds as the shifted thresholds. "
            "Individual seed values in the archive are rounded to six decimals.",
            build_sensitivity_table(sums),
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    import sys
    compile_empirical_tables_markdown(Path(sys.argv[1]), Path(sys.argv[2]))
