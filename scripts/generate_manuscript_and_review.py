"""Compile publication-ready Markdown/LaTeX tables and empirical audit report from master JSON."""

from pathlib import Path
from typing import Dict, List
import json


def _fmt_ms(d: Dict, metric: str = "ADE_1") -> str:
    mu = d.get(f"{metric}_mean")
    sd = d.get(f"{metric}_std", 0.0)
    if mu is None:
        return "—"
    return f"{mu:.4f} ± {sd:.4f}"


def build_pinns_main_ablation_table(sums: Dict) -> str:
    rows = [
        ("M0", "Motion Only (Masked GRU)"),
        ("M1a", "+ Equal-Weight 2D Pose Concat"),
        ("M1b", "+ Implicit Gate (No q-vector, Param-Matched)"),
        ("M1", "+ Explicit Quality Gate (q-vector, Eq. 6)"),
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
        ("Ext1_MCENET", "Ext-1: MCENET (ITSC'20)"),
        ("Ext2_HST", "Ext-2: HST (IEEE RA-L'23)"),
        ("Ext3_SocialTransmotion", "Ext-3: Social-Transmotion (ICLR'24)"),
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
        "| Model | Overall ADE_1 (m) ↓ | Overall FDE_1 (m) ↓ | Clear ADE_1 ↓ | Partial (Vehicle Lower-Occl.) ADE_1 ↓ | Heavy (sigma=0.20) ADE_1 ↓ |",
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
        blocks.append(f"#### {fam}\n")
        blocks.append(
            "| Hypothesis Comparison | N | Model ADE_1 | Ref ADE_1 | Abs Delta (m) | Rel Change (%) | Test | Raw p | Holm-Adjusted p | Sig (<0.05) | MPSES |"
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
        f"| Window | Concurrent Agents N | Mean Latency (ms) on {gpu} | P50 (ms) | P95 (ms) | Peak VRAM (MB) |",
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


def compile_empirical_tables_markdown(report_json: Path, out_md: Path) -> None:
    rep = json.loads(report_json.read_text(encoding="utf-8"))
    sums = rep["summaries"]
    md = "\n\n".join(
        [
            "# Complete Empirical Benchmark & Statistical Hypothesis Report",
            f"- **Hardware**: `{rep.get('gpu_name')}` (`x{rep.get('gpu_count')}`)",
            f"- **Dataset Diagnostics**: `{json.dumps(rep.get('diagnostics', {}))}`",
            "## 1. PINNS Short-Window (T=5 / 0.4s) Controlled Single-Factor Ablation & SOTA Table",
            build_pinns_main_ablation_table(sums),
            "## 2. PINNS Multi-Window Scaling (T=5 / 10 / 20)",
            build_multi_window_table(sums),
            "## 3. Cross-Dataset Generalization on JTA (0.4s Short Window, Table 1 Occlusion/Noise Regimes)",
            build_jta_cross_dataset_table(sums),
            "## 4. Pre-Registered Paired Statistical Tests (Shapiro-Wilk -> Paired-t/Wilcoxon + Holm-Bonferroni)",
            build_holm_tables(rep.get("statistical_tests", {})),
            "## 5. Multi-Agent Real-Time Inference Latency & Memory Footprint",
            build_latency_markdown(rep),
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    import sys
    compile_empirical_tables_markdown(Path(sys.argv[1]), Path(sys.argv[2]))
