"""Render 300-DPI Figure 22: JAAD 2.0 Post-Reappearance Visible Alarms, 12-Event Timeline, & Operating-Point Audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_fig22(results_dir: Path, out_png: Path) -> Path:
    j23 = json.loads((results_dir / "wave23_jaad_autopose_and_isofpr_leadtime_5seed_verified.json").read_text())
    j25 = json.loads((results_dir / "wave25_jaad_reappearance_per_event_audit_5seed_verified.json").read_text())
    ea = j25["jaad2_per_event_reappearance_and_visible_alarm_audit"]
    s_by_c = ea["strict_recovered_summary_by_class"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.7,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.6,
    })
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.2), dpi=300)
    fig.patch.set_facecolor("white")

    # Panel (a): Unrestricted Iso-FPR (22/431=5.10%) vs Strictly Post-Reappearance Visible-Only (18/344=5.23%)
    ax = axes[0, 0]
    cats = ["Start (N=12)\nUnrestr. 5.1%", "Start (N=12)\nPost-Reap Vis", "Stop (N=10)\nUnrestr. 5.1%", "Stop (N=10)\nPost-Reap Vis"]
    x = np.arange(4)
    kin_v = [
        100 * s_by_c["Start"]["models"]["Lead_M0_kin_only"]["iso_hits_unrestricted"] / 12,
        100 * s_by_c["Start"]["models"]["Lead_M0_kin_only"]["protocol2_visible_only_hits"] / 12,
        100 * s_by_c["Stop"]["models"]["Lead_M0_kin_only"]["iso_hits_unrestricted"] / 10,
        100 * s_by_c["Stop"]["models"]["Lead_M0_kin_only"]["protocol2_visible_only_hits"] / 10,
    ]
    ap_v = [
        100 * s_by_c["Start"]["models"]["Lead_AutoPose_rtmpose17"]["iso_hits_unrestricted"] / 12,
        100 * s_by_c["Start"]["models"]["Lead_AutoPose_rtmpose17"]["protocol2_visible_only_hits"] / 12,
        100 * s_by_c["Stop"]["models"]["Lead_AutoPose_rtmpose17"]["iso_hits_unrestricted"] / 10,
        100 * s_by_c["Stop"]["models"]["Lead_AutoPose_rtmpose17"]["protocol2_visible_only_hits"] / 10,
    ]
    hm_v = [
        100 * s_by_c["Start"]["models"]["Lead_HumanAttr_orient_gaze"]["iso_hits_unrestricted"] / 12,
        100 * s_by_c["Start"]["models"]["Lead_HumanAttr_orient_gaze"]["protocol2_visible_only_hits"] / 12,
        100 * s_by_c["Stop"]["models"]["Lead_HumanAttr_orient_gaze"]["iso_hits_unrestricted"] / 10,
        100 * s_by_c["Stop"]["models"]["Lead_HumanAttr_orient_gaze"]["protocol2_visible_only_hits"] / 10,
    ]
    cnt_lbls = [
        (2, 6, 7, 12), (1, 4, 7, 12), (0, 3, 4, 10), (0, 3, 3, 10)
    ]
    b1 = ax.bar(x - 0.25, kin_v, width=0.23, color="#AEB6BF", label="BBox Motion Only")
    b2 = ax.bar(x, ap_v, width=0.23, color="#1F618D", label="Auto RTMPose-17 Skeleton")
    b3 = ax.bar(x + 0.25, hm_v, width=0.23, color="#117A65", label="Human Orient+Gaze Attr")
    for gi in range(4):
        c0, c1, c2, den = cnt_lbls[gi]
        for b, ct in ((b1[gi], c0), (b2[gi], c1), (b3[gi], c2)):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.3, f"{ct}/{den}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x, cats, fontsize=7.9)
    ax.set_ylabel("Pre-Onset Recall (%)")
    ax.set_ylim(0, 72)
    ax.set_title("(a) JAAD Strict De-Occl Recall: All Pre-Onset vs. Post-Reappearance Visible")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    # Panel (b): Per-Event [t_last_occ, t_reappear, t_first_alarm, t_onset] Raster on All 12 Strict Start Events
    ax = axes[0, 1]
    ev12 = ea["per_event_strict_recovered_log"]["Start"]
    y_pos = np.arange(len(ev12))
    for yi, r in enumerate(ev12):
        f_on = r["f_onset_30hz"]
        t_reap = (r["f_reappear_30hz"] - f_on) / 30.0 * 1000.0
        # Shaded occluded zone [-1500 ms, t_reap]
        if t_reap > -1500.0:
            ax.barh(yi, t_reap - (-1500.0), left=-1500.0, height=0.56, color="#D5D8DC", alpha=0.75,
                    label="Occluded before t_reappear" if yi == 0 else None)
        ax.plot([t_reap, t_reap], [yi - 0.30, yi + 0.30], color="#7F8C8D", lw=1.6,
                label="t_reappear (occ -> none)" if yi == 0 else None)
        m_kin = r["per_model"]["Lead_M0_kin_only"]["f_first_alarm_30hz"]
        m_aut = r["per_model"]["Lead_AutoPose_rtmpose17"]["f_first_alarm_30hz"]
        if m_kin is not None:
            tk = (m_kin - f_on) / 30.0 * 1000.0
            ax.scatter([tk], [yi - 0.11], color="#566573", marker="s", s=34, zorder=4,
                       label="KinOnly 1st Alarm (2/12)" if yi == 9 else None)
        if m_aut is not None:
            ta = (m_aut - f_on) / 30.0 * 1000.0
            after_ok = r["per_model"]["Lead_AutoPose_rtmpose17"]["first_alarm_strictly_after_reappearance"]
            col = "#1F618D" if after_ok else "#E67E22"
            lbl = None
            if yi == 3:
                lbl = "AutoPose Post-Reappear Hit (4/12)"
            elif yi == 5:
                lbl = "AutoPose Pre-Occl Alarm (2/12)"
            ax.scatter([ta], [yi + 0.11], color=col, marker="o", s=42, zorder=5, label=lbl)

    ax.axvline(0.0, color="#C0392B", linestyle="--", lw=1.2, label="t_onset = 0 ms (Start begins)")
    ax.set_yticks(y_pos, [f"{r['video_id'][-4:]}:{r['ped_id'].split('_')[-1]}" for r in ev12], fontsize=7.3)
    ax.invert_yaxis()
    ax.set_xlim(-1550, 120)
    ax.set_xlabel("Time Relative to Start Onset (ms, <= 0 is Pre-Onset)")
    ax.set_title("(b) 12 Strict Start Events: [t_reappear, t_alarm, t_onset] (3/4 New >= t_reap)")
    ax.legend(frameon=False, fontsize=6.9, loc="lower left")
    ax.grid(axis="x", linestyle="--", alpha=0.30)

    # Panel (c): Common-Hit Alarm Lead-Time Delta with Explicit Comparison Pairs & Denominators (22/431 = 5.10% FPR)
    ax = axes[1, 0]
    all_p = j23["jaad2_sliding_isofpr_leadtime_5seed"]["iso_test_fpr"]["fpr_0p050"]["locked_test_all_events"]["apple_to_apple_pairs"]
    str_p = j23["jaad2_sliding_isofpr_leadtime_5seed"]["iso_test_fpr"]["fpr_0p050"]["locked_test_post_deocclusion_strict_recovered"]["apple_to_apple_pairs"]
    p2_p = ea["protocol2_visible_only_strict_paired"]
    bars_spec = [
        ("Strict Start:\nAuto vs Kin\n(n=2, +4 new)", str_p["autopose_vs_kin_only"]["Start"]["common_hit_lead_diff_ms_mean"], "#7F8C8D"),
        ("Strict Start:\nAuto+Veh vs Veh\n(n=5 common)", str_p["autopose_veh_vs_veh_only"]["Start"]["common_hit_lead_diff_ms_mean"], "#2874A6"),
        ("Vis-Only Cross:\nAuto+Veh vs Veh\n(n=4 common)", p2_p["autopose_veh_vs_veh_only"]["Turn"]["common_hit_lead_diff_ms_mean"], "#117A65"),
        ("All Test Start:\nAuto vs Kin\n(n=10 common)", all_p["autopose_vs_kin_only"]["Start"]["common_hit_lead_diff_ms_mean"], "#1F618D"),
        ("All Test Cross:\nAuto vs Kin\n(n=14 common)", all_p["autopose_vs_kin_only"]["Turn"]["common_hit_lead_diff_ms_mean"], "#D35400"),
    ]
    xc = np.arange(len(bars_spec))
    vals_c = [v for _, v, _ in bars_spec]
    cols_c = [c for _, _, c in bars_spec]
    ax.axhline(0.0, color="#2C3E50", lw=1.0, linestyle=":")
    bc = ax.bar(xc, vals_c, width=0.52, color=cols_c, alpha=0.90)
    for b, v in zip(bc, vals_c):
        ax.text(b.get_x() + b.get_width() / 2, max(v, 0.0) + 12, f"{v:+.0f} ms", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax.set_xticks(xc, [lb for lb, _, _ in bars_spec], fontsize=7.5)
    ax.set_ylabel("Mean Earlier Alarm on Common Hits (ms)")
    ax.set_ylim(-40, 590)
    ax.set_title("(c) Common-Hit Lead-Time Gain with Exact Pairs & Denominators (No Miss Penalty)")
    ax.grid(axis="y", linestyle="--", alpha=0.32)

    # Panel (d): Operating-Point Transparency — Iso-Test-FPR (22/431=5.10%) vs A-Priori Val-Calibrated Thresholds
    ax = axes[1, 1]
    sm = s_by_c["Start"]["models"]
    op_models = [
        ("KinOnly\n(Iso 22/431)", sm["Lead_M0_kin_only"]["iso_all_neg_fpr"] * 100, sm["Lead_M0_kin_only"]["iso_hits_unrestricted"] / 12 * 100, "#7F8C8D", "s"),
        ("KinOnly\n(Val 12/431)", sm["Lead_M0_kin_only"]["val_calibrated_test_fpr"] * 100, sm["Lead_M0_kin_only"]["val_calibrated_hits"] / 12 * 100, "#566573", "^"),
        ("AutoPose\n(Iso 22/431)", sm["Lead_AutoPose_rtmpose17"]["iso_all_neg_fpr"] * 100, sm["Lead_AutoPose_rtmpose17"]["iso_hits_unrestricted"] / 12 * 100, "#1F618D", "o"),
        ("AutoPose\n(Val 34/431)", sm["Lead_AutoPose_rtmpose17"]["val_calibrated_test_fpr"] * 100, sm["Lead_AutoPose_rtmpose17"]["val_calibrated_hits"] / 12 * 100, "#2E86C1", "D"),
        ("HumanAttr\n(Iso 22/431)", sm["Lead_HumanAttr_orient_gaze"]["iso_all_neg_fpr"] * 100, sm["Lead_HumanAttr_orient_gaze"]["iso_hits_unrestricted"] / 12 * 100, "#117A65", "P"),
    ]
    ax.axvline(5.104, color="#922B21", linestyle="--", lw=1.0, alpha=0.75, label="Matched Test FPR = 22/431 (5.10%)")
    for lbl, fx, ry, col, mk in op_models:
        ax.scatter([fx], [ry], color=col, marker=mk, s=78, zorder=5)
        dx = 0.22 if fx < 7.0 else -0.55
        dy = -7.5 if "Val 34" in lbl else (2.2 if "Val 12" not in lbl else -6.5)
        ax.annotate(f"{lbl}\n({fx:.2f}%, {ry:.1f}%)", xy=(fx, ry), xytext=(fx + dx, ry + dy), fontsize=7.2, color="#1C2833")
    ax.set_xlim(1.5, 9.8)
    ax.set_ylim(5, 72)
    ax.set_xlabel("Realized Test Negative False-Positive Rate (%, Denominator = 431)")
    ax.set_ylabel("Strict De-Occluded Start Recall (%, N = 12)")
    ax.set_title("(d) Threshold Calibration Audit: Matched 22/431 FPR vs. Val-Fixed Drift (p=0.125)")
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.32)

    fig.tight_layout(pad=1.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Rendered: {out_png}")
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-png", type=Path, default=Path("docs/figures/fig22_wave22_apa_leadtime_and_delayed_onset.png"))
    args = ap.parse_args()
    render_fig22(args.results_dir, args.out_png)


if __name__ == "__main__":
    main()
