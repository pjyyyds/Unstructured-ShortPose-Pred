"""Render 300-DPI Figure 16: Six-dataset O(2) synthesis, DAIR/JRDB neighbor graphs, & Table 1 cluster bootstrap CIs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _panel_ab(ax_a, ax_b, jr: dict, ext12: dict, dut: dict) -> None:
    d_0 = ext12["dair_v2x_54"]["by_transform"]["0deg"]
    j_nb = jr["jrdb_outdoor_1677"]["by_transform"]["0deg"]
    im = ext12["imptc_564"]["by_transform"]["0deg"]
    du = dut["dut_zero_shot_5seed_transfer"]["axes"]["0"]

    # Panel (a): External 10m O(2) Null-Sink Neighbor Graph across DUT (3,066), IMPTC (564), & JRDB (1,677) + DAIR (31)
    cats = ["DUT Campus\n(n=3,066)", "IMPTC DE\n(n=564)", "JRDB Outdoor\n(n=1,677)", "DAIR Strict\n(n=31)"]
    xa = np.arange(4)
    cv_v = [
        du["cv_measured"]["ade_m"],
        im["actual_time_cv"]["seq_equal_ade_m"],
        j_nb["last_two_cv"]["scene_equal_ade_m"],
        d_0["strict_31"]["cv_measured"]["ade_m"],
    ]
    ng_v = [
        du["o2_no_graph_bayes"]["ade_m"],
        im["full_clk_bayes"]["seq_equal_ade_m"],
        j_nb["no_graph_bayes"]["scene_equal_ade_m"],
        d_0["strict_31"]["ours_o2_full_t5_clk_bayes"]["ade_m"],
    ]
    gr_v = [
        du["o2_nullsink_real_bayes"]["ade_m"],
        im["full_graph_clk_bayes"]["seq_equal_ade_m"],
        j_nb["graph_bayes"]["scene_equal_ade_m"],
        d_0["strict_31"]["ours_o2_full_t2_clk_bayes"]["ade_m"],
    ]
    ax_a.bar(xa - 0.25, cv_v, width=0.23, color="#cbd5e1", label="Measured-Clock CV")
    ax_a.bar(xa + 0.00, ng_v, width=0.23, color="#64748b", label="O(2) No-Graph / T5->T2")
    ax_a.bar(xa + 0.25, gr_v, width=0.23, color="#0b4f4a", label="Ours_O2_Full (+10m Graph / T2)")
    for xi, v in zip(xa + 0.25, gr_v):
        ax_a.text(xi, v + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#0b4f4a")
    ax_a.set_xticks(xa, cats, fontsize=8.3)
    ax_a.set_ylim(0.0, max(cv_v) * 1.15)
    ax_a.set_ylabel("Zero-shot ADE (m, lower better)", fontsize=9.0)
    ax_a.set_title("(a) Six-benchmark external O(2) synthesis (DUT, IMPTC, JRDB, DAIR)", fontsize=10.0, fontweight="bold", loc="left", pad=7)
    ax_a.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax_a.spines["top"].set_visible(False); ax_a.spines["right"].set_visible(False)
    ax_a.legend(loc="upper left", fontsize=7.3, framealpha=0.95)

    # Panel (b): JRDB 1,677 outdoor 2.5 Hz windows across 3 Stanford scenes
    sc_order = ["hewlett-packard-intersection-2019-01-24_0", "huang-lane-2019-02-12_0", "tressider-2019-03-16_1"]
    sc_lbls = ["HP Intersection", "Huang Lane", "Tressider", "3-Scene Equal"]
    by_sc = jr["jrdb_outdoor_1677"]["by_scene_0deg"]
    xb = np.arange(4)
    j_cv = [by_sc[s]["last_two_cv"]["ade_m"] for s in sc_order] + [j_nb["last_two_cv"]["scene_equal_ade_m"]]
    j_ng = [by_sc[s]["no_graph_bayes"]["ade_m"] for s in sc_order] + [j_nb["no_graph_bayes"]["scene_equal_ade_m"]]
    j_gr = [by_sc[s]["graph_bayes"]["ade_m"] for s in sc_order] + [j_nb["graph_bayes"]["scene_equal_ade_m"]]
    ax_b.bar(xb - 0.25, j_cv, width=0.23, color="#cbd5e1", label="2.5 Hz Last-2 CV")
    ax_b.bar(xb + 0.00, j_ng, width=0.23, color="#64748b", label="Ours_O2_Full (No Graph)")
    ax_b.bar(xb + 0.25, j_gr, width=0.23, color="#0b4f4a", label="Ours_O2_Full (+ 10m O(2) Graph)")
    for xi, v in zip(xb + 0.25, j_gr):
        ax_b.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#0b4f4a")
    ax_b.set_xticks(xb, sc_lbls, fontsize=8.3)
    ax_b.set_ylim(0.0, max(max(j_cv), max(j_ng)) * 1.18)
    ax_b.set_ylabel("2.8 s forecast ADE (m, lower better)", fontsize=9.0)
    ax_b.set_title("(b) Stanford JRDB outdoor 2.5 Hz zero-shot (1,677 windows)", fontsize=10.2, fontweight="bold", loc="left", pad=7)
    ax_b.grid(axis="y", color="#e2e8f0", lw=0.7)
    ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
    ax_b.legend(loc="upper right", fontsize=7.4, framealpha=0.95)


def _panel_cd(ax_c, ax_d, cb: dict) -> None:
    refs = ["Ext1_MCENET", "Ours_Full", "Ext3_SocialTransmotion", "M3", "Ext2_HST", "M2", "M1", "M0", "M1p"]
    lbls = ["MCENET", "Ours_Full", "Soc-Trans.", "M3", "HST", "M2", "M1", "M0", "M1p"]
    for ax, tk, ttl in (
        (ax_c, "T5", "(c) Table 1 T=5 (0.4s): 21-clip & person clustered 95% CIs"),
        (ax_d, "T20", "(d) Table 1 T=20 (1.9s): 21-clip & person clustered 95% CIs"),
    ):
        comp = cb["durations"][tk]["clustered_comparisons"]
        present = [(r, l) for r, l in zip(refs, lbls) if r in comp]
        yc = np.arange(len(present))
        ax.axvline(0.0, color="#0f172a", ls="--", lw=1.2)
        for yi, (rk, rl) in enumerate(present):
            cc = comp[rk]["bayes_clip_cluster"]["window_weighted"]
            pc = comp[rk]["bayes_person_cluster"]["window_weighted"]
            m_cm = cc["mean_ade_m"] * 100.0
            lo_c, hi_c = cc["ci95_ade_m"][0] * 100.0, cc["ci95_ade_m"][1] * 100.0
            lo_p, hi_p = pc["ci95_ade_m"][0] * 100.0, pc["ci95_ade_m"][1] * 100.0
            ax.errorbar(m_cm, yi + 0.14, xerr=[[m_cm - lo_c], [hi_c - m_cm]], fmt="o", color="#0b4f4a", lw=1.8, ms=5.2, capsize=2.8, label="21-Clip Cluster 95% CI" if yi == 0 else None)
            ax.errorbar(m_cm, yi - 0.14, xerr=[[m_cm - lo_p], [hi_p - m_cm]], fmt="s", color="#c26a4a", lw=1.6, ms=4.8, capsize=2.8, label="Person-Track Cluster 95% CI" if yi == 0 else None)
        ax.set_yticks(yc, [l for _, l in present], fontsize=8.3)
        ax.set_ylim(-1.25, len(present) - 0.35)
        ax.set_xlabel("Ours_O2_Full(Bayes) minus baseline ADE (cm, <0 wins)", fontsize=8.8)
        ax.set_title(ttl, fontsize=10.0, fontweight="bold", loc="left", pad=7)
        ax.grid(axis="x", color="#e2e8f0", lw=0.7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(loc="lower left", fontsize=7.4, framealpha=0.95)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    res = root / "results"
    jr = json.loads((res / "jrdb_outdoor_o2_full_5seed_verified.json").read_text())["report"]
    cb = json.loads((res / "table1_cluster_bootstrap_o2_full_5seed_verified.json").read_text())["report"]
    ext12 = json.loads((res / "imptc_and_dair_o2_full_5seed_verified.json").read_text())["report"]
    dut = json.loads((res / "o2_hetero_graph_pinns_and_dut_5seed_verified.json").read_text())["report"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#334155", "axes.linewidth": 0.9})
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.6), dpi=300)
    fig.patch.set_facecolor("white")
    _panel_ab(axes[0, 0], axes[0, 1], jr, ext12, dut)
    _panel_cd(axes[1, 0], axes[1, 1], cb)
    fig.tight_layout(pad=1.7)
    out = root / "docs/figures/fig16_six_dataset_matrix_and_cluster_bootstrap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Rendered:", out)


if __name__ == "__main__":
    main()
