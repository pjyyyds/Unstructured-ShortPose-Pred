"""Plot the fraction of five-frame histories covering less than half a metre."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
SITE_LABELS = {"america_crossroad": "Wyoming crossing", "italy_park": "Pistoia parking plaza",
               "japan_crossroad": "Tokyo crossing", "thailand_street": "Surat Thani street"}


def draw_panel(ax, rows, title, subtitle, color):
    positions = list(range(len(rows)))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e4e9e8", linewidth=.7)
    for i, (_, short, total) in enumerate(rows):
        share = 100 * short / total
        ax.barh(i, share, height=.54, color=color if i == 0 else "#8badab", zorder=2)
        ax.text(103, i, f"{short:,} / {total:,}", ha="left", va="center", fontsize=8.4,
                weight="bold" if i == 0 else "regular", color="#1b3537")
    ax.set_yticks(positions, [r[0] for r in rows], fontsize=9)
    ax.get_yticklabels()[0].set_weight("bold")
    ax.set_xlim(0, 134)
    ax.set_ylim(len(rows)-.5, -.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.tick_params(axis="both", length=0, pad=5, colors="#33474b")
    ax.spines[:].set_visible(False)
    ax.set_title(title, loc="left", fontsize=14, weight="bold", color="#152f32", pad=27)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9.5, color="#546367")


def plot(output):
    p = json.loads((ROOT / "reviews/pinns_short_displacement/history_support.json").read_text())["design"]["site_distances"]
    d = json.loads((ROOT / "reviews/dut_shared_space/observed_distances.json").read_text())
    pinns = [(label, p[key]["cumulative"]["below_half_metre"], p[key]["windows"])
             for key, label in [("all", "All windows"), *SITE_LABELS.items()]]
    dut = [("All windows", d["short_windows"], d["windows"])]
    dut.extend((k.replace("roundabout_", "Clip "), v["below_0p5_m"], v["windows"]) for k,v in d["by_clip"].items())
    fig, axes = plt.subplots(2, 1, figsize=(8.1, 8.65), gridspec_kw={"height_ratios": [5, 12]})
    fig.subplots_adjust(left=.22, right=.96, top=.85, bottom=.10, hspace=.52)
    fig.text(.06,.97,"How far did pedestrians move in five observations?", fontsize=17, weight="bold", va="top", color="#152f32")
    fig.text(.06,.928,"Share of windows with cumulative ground displacement below 0.50 m", fontsize=11, color="#485e60")
    draw_panel(axes[0], pinns, "PINNS roadside cameras", "Five observations span 0.400 s. Four fixed sites.", "#08766e")
    draw_panel(axes[1], dut, "DUT campus shared space", "Five original images span 0.417 s. Eleven recordings of one location.", "#08766e")
    fig.text(.06,.038,"Counts at right show qualifying windows / all windows. Totals weight windows equally.", fontsize=9, color="#485e60")
    fig.savefig(output, dpi=240, facecolor="white", bbox_inches="tight", pad_inches=.21)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    plot(parser.parse_args().output)
