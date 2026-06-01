#!/usr/bin/env python3

"""
Parses two sets of training log files (regular + jumbled) and generates
paired violin + scatter plots for BLIP-L and ALIP-L metrics.

Template generated using Claude, modified by accg. 

Usage:
    python plot_training_metrics.py
"""

import re
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.size"] = 12

FILE_PATTERN_REGULAR = "../out_crossval_repeats/seed_*.txt"
FILE_PATTERN_JUMBLED = "../out_crossval_repeats/seed_*_jumbled.txt"

METRICS  = ["Accuracy","MCC", "AUROC","AUPRC"]
SECTIONS = ["BLIP-L", "ALIP-L"]

# Per-section positive rate (fraction of positive examples).
# Used for AUPRC and Accuracy random-guess baselines.
POSITIVE_RATE = {
    "BLIP-L": 0.17,
    "ALIP-L": 0.11,
}

def random_guess(section: str) -> dict:
    p = POSITIVE_RATE[section]
    return {
        "AUPRC":    p,
        "AUROC":    0.5,
        "Accuracy": p,   # uniform random; use max(p, 1-p) for majority-class
        "MCC":      0.0,
    }

VIOLIN_WIDTH  = 0.25
PAIR_GAP      = 0.06
GROUP_SPACING = 1.2
JITTER_SEED   = 0

COLORS = {
    "regular": "#4C72B0",
    "jumbled": "#DD8452",
}

######################################################################

def main():
    data_reg = collect(FILE_PATTERN_REGULAR, "regular")
    data_jum = collect(FILE_PATTERN_JUMBLED, "jumbled")

    rng = np.random.default_rng(JITTER_SEED)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3), constrained_layout=True)

    for ax, section in zip(axes, SECTIONS):
        violin_panel(ax, data_reg[section], data_jum[section], section, rng)

    out = "result_crossval_metrics.pdf"
    plt.savefig(out, dpi=300)
    print(f"\nSaved → {out}")
    plt.show()

######################################################################

# parse
def parse_file(filepath: str) -> dict:
    results = {s: defaultdict(list) for s in SECTIONS}
    current_section = None

    with open(filepath) as fh:
        for line in fh:
            line = line.strip()

            m = re.match(r"~~~TRAINING\s+(\S+)", line)
            if m:
                name = m.group(1)
                current_section = name if name in SECTIONS else None
                continue

            if current_section is None:
                continue

            m = re.match(
                r"(AUPRC|AUROC|Accuracy|MCC)\s*:\s*([0-9.eE+\-]+)",
                line, re.IGNORECASE,
            )
            if m:
                metric = m.group(1)
                for official in METRICS:
                    if official.lower() == metric.lower():
                        metric = official
                        break
                try:
                    results[current_section][metric].append(float(m.group(2)))
                except ValueError:
                    pass

    return results


def collect(pattern: str, label: str) -> dict:
    all_files = sorted(glob.glob(pattern))
    if label == "regular":
        files = [f for f in all_files if "_jumbled" not in Path(f).name]
    else:
        files = all_files

    if not files:
        raise FileNotFoundError(
            f"No files found for pattern '{pattern}'. "
            "Edit the FILE_PATTERN constants at the top of this script."
        )
    print(f"[{label:>8}]  {len(files)} file(s): {[Path(f).name for f in files]}")

    agg = {s: defaultdict(list) for s in SECTIONS}
    for fp in files:
        parsed = parse_file(fp)
        for section in SECTIONS:
            for metric, vals in parsed[section].items():
                agg[section][metric].extend(vals)
    return agg


# plots 
def draw_violin(ax, pos: float, values: list, color: str,
                rng: np.random.Generator):
    if not values:
        return

    if len(values) >= 2:
        parts = ax.violinplot(
            [values],
            positions=[pos],
            showmedians=True,
            showextrema=True,
            widths=VIOLIN_WIDTH,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.45)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_edgecolor(color)
                parts[key].set_linewidth(1.5)

    jitter = rng.uniform(-0.07, 0.07, size=len(values))
    ax.scatter(
        np.full(len(values), pos) + jitter,
        values,
        s=15, color=color, alpha=0.85,
        edgecolors="white", linewidths=0.35, zorder=3,
    )


def violin_panel(ax, data_reg: dict, data_jum: dict,
                 section: str, rng: np.random.Generator):
    half = PAIR_GAP / 2 + VIOLIN_WIDTH / 2
    pair_half_span = (half + VIOLIN_WIDTH / 2 + 0.02) * 0.2  # 0.5 = 50% of current width

    group_centres = np.arange(len(METRICS)) * GROUP_SPACING
    pos_reg = group_centres - half
    pos_jum = group_centres + half

    baselines = random_guess(section)

    for i, metric in enumerate(METRICS):
        draw_violin(ax, pos_reg[i], data_reg.get(metric, []),
                    COLORS["regular"], rng)
        draw_violin(ax, pos_jum[i], data_jum.get(metric, []),
                    COLORS["jumbled"], rng)

        # ── Random-guess reference line ──────────────────────────────────────
        baseline = baselines.get(metric)
        if baseline is not None:
            x_lo = group_centres[i] - pair_half_span
            x_hi = group_centres[i] + pair_half_span
            ax.hlines(
                baseline,
                xmin=x_lo, xmax=x_hi,
                colors="crimson", linewidths=1.4,
                linestyles="-", zorder=4,
            )

    # Vertical separators between metric groups
    for gc in group_centres:
        ax.axvline(gc, color="lightgrey",
                   linewidth=0.8, linestyle="--", zorder=0)

    ax.set_xticks(group_centres)
    ax.set_xticklabels(METRICS)
    ax.set_xlim(group_centres[0] - GROUP_SPACING / 2,
                group_centres[-1] + GROUP_SPACING / 2)
    ax.set_title(section, fontsize=12)
    ax.set_ylabel("Value")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_ylim(-0.45,1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    legend_handles = [
        mpatches.Patch(facecolor=COLORS["regular"], alpha=0.7, label="Trained model"),
        mpatches.Patch(facecolor=COLORS["jumbled"], alpha=0.7, label="Jumbled"),
        plt.Line2D([0], [0], color="crimson", linewidth=1.4,
                   linestyle="-", label="Random"),
    ]
    ax.legend(handles=legend_handles, fontsize=11, frameon=True, facecolor='white',
              loc="lower right", handlelength=1.0, handleheight=0.5)

######################################################################
if __name__ == "__main__":
    main()



