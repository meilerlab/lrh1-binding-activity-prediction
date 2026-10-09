#!/usr/bin/env python3

"""
Parses grouped-fold seed logs for pose-set ablations.
  seed_42.txt              all (base + rl + db)
  seed_42_base.txt         IUWtail only
  seed_42_base_rl.txt
  seed_42_base_db.txt
  seed_42_no_full.txt
  seed_42_no_4pld.txt
  seed_42_no_7tt8.txt
  seed_42_base_rl_no_full.txt
  seed_42_base_rl_no_4pld.txt
  seed_42_base_rl_no_7tt8.txt

Template generated with Cursor, modified by accg. 

Usage:
    python plot_crossval_metrics_ablation.py
"""

import re
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.size"] = 15

FILE_PATTERN = "../out_crossval_repeats_grouped/seed_*.txt"

METRICS  = ["Accuracy", "MCC", "AUROC", "AUPRC"]
SECTIONS = ["BLIP-L", "ALIP-L"]
TITLES = {"BLIP-L": "Binder prediction", "ALIP-L": "Activity prediction"}
ABLATIONS = [
    "all", "base", "base_rl", "base_db", "no_full", "no_4pld", "no_7tt8",
    "base_rl_no_full", "base_rl_no_4pld", "base_rl_no_7tt8",
]
ABLATION_RE = re.compile(
    r"^seed_\d+(?:_(base_rl_no_full|base_rl_no_4pld|base_rl_no_7tt8|base_rl|base_db|no_full|no_4pld|no_7tt8|base))?\.txt$"
)

# Positives / rows for each pose set. Accuracy and AUPRC use this rate.
# BLIP-L "all" is base + rl + db (359/2101); the other counts are from the logs.
POSITIVE_RATE = {
    "BLIP-L": {
        "all": 359/2101, "base": 32/1774,
        "base_rl": 194/1936, "base_db": 197/1939,
        "no_full": 250/1992, "no_4pld": 250/1992, "no_7tt8": 250/1992,
        "base_rl_no_full": 140/1882, "base_rl_no_4pld": 140/1882, "base_rl_no_7tt8": 140/1882,
    },
    "ALIP-L": {
        "all": 67/630, "base": 5/532,
        "base_rl": 37/581, "base_db": 39/582,
        "no_full": 48/598, "no_4pld": 46/598, "no_7tt8": 47/598,
        "base_rl_no_full": 26/565, "base_rl_no_4pld": 26/565, "base_rl_no_7tt8": 27/565,
    },
}

def random_guess(section: str, ablation: str) -> dict:
    p = POSITIVE_RATE[section][ablation]
    return {
        "AUPRC":    p,
        "AUROC":    0.5,
        "Accuracy": p,
        "MCC":      0.0,
    }

VIOLIN_WIDTH  = 0.11
GROUP_SPACING = 2.0
JITTER_SEED   = 0

COLORS = {
    "all":     "#4C72B0",
    "base":    "#DD8452",
    "base_rl": "#55A868",
    "base_db": "#C44E52",
    "no_full": "#8172B3",
    "no_4pld": "#CCB974",
    "no_7tt8": "#64B5CD",
    "base_rl_no_full": "#4A3F73",
    "base_rl_no_4pld": "#8A7340",
    "base_rl_no_7tt8": "#2E6F86",
}

LABELS = {
    "all":     "all",
    "base":    "base",
    "base_rl": "base+rl",
    "base_db": "base+db",
    "no_full": "no full",
    "no_4pld": "no 4pld",
    "no_7tt8": "no 7tt8",
    "base_rl_no_full": "base+rl, no full",
    "base_rl_no_4pld": "base+rl, no 4pld",
    "base_rl_no_7tt8": "base+rl, no 7tt8",
}

######################################################################

def main():
    data = collect(FILE_PATTERN)
    rng = np.random.default_rng(JITTER_SEED)

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.6), constrained_layout=True)
    for ax, section in zip(axes, SECTIONS):
        violin_panel(ax, data[section], section, rng)

    out = "result_crossval_metrics_ablation.pdf"
    plt.savefig(out, dpi=300)
    print(f"\nSaved → {out}")
    plt.show()

######################################################################

def parse_file(filepath: str) -> dict:
    results = {s: defaultdict(list) for s in SECTIONS}
    current_section = None

    with open(filepath) as fh:
        for line in fh:
            line = line.strip()

            m = re.match(r"~~~TRAINING\s+(BLIP-L|ALIP-L)\b", line)
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


def ablation_from_name(name: str):
    if "jumbled" in name:
        return None
    m = ABLATION_RE.match(name)
    if not m:
        return None
    return m.group(1) if m.group(1) else "all"


def collect(pattern: str) -> dict:
    files = sorted(glob.glob(pattern))
    by_ab = defaultdict(list)
    for fp in files:
        ab = ablation_from_name(Path(fp).name)
        if ab is None:
            continue
        by_ab[ab].append(fp)

    for ab in ABLATIONS:
        names = [Path(f).name for f in by_ab.get(ab, [])]
        print(f"[{ab:>16}]  {len(names)} file(s): {names}")

    agg = {s: {a: defaultdict(list) for a in ABLATIONS} for s in SECTIONS}
    for ab, fps in by_ab.items():
        for fp in fps:
            parsed = parse_file(fp)
            for section in SECTIONS:
                for metric, vals in parsed[section].items():
                    agg[section][ab][metric].extend(vals)
    return agg


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

    jitter = rng.uniform(-0.04, 0.04, size=len(values))
    ax.scatter(
        np.full(len(values), pos) + jitter,
        values,
        s=12, color=color, alpha=0.85,
        edgecolors="white", linewidths=0.35, zorder=3,
    )


def violin_panel(ax, data: dict, section: str, rng: np.random.Generator):
    n = len(ABLATIONS)
    offsets = (np.arange(n) - (n - 1) / 2.0) * (VIOLIN_WIDTH + 0.03)
    group_centres = np.arange(len(METRICS)) * GROUP_SPACING

    for i, metric in enumerate(METRICS):
        for j, ab in enumerate(ABLATIONS):
            pos = group_centres[i] + offsets[j]
            draw_violin(
                ax,
                pos,
                data[ab].get(metric, []),
                COLORS[ab],
                rng,
            )
            half = VIOLIN_WIDTH / 2
            ax.hlines(
                random_guess(section, ab)[metric],
                xmin=pos - half, xmax=pos + half,
                colors="crimson", linewidths=1.4,
                linestyles="-", zorder=4,
            )

    for gc in group_centres:
        ax.axvline(gc, color="lightgrey",
                   linewidth=0.8, linestyle="--", zorder=0)

    ax.set_xticks(group_centres)
    ax.set_xticklabels(METRICS)
    ax.set_xlim(group_centres[0] - GROUP_SPACING / 2,
                group_centres[-1] + GROUP_SPACING / 2)
    ax.set_title(TITLES[section], fontsize=15)
    ax.set_ylabel("Value")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_ylim(-0.45, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    legend_handles = [
        mpatches.Patch(facecolor=COLORS[a], alpha=0.7, label=LABELS[a])
        for a in ABLATIONS
    ] + [
        plt.Line2D([0], [0], color="crimson", linewidth=1.4,
                   linestyle="-", label="Random"),
    ]
    ax.legend(handles=legend_handles, fontsize=15, frameon=True,
              facecolor="white", loc="lower right",
              handlelength=1.0, handleheight=0.5, ncol=3)

######################################################################
if __name__ == "__main__":
    main()
