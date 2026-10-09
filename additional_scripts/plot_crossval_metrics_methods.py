#!/usr/bin/env python3

"""
Parses grouped-fold seed logs and writes both method-comparison figures.

    result_crossval_metrics_mlp_idx.png   MLP vs interface score
    result_crossval_metrics_methods.pdf   MLP, interface score, and the
                                          other baselines

Skips jumbled files and pose-set ablation logs.

Template generated with Cursor, modified by accg. 

Usage:
    python plot_crossval_metrics_methods_and_mlp_idx.py
"""

import re
import glob
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FILE_PATTERN = "../out_crossval_repeats_grouped/seed_*.txt"

SECTIONS = ["BLIP-L", "ALIP-L"]
TITLES = {"BLIP-L": "Binder prediction", "ALIP-L": "Activity prediction"}
METRICS = ["Accuracy", "MCC", "AUROC", "AUPRC"]
MLP_IDX_METHODS = ["mlp", "idx"]
ALL_METHODS = ["mlp", "idx", "logreg", "xgboost", "lightgbm", "rf"]
METHOD_RE = re.compile(
    r"^seed_\d+(?:_(idx|logreg|xgboost|lightgbm|rf))?\.txt$"
)

# Per-section positive rate (fraction of positive examples).
# Used for AUPRC and Accuracy random-guess baselines.
POSITIVE_RATE = {
    "BLIP-L": 0.17,
    "ALIP-L": 0.11,
}

VIOLIN_WIDTH = 0.18
GROUP_SPACING = 1.6
JITTER_SEED = 0

COLORS = {
    "mlp":      "#4C72B0",
    "idx":      "#8172B3",
    "logreg":   "#55A868",
    "xgboost":  "#C44E52",
    "lightgbm": "#CCB974",
    "rf":       "#64B5CD",
}

LABELS = {
    "mlp":      "MLP",
    "idx":      "Interface score",
    "logreg":   "LogReg",
    "xgboost":  "XGBoost",
    "lightgbm": "LightGBM",
    "rf":       "RF",
}

######################################################################

def main():
    data = collect(FILE_PATTERN, ALL_METHODS)
    plot_mlp_idx(data)
    plot_methods(data)

######################################################################

def random_guess(section: str) -> dict:
    p = POSITIVE_RATE[section]
    return {
        "AUROC":    0.5,
        "AUPRC":    p,
        "Accuracy": p,   # uniform random; use max(p, 1-p) for majority-class
        "MCC":      0.0,
    }


def plot_mlp_idx(data: dict):
    plt.rcParams["font.size"] = 12
    rng = np.random.default_rng(JITTER_SEED)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), constrained_layout=True)
    for ax, section in zip(axes, SECTIONS):
        violin_panel(
            ax, data[section], section, MLP_IDX_METHODS, METRICS, rng,
            title=TITLES[section], title_size=12,
        )

    axes[0].legend(
        handles=mlp_idx_legend_handles(),
        loc="lower left",
        fontsize=10,
        frameon=True,
        facecolor="white",
        handlelength=1.0,
        handleheight=0.5,
    )

    out = "result_crossval_metrics_mlp_idx.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved → {out}")


def plot_methods(data: dict):
    plt.rcParams["font.size"] = 16
    rng = np.random.default_rng(JITTER_SEED)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for ax, section in zip(axes, SECTIONS):
        violin_panel(
            ax, data[section], section, ALL_METHODS, METRICS, rng,
            title=section, title_size=16,
        )
        ax.legend(
            handles=methods_legend_handles(),
            fontsize=16, frameon=True,
            facecolor="white", loc="lower right",
            handlelength=1.0, handleheight=0.5, ncol=2,
        )

    out = "result_crossval_metrics_methods.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved → {out}")

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
                r"(AUROC|AUPRC|Accuracy|MCC)\s*:\s*([0-9.eE+\-]+)",
                line, re.IGNORECASE,
            )
            if m:
                metric = m.group(1)
                for official in ("AUROC", "AUPRC", "Accuracy", "MCC"):
                    if official.lower() == metric.lower():
                        metric = official
                        break
                try:
                    results[current_section][metric].append(float(m.group(2)))
                except ValueError:
                    pass

    return results


def method_from_name(name: str):
    # seed_42.txt → mlp; seed_42_logreg.txt → logreg; *_jumbled.txt → None
    if "jumbled" in name:
        return None
    m = METHOD_RE.match(name)
    if not m:
        return None
    return m.group(1) if m.group(1) else "mlp"


def collect(pattern: str, methods: list) -> dict:
    files = sorted(glob.glob(pattern))
    by_method = defaultdict(list)
    for fp in files:
        method = method_from_name(Path(fp).name)
        if method not in methods:
            continue
        by_method[method].append(fp)

    for method in methods:
        names = [Path(f).name for f in by_method.get(method, [])]
        print(f"[{method:>8}]  {len(names)} file(s): {names}")

    # data[section][method][metric] = list of fold values
    agg = {s: {m: defaultdict(list) for m in methods} for s in SECTIONS}
    for method, fps in by_method.items():
        for fp in fps:
            parsed = parse_file(fp)
            for section in SECTIONS:
                for metric, vals in parsed[section].items():
                    agg[section][method][metric].extend(vals)
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
            pc.set_alpha(0.65)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_edgecolor(color)
                parts[key].set_linewidth(1.5)

    jitter = rng.uniform(-0.05, 0.05, size=len(values))
    ax.scatter(
        np.full(len(values), pos) + jitter,
        values,
        s=15, color=color, alpha=0.85,
        edgecolors="white", linewidths=0.35, zorder=3,
    )


def violin_panel(ax, data: dict, section: str, methods: list, metrics: list,
                 rng: np.random.Generator, title: str, title_size: int):
    n = len(methods)
    offsets = (np.arange(n) - (n - 1) / 2.0) * (VIOLIN_WIDTH + 0.04)
    group_centres = np.arange(len(metrics)) * GROUP_SPACING
    baselines = random_guess(section)

    for i, metric in enumerate(metrics):
        baseline = baselines[metric]
        half = VIOLIN_WIDTH / 2
        for j, method in enumerate(methods):
            pos = group_centres[i] + offsets[j]
            draw_violin(
                ax,
                pos,
                data[method].get(metric, []),
                COLORS[method],
                rng,
            )
            ax.hlines(
                baseline,
                xmin=pos - half, xmax=pos + half,
                colors="crimson", linewidths=1.4,
                linestyles="-", zorder=4,
            )

    for gc in group_centres:
        ax.axvline(gc, color="lightgrey",
                   linewidth=0.8, linestyle="--", zorder=0)

    ax.set_xticks(group_centres)
    ax.set_xticklabels(metrics)
    ax.set_xlim(group_centres[0] - GROUP_SPACING / 2,
                group_centres[-1] + GROUP_SPACING / 2)
    ax.set_title(title, fontsize=title_size)
    ax.set_ylabel("Value")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_ylim(-0.45, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def mlp_idx_legend_handles():
    return [
        plt.Line2D([0], [0], color="crimson", linewidth=1.4,
                   linestyle="-", label="Random"),
        mpatches.Patch(facecolor=COLORS["mlp"], alpha=0.7, label=LABELS["mlp"]),
        mpatches.Patch(facecolor=COLORS["idx"], alpha=0.7, label=LABELS["idx"]),
    ]


def methods_legend_handles():
    return [
        mpatches.Patch(facecolor=COLORS[m], alpha=0.7, label=LABELS[m])
        for m in ALL_METHODS
    ] + [
        plt.Line2D([0], [0], color="crimson", linewidth=1.4,
                   linestyle="-", label="Random"),
    ]

######################################################################
if __name__ == "__main__":
    main()
