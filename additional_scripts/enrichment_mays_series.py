#!/usr/bin/env python3

"""Enrichment factor for the pooled Mays congeneric series.

Loads the design compounds the same way as predict_testset_rjw100_series.py
and scores them with the saved BLiP-L and ALiP-L models. Enrichment is
computed once on that pooled list. 

Mays compounds (numeric ec50_avg) are active and Mays_ia compounds (ec50_avg
labeled "ia") are inactive.

Interface score is ranked more favorable first (lower REU). Binder and
activity predictions are ranked more favorable first (higher probability).

    EF(n) = (a / n) / (A / N)

N is the number of pooled compounds, A is the number of Mays actives, n is
the size of the top-ranked slice, and a is the number of actives in that
slice. Ties at the cutoff are averaged the same way as
enrichment_counts_tieavg in analyze_vu98k_preds.py, so a can be fractional.
Random ranking gives EF = 1.

Template generated with Cursor, modified by accg. 

Usage, from the repository root or from additional_scripts:

    python additional_scripts/enrichment_mays_series.py
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

plt.rcParams["font.size"] = 12

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / "data" / "xtal_rjw100_series"
sys.path.insert(0, str(REPO))

from predict_testset_rjw100_series import predHoldOutSet  # noqa: E402

ETERMS = [
    "if_X_fa_atr",
    "if_X_fa_elec",
    "if_X_fa_rep",
    "if_X_fa_sol",
    "if_X_hbond_bb_sc",
    "if_X_hbond_sc",
]
METRICS = [
    ("interface_delta_X", True, "Interface score"),
    ("sigmoid_binder_pred", False, "Binder prediction"),
    ("sigmoid_activity_pred", False, "Activity prediction"),
]
TOP_K = (2, 5)
FRACTIONS = (0.10, 0.20)
METRIC_COLORS = {
    "Interface score": "peru",
    "Binder prediction": "teal",
    "Activity prediction": "limegreen",
}


def series_label(ec50_avg):
    """Map an experimental label to mays (active) or mays_ia (inactive)."""
    text = str(ec50_avg).strip().lower()
    if text == "ia":
        return "mays_ia"
    if pd.notna(pd.to_numeric(text, errors="coerce")):
        return "mays"
    return None


def load_mays():
    """Read the pooled Mays and Mays_ia analogs and append model predictions."""
    eterms_csv = DATA / "out_eterms_design.csv"
    master_csv = DATA / "master_design.csv"
    ifname = str(eterms_csv)
    refmap = [str(master_csv), "ligand", ifname, "description"]
    df = predHoldOutSet(
        "design",
        str(REPO / "blip-l.pth"),
        str(REPO / "alip-l.pth"),
        ETERMS,
        ifname,
        refmap,
    )
    df = df.copy()
    df["sigmoid_binder_pred"] = 1 / (1 + np.exp(-df["binder_pred"]))
    df["sigmoid_activity_pred"] = 1 / (1 + np.exp(-df["activity_pred"]))
    df["series"] = df["ec50_avg"].map(series_label)
    df = df.loc[df["series"].isin(["mays", "mays_ia"])].copy()
    df["active"] = (df["series"] == "mays").astype(int)
    df["label"] = df["description"].map(short_label)
    n_mays = int((df["series"] == "mays").sum())
    n_mays_ia = int((df["series"] == "mays_ia").sum())
    if n_mays == 0 or n_mays_ia == 0:
        sys.exit(
            "Enrichment factor needs both classes in one ranked list. "
            f"Found Mays actives={n_mays}, Mays_ia inactives={n_mays_ia}."
        )
    return df


def short_label(description):
    """Map a design filename to the R-group label used in the manuscript."""
    name = str(description)
    parts = name.split("_")
    # 5l11_XR1_endo_6_0001 -> R1_6N ; 5l11_XR2_10_0001 -> R2_10
    if len(parts) >= 4 and parts[1].startswith("XR") and parts[2] in ("endo", "exo"):
        stereo = "N" if parts[2] == "endo" else "X"
        return f"R{parts[1][-1]}_{int(parts[3])}{stereo}"
    if len(parts) >= 3 and parts[1].startswith("XR"):
        return f"R{parts[1][-1]}_{int(parts[2])}"
    return name


def expected_actives(df, sort_col, ascending, n):
    """Actives recovered in the top n, averaging compounds tied at the cutoff."""
    ordered = df.sort_values(sort_col, ascending=ascending, kind="mergesort")
    ordered = ordered.reset_index(drop=True)
    cutoff = ordered.iloc[n - 1][sort_col]
    if ascending:
        before = ordered[sort_col] < cutoff
    else:
        before = ordered[sort_col] > cutoff
    tied = ordered[sort_col] == cutoff
    remaining = n - int(before.sum())
    hits_before = ordered.loc[before, "active"].sum()
    tied_hit_fraction = ordered.loc[tied, "active"].mean()
    return float(hits_before + remaining * tied_hit_fraction)


def enrichment_factor(df, sort_col, ascending, n):
    """EF(n) = (a/n) / (A/N). Returns the factor and the actives recovered."""
    n_total = len(df)
    n_actives = int(df["active"].sum())
    recovered = expected_actives(df, sort_col, ascending, n)
    prevalence = n_actives / n_total
    factor = (recovered / n) / prevalence
    return factor, recovered


def slice_sizes(n_total):
    """Top-k sizes and top-percent sizes. Shared n values get one row."""
    named = {}
    def add(name, n):
        named.setdefault(n, []).append(name)

    for k in TOP_K:
        if 1 <= k <= n_total:
            add(f"Top {k}", k)
    for fraction in FRACTIONS:
        n = max(1, min(n_total, int(math.floor(fraction * n_total))))
        add(f"top {fraction:.0%}", n)
    return [(", ".join(names), n) for n, names in named.items()]


def enrichment_table(df):
    n_total = len(df)
    n_actives = int(df["active"].sum())
    rows = []
    for sort_col, ascending, metric in METRICS:
        for slice_name, n in slice_sizes(n_total):
            factor, recovered = enrichment_factor(df, sort_col, ascending, n)
            rows.append(
                {
                    "metric": metric,
                    "N": n_total,
                    "actives": n_actives,
                    "slice": slice_name,
                    "n": n,
                    "actives_recovered": recovered,
                    "EF": factor,
                }
            )
    return pd.DataFrame(rows)


def print_ranking(df):
    n_mays = int((df["series"] == "mays").sum())
    n_mays_ia = int((df["series"] == "mays_ia").sum())
    print(
        f"\nPooled congeneric series. N={len(df)}, "
        f"Mays actives={n_mays}, Mays_ia inactives={n_mays_ia}"
    )
    for sort_col, ascending, metric in METRICS:
        ordered = df.sort_values(sort_col, ascending=ascending, kind="mergesort")
        top = ordered.head(5)
        labels = [
            f"{row.label} ({'active' if row.active else 'inactive'}, EC50={row.ec50_avg})"
            for row in top.itertuples()
        ]
        print(f"  {metric} top 5: {'; '.join(labels)}")


def plot_enrichment(table):
    """Scatter of enrichment factor, styled like the VU98k enrichment plot."""
    slices = list(dict.fromkeys(table["slice"]))
    metrics = [name for _, _, name in METRICS]
    markers = {
        "Interface score": "s",
        "Binder prediction": "^",
        "Activity prediction": "H",
    }
    x = np.arange(len(slices))
    width = 0.18

    fig, ax = plt.subplots(1, figsize=(5, 3))
    for i, metric in enumerate(metrics):
        vals = [
            table.loc[
                (table["metric"] == metric) & (table["slice"] == sl), "EF"
            ].iloc[0]
            for sl in slices
        ]
        offset = (i - (len(metrics) - 1) / 2) * width
        ax.scatter(
            x + offset,
            vals,
            label=metric,
            marker=markers[metric],
            c=METRIC_COLORS[metric],
            edgecolor="k",
            s=40,
            zorder=3,
        )

    tick_labels = []
    for sl in slices:
        n = int(table.loc[table["slice"] == sl, "n"].iloc[0])
        tick_labels.append(f"{sl}\n(n={n})")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Enrichment factor")
    ax.set_xlabel("Ranked slice")
    ax.set_ylim([0,2.55])
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "plot_mays_enrichment_factor.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main():
    if not (DATA / "out_eterms_design.csv").is_file():
        sys.exit(f"Missing Mays energy terms: {DATA / 'out_eterms_design.csv'}")

    design = load_mays()
    if len(design) == 0:
        sys.exit("No design compounds loaded.")
    print_ranking(design)
    result = enrichment_table(design)
    out_png = plot_enrichment(result)
    print(f"\nWrote {out_png}")


if __name__ == "__main__":
    main()
