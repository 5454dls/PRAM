"""Common code of the figure scripts: paths, style, palette and the record of plotted values.

Figure format: vector PDF and 600 dpi PNG, width 174 mm (two columns), Arial 8 pt.
Input: the result csv files of this repository (folder results_csv/ next to code/), or the folder given in the
environment variable PRAM_FIGURE_CSV (for example notebooks/out_pram/github_csv written by notebook 01).
Output: the folder given in PRAM_FIGURE_OUT (default ./figure_out).
"""
import hashlib
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_DIR = Path(os.environ.get("PRAM_FIGURE_CSV", str(HERE.parents[1] / "results_csv")))
CSV2 = CSV5 = CSV_DIR
OUT = Path(os.environ.get("PRAM_FIGURE_OUT", "figure_out"))
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
W_DOUBLE = 174 * MM          # two-column width

# Okabe-Ito palette (safe for color-vision deficiency)
C = dict(
    black="#000000", grey="#6e6e6e", orange="#E69F00", sky="#56B4E9", green="#009E73",
    blue="#0072B2", vermilion="#D55E00", purple="#CC79A7",
)

# Strategy names and styles. The same strategy has the same color and marker in Figures 2, 4 and S1.
STRAT = {
    "b1_base":   dict(label="Frozen logistic",        color=C["black"],     marker="o", ls="-",  lw=1.0),
    "b5_base":   dict(label="Frozen CatBoost",        color=C["grey"],      marker="s", ls="--", lw=1.0),
    "recal_iso": dict(label="Isotonic recalibration", color=C["green"],     marker="^", ls="-.", lw=1.0),
    "pram_b1":   dict(label="PRAM on logistic",       color=C["blue"],      marker="D", ls="-",  lw=1.8),
    "pram_b5":   dict(label="PRAM on CatBoost",       color=C["sky"],       marker="v", ls="-",  lw=1.2),
    "local_lr":  dict(label="Local logistic",         color=C["vermilion"], marker="X", ls=":",  lw=1.3),
    "pooled_lr": dict(label="Pooled logistic",        color=C["purple"],    marker="P", ls="--", lw=1.0),
    "local_cb":  dict(label="Local CatBoost",         color=C["orange"],    marker="<", ls=":",  lw=1.3),
}
MAIN8 = ["b1_base", "b5_base", "recal_iso", "pram_b1", "pram_b5", "local_lr", "pooled_lr", "local_cb"]

OUTCOME = {"AKI_168h": "Acute kidney injury", "mortality_168h": "168 h mortality"}
COHORT = {"eICU": "eICU-CRD", "MIMIC-III": "MIMIC-III", "MIMIC-IV": "MIMIC-IV"}


def style():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "figure.dpi": 100,
        "axes.grid": False,
    })


def read(name, **kw):
    return pd.read_csv(CSV2 / (name + ".csv"), encoding="utf-8-sig", **kw)


def panel_letter(ax, s, dx=-0.0, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def save(fig, stem):
    fig.savefig(OUT / (stem + ".pdf"))
    fig.savefig(OUT / (stem + ".png"), dpi=600)
    plt.close(fig)


def write_values(df, stem):
    """Values used in the figure. Written with a fixed number of decimals so that two runs give the same hash."""
    p = OUT / (stem + "_values.csv")
    df.to_csv(p, index=False, float_format="%.6f", encoding="utf-8", lineterminator="\n")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    print("values", p.name, len(df), "rows sha256", h)
    return h
