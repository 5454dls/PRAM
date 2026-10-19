"""Supplementary Fig. S1  eICU-CRD decision curves, 42 columns, bank of 5,000 patients, thresholds 2-30%. Frozen logistic versus PRAM (logistic).

Input (folder results_csv/ or PRAM_FIGURE_CSV)
  A3_dca_curves_matrix.csv  columns: fs, outcome, target, bank, method, threshold, nb_median, nb_treat_all
                            filter: fs=F42, target=eICU, bank=5000, method in b1_base / pram_b1, threshold 0.02-0.30
The legend uses the term PRAM of the main text. The figure draws the csv values as they are.
"""
from figure_common import *

OUTS = ["AKI_168h", "mortality_168h"]
YLIM = {"AKI_168h": (-0.012, 0.045), "mortality_168h": (-0.022, 0.036)}


def main():
    style()
    d = read("A3_dca_curves_matrix")
    d["bank"] = d["bank"].astype(str)
    d = d[(d.fs == "F42") & (d.target == "eICU") & (d.bank == "5000") & d.method.isin(["b1_base", "pram_b1"])
          & (d.threshold >= 0.02 - 1e-9) & (d.threshold <= 0.30 + 1e-9)]
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 2.9))
    rows = []
    for j, oc in enumerate(OUTS):
        ax = axes[j]
        g = d[d.outcome == oc]
        ta = g[g.method == "b1_base"].sort_values("threshold")
        ax.plot(ta.threshold, ta.nb_treat_all, color=C["grey"], lw=1.0, ls=(0, (1, 1.5)), label="Treat all")
        ax.axhline(0, color=C["black"], lw=0.9, label="Treat none")
        for t_, v in zip(ta.threshold, ta.nb_treat_all):
            rows.append(dict(panel="ab"[j], outcome=OUTCOME[oc], series="Treat all", threshold=t_, net_benefit=v))
            rows.append(dict(panel="ab"[j], outcome=OUTCOME[oc], series="Treat none", threshold=t_, net_benefit=0.0))
        for m in ["b1_base", "pram_b1"]:
            s = STRAT[m]
            gm = g[g.method == m].sort_values("threshold")
            ax.plot(gm.threshold, gm.nb_median, color=s["color"], ls="-", lw=s["lw"], marker=s["marker"], ms=3.0,
                    mec="white", mew=0.3, label="PRAM" if m == "pram_b1" else s["label"], zorder=4)
            for t_, v in zip(gm.threshold, gm.nb_median):
                rows.append(dict(panel="ab"[j], outcome=OUTCOME[oc], series="PRAM" if m == "pram_b1" else s["label"],
                                 threshold=t_, net_benefit=v))
        ax.set_ylim(*YLIM[oc])
        ax.set_xlim(0.015, 0.305)
        ax.set_xticks(np.arange(0.02, 0.301, 0.04))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%d" % round(v * 100)))
        ax.set_xlabel("Threshold probability (%)")
        ax.set_title(OUTCOME[oc], loc="left", fontsize=8.5)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel("Net benefit")
        panel_letter(ax, "ab"[j], dx=-0.16, dy=1.02)
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(x) for x in ["Frozen logistic", "PRAM", "Treat all", "Treat none"]]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.2)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.91, bottom=0.26, wspace=0.18)
    save(fig, "FigS1")
    write_values(pd.DataFrame(rows), "FigS1")
    # for comparison with the legend
    for oc in OUTS:
        for m in ["b1_base", "pram_b1"]:
            gm = d[(d.outcome == oc) & (d.method == m)].set_index("threshold").nb_median
            print(oc, m, ["%+.4f" % gm.loc[round(t, 2)] for t in (0.10, 0.15, 0.20, 0.30)])


if __name__ == "__main__":
    main()
