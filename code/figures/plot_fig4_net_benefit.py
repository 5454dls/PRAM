"""Fig. 4  a: decision curves of the whole pool of eICU-CRD banks, outcome-specific threshold range. b: fraction of hospitals whose first strategy by AUROC differs from the first strategy by net benefit.

Input (folder results_csv/ or PRAM_FIGURE_CSV)
  A3_dca_curves_matrix.csv       columns: fs, outcome, target, bank, method, threshold, nb_median, nb_treat_all
                                 filter: fs=F42, target=eICU, bank=full, the 8 main strategies
                                 threshold range: AKI 0.02-0.10, death 0.10-0.30 (range of the Methods)
  A3_disagreement_by_threshold.csv  columns: fs, outcome, threshold, share_disagree, n_hospitals
                                 filter: fs=F42, thresholds 0.02-0.30
The representative thresholds 6% (AKI) and 20% (death) are marked with vertical dotted lines. The open circle of b is also placed at the same
reporting threshold (AKI 6%, death 20%). Both 6% and 20% are on the csv grid (steps of 0.01).
"""
from figure_common import *

RANGE = {"AKI_168h": (0.02, 0.10), "mortality_168h": (0.10, 0.30)}
REP = {"AKI_168h": 0.06, "mortality_168h": 0.20}
OUTS = ["AKI_168h", "mortality_168h"]


def main():
    style()
    d = read("A3_dca_curves_matrix")
    d["bank"] = d["bank"].astype(str)
    d = d[(d.fs == "F42") & (d.target == "eICU") & (d.bank == "full")]
    dis = read("A3_disagreement_by_threshold")
    dis = dis[(dis.fs == "F42") & (dis.threshold >= 0.02 - 1e-9) & (dis.threshold <= 0.30 + 1e-9)]
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 5.4))
    rows = []
    for j, oc in enumerate(OUTS):
        lo, hi = RANGE[oc]
        ax = axes[0, j]
        g = d[(d.outcome == oc) & (d.threshold >= lo - 1e-9) & (d.threshold <= hi + 1e-9)]
        ta = g[g.method == "b1_base"].sort_values("threshold")
        ax.plot(ta.threshold, ta.nb_treat_all, color=C["black"], lw=0.9, ls=(0, (1, 1.5)), zorder=2)
        ax.axhline(0, color=C["black"], lw=0.9, zorder=2)
        for t_, v in zip(ta.threshold, ta.nb_treat_all):
            rows.append(dict(panel="a", outcome=OUTCOME[oc], series="Treat all", threshold=t_, value=v))
        for t_ in ta.threshold:
            rows.append(dict(panel="a", outcome=OUTCOME[oc], series="Treat none", threshold=t_, value=0.0))
        for m in MAIN8:
            s = STRAT[m]
            gm = g[g.method == m].sort_values("threshold")
            ax.plot(gm.threshold, gm.nb_median, color=s["color"], ls=s["ls"], lw=s["lw"], marker=s["marker"],
                    ms=3.0, mec="white", mew=0.3, zorder=5 if m == "pram_b1" else 3)
            for t_, v in zip(gm.threshold, gm.nb_median):
                rows.append(dict(panel="a", outcome=OUTCOME[oc], series=s["label"], threshold=t_, value=v))
        ax.axvline(REP[oc], color=C["grey"], lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.text(REP[oc], 1.0, "%d%%" % round(REP[oc] * 100), transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=8, color=C["grey"])
        ax.set_xlim(lo - 0.003 * (hi - lo) / 0.08, hi + 0.003 * (hi - lo) / 0.08)
        ax.set_xticks(np.arange(lo, hi + 1e-9, 0.01 if oc == "AKI_168h" else 0.05))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%d" % round(v * 100)))
        YL = {"AKI_168h": (-0.012, 0.047), "mortality_168h": (-0.022, 0.016)}[oc]
        ax.set_ylim(*YL)
        tmin = float(ta.nb_treat_all.min())
        tmax = float(ta.nb_treat_all.max())
        if tmax < YL[0] + 0.004:
            ax.text(0.98, 0.04, "Treat all: %.2f to %.2f,\nbelow the axis" % (tmin, tmax), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color=C["grey"])
        elif tmin < YL[0]:
            ax.text(0.98, 0.04, "Treat all falls to %.2f\nat %d%%" % (tmin, round(hi * 100)), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color=C["grey"])
        ax.set_xlabel("Threshold probability (%)")
        ax.set_title("%s, %d to %d%%" % (OUTCOME[oc], round(lo * 100), round(hi * 100)), loc="left", fontsize=8.5, pad=11)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel("Net benefit")
        # b
        ax = axes[1, j]
        gd = dis[dis.outcome == oc].sort_values("threshold")
        ax.axvspan(lo, hi, color=C["sky"], alpha=0.18, lw=0)
        ax.plot(gd.threshold, gd.share_disagree * 100, color=C["blue"], lw=1.4, marker="o", ms=2.8, mec="white",
                mew=0.3, zorder=3)
        for t_, v in zip(gd.threshold, gd.share_disagree):
            rows.append(dict(panel="b", outcome=OUTCOME[oc], series="Share of hospitals, different first strategy (%)",
                             threshold=t_, value=v * 100))
        tr = REP[oc]                                   # the open circle is at the reporting threshold of the outcome (AKI 6%, death 20%)
        vr = float(gd[np.isclose(gd.threshold, tr)].share_disagree.iloc[0]) * 100
        rows.append(dict(panel="b", outcome=OUTCOME[oc], series="Open circle: share at reporting threshold (%)",
                         threshold=tr, value=vr))
        ax.plot([tr], [vr], marker="o", ms=6, mfc="white", mec=C["blue"], mew=1.3, zorder=5)
        ax.annotate("%.0f%% at %d%%" % (vr, round(tr * 100)), (tr, vr), xytext=(tr, 38), ha="center", fontsize=8,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=C["grey"]))
        ax.set_title("%s" % OUTCOME[oc], loc="left", fontsize=8.5, pad=4)
        ax.set_ylim(30, 100)
        ax.set_xlim(0.015, 0.305)
        ax.set_xticks(np.arange(0.02, 0.301, 0.04))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%d" % round(v * 100)))
        ax.set_xlabel("Threshold probability (%)")
        if j == 0:
            ax.set_ylabel("Hospitals with a different\nfirst strategy (%)")
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        ax.set_axisbelow(True)
        ax.text(0.5 * (lo + hi), 97, "threshold range", ha="center", va="top", fontsize=8, color=C["grey"])
    panel_letter(axes[0, 0], "a", dx=-0.17, dy=1.10)
    panel_letter(axes[1, 0], "b", dx=-0.17, dy=1.10)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C["black"], lw=0.9, ls=(0, (1, 1.5)), label="Treat all"),
         Line2D([], [], color=C["black"], lw=0.9, label="Treat none")]
    for m in MAIN8:
        s = STRAT[m]
        h.append(Line2D([], [], color=s["color"], ls=s["ls"], lw=s["lw"], marker=s["marker"], ms=3.0, mec="white",
                        mew=0.3, label=s["label"]))
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, columnspacing=1.0, handlelength=2.0,
               bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.115, right=0.99, top=0.855, bottom=0.085, wspace=0.20, hspace=0.50)
    save(fig, "Fig4")
    write_values(pd.DataFrame(rows), "Fig4")
    # output for comparison: the range of fractions given in the text
    for oc in OUTS:
        lo, hi = RANGE[oc]
        gd = dis[(dis.outcome == oc) & (dis.threshold >= lo - 1e-9) & (dis.threshold <= hi + 1e-9)]
        print(oc, "range %.2f-%.2f" % (lo, hi), "disagree min %.1f max %.1f" % (gd.share_disagree.min() * 100,
              gd.share_disagree.max() * 100), "at20 %.1f" % (float(dis[(dis.outcome == oc) &
              np.isclose(dis.threshold, 0.2)].share_disagree.iloc[0]) * 100),
              "at reporting %.0f%% %.1f" % (REP[oc] * 100, float(dis[(dis.outcome == oc) &
              np.isclose(dis.threshold, REP[oc])].share_disagree.iloc[0]) * 100))


if __name__ == "__main__":
    main()
