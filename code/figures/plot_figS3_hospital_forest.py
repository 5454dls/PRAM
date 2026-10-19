"""Supplementary Fig. S3  AUROC change per eICU-CRD hospital, PRAM on logistic, 42 columns. Top: forest plot ordered by the estimate (standard DeLong 95% interval), bottom: estimate versus standard error.

Input (folder results_csv/ or PRAM_FIGURE_CSV)
  A6_variance_diagnosis_by_hospital.csv  columns: outcome, site, n_site, d_auroc, var_delong_standard
        interval = d_auroc +- 1.96 * sqrt(var_delong_standard)   (paired DeLong 95% interval with the standard variance)
  hospital_summary.csv  columns: fs=F42, method=pram_b1, delta_stays_weighted  (solid line = mean weighted by stays)
  hospital_results_F42.csv  (comparison: d_auroc_b1 of pram_b1 must equal d_auroc of A6)
"""
from figure_common import *

OUTS = ["AKI_168h", "mortality_168h"]


def main():
    style()
    v = read("A6_variance_diagnosis_by_hospital")
    hs = read("hospital_summary")
    hr = read("hospital_results_F42", usecols=["outcome", "site", "method", "d_auroc_b1"])
    hr = hr[hr.method == "pram_b1"]
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 6.6), gridspec_kw={"height_ratios": [1.55, 1]})
    rows = []
    for j, oc in enumerate(OUTS):
        g = v[v.outcome == oc].sort_values("d_auroc").reset_index(drop=True)
        chk = hr[hr.outcome == oc].set_index("site").d_auroc_b1.reindex(g.site).to_numpy()
        assert np.nanmax(np.abs(chk - g.d_auroc.to_numpy())) < 1e-6, "the change of A6 and of hospital_results differ"
        se = np.sqrt(g.var_delong_standard.to_numpy())
        d = g.d_auroc.to_numpy()
        pooled = float(hs[(hs.fs == "F42") & (hs.outcome == oc) & (hs.method == "pram_b1")].delta_stays_weighted.iloc[0])
        ax = axes[0, j]
        y = np.arange(len(g))
        ax.hlines(y, d - 1.96 * se, d + 1.96 * se, color="#8c8c8c", lw=0.7)
        ax.plot(d, y, "o", ms=2.8, color=C["blue"], mec="none")
        ax.axvline(0, color=C["black"], lw=0.8, ls=(0, (3, 2)))
        ax.axvline(pooled, color=C["vermilion"], lw=1.5)
        ax.set_yticks([])
        ax.set_ylim(-3.2, len(g) + 0.5)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
        ax.set_title("%s, %d hospitals" % (OUTCOME[oc], len(g)), loc="left", fontsize=8.5)
        ax.set_xlabel("Change in AUROC against the frozen logistic model")
        if j == 0:
            ax.set_ylabel("Hospitals, ordered by the estimate")
        ax.spines["left"].set_visible(False)
        ax.text(pooled + 0.002, -1.4, "%+.4f" % pooled, color=C["vermilion"], fontsize=8, ha="left", va="center")
        ax2 = axes[1, j]
        ax2.plot(d, se, "o", ms=3, color=C["blue"], alpha=0.8, mec="none")
        ax2.axvline(pooled, color=C["vermilion"], lw=1.5)
        ax2.axvline(0, color=C["black"], lw=0.8, ls=(0, (3, 2)))
        ax2.invert_yaxis()
        ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
        ax2.set_xlim(-0.035, 0.092) if oc == "AKI_168h" else ax2.set_xlim(-0.028, 0.083)
        ax2.set_xlabel("Change in AUROC")
        if j == 0:
            ax2.set_ylabel("Standard error (standard DeLong)")
        ax2.grid(axis="y", color="#dddddd", lw=0.5)
        ax2.set_axisbelow(True)
        for k in range(len(g)):
            rows.append(dict(outcome=OUTCOME[oc], rank_by_estimate=k + 1, hospital=g.site[k], n_stays=int(g.n_site[k]),
                             change_in_auroc=d[k], standard_error=se[k], ci_low=d[k] - 1.96 * se[k],
                             ci_high=d[k] + 1.96 * se[k], stay_weighted_mean=pooled))
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="none", color=C["blue"], ms=3.5, label="Hospital estimate, with 95% interval"),
         Line2D([], [], color=C["vermilion"], lw=1.5, label="Mean weighted by stays"),
         Line2D([], [], color=C["black"], lw=0.8, ls=(0, (3, 2)), label="Zero")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.095, right=0.99, top=0.95, bottom=0.10, wspace=0.17, hspace=0.30)
    save(fig, "FigS3")
    write_values(pd.DataFrame(rows), "FigS3")


if __name__ == "__main__":
    main()
