"""Fig. 3  AUROC change per eICU-CRD hospital (against the frozen logistic). 2 outcomes x 2 variable sets, hospital points + median + 10th to 90th percentile bar.

Input (folder results_csv/ or PRAM_FIGURE_CSV)
  hospital_results_F42.csv, hospital_results_F33.csv
      columns: outcome, site, method, d_auroc_b1 (= AUROC of the method minus AUROC of the frozen logistic in that hospital)
      filter: method in pram_b1 / local_lr / local_cb
  hospital_summary.csv  (for comparison: delta_median, delta_p10, delta_p90)
The bar is not a confidence interval (see the legend). The horizontal scatter of the points is jitter fixed with seed 0 and is not a value.
"""
from figure_common import *

METHODS = ["pram_b1", "local_lr", "local_cb"]
FSETS = [("F42", "42 variables"), ("F33", "33 variables")]
OUTS = ["AKI_168h", "mortality_168h"]


def main():
    style()
    rng = np.random.default_rng(0)
    summ = read("hospital_summary")
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 5.0), sharey=True)
    pts, stats = [], []
    for i, (fs, fl) in enumerate(FSETS):
        h = read("hospital_results_%s" % fs, usecols=["outcome", "site", "method", "d_auroc_b1"])
        for j, oc in enumerate(OUTS):
            ax = axes[i, j]
            ax.axhline(0, color=C["black"], lw=0.8, zorder=1)
            ax.grid(axis="y", color="#dddddd", lw=0.5)
            ax.set_axisbelow(True)
            for k, m in enumerate(METHODS):
                g = h[(h.outcome == oc) & (h.method == m)].sort_values("site")
                d = g.d_auroc_b1.to_numpy()
                jit = rng.uniform(-0.28, 0.28, len(d))
                col = STRAT[m]["color"]
                ax.scatter(k + jit, d, s=7, color=col, alpha=0.45, lw=0, zorder=2)
                med, p10, p90 = np.median(d), np.percentile(d, 10), np.percentile(d, 90)
                ax.plot([k, k], [p10, p90], color="black", lw=1.6, solid_capstyle="butt", zorder=4)
                ax.plot([k - 0.12, k + 0.12], [p10, p10], color="black", lw=1.2, zorder=4)
                ax.plot([k - 0.12, k + 0.12], [p90, p90], color="black", lw=1.2, zorder=4)
                ax.plot([k], [med], marker="D", ms=5.2, mfc="white", mec="black", mew=1.2, zorder=5)
                for s_, dv in zip(g.site, d):
                    pts.append(dict(variable_set=fs, outcome=OUTCOME[oc], strategy=STRAT[m]["label"], hospital=s_,
                                    change_in_auroc=dv))
                sm = summ[(summ.fs == fs) & (summ.outcome == oc) & (summ.method == m)].iloc[0]
                stats.append(dict(variable_set=fs, outcome=OUTCOME[oc], strategy=STRAT[m]["label"], n_hospitals=len(d),
                                  median=med, p10=p10, p90=p90, summary_csv_median=sm.delta_median,
                                  summary_csv_p10=sm.delta_p10, summary_csv_p90=sm.delta_p90))
                assert abs(med - sm.delta_median) < 1e-6 and abs(p10 - sm.delta_p10) < 1e-6 \
                    and abs(p90 - sm.delta_p90) < 1e-6, (fs, oc, m)
            n = len(g)
            ax.set_title("%s, %s (%d hospitals)" % (OUTCOME[oc], fl, n), loc="left", fontsize=8.5)
            ax.set_xticks(range(3))
            ax.set_xticklabels(["PRAM on\nlogistic", "Local\nlogistic", "Local\nCatBoost"])
            ax.set_xlim(-0.55, 2.55)
            ax.set_ylim(-0.29, 0.20)
            if j == 0:
                ax.set_ylabel("Change in AUROC\nagainst frozen logistic")
    from matplotlib.lines import Line2D
    h_ = [Line2D([], [], marker="o", ls="none", color="#888888", ms=3.5, alpha=0.7, label="One hospital"),
          Line2D([], [], marker="D", ls="none", mfc="white", mec="black", mew=1.2, ms=5.2, label="Median of hospitals"),
          Line2D([], [], color="black", lw=1.6, label="10th to 90th percentile")]
    fig.legend(handles=h_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.108, right=0.995, top=0.95, bottom=0.145, wspace=0.07, hspace=0.45)
    save(fig, "Fig3")
    pts = pd.DataFrame(pts)
    pts.insert(0, "kind", "hospital_point")
    st = pd.DataFrame(stats)
    st.insert(0, "kind", "summary")
    write_values(pd.concat([pts, st], ignore_index=True), "Fig3")


if __name__ == "__main__":
    main()
