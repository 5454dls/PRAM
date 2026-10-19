"""Fig. 2  AUROC change of the strategies by bank size (against the frozen logistic), 3 cohorts x 2 outcomes.

Input (folder results_csv/ or PRAM_FIGURE_CSV)
  matrix_draws_F42.csv   columns: target, outcome, bank, rep, method, auroc
                         filter: method in the 8 main strategies, bank in 500/1000/2000/5000/full
                         change = (method auroc - b1_base auroc) in the same rep, median per bank size (10 reps, 1 for full)
  R5_X_S5A_values.csv    columns: fs, outcome, target, bank, n_draws, d_median, d_p05, ...  (F42, 100 single-bank draws)

The 5th percentile marks (x) come from d_p05 of R5_X_S5A_values.csv (fs F42, n_draws=100). The median lines come from
matrix_draws_F42.csv. The 42-column set is drawn and no band is drawn (only p05 is in the summary).
usage: python fig2_bank_size.py            (default, 42 columns)
"""
import sys

from figure_common import *

FS = "F42"
STEM = "Fig2"
BANKS = ["500", "1000", "2000", "5000", "full"]
XLAB = ["500", "1,000", "2,000", "5,000", "Whole\npool"]
TARGETS = ["eICU", "MIMIC-III", "MIMIC-IV"]
OUTS = ["AKI_168h", "mortality_168h"]
SHOW = [m for m in MAIN8 if m != "b1_base"]          # the frozen logistic is the 0 line


def load():
    m = read("matrix_draws_%s" % FS, usecols=["target", "outcome", "bank", "rep", "method", "auroc"])
    m["bank"] = m["bank"].astype(str)
    m = m[m.method.isin(MAIN8) & m.bank.isin(BANKS)]
    piv = m.pivot_table(index=["target", "outcome", "bank", "rep"], columns="method", values="auroc")
    d = piv.sub(piv["b1_base"], axis=0).drop(columns="b1_base")
    med = d.groupby(["target", "outcome", "bank"]).median()
    nd = piv.groupby(["target", "outcome", "bank"]).size().rename("n_draws")
    return med, nd


def tail_marks():
    """(target, outcome, bank) -> (kind, p05, nan, n_draws). 100 single-bank draws."""
    out = {}
    t = pd.read_csv(CSV5 / "R5_X_S5A_values.csv", encoding="utf-8-sig")
    t = t[t.fs == FS]
    for r in t.itertuples():
        assert int(r.n_draws) == 100, (r.target, r.outcome, r.bank, r.n_draws)
        out[(r.target, r.outcome, str(r.bank))] = ("p05", r.d_p05, np.nan, int(r.n_draws))
    return out


def main():
    style()
    med, nd = load()
    tails = tail_marks()
    fig, axes = plt.subplots(2, 3, figsize=(W_DOUBLE, 4.3), sharex=True, sharey=True)
    x = np.arange(len(BANKS))
    rows = []
    for i, oc in enumerate(OUTS):
        for j, tg in enumerate(TARGETS):
            ax = axes[i, j]
            ax.axhline(0, color=C["black"], lw=0.8)
            for m in SHOW:
                s = STRAT[m]
                y = [med.loc[(tg, oc, b), m] for b in BANKS]
                ax.plot(x, y, color=s["color"], ls=s["ls"], lw=s["lw"], marker=s["marker"], ms=3.6,
                        mec="white", mew=0.4, zorder=5 if m == "pram_b1" else 3)
                for b, v in zip(BANKS, y):
                    rows.append(dict(panel="%s | %s" % (COHORT[tg], OUTCOME[oc]), strategy=s["label"], bank=b,
                                     kind="median_change_auroc",
                                     value=v))
            for k, b in enumerate(BANKS[:4]):
                t = tails.get((tg, oc, b))
                if t is not None:
                    ax.plot([x[k]], [t[1]], marker="x", color=C["blue"], ms=4.5, mew=1.1, zorder=6, ls="none")
                    rows.append(dict(panel="%s | %s" % (COHORT[tg], OUTCOME[oc]), strategy="PRAM on logistic",
                                     bank=b, kind="p05_of_100", value=t[1]))
            ax.axvline(3.5, color=C["grey"], lw=0.5, ls=(0, (2, 2)))
            ax.set_xticks(x)
            ax.set_xticklabels(XLAB)
            ax.set_xlim(-0.35, 4.35)
            ax.set_ylim(-0.105, 0.085)
            ax.grid(axis="y", color="#dddddd", lw=0.5)
            ax.set_axisbelow(True)
            ax.set_title("%s, %s" % (COHORT[tg], OUTCOME[oc].lower() if oc == "AKI_168h" else OUTCOME[oc]),
                         loc="left", fontsize=8.5)
            if j == 0:
                ax.set_ylabel("Change in AUROC\nagainst frozen logistic")
            if i == 1:
                ax.set_xlabel("Local patients in the bank")
    # legend
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C["black"], lw=0.8, label="Frozen logistic (0)")]
    for m in SHOW:
        s = STRAT[m]
        h.append(Line2D([], [], color=s["color"], ls=s["ls"], lw=s["lw"], marker=s["marker"], ms=3.6,
                        mec="white", mew=0.4, label=s["label"]))
    h.append(Line2D([], [], color=C["blue"], marker="x", ls="none", ms=4.5, mew=1.1,
                    label="5th percentile of 100\nsingle PRAM banks"))
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, columnspacing=1.0, handlelength=2.0,
               bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.108, right=0.995, top=0.95, bottom=0.255, wspace=0.07, hspace=0.30)
    save(fig, STEM)
    df = pd.DataFrame(rows)
    df.insert(0, "variable_set", FS)
    write_values(df, STEM)


if __name__ == "__main__":
    main()
