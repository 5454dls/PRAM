"""Fig. 1  PRAM. a: flow at the time of prediction, b: Algorithm 1.

No input csv. All values in the figure are constants of the method description (they are written to Fig1_values.csv with the place in the paper that states each of them).
  k = 50 neighbors, cosine distance, inverse-distance weighting (1e-6 in the denominator), mixing weight a on the grid 0 to 0.50 in steps of 0.05,
  3-fold cross-validation inside the bank, selection criterion = maximum mean AUROC, when the bank is replaced a is chosen again and the model parameters do not change.
"""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from figure_common import *

GREY_FILL = "#eeeeee"
BLUE_FILL = "#dcecf7"
VERM_FILL = "#fbe5d6"


def box(ax, x, y, w, h, text, fc, ec, fs=8, bold=False, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            linespacing=1.25)


def arrow(ax, p, q, color="black", lw=1.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color,
                                 connectionstyle="arc3,rad=%s" % rad, shrinkA=0, shrinkB=0))


def main():
    style()
    fig = plt.figure(figsize=(W_DOUBLE, 5.7))
    axa = fig.add_axes([0.0, 0.46, 1.0, 0.54])
    axb = fig.add_axes([0.0, 0.0, 1.0, 0.44])
    for ax in (axa, axb):
        ax.set_xlim(0, 100)
        ax.axis("off")
    axa.set_ylim(0, 50)
    axb.set_ylim(0, 48)

    # ---- a ----
    axa.text(0.5, 49.5, "a", fontsize=10, fontweight="bold", va="top")
    box(axa, 3, 20, 14, 10, "New patient\n$x$", "white", "black")
    # top: frozen model
    box(axa, 27, 35, 24, 11, "Frozen model $f$\nunchanged", GREY_FILL, "black")
    box(axa, 58, 35, 14, 11, "$p_{base}(x)$", "white", "black")
    # bottom: bank
    box(axa, 27, 4, 24, 11, "Bank $B$\nlocal patients with\nknown outcomes", BLUE_FILL, C["blue"])
    box(axa, 27, 20, 24, 10, "50 nearest bank patients\nby cosine distance", BLUE_FILL, C["blue"])
    box(axa, 58, 20, 17, 10, "Mean of their\noutcomes, weight\n1 / distance", BLUE_FILL, C["blue"])
    box(axa, 58, 4, 14, 11, "$p_{retr}(x)$", "white", C["blue"])
    # mixing
    box(axa, 80, 15, 19, 20, "Mix\n$p_{mix}(x) =$\n$(1-a)\\,p_{base}(x)$\n$+\\,a\\,p_{retr}(x)$", VERM_FILL, C["vermilion"],
        bold=False, lw=1.2)
    # arrows
    arrow(axa, (17, 27), (27, 40.5))
    arrow(axa, (17, 25), (27, 25))
    arrow(axa, (39, 15), (39, 20))
    arrow(axa, (51, 40.5), (58, 40.5))
    arrow(axa, (51, 25), (58, 25))
    arrow(axa, (66.5, 20), (66.5, 15))
    arrow(axa, (72, 40.5), (80, 30), rad=-0.1)
    arrow(axa, (72, 9.5), (80, 20), rad=0.1)
    axa.text(81, 12.5, "$a$: 0 to 0.50, step 0.05,"+chr(10)+"chosen by AUROC in"+chr(10)+"3-fold cross-validation"+chr(10)+"inside $B$",
             ha="left", va="top", fontsize=8, color=C["vermilion"], linespacing=1.25)
    axa.text(27, 48.3, "No parameter of $f$ is refit.", fontsize=8, color=C["grey"], va="bottom")
    axa.text(41, 17.5, "Adapting to a hospital replaces $B$.", fontsize=8, color=C["blue"], va="center")
    axa.set_ylim(0, 52)

    # ---- b ----
    axb.text(0.5, 47.5, "b", fontsize=10, fontweight="bold", va="top")
    axb.add_patch(FancyBboxPatch((3, 1), 94, 44.5, boxstyle="round,pad=0.0,rounding_size=1.2", fc="white", ec="black", lw=0.9))
    lines = [
        ("Algorithm 1. PRAM", True),
        ("Input: frozen model $f$; bank $B$ of local patients $(x, y)$ with known outcomes; weights $A = \\{0, 0.05, \\ldots, 0.50\\}$; $k = 50$.", False),
        ("Select $a$:", True),
        ("   1. Split $B$ into three folds, stratified by outcome.", False),
        ("   2. For each patient $x$ in a fold, find the $k$ nearest patients in the other two folds by cosine distance $d$ and set", False),
        ("       $p_{retr}(x) = \\sum y\\,/\\,(d + 10^{-6})\\ /\\ \\sum 1\\,/\\,(d + 10^{-6})$.", False),
        ("   3. For each $a$ in $A$, compute the AUROC of $(1 - a)\\,f(x) + a\\,p_{retr}(x)$ in each fold, and average over folds.", False),
        ("   4. Choose the $a$ with the highest mean AUROC.", False),
        ("Predict:", True),
        ("   For a new patient $x$, find the $k$ nearest patients in $B$, compute $p_{retr}(x)$ in the same way and return $(1 - a)\\,f(x) + a\\,p_{retr}(x)$.", False),
    ]
    y = 42.5
    for t, b in lines:
        axb.text(5.5, y, t, fontsize=8, fontweight="bold" if b else "normal", va="center", ha="left")
        y -= 4.2
    save(fig, "Fig1")

    vals = [
        ("nearest neighbours k", 50, "Methods, PRAM: 'The 50 bank patients nearest to x'"),
        ("distance", "cosine", "Methods, PRAM"),
        ("neighbour weight", "inverse distance, +1e-6 in denominator", "Fig. 1 legend, Algorithm 1"),
        ("mixing weight grid", "0 to 0.50, step 0.05", "Methods, PRAM"),
        ("weight selection", "highest mean AUROC, 3-fold cross-validation inside the bank, stratified by outcome",
         "Methods, PRAM; Fig. 1 legend"),
        ("model refit", "none", "Methods, PRAM: 'No model parameter is refit.'"),
    ]
    write_values(pd.DataFrame(vals, columns=["item", "value_drawn", "source_sentence"]), "Fig1")


if __name__ == "__main__":
    main()
