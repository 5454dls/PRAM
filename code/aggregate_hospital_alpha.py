"""Hospital-level statistics of the unlabeled-hospital network (A7) for every mixing weight alpha.

Input : A7_network_F42.csv and A7_network_F33.csv written by notebook 01 (one row = querying hospital x alpha x partner prevalence factor).
Output: P3_hospital_stats_all_alpha.csv, one row per (variable set, outcome, alpha, metric) with the hospital bootstrap interval of the mean,
        the exact sign test and the Wilcoxon signed-rank test over hospitals (Supplementary Table S11 Panel C and Table 6).

The hospital is the sampling unit. Zero differences (|d| <= 1e-12) are dropped from both tests; with fewer than 6 nonzero differences the p values are NaN.
Improvement direction: AUROC and net benefit positive, ECE and the distance of the calibration slope from 1 negative.
No correction for multiple comparison is applied.

usage: python -X utf8 aggregate_hospital_alpha.py [--in-dir DIR] [--out-dir DIR]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pram_config import BOOT_SEED as SEED, N_BOOT  # noqa: E402
from aggregate_claims import wilcoxon_p  # noqa: E402

ZERO_TOL = 1e-12                  # a difference of 0
MIN_NONZERO = 6                   # with fewer nonzero differences the sign test and Wilcoxon are NaN
SIGN = {"AUROC": 1, "ECE": -1, "net benefit": 1, "distance of the calibration slope from 1": -1}


def unit_stats(v, sign, rng, n_boot=N_BOOT):
    """Statistics of the per-hospital values v (differences). sign=+1 if positive is improvement, -1 if negative is improvement.
    95% hospital bootstrap interval of the mean (hospitals resampled with replacement, percentile), sign test (zeros excluded, exact binomial, two-sided),
    Wilcoxon (zeros excluded, two-sided). Fraction of improved hospitals = improved hospitals / valid hospitals (a difference of 0 is not an improvement).
    If there are fewer than 6 nonzero differences both p values are NaN.
    Rule for ties (0): |v| <= ZERO_TOL (1e-12) counts as 0 (including floating-point noise of size 1e-16, counted in n_tiny).
    share_improved_strict is for reference: the fraction counting strictly above 0 as improved."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    k = len(v)
    if k == 0:
        return dict(k=0, median=np.nan, mean=np.nan, lo=np.nan, hi=np.nan, n_improved=0, share_improved=np.nan, share_improved_strict=np.nan,
                    n_zero=0, n_tiny=0, n_nonzero=0, p_sign=np.nan, p_wilcoxon=np.nan)
    idx = rng.integers(0, k, size=(n_boot, k))
    lo, hi = np.percentile(v[idx].mean(1), [2.5, 97.5])
    nz = v[np.abs(v) > ZERO_TOL]
    n_imp = int((sign * v > ZERO_TOL).sum())
    n_imp_nz = int((sign * nz > 0).sum())
    p_sign = float(stats.binomtest(n_imp_nz, len(nz), 0.5).pvalue) if len(nz) >= MIN_NONZERO else np.nan
    return dict(k=k, median=float(np.median(v)), mean=float(v.mean()), lo=float(lo), hi=float(hi), n_improved=n_imp,
                share_improved=n_imp / k, share_improved_strict=float((sign * v > 0).mean()), n_zero=int((np.abs(v) <= ZERO_TOL).sum()),
                n_tiny=int(((np.abs(v) > 0) & (np.abs(v) <= ZERO_TOL)).sum()), n_nonzero=len(nz), p_sign=p_sign, p_wilcoxon=wilcoxon_p(nz))


def a7_frame(net):
    d = net.copy()
    d["alpha"] = d["alpha"].round(2)
    d["d_sld"] = (d["slope"] - 1).abs() - (d["base_slope"] - 1).abs()
    return d


def a7_stats(net, fs, rng):
    rows = []
    for oc in sorted(net["outcome"].unique()):
        for al in sorted(net["alpha"].unique()):
            g = net[(net["outcome"] == oc) & (net["alpha"] == al)]
            for met, col in (("AUROC", "d_auroc"), ("ECE", "d_ece"), ("net benefit", "d_nb_rep"), ("distance of the calibration slope from 1", "d_sld")):
                rows.append(dict(fs=fs, outcome=oc, alpha=al, metric=met, **unit_stats(g[col], SIGN[met], rng)))
    return pd.DataFrame(rows)


def run(in_dir, out_dir):
    nets = {fs: a7_frame(pd.read_csv(Path(in_dir) / f"A7_network_{fs}.csv", encoding="utf-8-sig")) for fs in ("F42", "F33")}
    rng = np.random.default_rng(SEED)
    allst = pd.concat([a7_stats(nets[fs], fs, rng) for fs in ("F42", "F33")], ignore_index=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(out_dir) / "P3_hospital_stats_all_alpha.csv"
    allst.to_csv(out, index=False, encoding="utf-8-sig")
    return out, allst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.environ.get("PRAM_CSV_DIR", "notebooks/out_pram/github_csv"))
    ap.add_argument("--out-dir", default=os.environ.get("PRAM_CLAIMS_OUT", "out_claims"))
    a = ap.parse_args()
    out, allst = run(a.in_dir, a.out_dir)
    print("wrote", out, len(allst), "rows")


if __name__ == "__main__":
    main()
