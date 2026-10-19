# -*- coding: utf-8 -*-
"""PRAM claims map (aggregate statistics from the result csv files).

Hospital-level verdict columns (section B, function verdict_row): the lower limit of the AUROC difference is above -0.01 and the upper limit of the ECE difference is below 0 (a second column uses the upper limit of the SLD difference instead of ECE). The margin is a post hoc definition.
This script reads only the aggregate csv files written by the notebooks
(no model rerun, no patient-level data).

A. Cell level: retrieval versus the frozen base model of its own, paired differences. Cell classes G/N/V/S/F. A distribution between draws, not a test.
B. Hospital level: the hospital is the sampling unit. Stay-weighted mean + hospital bootstrap, sign test, Wilcoxon, per-hospital DeLong.
C. Tail: A2_tail_draws_F33 (eICU F33, per-draw DeLong interval).
D. Main table comparison (eICU): main_table_F42_F33 + check against reference values.

The word significant is used only where there is a test (B, C). It is not attached to numbers of section A.
The non-inferiority margin 0.01 (AUROC) and the net benefit margin 0.001 are post hoc definitions. Sensitivity 0.005 and 0.02 are given alongside.

usage: python -X utf8 aggregate_claims.py            (run; writes csv files and claims_map.out.txt to the output folder)
      python -X utf8 aggregate_claims.py --selftest   (check with simulated csv only)
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pram_config import BOOT_SEED as SEED, N_BOOT  # noqa: E402

# ===================== constants (edit only here) =====================
IN_DIR = Path(os.environ.get("PRAM_CSV_DIR", "notebooks/out_pram/github_csv"))      # aggregate csv written by notebook 01 (the notebooks write next to the notebook files)
OUT_DIR = Path(os.environ.get("PRAM_CLAIMS_OUT", "out_claims"))
CELLS_PY = Path(__file__).resolve().parent / "pram_config.py"     # THRESH_RANGE is read from here

FS_LIST = ["F42", "F33"]
MARGIN = 0.01                    # AUROC non-inferiority margin (post hoc definition)
MARGINS_SENS = [0.005, 0.01, 0.02]
NB_MARGIN = 0.001                # net benefit margin (post hoc definition)
TAIL_NI_SHARE = 0.05             # NI_tail = share(dAUC < -margin) <= this value
TAIL_ECE_SHARE = 0.80            # ECE_robust = share(dECE < 0) >= this value
Z = 1.96
ZERO_TOL = 1e-12                 # a difference of 0
D_COL_TOL = 1.0e-6 + 1e-9        # the csv is rounded to 6 digits, so it differs from a direct computation by at most 1e-6 in a column value
MIN_NONZERO = 6                  # if there are fewer nonzero differences than this, the sign test and Wilcoxon are NaN
SCALE_MAX = 5.0                  # if the variance scale ratio is below this, standard scale
SCALE_ORIG = 10.0                # above this, the scale of the original implementation (about 22 to 36 times the standard)
EXPECT_NULL_ONE_SIDE = 0.025
TAIL_XCHECK_TOL = 0.01

NB_MS = 0.2                      # main threshold nb@0.2
D_BANKS = ["500", "1000", "full"]

PAIRS = [("pram_b1", "b1_base"), ("pram_b5", "b5_base"), ("pram_platt_b1", "b1_base"),
         ("pram_platt_b5", "b5_base"), ("platt_b1", "b1_base")]
HOSP_PAIRS = [("pram_b1", "b1_base"), ("pram_b5", "b5_base"), ("pram_platt_b1", "b1_base"),
              ("platt_b1", "b1_base")]
STRATS = ["pram_b1", "pram_b5", "local_lr", "local_cb", "pooled_lr", "platt_b1", "recal_iso"]
KEY = ["fs", "outcome", "target", "bank", "rep"]
CELL = ["fs", "outcome", "target", "bank"]
BANK_ORDER = ["500", "1000", "2000", "5000", "full"]
TARGET_ORDER = ["eICU", "MIMIC-III", "MIMIC-IV"]
DIFFS = ["dAUC", "dECE", "dSLD", "dNBs", "dNB20"]

# reading values (self-consistency check). (fs, outcome) -> (lower, upper), tolerance +-0.0008
READING = {("F42", "AKI_168h"): (0.0040, 0.0150), ("F42", "mortality_168h"): (0.0135, 0.0267),
           ("F33", "AKI_168h"): (0.0031, 0.0145), ("F33", "mortality_168h"): (-0.0015, 0.0049)}
READING_TOL = 0.0008

RULE_COLS = {"median": ("NI", "ECE_better"), "NItail": ("NI_tail", "ECE_better"),
             "ECErobust": ("NI", "ECE_robust"), "both": ("NI_tail", "ECE_robust")}

MT_METHODS = ["b1_base", "b5_base", "pram_b1", "pram_b5"]
MT_METRICS = ["auroc", "ece", "slope", "nb_20", "nb_site"]


# ===================== helpers =====================
def nb_col(t):
    return f"nb@{float(t)}"


def read_csv(path, **kw):
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"bank": str}, **kw)


def parse_thresholds(py_path):
    """reads THRESH_RANGE from a code file and builds T_SITE (range midpoint, second decimal place).
    that file is read with a regular expression for a one-line definition."""
    text = Path(py_path).read_text(encoding="utf-8")
    hits = re.findall(r"^THRESH_RANGE\s*=\s*(\{[^\n#]*\})", text, flags=re.M)
    if len(hits) != 1:
        raise RuntimeError(f"one-line definitions of THRESH_RANGE: {len(hits)} ({py_path})")
    rng = ast.literal_eval(hits[0])
    return rng, {oc: round(float(np.mean(r)), 2) for oc, r in rng.items()}


def md(df, fmt=4):
    def f(x):
        if isinstance(x, (bool, np.bool_)):
            return "T" if x else "F"
        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return "NaN"
            if x != 0 and abs(x) < 1e-3:
                return f"{x:.2e}"
            return f"{x:.{fmt}f}"
        return str(x).replace("|", "/")      # so that it does not collide with the markdown table column separator
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(f(v) for v in row) + " |")
    return "\n".join(out)


class Report:
    def __init__(self):
        self.lines = []

    def h(self, t, lvl=2):
        self.lines += ["", "#" * lvl + " " + t, ""]

    def p(self, t):
        self.lines.append(t)

    def t(self, df, fmt=4):
        self.lines += [md(df, fmt), ""]

    def text(self):
        return "\n".join(self.lines)


def cat_order(df):
    df = df.copy()
    if "bank" in df:
        df["bank"] = pd.Categorical(df["bank"].astype(str), BANK_ORDER, ordered=True)
    if "target" in df:
        df["target"] = pd.Categorical(df["target"], TARGET_ORDER, ordered=True)
    return df


# ===================== A. cell level =====================
def extraction_diffs(m, meth, base, t_site):
    """pairs the method and the frozen model by the same draw (fs,outcome,target,bank,rep) to make the differences."""
    nbcols = sorted({nb_col(v) for v in t_site.values()} | {nb_col(NB_MS)})
    need = KEY + ["auroc", "ece", "slope"] + nbcols
    a = m.loc[m["method"] == meth, need]
    b = m.loc[m["method"] == base, need]
    j = a.merge(b, on=KEY, how="inner", suffixes=("_m", "_b"), validate="one_to_one")
    info = dict(pair=f"{meth}|{base}", n_method=len(a), n_base=len(b), n_merged=len(j),
                n_method_without_base=len(a) - len(j), n_base_without_method=len(b) - len(j))
    assert set(j["outcome"]) <= set(t_site), f"outcomes not in T_SITE: {set(j['outcome']) - set(t_site)}"
    j["dAUC"] = j["auroc_m"] - j["auroc_b"]
    j["dECE"] = j["ece_m"] - j["ece_b"]                                  # negative is improvement
    j["dSLD"] = (j["slope_m"] - 1).abs() - (j["slope_b"] - 1).abs()      # negative is improvement
    nbs = np.full(len(j), np.nan)
    for oc, t in t_site.items():
        mask = (j["outcome"] == oc).to_numpy()
        c = nb_col(t)
        nbs[mask] = (j.loc[mask, c + "_m"] - j.loc[mask, c + "_b"]).to_numpy()
    j["dNBs"] = nbs
    j["dNB20"] = j[nb_col(NB_MS) + "_m"] - j[nb_col(NB_MS) + "_b"]
    return j, info


def summarize_cells(j, margin):
    rows = []
    for key, x in j.groupby(CELL, sort=False):
        r = dict(zip(CELL, key))
        r["n_draws"] = len(x)
        for c in DIFFS:
            v = x[c].to_numpy(float)
            r[f"{c}_med"] = float(np.median(v))
            r[f"{c}_p05"] = float(np.percentile(v, 5))
            r[f"{c}_p95"] = float(np.percentile(v, 95))
        r["share_dAUC_lt0"] = float((x["dAUC"] < 0).mean())
        r["share_dAUC_lt_negmargin"] = float((x["dAUC"] < -margin).mean())
        r["share_dECE_lt0"] = float((x["dECE"] < 0).mean())
        r["share_dSLD_lt0"] = float((x["dSLD"] < 0).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def letter(auc_up, ni, ece_better):
    if not ni:
        return "F"
    if auc_up:
        return "G" if ece_better else "V"
    return "N" if ece_better else "S"


def classify_cells(cs, margin, nb_margin=NB_MARGIN):
    c = cs.copy()
    c["AUC_up"] = c["dAUC_med"] > 0
    c["NI"] = c["dAUC_med"] >= -margin
    c["ECE_better"] = c["dECE_med"] < 0
    c["SL_better"] = c["dSLD_med"] < 0
    c["NB_ok"] = c["dNBs_med"] >= -nb_margin
    c["NI_tail"] = c["share_dAUC_lt_negmargin"] <= TAIL_NI_SHARE
    c["ECE_robust"] = c["share_dECE_lt0"] >= TAIL_ECE_SHARE
    for rule, (ni_c, ece_c) in RULE_COLS.items():
        c[f"cls_{rule}"] = [letter(a, n, e) for a, n, e in zip(c["AUC_up"], c[ni_c], c[ece_c])]
    return c


def count_table(c, by, rule="median"):
    ni_col, ece_col = RULE_COLS[rule]
    cls = f"cls_{rule}"
    rows = []
    for key, x in c.groupby(by, sort=True, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        r["rule"] = rule
        r["n_cells"] = len(x)
        r["n_full1"] = int((x["n_draws"] == 1).sum())
        for L in "GNVSF":
            r[L] = int((x[cls] == L).sum())
        ni = x[ni_col]
        r["NI"] = int(ni.sum())
        r["NI_ECE"] = int((ni & x[ece_col]).sum())
        r["NI_SL"] = int((ni & x["SL_better"]).sum())
        r["NI_ECE_or_SL"] = int((ni & (x[ece_col] | x["SL_better"])).sum())
        r["NI_NBok"] = int((ni & x["NB_ok"]).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def margin_sensitivity(c):
    rows = []
    for (fs, pair), x in c.groupby(["fs", "pair"], sort=False):
        r = {"fs": fs, "pair": pair, "n_cells": len(x)}
        for mm in MARGINS_SENS:
            r[f"NI_margin_{mm}"] = int((x["dAUC_med"] >= -mm).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def grid_table(c):
    rows = []
    for (fs, pair), x in c.groupby(["fs", "pair"], sort=False):
        for (oc, tg), y in x.groupby(["outcome", "target"], sort=True, observed=True):
            r = {"fs": fs, "pair": pair, "outcome": oc, "target": str(tg)}
            for b in BANK_ORDER:
                z = y[y["bank"].astype(str) == b]
                r[b] = "." if z.empty else z["cls_median"].iloc[0] + ("*" if z["n_draws"].iloc[0] == 1 else "")
            rows.append(r)
    return pd.DataFrame(rows)


def run_A(mats, t_site, margin):
    cells, infos = [], []
    for fs, m in mats.items():
        present = set(m["method"])
        for meth, base in PAIRS:
            if meth not in present or base not in present:
                infos.append(dict(pair=f"{meth}|{base}", fs=fs, n_method=0, n_base=0, n_merged=0,
                                  n_method_without_base=0, n_base_without_method=0))
                continue
            j, info = extraction_diffs(m, meth, base, t_site)
            info["fs"] = fs
            infos.append(info)
            cs = summarize_cells(j, margin)
            cs["pair"] = f"{meth}|{base}"
            cells.append(cs)
    c = classify_cells(pd.concat(cells, ignore_index=True), margin)
    c["shift"] = np.where(c["target"] == "MIMIC-IV", "none(MIMIC-IV)", "shifted(eICU,MIMIC-III)")
    return cat_order(c), pd.DataFrame(infos)


# ===================== B. hospital level =====================
HMETRICS = ["dAUC", "dECE", "dSLD", "dNBs"]
IMPROVE_SIGN = {"dAUC": 1, "dNBs": 1, "dECE": -1, "dSLD": -1}


def improved(metric, v):
    """improvement direction: AUROC and net benefit positive, ECE and |slope-1| negative."""
    return IMPROVE_SIGN[metric] * np.asarray(v, float) > ZERO_TOL


def hosp_frame(h, meth, base, t_site):
    nbc = sorted({nb_col(v) for v in t_site.values()})
    cols = ["outcome", "site", "n_site", "auroc", "ece", "slope"] + nbc
    dcol, vcol = ("d_auroc_b1", "var_b1") if base == "b1_base" else ("d_auroc_b5", "var_b5")
    a = h.loc[h["method"] == meth, cols + [dcol, vcol]]
    b = h.loc[h["method"] == base, cols]
    j = a.merge(b, on=["outcome", "site"], suffixes=("", "_b"), validate="one_to_one")
    assert (j["n_site"] == j["n_site_b"]).all(), "the same hospital but n_site differs"
    j["dAUC"] = j["auroc"] - j["auroc_b"]
    col_gap = float(np.max(np.abs(j["dAUC"] - j[dcol]))) if len(j) else 0.0
    assert col_gap <= D_COL_TOL, f"{meth}|{base}: direct computation of dAUC and {dcol} column {col_gap:.2e} mismatch"
    j["dECE"] = j["ece"] - j["ece_b"]
    j["dSLD"] = (j["slope"] - 1).abs() - (j["slope_b"] - 1).abs()
    nbs = np.full(len(j), np.nan)
    for oc, t in t_site.items():
        mask = (j["outcome"] == oc).to_numpy()
        c = nb_col(t)
        nbs[mask] = (j.loc[mask, c] - j.loc[mask, c + "_b"]).to_numpy()
    j["dNBs"] = nbs
    j["d_col"] = j[dcol]
    j["var_col"] = j[vcol]
    j.attrs["col_gap"] = col_gap
    return j


def wilcoxon_p(nz):
    if len(nz) < MIN_NONZERO:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(stats.wilcoxon(nz, zero_method="wilcox").pvalue)
    except ValueError:
        return np.nan


def hospital_stats(fr, rng, n_boot=N_BOOT):
    k = len(fr)
    w = fr["n_site"].to_numpy(float)
    idx = rng.integers(0, k, size=(n_boot, k))     # hospitals resampled with replacement, the same replicates used for all metrics
    rows = []
    for met in HMETRICS:
        v = fr[met].to_numpy(float)
        ok = ~np.isnan(v)
        we = np.where(ok, w, 0.0)
        vf = np.where(ok, v, 0.0)
        est = float((we * vf).sum() / we.sum())
        bs = (we[idx] * vf[idx]).sum(1) / we[idx].sum(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        vv = v[ok]
        nz = vv[np.abs(vv) > ZERO_TOL]
        imp_nz = int(improved(met, nz).sum())
        p_b = float(stats.binomtest(imp_nz, len(nz), 0.5).pvalue) if len(nz) >= MIN_NONZERO else np.nan
        n_imp = int(improved(met, vv).sum())
        rows.append(dict(metric=met, k=int(ok.sum()), wmean=est, lo=float(lo), hi=float(hi),
                         n_improved=n_imp, share_improved=n_imp / max(len(vv), 1),
                         n_zero=int((np.abs(vv) <= ZERO_TOL).sum()), n_nonzero=len(nz),
                         p_binom=p_b, p_wilcoxon=wilcoxon_p(nz)))
    return pd.DataFrame(rows)


def verdict_row(st):
    a = st[st["metric"] == "dAUC"].iloc[0]
    e = st[st["metric"] == "dECE"].iloc[0]
    s = st[st["metric"] == "dSLD"].iloc[0]
    r = {"auc_wmean": a["wmean"], "auc_lo": a["lo"], "auc_hi": a["hi"], "ece_hi": e["hi"], "sld_hi": s["hi"]}
    for mm in MARGINS_SENS:
        r[f"auc_lo_gt_neg{mm}"] = bool(a["lo"] > -mm)
    r["AUC_lo_gt_neg0.01_and_ECE_hi_lt0"] = bool(a["lo"] > -0.01 and e["hi"] < 0)
    r["AUC_lo_gt_neg0.01_and_dSLD_hi_lt0"] = bool(a["lo"] > -0.01 and s["hi"] < 0)
    return r


def var_provenance(hp, a6, tol=1e-9):
    """checks against A6 whether var_b1 of the pram_b1 rows is the standard DeLong variance or the original implementation variance."""
    j = hp.merge(a6, on=["outcome", "site"], how="inner")
    n = len(j)
    out = dict(n_matched_A6=n)
    if n == 0:
        out.update(status="unknown", n_std=0, n_orig=0, n_d_match=0, ratio_to_std=np.nan)
        return out
    out["n_d_match"] = int((np.abs(j["d_col"] - j["d_auroc"]) <= D_COL_TOL).sum())
    out["n_std"] = int((np.abs(j["var_col"] - j["var_delong_standard"]) <= tol).sum())
    out["n_orig"] = int((np.abs(j["var_col"] - j["var_delong_original"]) <= tol).sum())
    out["ratio_to_std"] = float(np.median(j["var_col"]) / np.median(j["var_delong_standard"]))
    if out["n_std"] == n:
        out["status"] = "standard_direct"
    elif out["n_orig"] == n:
        out["status"] = "original_direct"
    elif out["ratio_to_std"] < SCALE_MAX:
        out["status"] = "standard_scale"
    elif out["ratio_to_std"] > SCALE_ORIG:
        out["status"] = "original_scale"
    else:
        out["status"] = "unknown"
    return out


def delong_counts(d, var, margin):
    d = np.asarray(d, float)
    var = np.asarray(var, float)
    ok = ~np.isnan(d) & ~np.isnan(var) & (var >= 0)
    n_excl = int((~ok).sum())
    d, var = d[ok], var[ok]
    se = np.sqrt(var)
    ident = (var == 0) & (np.abs(d) <= ZERO_TOL)
    up = (~ident) & (d - Z * se > 0)
    down = (~ident) & (d + Z * se < 0)
    ni = d - Z * se > -margin
    inf = d + Z * se < -margin
    unc = ~(ident | up | down)
    n = len(d)
    r = dict(n=n, n_excluded=n_excl, n_identical=int(ident.sum()), n_sig_up=int(up.sum()),
             n_sig_down=int(down.sum()), n_uncertain=int(unc.sum()), n_NI=int(ni.sum()),
             n_clear_inferior=int(inf.sum()))
    for key in ["identical", "sig_up", "sig_down", "uncertain", "NI", "clear_inferior"]:
        r[f"share_{key}"] = r[f"n_{key}"] / n if n else np.nan
    return r


def run_B(hosp, a6, t_site):
    rng = np.random.default_rng(SEED)
    stat_rows, verdict_rows, delong_rows, gap_rows = [], [], [], []
    frames, prov = {}, {}
    for fs, h in hosp.items():
        for meth, base in HOSP_PAIRS:
            fr = hosp_frame(h, meth, base, t_site)
            frames[(fs, meth, base)] = fr
            gap_rows.append(dict(fs=fs, pair=f"{meth}|{base}", k=len(fr), max_abs_gap_vs_column=fr.attrs["col_gap"]))
        prov[fs] = var_provenance(frames[(fs, "pram_b1", "b1_base")], a6)
    pb1_med = {(fs, oc): float(np.median(g["var_col"])) for fs in hosp
               for oc, g in frames[(fs, "pram_b1", "b1_base")].groupby("outcome")}
    for fs in hosp:
        for meth, base in HOSP_PAIRS:
            fr = frames[(fs, meth, base)]
            pair = f"{meth}|{base}"
            for oc, g in fr.groupby("outcome", sort=False):
                g = g.reset_index(drop=True)
                st = hospital_stats(g, rng)
                st.insert(0, "pair", pair)
                st.insert(0, "outcome", oc)
                st.insert(0, "fs", fs)
                stat_rows.append(st)
                verdict_rows.append(dict(fs=fs, outcome=oc, pair=pair, k=len(g), **verdict_row(st)))
                # the DeLong test only when confirmed with the standard variance
                if meth == "pram_b1":
                    status = prov[fs]["status"]
                else:
                    ratio = float(np.median(g["var_col"])) / pb1_med[(fs, oc)] if pb1_med[(fs, oc)] > 0 else np.nan
                    ok_scale = ratio < SCALE_MAX and prov[fs]["status"].startswith("standard")
                    status = "standard_scale_indirect" if ok_scale else "unverified"
                r = dict(fs=fs, outcome=oc, pair=pair, var_status=status, var_median=float(np.median(g["var_col"])))
                if status.startswith("standard"):
                    r.update(delong_counts(g["d_col"], g["var_col"], MARGIN))
                    for mm in MARGINS_SENS:
                        r[f"n_NI_margin_{mm}"] = delong_counts(g["d_col"], g["var_col"], mm)["n_NI"]
                else:
                    r["note"] = "not tested, variance overestimated or unconfirmed"
                delong_rows.append(r)
    return dict(stats=pd.concat(stat_rows, ignore_index=True), verdicts=pd.DataFrame(verdict_rows),
                delong=pd.DataFrame(delong_rows), prov=prov, gaps=pd.DataFrame(gap_rows))


def selfcheck(stats_df, hs, tol_hs=1e-6, readings=READING, tol_read=READING_TOL):
    rows, ok_all = [], True
    for (fs, oc), (rlo, rhi) in readings.items():
        s = stats_df[(stats_df["fs"] == fs) & (stats_df["outcome"] == oc)
                     & (stats_df["pair"] == "pram_b1|b1_base") & (stats_df["metric"] == "dAUC")]
        h = hs[(hs["fs"] == fs) & (hs["outcome"] == oc) & (hs["method"] == "pram_b1")]
        if s.empty or h.empty:
            rows.append(dict(fs=fs, outcome=oc, wmean=np.nan, hs_delta=np.nan, gap_hs=np.nan, lo=np.nan, hi=np.nan,
                             read_lo=rlo, read_hi=rhi, pass_hs=False, pass_read=False))
            ok_all = False
            continue
        s, h = s.iloc[0], h.iloc[0]
        gap = abs(s["wmean"] - h["delta_stays_weighted"])
        p_hs = bool(gap <= tol_hs + 1e-12)
        p_rd = bool(abs(s["lo"] - rlo) <= tol_read and abs(s["hi"] - rhi) <= tol_read)
        ok_all &= p_hs and p_rd
        rows.append(dict(fs=fs, outcome=oc, wmean=s["wmean"], hs_delta=h["delta_stays_weighted"], gap_hs=gap,
                         lo=s["lo"], hi=s["hi"], read_lo=rlo, read_hi=rhi, pass_hs=p_hs, pass_read=p_rd))
    return pd.DataFrame(rows), ok_all


def dispersion(hosp, hs):
    """spread of the hospital-level AUROC change. The change is versus b1_base, and for pram_b5 the rows versus b5_base are separate."""
    rows = []
    for fs, h in hosp.items():
        for oc in sorted(h["outcome"].unique()):
            auc = {m: g.set_index("site")["auroc"] for m, g in h[h["outcome"] == oc].groupby("method")}
            for meth, base in [(s, "b1_base") for s in STRATS] + [("pram_b5", "b5_base")]:
                d = (auc[meth] - auc[base].reindex(auc[meth].index)).to_numpy(float)
                p10, p90 = np.percentile(d, [10, 90])
                rows.append(dict(fs=fs, outcome=oc, strategy=meth, vs=base, k=len(d), sd=float(np.std(d, ddof=1)),
                                 p10=float(p10), p90=float(p90), width=float(p90 - p10),
                                 share_abs_lt_margin=float((np.abs(d) < MARGIN).mean()),
                                 share_lt_neg01=float((d < -0.01).mean()), share_lt_neg02=float((d < -0.02).mean())))
    disp = pd.DataFrame(rows)
    if "I2_stays_weighted" not in hs.columns:      # column absent from hospital_summary.csv
        hs = hs.assign(I2_stays_weighted=np.nan)
    i2 = hs[["fs", "outcome", "method", "I2_stays_weighted", "I2_invvar"]].rename(columns={"method": "strategy"})
    disp = disp.merge(i2, on=["fs", "outcome", "strategy"], how="left")
    ratios = []
    for (fs, oc), g in disp[disp["vs"] == "b1_base"].groupby(["fs", "outcome"], sort=False):
        g = g.set_index("strategy")
        r = dict(fs=fs, outcome=oc)
        for loc in ["local_lr", "local_cb"]:
            if loc in g.index and "pram_b1" in g.index:
                r[f"width_ratio_{loc}_over_pram_b1"] = g.loc[loc, "width"] / g.loc["pram_b1", "width"]
                r[f"sd_ratio_{loc}_over_pram_b1"] = g.loc[loc, "sd"] / g.loc["pram_b1", "sd"]
        ratios.append(r)
    return disp, pd.DataFrame(ratios)


# ===================== C. tail =====================
def tail_table(t, margin=MARGIN):
    rows = []
    for (oc, bank, meth), g in t.groupby(["outcome", "bank", "method"], sort=False):
        d = g["d_auroc_base"].to_numpy(float)
        var = g["var_auroc_base"].to_numpy(float)
        ok = ~np.isnan(d) & ~np.isnan(var) & (var >= 0)
        d, var = d[ok], var[ok]
        n = len(d)
        r = dict(outcome=oc, bank=str(bank), method=meth, n=n, n_excluded=int((~ok).sum()))
        if n:
            se = np.sqrt(var)
            lo, hi = d - Z * se, d + Z * se
            r.update(share_d_lt0=float((d < 0).mean()), share_d_lt_neg01=float((d < -margin).mean()),
                     share_ci_below0=float((hi < 0).mean()), share_ci_below_neg01=float((hi < -margin).mean()),
                     share_ni=float((lo > -margin).mean()), share_ci_above0=float((lo > 0).mean()),
                     se_median=float(np.median(se)))
            r["diff_below_vs_2.5pct"] = r["share_ci_below0"] - EXPECT_NULL_ONE_SIDE
            r["diff_above_vs_2.5pct"] = r["share_ci_above0"] - EXPECT_NULL_ONE_SIDE
        rows.append(r)
    return pd.DataFrame(rows)


def tail_crosscheck(tt, summ):
    """whether A2_tail_summary (new computation, F33 eICU) and the direct computation agree within 0.01."""
    s = summ[(summ["source"] == "new computation") & (summ["fs"] == "F33") & (summ["target"] == "eICU")]
    m = s.merge(tt, on=["outcome", "bank", "method"], how="inner")
    pairs = [("share_below_zero", "share_d_lt0"), ("draws_ci_wholly_below_zero", "share_ci_below0"),
             ("draws_ci_wholly_above_zero", "share_ci_above0")]
    rows = []
    for _, r in m.iterrows():
        row = dict(outcome=r["outcome"], bank=r["bank"], method=r["method"])
        okall = True
        for sc, dc in pairs:
            row[f"{dc}_direct"] = r[dc]
            row[f"{sc}_summary"] = r[sc]
            okall &= bool(abs(r[sc] - r[dc]) <= TAIL_XCHECK_TOL + 1e-12)
        row["match"] = okall
        rows.append(row)
    return pd.DataFrame(rows)


# ===================== D. main table comparison =====================
def main_long(mt, mats, t_site):
    """eICU main table + banks not in the main table (500) from the median of matrix_draws."""
    x = mt[mt["target"] == "eICU"].copy()
    x["source"] = "main_table"
    x = x[["fs", "outcome", "bank", "method"] + MT_METRICS + ["source"]]
    x["bank"] = x["bank"].astype(str)
    have = set(x["bank"])
    extra = []
    for fs, m in mats.items():
        for oc in m["outcome"].unique():
            for bank in [b for b in D_BANKS if b not in have]:
                g = m[(m["target"] == "eICU") & (m["outcome"] == oc) & (m["bank"] == bank) & m["method"].isin(MT_METHODS)]
                for meth, gg in g.groupby("method"):
                    extra.append(dict(fs=fs, outcome=oc, bank=bank, method=meth, auroc=gg["auroc"].median(),
                                      ece=gg["ece"].median(), slope=gg["slope"].median(),
                                      nb_20=gg[nb_col(NB_MS)].median(), nb_site=gg[nb_col(t_site[oc])].median(),
                                      source="matrix median (bank not in the main table)"))
    if extra:
        x = pd.concat([x, pd.DataFrame(extra)], ignore_index=True)
    return x


def verify_matrix_vs_main(mt, mats, t_site):
    """whether the main table is the median or the mean of matrix_draws (banks in the main table, same cell)."""
    rows = []
    for fs, m in mats.items():
        for oc in m["outcome"].unique():
            for bank in sorted(set(mt["bank"].astype(str)) & set(m["bank"])):
                for meth in MT_METHODS:
                    g = m[(m["target"] == "eICU") & (m["outcome"] == oc) & (m["bank"] == bank) & (m["method"] == meth)]
                    r = mt[(mt["fs"] == fs) & (mt["target"] == "eICU") & (mt["outcome"] == oc)
                           & (mt["bank"].astype(str) == bank) & (mt["method"] == meth)]
                    if g.empty or r.empty:
                        continue
                    r = r.iloc[0]
                    rows.append(dict(fs=fs, outcome=oc, bank=bank, method=meth, n_draws_matrix=len(g),
                                     gap_median_auroc=abs(g["auroc"].median() - r["auroc"]),
                                     gap_mean_auroc=abs(g["auroc"].mean() - r["auroc"]),
                                     gap_median_nb_site=abs(g[nb_col(t_site[oc])].median() - r["nb_site"]),
                                     gap_median_nb20=abs(g[nb_col(NB_MS)].median() - r["nb_20"])))
    return pd.DataFrame(rows)


def lookup(L):
    d = {(r.fs, r.outcome, str(r.bank), r.method): r for r in L.itertuples(index=False)}

    def get(oc, bank, fs, meth, metric):
        r = d.get((fs, oc, str(bank), meth))
        return np.nan if r is None else float(getattr(r, metric))
    return get


def diff_tables(L):
    get = lookup(L)
    rows = []
    for oc in sorted(L["outcome"].unique()):
        for bank in [b for b in BANK_ORDER if b in set(L["bank"].astype(str))]:
            for pm, bm in [("pram_b1", "b1_base"), ("pram_b5", "b5_base")]:
                defs = [(f"{pm}(F33) - {bm}(F33)", ("F33", pm), ("F33", bm)),
                        (f"{pm}(F42) - {bm}(F42)", ("F42", pm), ("F42", bm)),
                        (f"{bm}(F33) - {bm}(F42)", ("F33", bm), ("F42", bm)),
                        (f"{pm}(F42) - {bm}(F33)", ("F42", pm), ("F33", bm))]
                for name, (fa, ma), (fb, mb) in defs:
                    r = dict(outcome=oc, bank=bank, diff=name)
                    for met in MT_METRICS:
                        r[met] = get(oc, bank, fa, ma, met) - get(oc, bank, fb, mb, met)
                    rows.append(r)
    out = pd.DataFrame(rows)
    out["_b"] = out["bank"].map({b: i for i, b in enumerate(BANK_ORDER)})
    return out.sort_values(["outcome", "_b"], kind="stable").drop(columns="_b").reset_index(drop=True)


def claims_check(L, mt):
    """reference values (for checking). Read for death (mortality_168h), with the AKI values given alongside for reference."""
    get = lookup(L)
    mo, aki = "mortality_168h", "AKI_168h"
    spec = [
        ("death full F33 b1_base AUROC", ("full", "F33", "b1_base", "auroc"), 0.8117, 4),
        ("death full F33 b1_base ECE", ("full", "F33", "b1_base", "ece"), 0.0247, 4),
        ("death full F33 b1_base slope", ("full", "F33", "b1_base", "slope"), 0.744, 3),
        ("death full F33 pram_b1 AUROC", ("full", "F33", "pram_b1", "auroc"), 0.8207, 4),
        ("death full F33 pram_b1 ECE", ("full", "F33", "pram_b1", "ece"), 0.0163, 4),
        ("death full F33 pram_b1 slope", ("full", "F33", "pram_b1", "slope"), 0.921, 3),
        ("death full F33 b1_base nb@20%", ("full", "F33", "b1_base", "nb_20"), -0.0008, 4),
        ("death full F33 pram_b1 nb@20%", ("full", "F33", "pram_b1", "nb_20"), 0.0020, 4),
        ("death 1000 F33 pram_b1 ECE", ("1000", "F33", "pram_b1", "ece"), 0.0196, 4),
        ("death 1000 F33 pram_b1 slope", ("1000", "F33", "pram_b1", "slope"), 0.835, 3),
        ("CatBoost 33 columns (death full) b5_base AUROC", ("full", "F33", "b5_base", "auroc"), 0.8224, 4),
        ("CatBoost 33 columns (death full) pram_b5 AUROC", ("full", "F33", "pram_b5", "auroc"), 0.8257, 4),
        ("CatBoost 33 columns (death full) b5_base ECE", ("full", "F33", "b5_base", "ece"), 0.0158, 4),
        ("CatBoost 33 columns (death full) pram_b5 ECE", ("full", "F33", "pram_b5", "ece"), 0.0121, 4),
        ("CatBoost 42 columns (death full) pram_b5 AUROC", ("full", "F42", "pram_b5", "auroc"), 0.8310, 4),
    ]
    rows = []
    for label, (bank, fs, meth, met), claim, dec in spec:
        v, va = get(mo, bank, fs, meth, met), get(aki, bank, fs, meth, met)
        if np.isnan(v):
            status = "[not available] no cell"
        else:
            status = "OK" if abs(v - claim) <= 0.5 * 10 ** (-dec) + 1e-12 else "mismatch"
        rows.append(dict(claim=label, claimed=claim, computed_mortality=v, gap=v - claim, status=status,
                         computed_AKI_ref=va))
    # retrieval at bank 1000 +0.0015 = pram_b1 - b1_base (F33, death). The value of the dAUROC_vs_b1 column of the main table is given in the ref cell
    v = get(mo, "1000", "F33", "pram_b1", "auroc") - get(mo, "1000", "F33", "b1_base", "auroc")
    r = mt[(mt["fs"] == "F33") & (mt["target"] == "eICU") & (mt["outcome"] == mo) & (mt["bank"].astype(str) == "1000")
           & (mt["method"] == "pram_b1")]
    col = float(r["dAUROC_vs_b1"].iloc[0]) if len(r) else np.nan
    st = "[not available] no cell" if np.isnan(v) else ("OK" if abs(v - 0.0015) <= 0.5e-4 + 1e-12 else "mismatch")
    rows.append(dict(claim="death 1000 F33 pram_b1 - b1_base AUROC (ref cell = dAUROC_vs_b1 column of the main table)", claimed=0.0015,
                     computed_mortality=v, gap=v - 0.0015, status=st, computed_AKI_ref=col))
    return pd.DataFrame(rows)


# ===================== assembly =====================
def load_inputs(d):
    d = Path(d)
    need_m = ["fs", "outcome", "target", "bank", "rep", "method", "auroc", "ece", "slope"]
    need_h = ["outcome", "site", "method", "n_site", "auroc", "ece", "slope", "d_auroc_b1", "var_b1",
              "d_auroc_b5", "var_b5"]
    mats = {fs: read_csv(d / f"matrix_draws_{fs}.csv") for fs in FS_LIST}
    hosp = {fs: read_csv(d / f"hospital_results_{fs}.csv") for fs in FS_LIST}
    for fs in FS_LIST:
        assert not set(need_m) - set(mats[fs].columns), f"matrix_draws_{fs} no column: {set(need_m) - set(mats[fs].columns)}"
        assert not set(need_h) - set(hosp[fs].columns), f"hospital_results_{fs} no column: {set(need_h) - set(hosp[fs].columns)}"
    return dict(mats=mats, hosp=hosp, a6=read_csv(d / "A6_variance_diagnosis_by_hospital.csv"),
                hs=read_csv(d / "hospital_summary.csv"), tail=read_csv(d / "A2_tail_draws_F33.csv"),
                tail_sum=read_csv(d / "A2_tail_summary.csv"), main=read_csv(d / "main_table_F42_F33.csv"))


def run(inp, out_dir, t_site, readings=READING, header_extra=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    R, RA, t = Report(), Report(), {}

    def save(name, df):
        df.to_csv(out_dir / name, index=False, encoding="utf-8-sig")
        t[name] = df

    R.p("# PRAM claims map")
    R.p("The only input is the aggregate csv files. No model rerun. The non-inferiority margin AUROC 0.01 and the net benefit margin 0.001 are "
        "**post hoc definitions, sensitivity 0.005 and 0.02 given alongside**.")
    if header_extra:
        R.p(header_extra)
    R.p(f"representative threshold T_SITE (midpoint of THRESH_RANGE in `pram_config.py`): {t_site} . net benefit column {[nb_col(v) for v in t_site.values()]} and {nb_col(NB_MS)}.")
    R.h("0. input check")
    for fs, m in inp["mats"].items():
        R.p(f"- matrix_draws_{fs}: {len(m)} rows, bank dtype={m['bank'].dtype}, unique values "
            f"{sorted(m['bank'].unique(), key=BANK_ORDER.index)}, target {sorted(m['target'].unique())}, "
            f"outcome {sorted(m['outcome'].unique())}")
        for oc in m["outcome"].unique():
            assert nb_col(t_site[oc]) in m.columns, f"{nb_col(t_site[oc])} no column"
        assert nb_col(NB_MS) in m.columns
    R.p("- the assert that the nb@ threshold column exists passed.")

    # ---------------- A ----------------
    c, info = run_A(inp["mats"], t_site, MARGIN)
    save("A_cells.csv", c.assign(bank=c["bank"].astype(str), target=c["target"].astype(str)))
    save("A_pair_merge_info.csv", info)
    RA.h("A. Cell level: retrieval versus the frozen model of its own (no model rerun)")
    RA.p("**The cell classification is a distribution between draws and not a test.** The variation between bank draws is bank-draw variation, not test-sample uncertainty.")
    RA.p(f"one draw = (fs,outcome,target,bank,rep). Differences: dAUC = AUROC method - frozen, dECE = ECE method - frozen (negative is improvement), "
         f"dSLD = |slope-1| method - frozen (negative is improvement), dNBs = difference of nb@T_SITE, dNB20 = difference of nb@0.2. "
         f"classification (median rule): AUC_up = median dAUC > 0, NI = median dAUC >= -{MARGIN}, ECE_better = median dECE < 0, "
         f"SL_better = median dSLD < 0, NB_ok = median dNBs >= -{NB_MARGIN}. "
         f"G = AUC_up & ECE_better, N = (not AUC_up) & NI & ECE_better, V = AUC_up & not ECE_better, "
         f"S = (not AUC_up) & NI & not ECE_better, F = not NI. "
         f"tail variant: NI_tail = share(dAUC < -{MARGIN}) <= {TAIL_NI_SHARE}, ECE_robust = share(dECE < 0) >= {TAIL_ECE_SHARE}. "
         f"rule names: median (median rule) | NItail (NI only by tail) | ECErobust (ECE only by tail) | both.")
    RA.p("**rows before and after merging** (including the number of draws that lost their pair):")
    RA.t(info[["fs", "pair", "n_method", "n_base", "n_merged", "n_method_without_base", "n_base_without_method"]])
    cnt1 = pd.concat([count_table(c, ["pair", "fs"], r) for r in RULE_COLS], ignore_index=True)
    save("A_counts_pair_fs.csv", cnt1)
    RA.p("**(1) G/N/V/S/F per pair x fs** (n_full1 = number of full cells with n_draws=1, NI_ECE = NI and ECE improves, NI_SL = NI and slope improves, "
         "NI_NBok = NI and net benefit within the margin). Median rule:")
    RA.t(cnt1[cnt1["rule"] == "median"].drop(columns="rule"))
    RA.p("tail-variant rules (the same classification by another criterion):")
    RA.t(cnt1[cnt1["rule"] != "median"][["pair", "fs", "rule", "n_cells", "G", "N", "V", "S", "F", "NI", "NI_ECE",
                                         "NI_SL", "NI_ECE_or_SL", "NI_NBok"]])
    cnt_bank = count_table(c, ["pair", "fs", "bank"], "median")
    cnt_tgt = count_table(c, ["pair", "fs", "shift"], "median")
    save("A_counts_by_bank.csv", cnt_bank.assign(bank=cnt_bank["bank"].astype(str)))
    save("A_counts_by_target.csv", cnt_tgt)
    RA.p("**(2) by bank size** (median rule):")
    RA.t(cnt_bank.drop(columns="rule"))
    RA.p("**(2) by target** (with shift eICU and MIMIC-III versus without shift MIMIC-IV, median rule):")
    RA.t(cnt_tgt.drop(columns="rule"))
    grid = grid_table(c)
    save("A_grid.csv", grid)
    RA.p("**(3) class letter grid** (rows = outcome x target, columns = bank, `*` = n_draws=1, median rule):")
    RA.t(grid)
    fl = c.loc[c["cls_median"] == "F", ["fs", "pair", "outcome", "target", "bank", "n_draws", "dAUC_med", "dAUC_p05",
                                        "dAUC_p95", "share_dAUC_lt_negmargin", "dECE_med"]].copy()
    fl["bank"], fl["target"] = fl["bank"].astype(str), fl["target"].astype(str)
    save("A_F_cells.csv", fl)
    RA.p(f"**(4) all F cells** ({len(fl)} cells, median rule):")
    RA.t(fl)
    ms = margin_sensitivity(c)
    save("A_margin_sensitivity.csv", ms)
    RA.p("**(5) margin sensitivity: number of NI cells** (median dAUC >= -margin):")
    RA.t(ms)
    assert "significant" not in RA.text().replace("not a test", ""), "the word significant appeared in section A"
    R.lines += RA.lines

    # ---------------- B ----------------
    B = run_B(inp["hosp"], inp["a6"], t_site)
    save("B_hospital_stats.csv", B["stats"])
    save("B_hospital_verdicts.csv", B["verdicts"])
    save("B_hospital_delong.csv", B["delong"])
    save("B_hospital_d_vs_column.csv", B["gaps"])
    R.h("B. Hospital level: the hospital is the sampling unit")
    R.h("B-0. check of the source of the var column (is var_b1 the standard DeLong variance)", 3)
    R.t(pd.DataFrame([dict(fs=fs, **p) for fs, p in B["prov"].items()]))
    R.p("A6 has no fs column, and the number of rows whose d_auroc equals d of pram_b1 (n_d_match) tells which fs the data are from. "
        "status: standard_direct = var equals the A6 standard variance within 1e-9 in all rows. standard_scale = no direct match but the scale is on the standard side"
        "(the original implementation is about 22 to 36 times the standard). For pairs other than pram_b1 (pram_b5, pram_platt_b1, platt_b1) a direct proof from A6 is not possible, "
        "so they are regarded as standard scale only by the scale relative to the median var of pram_b1 of the same fs (below 5 times) (indirect, standard_scale_indirect).")
    R.p("maximum difference between the directly computed dAUC and the d_auroc column of the csv (tolerance 1e-6 for 6-digit rounding):")
    R.t(B["gaps"])
    sc, ok_all = selfcheck(B["stats"], inp["hs"], readings=readings)
    save("B_selfcheck.csv", sc)
    R.h("B-1. self-consistency check (weighted mean of pram_b1 AUROC)", 3)
    R.t(sc)
    if not ok_all:
        R.p("**Self-consistency check failed. Stops here.** Look at the lines where pass_hs / pass_read is False in the table above.")
        (out_dir / "claims_map.out.txt").write_text(R.text(), encoding="utf-8")
        return R.text(), t, False
    R.p("self-consistency check passed (within 1e-6 of hospital_summary delta_stays_weighted, within +-0.0008 of the reading interval).")

    R.h(f"B-2. hospital-level paired differences: stay-weighted mean + hospital bootstrap 95% interval (B={N_BOOT}, seed={SEED})", 3)
    R.p("improvement direction: AUROC and net benefit positive, ECE and |slope-1| negative. Fraction of improved hospitals = improved / all (a difference of 0 is not an improvement). "
        "p_binom = exact binomial sign test (zeros excluded, NaN if fewer than 6 nonzero), p_wilcoxon = signed-rank (zero_method=wilcox, NaN if fewer than 6 nonzero). "
        "the hospital test samples are small, so the power is low.")
    R.t(B["stats"][["fs", "outcome", "pair", "metric", "k", "wmean", "lo", "hi", "n_improved", "share_improved",
                    "n_zero", "p_binom", "p_wilcoxon"]])
    R.h("B-3. combined decision (AUROC interval lower bound versus margin, ECE and dSLD interval upper bound)", 3)
    R.t(B["verdicts"])
    R.h("B-4. per-hospital DeLong (only when confirmed with the standard variance)", 3)
    R.p("classes: same (var=0 and d=0) | significant increase (d-1.96se>0) | significant decrease (d+1.96se<0) | undetermined. Non-inferiority (d-1.96se>-margin) is given separately from the classes above. "
        f"clearly inferior = d+1.96se<-margin (margin={MARGIN}, a subset of significant decrease). Denominator = number of valid hospitals. "
        "under the null the one-sided expectation is 2.5% (approximate). The hospital test samples are small, so the power is low.")
    R.t(B["delong"])
    disp, ratios = dispersion(inp["hosp"], inp["hs"])
    save("B_dispersion.csv", disp)
    save("B_dispersion_ratios.csv", ratios)
    R.h("B-5. spread (hospital-level AUROC change versus b1_base. For pram_b5 the rows versus b5_base are separate)", 3)
    R.p(f"the p10 to p90 width is np.percentile with linear interpolation. share_abs_lt_margin = |d|<{MARGIN} fraction of hospitals. "
        "the two I2 columns are the display values of hospital_summary (not changed).")
    R.t(disp)
    R.p("width ratio and SD ratio (local / pram_b1, versus b1_base):")
    R.t(ratios)

    # ---------------- C ----------------
    tt = tail_table(inp["tail"])
    save("C_tail.csv", tt)
    R.h("C. Tail: A2_tail_draws_F33 (eICU F33, per-draw DeLong interval d +- 1.96 se)")
    R.p("d = paired AUROC difference of the method minus frozen logistic (b1_base) (pram_b5 and platt_b5 are also versus logistic). Base rows have NaN d and are excluded. "
        "the draws are repeated draws from a bank and not independent of each other (the interval contains only the test-sample uncertainty). Under the null the one-sided expectation is 2.5%.")
    R.p("valid n and number excluded per method:")
    R.t(tt.groupby("method")[["n", "n_excluded"]].sum().reset_index())
    R.t(tt[tt["n"] > 0])
    xc = tail_crosscheck(tt, inp["tail_sum"])
    save("C_crosscheck_summary.csv", xc)
    R.p("**cross-check with A2_tail_summary (new computation)** (within 0.01 = match):")
    R.t(xc[xc["method"] == "pram_b1"])
    R.p(f"cross-check of all methods: {int(xc['match'].sum())}/{len(xc)} cells agree.")
    if (~xc["match"]).any():
        R.p("cells that do not agree (candidates for a definition difference: whether share_below_zero is d<0 or d<=0, whether the interval is DeLong or another variance):")
        R.t(xc[~xc["match"]])
    R.p("median DeLong se at pram_b1, banks 500 and 1000 (size of the test-sample uncertainty):")
    R.t(tt[(tt["method"] == "pram_b1") & tt["bank"].isin(["500", "1000"])]
        [["outcome", "bank", "n", "se_median", "share_ci_below0", "share_ci_above0"]])

    # ---------------- D ----------------
    L = main_long(inp["main"], inp["mats"], t_site)
    save("D_main_long.csv", L)
    ver = verify_matrix_vs_main(inp["main"], inp["mats"], t_site)
    wide = L[L["method"].isin(MT_METHODS)]
    pv = wide.pivot_table(index=["outcome", "bank", "method"], columns="fs", values=MT_METRICS)
    pv.columns = [f"{a}_{b}" for a, b in pv.columns]
    pv = pv[[f"{m}_{fs}" for m in MT_METRICS for fs in ["F42", "F33"] if f"{m}_{fs}" in pv.columns]].reset_index()
    pv["_b"] = pv["bank"].map({b: i for i, b in enumerate(BANK_ORDER)})
    pv["_m"] = pv["method"].map({m: i for i, m in enumerate(MT_METHODS)})
    pv = pv.sort_values(["outcome", "_b", "_m"]).drop(columns=["_b", "_m"])
    save("D_main_wide.csv", pv)
    dt = diff_tables(L[L["method"].isin(MT_METHODS)])
    save("D_diffs.csv", dt)
    cl = claims_check(L, inp["main"])
    save("D_claims_check.csv", cl)
    R.h("D. Main table comparison (eICU, banks 500, 1000 and full)")
    R.p(f"banks in the main table (main_table_F42_F33.csv) are {sorted(set(inp['main']['bank'].astype(str)), key=BANK_ORDER.index)}. "
        "banks not in it are filled with the median of matrix_draws and stated in the source column (D_main_long.csv).")
    R.p("check whether the main table is the median or the mean of matrix_draws (maximum AUROC difference in the same cell, the smaller one is the definition):")
    R.t(ver.groupby("bank")[["gap_median_auroc", "gap_mean_auroc", "gap_median_nb_site", "gap_median_nb20"]].max().reset_index())
    R.p("side by side by fs (column suffix = fs):")
    R.t(pv)
    R.p("difference table (diff name = first - second):")
    R.t(dt)
    R.p("check against reference values (OK if equal at the rounded decimal place):")
    R.t(cl)
    (out_dir / "claims_map.out.txt").write_text(R.text(), encoding="utf-8")
    return R.text(), t, True


# ===================== internal check =====================
def _fake_matrix(fs="F42"):
    """12 cells of simulated matrix_draws. For each cell (center dAUC, spacing, center dECE, spacing, center dSLD, center dNBs, expected 4-rule letters)."""
    S = [
        ("AKI_168h", "eICU", "500", 0.01, 0.0005, -0.01, 0.001, -0.10, 0.002, "GGGG"),
        ("AKI_168h", "eICU", "1000", 0.003, 0.0005, -0.002, 0.002, 0.05, -0.002, "GGVV"),
        ("AKI_168h", "eICU", "full", 0.02, 0.0, -0.01, 0.0, -0.02, 0.0, "GGGG"),
        ("AKI_168h", "MIMIC-IV", "500", -0.004, 0.001, -0.003, 0.0005, -0.02, 0.0, "NNNN"),
        ("AKI_168h", "MIMIC-IV", "1000", 0.0, 0.004, -0.001, 0.0001, -0.03, 0.0, "NFNF"),
        ("AKI_168h", "MIMIC-IV", "full", -0.03, 0.0, -0.01, 0.0, -0.02, 0.0, "FFFF"),
        ("mortality_168h", "eICU", "500", 0.005, 0.001, 0.002, 0.0005, 0.01, 0.0, "VVVV"),
        ("mortality_168h", "eICU", "1000", 0.001, 0.0005, 0.0, 0.001, 0.0, 0.0, "VVVV"),
        ("mortality_168h", "eICU", "full", -0.0049, 0.0, 0.004, 0.0, -0.02, 0.0, "SSSS"),
        ("mortality_168h", "MIMIC-IV", "500", -0.0049, 0.003, 0.004, 0.0005, 0.02, 0.0, "SFSF"),
        ("mortality_168h", "MIMIC-IV", "1000", -0.015, 0.0005, -0.01, 0.001, -0.01, 0.0, "FFFF"),
        ("mortality_168h", "MIMIC-IV", "full", -0.0099, 0.0, -0.005, 0.0, -0.02, 0.0, "NNNN"),
    ]
    rows = []
    for oc, tg, bank, cA, sA, cE, sE, cS, cNB, _ in S:
        for rep, k in enumerate([0] if bank == "full" else range(-5, 6)):
            dA, dE, dS = cA + k * sA, cE + k * sE, cS
            for meth in ("b1_base", "pram_b1"):
                is_m = meth == "pram_b1"
                slope_m = (1 + (0.2 + dS)) if rep % 2 == 0 else (1 - (0.2 + dS))   # test of absolute-value handling with mixed slope>1 and <1
                if oc == "AKI_168h":
                    nb06 = 0.01 + (cNB if is_m else 0.0)
                    nb20 = 0.02 + (-0.5 if is_m else 0.0)          # decoy for columns that are not the representative threshold
                else:
                    nb06 = 0.01 + (0.9 if is_m else 0.0)           # decoy for columns that are not the representative threshold
                    nb20 = 0.02 + (cNB if is_m else 0.0)
                rows.append({"fs": fs, "outcome": oc, "target": tg, "bank": bank, "rep": rep, "method": meth,
                             "auroc": 0.5 + (dA if is_m else 0.0), "auprc": 0.1, "brier": 0.05,
                             "ece": 0.05 + (dE if is_m else 0.0), "slope": slope_m if is_m else 0.8,
                             "intercept": 0.0, "slope_orig": 1.0, "mean_pred": 0.1, "obs_rate": 0.1,
                             "nb@0.06": nb06, "nb@0.2": nb20})
    extra = dict(rows[1])            # draw that lost its pair: rep=99 in the first cell has only pram_b1 and no frozen
    extra["rep"] = 99
    rows.append(extra)
    return pd.DataFrame(rows), [s[-1] for s in S], [(s[0], s[1], s[2]) for s in S]


def _fake_hospital(spec):
    """spec: {outcome: (muA, muE, muS, muNB)}. 40 hospitals, symmetric quantile noise (the number of improved hospitals can be counted by hand)."""
    z = stats.norm.ppf((np.arange(40) + 0.5) / 40)
    perm = np.random.default_rng(7).permutation(40)
    rows = []
    for oc, (muA, muE, muS, muNB) in spec.items():
        for i in range(40):
            dA = muA + 0.008 * z[i]
            dE = muE + 0.003 * z[perm[i]]
            dS = muS + 0.02 * z[perm[(i * 3) % 40]]
            dNB = muNB + 0.0005 * z[perm[(i * 7) % 40]]
            base = dict(outcome=oc, site=f"H{i:03d}", n_site=1000 + 20 * (i % 7), n_events=100, prev=0.1, auroc=0.7,
                        auprc=0.1, brier=0.05, ece=0.04, slope=0.8, intercept=0.0, mean_pred=0.1, obs_rate=0.1,
                        **{"nb@0.06": 0.01, "nb@0.2": 0.02})
            for meth in ["b1_base", "b5_base", "pram_b1", "pram_b5", "platt_b1", "pram_platt_b1", "local_lr",
                         "local_cb", "pooled_lr", "recal_iso"]:
                r = dict(base, method=meth)
                if meth == "pram_b1":
                    r.update(auroc=round(0.7 + dA, 6), ece=0.04 + dE, slope=1 - (0.2 + dS),
                             **{"nb@0.06": 0.01 + dNB, "nb@0.2": 0.02 + dNB})
                elif meth in ("local_lr", "local_cb"):
                    r["auroc"] = round(0.67 + 0.02 * z[i], 6)
                elif meth == "pram_b5":
                    r["auroc"] = 0.705
                rows.append(r)
    df = pd.DataFrame(rows)
    key = list(zip(df["outcome"], df["site"]))
    for dc, vc, bm in (("d_auroc_b1", "var_b1", "b1_base"), ("d_auroc_b5", "var_b5", "b5_base")):
        b = df[df["method"] == bm].set_index(["outcome", "site"])["auroc"]
        df[dc] = (df["auroc"].to_numpy() - b.reindex(key).to_numpy()).round(6)
        df[vc] = np.where(df["method"] == bm, 0.0, 1e-4)
    return df


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


def selftest():
    t_site = {"AKI_168h": 0.06, "mortality_168h": 0.2}
    tmp = Path(tempfile.mkdtemp(prefix="pram_claims_selftest_")).resolve()

    print("[0] THRESH_RANGE parser")
    fake_py = tmp / "fake_cells.py"
    fake_py.write_text('X = 1\nTHRESH_RANGE = {"AKI_168h": (0.02, 0.10), "mortality_168h": (0.10, 0.30)}\n', encoding="utf-8")
    _check(parse_thresholds(fake_py)[1] == {"AKI_168h": 0.06, "mortality_168h": 0.2}, "T_SITE parsing AKI 0.06 | death 0.2")
    _check(nb_col(0.06) == "nb@0.06" and nb_col(0.2) == "nb@0.2", "nb column name format")

    # ---------------- (a) cells ----------------
    print("[a] cell classification")
    m, expect, keys = _fake_matrix()
    fake_dir = tmp / "in"
    fake_dir.mkdir()
    m.to_csv(fake_dir / "matrix_draws_F42.csv", index=False, encoding="utf-8-sig")
    m.assign(fs="F33").to_csv(fake_dir / "matrix_draws_F33.csv", index=False, encoding="utf-8-sig")
    mm = read_csv(fake_dir / "matrix_draws_F42.csv")
    _check(set(mm["bank"]) == {"500", "1000", "full"}, "bank is a string (mixed with full)")
    j, info = extraction_diffs(mm, "pram_b1", "b1_base", t_site)
    _check(info["n_method"] == info["n_base"] + 1 and info["n_merged"] == info["n_base"], "rows before and after merging | 1 draw that lost its pair")
    _check(info["n_method_without_base"] == 1 and info["n_base_without_method"] == 0, "number of draws that lost their pair reported")
    c = classify_cells(summarize_cells(j, MARGIN), MARGIN)
    _check(len(c) == 12, "12 cells")
    _check((c[c["bank"] == "full"]["n_draws"] == 1).all() and (c["bank"] == "full").sum() == 4, "full cells n_draws=1, 4 cells")
    first = c[(c["outcome"] == "AKI_168h") & (c["target"] == "eICU") & (c["bank"] == "500")].iloc[0]
    _check(first["n_draws"] == 11, "the draw that lost its pair does not enter n_draws of the cell (11)")
    for (oc, tg, bank), exp in zip(keys, expect):
        row = c[(c["outcome"] == oc) & (c["target"] == tg) & (c["bank"] == bank)].iloc[0]
        got = row["cls_median"] + row["cls_NItail"] + row["cls_ECErobust"] + row["cls_both"]
        _check(got == exp, f"{oc}/{tg}/{bank} classification {got} (expected {exp})")
    cc = cat_order(c.assign(pair="pram_b1|b1_base"))
    cnt = count_table(cc, ["pair", "fs"], "median").iloc[0]
    _check((cnt["G"], cnt["N"], cnt["V"], cnt["S"], cnt["F"]) == (3, 3, 2, 2, 2), "median rule G3 N3 V2 S2 F2")
    _check((cnt["NI"], cnt["NI_ECE"], cnt["NI_SL"], cnt["NI_ECE_or_SL"], cnt["NI_NBok"]) == (10, 6, 6, 7, 9),
           "NI10 NI_ECE6 NI_SL6 NI_ECE_or_SL7 NI_NBok9")
    for rule, e in {"NItail": (3, 2, 2, 1, 4), "ECErobust": (2, 3, 3, 2, 2), "both": (2, 2, 3, 1, 4)}.items():
        r = count_table(cc, ["pair", "fs"], rule).iloc[0]
        _check((r["G"], r["N"], r["V"], r["S"], r["F"]) == e, f"{rule} number of rules {e}")
    ms = margin_sensitivity(cc).iloc[0]
    _check((ms["NI_margin_0.005"], ms["NI_margin_0.01"], ms["NI_margin_0.02"]) == (9, 10, 11), "margin sensitivity NI 9/10/11")
    a_cell = c[(c["outcome"] == "AKI_168h") & (c["target"] == "eICU") & (c["bank"] == "500")].iloc[0]
    m_cell = c[(c["outcome"] == "mortality_168h") & (c["target"] == "eICU") & (c["bank"] == "500")].iloc[0]
    _check(abs(a_cell["dNBs_med"] - 0.002) < 1e-12 and abs(a_cell["dNB20_med"] + 0.5) < 1e-12,
           "for AKI nb@0.06 is dNBs and nb@0.2 is dNB20")
    _check(abs(m_cell["dNBs_med"]) < 1e-12 and abs(m_cell["dNB20_med"]) < 1e-12, "for death nb@0.2 is dNBs (decoy column ignored)")
    _check(abs(a_cell["dSLD_med"] + 0.10) < 1e-9, "median dSLD -0.10 (absolute values are used even if slope>1 and <1 are mixed)")
    # (d) sign rule inversion error
    _check(improved("dECE", [-0.01, 0.01]).tolist() == [True, False], "sign (d): negative is improvement for ECE")
    _check(improved("dSLD", [-0.01, 0.01]).tolist() == [True, False], "sign (d): negative is improvement for dSLD")
    _check(improved("dAUC", [0.01, -0.01]).tolist() == [True, False], "sign (d): positive is improvement for AUROC")
    _check(improved("dNBs", [0.01, -0.01]).tolist() == [True, False], "sign (d): positive is improvement for net benefit")
    _check((c.loc[c["cls_median"] == "G", "dECE_med"] < 0).all() and (c.loc[c["cls_median"] == "V", "dECE_med"] >= 0).all(),
           "sign (d): G cells have negative median dECE, V cells zero or above")

    # ---------------- (b) hospitals ----------------
    print("[b] hospitals")
    spec = {"AKI_168h": (0.01, -0.01, -0.05, 0.001), "mortality_168h": (0.0, 0.005, 0.05, 0.0)}
    spec3 = {"AKI_168h": (-0.015, -0.01, -0.05, 0.001), "mortality_168h": (0.01, -0.01, -0.05, 0.001)}
    h42, h33 = _fake_hospital(spec), _fake_hospital(spec3)
    h42.to_csv(fake_dir / "hospital_results_F42.csv", index=False, encoding="utf-8-sig")
    h33.to_csv(fake_dir / "hospital_results_F33.csv", index=False, encoding="utf-8-sig")
    fr = hosp_frame(read_csv(fake_dir / "hospital_results_F42.csv"), "pram_b1", "b1_base", t_site)
    _check(fr.attrs["col_gap"] <= D_COL_TOL, "direct dAUC equals the d_auroc_b1 column")
    rng = np.random.default_rng(SEED)
    # (true value, three non-inferiority margins, number of improved hospitals dAUC/dECE/dSLD/dNBs, combined ECE, combined dSLD)
    exp_h = {"AKI_168h": (0.01, (True, True, True), (36, 40, 40, 39), True, True),
             "mortality_168h": (0.0, (True, True, True), (20, 2, 0, None), False, False)}
    for oc, (mu, ni_exp, imp_exp, comb_ece, comb_sld) in exp_h.items():
        st = hospital_stats(fr[fr["outcome"] == oc].reset_index(drop=True), rng)
        a = st[st["metric"] == "dAUC"].iloc[0]
        _check(a["lo"] <= mu <= a["hi"], f"{oc} interval [{a['lo']:.4f},{a['hi']:.4f}] contains the true value {mu} contained")
        v = verdict_row(st)
        _check((v["auc_lo_gt_neg0.005"], v["auc_lo_gt_neg0.01"], v["auc_lo_gt_neg0.02"]) == ni_exp, f"{oc} three non-inferiority margins {ni_exp}")
        _check(v["AUC_lo_gt_neg0.01_and_ECE_hi_lt0"] == comb_ece, f"{oc} combined (ECE) {comb_ece}")
        _check(v["AUC_lo_gt_neg0.01_and_dSLD_hi_lt0"] == comb_sld, f"{oc} combined (dSLD) {comb_sld}")
        got = tuple(int(st[st["metric"] == mt_].iloc[0]["n_improved"]) for mt_ in HMETRICS)
        _check(got[:3] == imp_exp[:3] and (imp_exp[3] is None or got[3] == imp_exp[3]), f"{oc} number of improved hospitals {got}")
        if oc == "mortality_168h":
            _check(a["p_binom"] == 1.0 and a["p_wilcoxon"] > 0.5, "mean 0 scenario: sign test p=1, Wilcoxon p>0.5")
        else:
            _check(a["p_binom"] < 1e-5 and a["p_wilcoxon"] < 1e-4, "mean +0.01 scenario: both p small")
    fr3 = hosp_frame(read_csv(fake_dir / "hospital_results_F33.csv"), "pram_b1", "b1_base", t_site)
    st3 = hospital_stats(fr3[fr3["outcome"] == "AKI_168h"].reset_index(drop=True), np.random.default_rng(SEED))
    a3, v3 = st3[st3["metric"] == "dAUC"].iloc[0], verdict_row(st3)
    _check(a3["lo"] <= -0.015 <= a3["hi"], f"mean -0.015 scenario interval [{a3['lo']:.4f},{a3['hi']:.4f}] contains the true value")
    _check((v3["auc_lo_gt_neg0.005"], v3["auc_lo_gt_neg0.01"], v3["auc_lo_gt_neg0.02"]) == (False, False, True), "mean -0.015: non-inferiority F F T")
    _check(v3["AUC_lo_gt_neg0.01_and_ECE_hi_lt0"] is False, "mean -0.015: even if ECE improves the combined decision is False")
    _check(int(a3["n_improved"]) == 1, "mean -0.015: 1 improved hospital")
    z0 = pd.DataFrame({"n_site": [100] * 40, "dAUC": [0.0] * 35 + [0.01] * 5, "dECE": [0.0] * 40, "dSLD": [0.0] * 40,
                       "dNBs": [0.0] * 40})
    r0 = hospital_stats(z0, np.random.default_rng(1), n_boot=200).query("metric=='dAUC'").iloc[0]
    _check(r0["n_nonzero"] == 5 and np.isnan(r0["p_binom"]) and np.isnan(r0["p_wilcoxon"]), "5 nonzero: both p are NaN")
    z0["dAUC"] = [0.0] * 34 + [0.01] * 6
    r6 = hospital_stats(z0, np.random.default_rng(1), n_boot=200).query("metric=='dAUC'").iloc[0]
    _check(r6["n_nonzero"] == 6 and abs(r6["p_binom"] - 2 / 64) < 1e-12 and abs(r6["p_wilcoxon"] - 2 / 64) < 1e-12,
           "6 nonzero: p computed (2/64)")
    d_ = [0.05, -0.05, 0.0, 0.0, 0.0, -0.02, -0.005, 0.001, 0.0, 0.0]
    v_ = [1e-4, 1e-4, 1e-4, 1e-6, 0.0, 1e-6, 2e-6, 4e-6, np.nan, 0.0]
    dc = delong_counts(d_, v_, 0.01)
    _check((dc["n"], dc["n_excluded"], dc["n_identical"], dc["n_sig_up"], dc["n_sig_down"], dc["n_uncertain"],
            dc["n_NI"], dc["n_clear_inferior"]) == (9, 1, 2, 1, 3, 3, 6, 2), "DeLong classes 9/1/2/1/3/3/6/2 (var=0 and d=0 same, NaN excluded, 1 significant decrease that is also non-inferior)")
    _check(dc["n_identical"] + dc["n_sig_up"] + dc["n_sig_down"] + dc["n_uncertain"] == dc["n"], "exclusive classes sum = valid n")
    a6 = pd.DataFrame({"outcome": fr["outcome"], "site": fr["site"], "d_auroc": fr["d_col"],
                       "var_delong_original": 21.8 * fr["var_col"], "var_delong_standard": fr["var_col"]})
    _check(var_provenance(fr, a6)["status"] == "standard_direct", "variance source: standard direct match")
    vc = fr["var_col"].to_numpy()
    _check(var_provenance(fr, a6.assign(var_delong_original=vc, var_delong_standard=vc / 21.8))["status"] == "original_direct",
           "variance source: original implementation direct match")
    _check(var_provenance(fr, a6.assign(var_delong_standard=vc * 0.5, var_delong_original=vc * 11))["status"] == "standard_scale",
           "variance source: direct mismatch, standard scale")
    _check(var_provenance(fr, a6.assign(var_delong_standard=vc / 21.8, var_delong_original=vc * 2.0))["status"] == "original_scale",
           "variance source: direct mismatch, overestimated scale")
    stx = pd.DataFrame([dict(fs="F42", outcome="AKI_168h", pair="pram_b1|b1_base", metric="dAUC", wmean=0.0092, lo=0.0042, hi=0.0148)])
    hsx = pd.DataFrame([dict(fs="F42", outcome="AKI_168h", method="pram_b1", delta_stays_weighted=0.0092)])
    rd = {("F42", "AKI_168h"): (0.0040, 0.0150)}
    _check(selfcheck(stx, hsx, readings=rd)[1], "self-consistency passing scenario")
    _check(not selfcheck(stx, hsx, readings={("F42", "AKI_168h"): (0.0100, 0.0150)})[1], "fails if the reading interval deviates by more than 0.0008")
    _check(not selfcheck(stx, hsx.assign(delta_stays_weighted=0.0093), readings=rd)[1], "fails if it deviates from hospital_summary by more than 1e-6")

    # ---------------- (c) tail ----------------
    print("[c] tail")
    d_t = [0.05, -0.05, 0.0, -0.03, 0.02, -0.015, -0.005, 0.0, 0.01, np.nan]
    v_t = [1e-4, 1e-4, 1e-4, 2.5e-5, 2.5e-5, 4e-6, 4e-6, 1e-6, np.nan, 1e-4]
    mk = lambda meth, i, d, v: dict(fs="F33", outcome="AKI_168h", target="eICU", bank=500, fold=0, draw=i, method=meth,  # noqa: E731
                                    n_events_bank=30, auroc=0.7, d_auroc_base=d, var_auroc_base=v)
    rows = [mk("pram_b1", i, d, v) for i, (d, v) in enumerate(zip(d_t, v_t))]
    rows += [mk("b1_base", i, np.nan, np.nan) for i in range(3)] + [mk("ct_b1", i, 0.0, 0.0) for i in range(4)]
    pd.DataFrame(rows).to_csv(fake_dir / "A2_tail_draws_F33.csv", index=False, encoding="utf-8-sig")
    tdr = read_csv(fake_dir / "A2_tail_draws_F33.csv")
    tt = tail_table(tdr)
    r = tt[tt["method"] == "pram_b1"].iloc[0]
    _check(r["n"] == 8 and r["n_excluded"] == 2, "tail: valid 8, NaN excluded 2")
    _check(abs(r["share_d_lt0"] - 0.5) < 1e-12 and abs(r["share_d_lt_neg01"] - 0.375) < 1e-12, "share(d<0)=0.5, share(d<-0.01)=0.375")
    _check(abs(r["share_ci_below0"] - 0.5) < 1e-12 and abs(r["share_ci_below_neg01"] - 0.375) < 1e-12, "whole interval below 0 0.5, below -0.01 0.375")
    _check(abs(r["share_ni"] - 0.5) < 1e-12 and abs(r["share_ci_above0"] - 0.25) < 1e-12, "non-inferiority 0.5, whole interval above 0 0.25")
    _check(abs(r["se_median"] - 0.005) < 1e-12, "se median 0.005")
    _check(abs(r["diff_below_vs_2.5pct"] - 0.475) < 1e-12 and abs(r["diff_above_vs_2.5pct"] - 0.225) < 1e-12, "difference from the null 2.5%")
    rb = tt[tt["method"] == "b1_base"].iloc[0]
    _check(rb["n"] == 0 and rb["n_excluded"] == 3, "all base rows excluded")
    rc = tt[tt["method"] == "ct_b1"].iloc[0]
    _check(rc["share_ci_below0"] == 0 and rc["share_ci_above0"] == 0 and rc["share_ni"] == 1.0, "d=0 var=0: not below 0 | non-inferiority holds")
    summ = pd.DataFrame([dict(fs="F33", source="new computation", outcome="AKI_168h", target="eICU", bank="500", method="pram_b1",
                              share_below_zero=0.5, draws_ci_wholly_below_zero=0.5, draws_ci_wholly_above_zero=0.25),
                         dict(fs="F33", source="new computation", outcome="AKI_168h", target="eICU", bank="500", method="ct_b1",
                              share_below_zero=0.3, draws_ci_wholly_below_zero=0.0, draws_ci_wholly_above_zero=0.0)])
    xc = tail_crosscheck(tt, summ)
    _check(bool(xc[xc["method"] == "pram_b1"]["match"].iloc[0]) and not bool(xc[xc["method"] == "ct_b1"]["match"].iloc[0]),
           "cross-check: equal values are marked match, different values are marked mismatch")

    # ---------------- (D) main table ----------------
    print("[D] main table | values for checking")
    mt_rows = []
    for fs, off in (("F33", 0.0), ("F42", 0.01)):
        for oc in ("AKI_168h", "mortality_168h"):
            for bank in ("1000", "full"):
                for meth, a_ in (("b1_base", 0.8117), ("pram_b1", 0.8207), ("b5_base", 0.8224), ("pram_b5", 0.8257)):
                    mt_rows.append(dict(fs=fs, outcome=oc, target="eICU", bank=bank, method=meth, auroc=a_ + off,
                                        slope=0.744 if meth == "b1_base" else 0.921, intercept=0, ece=0.0247, nb_site=0.0,
                                        nb_20=-0.0008, n_draws=10, dAUROC_vs_b1=0.0015, dAUROC_q25=0, dAUROC_q75=0))
    mtf = pd.DataFrame(mt_rows)
    sel = (mtf.fs == "F33") & (mtf.outcome == "mortality_168h") & (mtf.bank == "1000") & (mtf.method == "pram_b1")
    mtf.loc[sel, "auroc"] = 0.8117 + 0.0015
    L = main_long(mtf, {"F42": mm, "F33": mm.assign(fs="F33")}, t_site)
    _check((L["bank"] == "500").any(), "bank 500, not in the main table, filled with the matrix median")
    cl_ = claims_check(L, mtf)
    st_ = dict(zip(cl_["claim"], cl_["status"]))
    _check(st_["death full F33 b1_base AUROC"] == "OK" and st_["death full F33 pram_b1 AUROC"] == "OK", "values for checking: correct values are OK")
    _check(st_["death full F33 b1_base ECE"] == "OK" and st_["death full F33 b1_base slope"] == "OK", "values for checking: ECE and slope OK")
    _check(st_["CatBoost 42 columns (death full) pram_b5 AUROC"] == "mismatch", "values for checking: simulated 0.8357 versus 0.8310 is a mismatch")
    _check(st_["death full F33 pram_b1 ECE"] == "mismatch", "values for checking: ECE 0.0247 versus 0.0163 is a mismatch")
    _check([v for k, v in st_.items() if k.startswith("death 1000 F33 pram_b1 - b1_base")] == ["OK"], "bank 1000 +0.0015 confirmed")
    dt = diff_tables(L[L["method"].isin(MT_METHODS)])
    x = dt[(dt["outcome"] == "mortality_168h") & (dt["bank"] == "full") & (dt["diff"] == "b1_base(F33) - b1_base(F42)")]
    _check(abs(x["auroc"].iloc[0] + 0.01) < 1e-9, "difference table: b1_base(F33) - b1_base(F42) = -0.01")

    # ---------------- end to end: run() on a simulated folder ----------------
    print("[run] simulated folder end to end")
    pd.DataFrame({"outcome": fr["outcome"], "site": fr["site"], "n_site": fr["n_site"], "d_auroc": fr["d_col"],
                  "var_delong_original": 21.8 * fr["var_col"], "var_delong_standard": fr["var_col"],
                  "var_boot": fr["var_col"]}).to_csv(fake_dir / "A6_variance_diagnosis_by_hospital.csv", index=False, encoding="utf-8-sig")
    hs_rows = []
    for fs, h in (("F42", h42), ("F33", h33)):
        for (oc, meth), g in h.groupby(["outcome", "method"]):
            w = g["n_site"].to_numpy(float)
            hs_rows.append(dict(fs=fs, outcome=oc, method=meth, k=len(g),
                                delta_stays_weighted=float((w * g["d_auroc_b1"]).sum() / w.sum()),
                                I2_stays_weighted=0.0, I2_invvar=10.0))
    pd.DataFrame(hs_rows).to_csv(fake_dir / "hospital_summary.csv", index=False, encoding="utf-8-sig")
    summ.to_csv(fake_dir / "A2_tail_summary.csv", index=False, encoding="utf-8-sig")
    mtf.to_csv(fake_dir / "main_table_F42_F33.csv", index=False, encoding="utf-8-sig")
    out = tmp / "out"
    text, tabs, ok = run(load_inputs(fake_dir), out, t_site, readings={}, header_extra="[internal check simulated input]")
    _check(ok and (out / "claims_map.out.txt").exists(), "run() ran to the end and wrote out.txt")
    _check(len(tabs["B_hospital_stats.csv"]) == 2 * 4 * 2 * 4, "number of hospital statistics rows = fs2 x pairs4 x outcomes2 x metrics4")
    bs = tabs["B_hospital_stats.csv"]
    r = bs[(bs.fs == "F42") & (bs.outcome == "AKI_168h") & (bs.pair == "pram_b1|b1_base") & (bs.metric == "dAUC")].iloc[0]
    _check(r["lo"] <= 0.01 <= r["hi"], "run(): the F42 AKI interval contains the true value 0.01")
    dl = tabs["B_hospital_delong.csv"]
    _check((dl[dl["pair"] == "pram_b1|b1_base"]["var_status"] == "standard_direct").all(), "run(): pram_b1 var standard direct check")
    a_part = text.split("## A.")[1].split("## B. Hospital level")[0]
    _check("significant" not in a_part.replace("not a test", ""), "no significant in the output of section A")
    _check("post hoc definitions, sensitivity 0.005 and 0.02 given alongside" in text, "post hoc definition wording at the head of the output")
    for p in out.iterdir():
        _check(str(p.resolve()).startswith(str(tmp)), f"files written are inside the temporary folder: {p.name}")
    print("all internal checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if a.selftest:
        return selftest()
    _, t_site = parse_thresholds(CELLS_PY)
    text, _, ok = run(load_inputs(IN_DIR), OUT_DIR, t_site)
    print(text)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
