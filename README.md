# PRAM: post-hoc retrieval augmentation of a frozen ICU prediction model

Code and aggregate results of the analysis of PRAM (Post-hoc Retrieval Augmentation for Adapting a Frozen ICU Prediction Model to a New Hospital Without Retraining). A frozen model keeps its parameters. For a new hospital, its prediction is mixed with the outcomes of the most similar patients in a small, replaceable local bank, with the mixing weight chosen inside the bank.

The three source databases are not redistributed (see Data). The repository holds the code, a description of the inputs, and the aggregate result grids as CSV files.

## Contents

| Path | What it holds |
|---|---|
| `notebooks/01_main_grid_and_hospital_level.ipynb` | development models, strategy grid over bank sizes (main analysis and the 33-variable set), hospital level, temporal validation, tail risk, network of unlabeled hospitals, TabPFN, update hierarchy, rescoring with the standard DeLong variance |
| `notebooks/02_closed_testing.ipynb` | closed testing as published (Vergouwe et al., 2017) next to the variant used in the paper; reads the results of notebook 01 |
| `notebooks/03_supplementary_analyses.ipynb` | importance weighting, CORAL, anchor regression, full-bank test for 33 variables, 100-draw single-bank intervals, partner prevalence, within-hospital bootstrap, mixing-weight criteria, upper limit of the mixing weight, federated analyses |
| `code/pram_config.py` | seeds, bootstrap repetitions, threshold range and input folders, defined once and imported by every notebook and script |
| `code/aggregate_claims.py` | cell-level, hospital-level and tail summaries computed from the csv files of notebook 01 (`derived/` files) |
| `code/aggregate_hospital_alpha.py` | hospital-level statistics of the network analysis for every mixing weight (`derived/P3_hospital_stats_all_alpha.csv`) |
| `code/figures/` | scripts of Figures 1 to 4 and Supplementary Figures S1 and S3 |
| `results_csv/` | aggregate result grids (see `results_csv/README.md` and `results_csv/table_map.csv`) |
| `requirements.txt` | Python packages |

## Data

The three databases are hosted on PhysioNet under credentialed access. For each, create a PhysioNet account, complete the required human-subjects research training, sign the data use agreement of the database, then download the release used here:

- MIMIC-IV v2.2 (https://physionet.org/content/mimiciv/)
- MIMIC-III Clinical Database v1.4 (https://physionet.org/content/mimiciii/)
- eICU Collaborative Research Database v2.0 (https://physionet.org/content/eicu-crd/)

The databases are not redistributed in this repository. The notebooks read the prepared input tables below, one row per ICU stay.

### Prepared inputs (`PRAM_DATA_ROOT`)

One folder, read-only, with these files. Only names, types and roles are given.

| File | Columns | Role |
|---|---|---|
| `mimic4/X_folded.parquet` | `stay_id` (integer, ICU stay), optional `fold_id` (integer, ignored), and the 42 feature columns below | development cohort (MIMIC-IV) |
| `mimic4/binary_outcomes_mimic4.csv` | `stay_id`, `aki_168h` (0 or 1), `death_168h` (0 or 1) | outcomes of the development cohort |
| `mimic3/X_external.parquet`, `mimic3/binary_outcomes_mimic3.csv` | the same columns | MIMIC-III cohort |
| `eicu/X_external.parquet`, `eicu/binary_outcomes_eicu.csv` | the same columns; `stay_id` is `patientunitstayid` | eICU-CRD cohort |

The 42 feature columns (aggregates of the first 24 hours; a column that a database does not have is read as missing):

`age, sex_male, baseline_scr, hr_min, hr_max, hr_last, rr_min, rr_max, rr_last, spo2_min, spo2_max, spo2_last, temp_c_min, temp_c_max, temp_c_last, sbp_min, sbp_max, sbp_last, dbp_min, dbp_max, dbp_last, weight_kg_min, weight_kg_max, weight_kg_last, gcs_total_min, gcs_total_max, gcs_total_last, bun_min, bun_max, bun_last, calcium_min, calcium_max, calcium_last, inr_min, inr_max, inr_last, scr_feat_min, scr_feat_max, scr_feat_last, bun_cr_ratio_min, bun_cr_ratio_max, bun_cr_ratio_last`

A column with only the values 0 and 1 is treated as binary and all others as numeric.

`stay_id` must be unique within a file, and the feature and outcome files are linked by it. No split column is read: the notebooks split the MIMIC-IV development cohort themselves (stratified 56 / 14 / 30 percent for training, validation and test, seed `SEED` of `code/pram_config.py`). A MIMIC-IV stay without a row in the outcome file is given outcome 0 and counted in a printed warning; a MIMIC-III or eICU stay without one is dropped. Units follow the column names (`temp_c` in degrees Celsius, `weight_kg` in kilograms, `age` in years, `sex_male` 0 or 1).

### Raw tables read directly

These are read from the database releases, not from the prepared folder. Each is optional; an analysis that needs a missing table is skipped and reported in the input check table printed by the first cell.

| Table | Columns used | Folder (variable) | Used for |
|---|---|---|---|
| eICU `patient.csv` or `patient.csv.gz` | `patientunitstayid`, `hospitalid`, `hospitaldischargeyear` | `PRAM_EICU_ROOT` (also found in `PRAM_DATA_ROOT/eicu`) | hospital-level analyses, eICU temporal validation |
| MIMIC-IV `hosp/patients.csv(.gz)` | `subject_id`, `anchor_year_group`, optional `anchor_year` | `PRAM_MIMIC4_ROOT` | MIMIC-IV temporal validation and period environments |
| MIMIC-IV `icu/icustays.csv(.gz)` | `stay_id`, `subject_id`, optional `intime` | `PRAM_MIMIC4_ROOT` | the same |
| eICU `vitalPeriodic.csv.gz`, `nurseCharting.csv.gz` | `patientunitstayid`, `observationoffset`, `temperature`; `nursingchartoffset`, `nursingchartcelltypevallabel`, `nursingchartcelltypevalname`, `nursingchartvalue` | `PRAM_EICU_ROOT` | share of stays with a temperature record (notebook 03) |

## Environment

Python 3.11. Package versions are in `requirements.txt`. A GPU is needed only for the TabPFN analysis (without a GPU the target cohort is reduced and the notebook prints the expected time first).

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyterlab
```

## How to run

1. Set the input folders before starting Jupyter, for example

   ```
   export PRAM_DATA_ROOT=/path/to/prepared_data
   export PRAM_MIMIC4_ROOT=/path/to/mimiciv
   export PRAM_EICU_ROOT=/path/to/eicu-crd
   ```

   (Windows: `set` instead of `export`.) The defaults are `./prepared_data`, `./mimic4_raw` and `./eicu_raw`, relative to the folder of the notebook.
2. Open the notebooks of `notebooks/` in this order and run each with Restart and Run All, starting from the same working folder, which is the folder of the notebook files (`notebooks/`; the notebooks write to `out_pram/` there, which holds the finished units):
   1. `01_main_grid_and_hospital_level.ipynb` computes everything from the prepared inputs: development models, bank grid, hospital level, then the other sections. It is the longest notebook.
   2. `02_closed_testing.ipynb` needs the finished grid and hospital units of notebook 01.
   3. `03_supplementary_analyses.ipynb` reads the results of notebooks 01 and 02 where they are present and recomputes what is missing.
3. Every notebook can be interrupted and run again: finished units are loaded and only the rest is computed, also when cells are added at the end. The first cell prints an input check table; the pre-check cells print the expected time.
4. The notebooks write csv files to `out_pram/github_csv/` and a zip of the aggregate results to the working folder (`notebooks/`). The scripts read that folder by default when they are run from the repository root:

   ```
   python code/aggregate_claims.py                   # PRAM_CSV_DIR (default notebooks/out_pram/github_csv) -> out_claims/
   python code/aggregate_hospital_alpha.py
   python code/figures/plot_fig2_bank_size.py             # figures read PRAM_FIGURE_CSV (default results_csv/) and write to figure_out/
   ```

   `python code/aggregate_claims.py --selftest` checks the script on simulated csv files.

Settings that must be identical across the analysis (seeds, number of bootstrap resamples, threshold range) are in `code/pram_config.py`: `SEED = 42` for model fitting, splitting and bank draws; `BOOT_SEED = 20261003` and `N_BOOT = 5000` for the bootstrap of hospitals; `N_BOOT_WITHIN = N_BOOT` for the within-hospital bootstrap of the variance diagnosis (Supplementary Table S8, Panel F).

## Manuscript items and the code and files behind them

Cell numbers count from 0 and include markdown cells. A cell writes the csv files named in the same row. Files in `derived/` and `legacy/` are in `results_csv/`. Supplementary Methods and Supplementary Tables S13 and S16 have no result file, and Supplementary Figures S2 and S4 are not produced by code in this repository.

| Item | Result files | Code |
|---|---|---|
| Table S1 | legacy folder | legacy/ |
| Table S2 | main_table_F42_F33.csv, A8_compare_F42_F33.csv, R5_X_fz_table.csv, A8_network_F42_F33.csv, R8_C_network_summary.csv | notebook 01 cell 15, 39; notebook 03 cell 72, 95 |
| Table S3 | legacy folder | legacy/ |
| Table S4 | A8_compare_F42_F33.csv, matrix_draws_F42.csv, matrix_draws_F33.csv, derived/A_grid.csv, derived/A_counts_pair_fs.csv, derived/A_margin_sensitivity.csv | notebook 01 cell 13, 15; code/aggregate_claims.py |
| Table S5 | R5_X_S5A_values.csv, R5_tail_ci_summary.csv, R5_tail_draws.csv, R5_whole_table_F33.csv, A2_tail_summary.csv, A2_tail_draws_F33.csv, RESCORE_table2_q.csv, RESCORE_draws_F42.csv | notebook 01 cell 23, 24, 47; notebook 03 cell 40, 42, 52; code/aggregate_claims.py |
| Table S6 | A3_dca_curves_matrix.csv, A3_disagreement_by_threshold.csv, A3_range_summary.csv, A3_area_winners.csv, main_table_F42_F33.csv | notebook 01 cell 15, 31 |
| Table S7 | A4_update_hierarchy_table.csv, A4_closed_test_step_shares.csv, A5_combination_table.csv, CTPUB_S7A_side_by_side.csv, CTPUB_S7B_table.csv, CTPUB_S7C_flags.csv, CTPUB_step_shares.csv, legacy folder | notebook 01 cell 41, 43; notebook 02 cell 20; legacy/ |
| Table S8 | hospital_results_F42.csv, hospital_results_F33.csv, hospital_summary.csv, A6_meta_reml_hk.csv, A6_variance_diagnosis_by_hospital.csv, CTPUB_hospital_summary.csv, hospital_events_summary.csv, derived/B_hospital_stats.csv, derived/B_hospital_delong.csv | notebook 01 cell 26, 27, 35, 37 (Panel F); notebook 02 cell 20; code/aggregate_claims.py; notebook 03 cell 82 (Panel F, written there as R5S8_A6_variance_diagnosis_by_hospital.csv) |
| Table S9 | A1_policy_contrast.csv, A1_period_summary.csv, A1_mimic4_period_counts.csv, R6_N2_period_pairs.csv | notebook 01 cell 20, 21; notebook 03 cell 88 |
| Table S10 | A9_tabpfn_summary.csv, A9_tabpfn_retrieval_gain.csv, A9_tabpfn_draws.csv, R5_A9_N_inverted.csv, legacy folder (rank correlations only; the rows of Panel A have no CSV file) | notebook 01 cell 17; notebook 03 cell 50; legacy/ |
| Table S11 | R8_A_scheme_summary.csv, R8_B_method_summary.csv, R8_C_network_summary.csv, R8_C_prevalence_correlation.csv, R8_C_prevalence_scenarios.csv, R8_split_frame_effect.csv, A7_summary_F42.csv, A7_summary_F33.csv, A7_network_F42.csv, A7_network_F33.csv, A7_correlations.csv, R5_X_net_summary.csv, derived/P3_hospital_stats_all_alpha.csv, legacy folder | notebook 01 cell 39; notebook 03 cell 74, 93, 94, 95, 96; code/aggregate_hospital_alpha.py; legacy/ |
| Table S12 | legacy folder | legacy/ |
| Table S14 | legacy folder, RESCORE_coldstart_S20.csv, A4_update_hierarchy_table.csv, hospital_results_F42.csv, hospital_results_F33.csv, R6_N1_hospital_criteria.csv, R6_N1_hospital_pairdiff.csv, R6_N1_hospital_alpha_distribution.csv, R6_N1_whole_criteria.csv, R6_N1_whole_pairdiff.csv, R7_N1_hospital_arms.csv, R7_N1_hospital_pairdiff.csv, R7_N1_hospital_alpha_distribution.csv, R7_N1_whole_arms.csv, R7_N1_whole_pairdiff.csv, R7_single_bank_alpha_distribution.csv, R7_single_bank_alpha_summary.csv | notebook 01 cell 26, 41, 49; notebook 03 cell 87, 90, 91, 92; legacy/ |
| Table S15 | R7_N1_hospital_arms.csv, R7_N1_hospital_pairdiff.csv, R7_N1_whole_arms.csv, R7_N1_whole_pairdiff.csv | notebook 03 cell 90, 91 |
| Table S17 | R5_B_cells_all.csv, R5_B_margin_split.csv, R5_B_hosp_boot.csv, R5_B_hosp_dom.csv, R5_B1c_weight_compare.csv, R5_B1c_penalty_share.csv, R5_B3_ar_fit.csv, R5_B3_ar_eval.csv, R5_B3_fusion_cells.csv, legacy folder | notebook 03 cell 56, 60, 70; legacy/ |
| Fig. S1 | as Table S6 | code/figures/plot_figS1_decision_curves.py |
| Fig. S2 | as Table S6 |  |
| Fig. S3 | as Table S8 | code/figures/plot_figS3_hospital_forest.py |
| Fig. S4 | as Table S12 |  |
| Table 1 (main) | legacy/S05_PanelB_smd_by_variable.csv; A1_period_summary.csv | notebook 01 cell 21; legacy/ |
| Table 2 (main) | A4_update_hierarchy_table.csv; hospital_results_F42.csv; hospital_results_F33.csv; legacy/Main02_bank_size_slope.csv; RESCORE_coldstart_S20.csv | notebook 01 cell 26, 41, 49; legacy/ |
| Table 3 (main) | main_table_F42_F33.csv; RESCORE_table2_q.csv; R5_whole_table_F33.csv; R5_X_fz_table.csv; A5_combination_table.csv | notebook 01 cell 15, 43, 47; notebook 03 cell 40, 72 |
| Table 4 (main) | A1_policy_contrast.csv; A1_period_summary.csv; R6_N2_period_pairs.csv; legacy/Main04_PanelA_time_memory.csv | notebook 01 cell 21; notebook 03 cell 88; legacy/ |
| Table 5 (main) | A9_tabpfn_retrieval_gain.csv; A8_compare_F42_F33.csv | notebook 01 cell 15, 17; code/aggregate_claims.py |
| Table 6 (main) | A7_summary_F42.csv; A4_update_hierarchy_table.csv; main_table_F42_F33.csv | notebook 01 cell 15, 39, 41 |
| Fig. 2 (main) | main_table_F42_F33.csv; matrix_draws_F42.csv | code/figures/plot_fig2_bank_size.py; notebook 01 cell 13, 15 |
| Fig. 3 (main) | hospital_results_F42.csv; hospital_results_F33.csv; hospital_summary.csv | code/figures/plot_fig3_hospital.py; notebook 01 cell 26, 27 |
| Fig. 4 (main) | A3_dca_curves_matrix.csv; A3_disagreement_by_threshold.csv | code/figures/plot_fig4_net_benefit.py; notebook 01 cell 31 |
| Table S14: choice of the mixing-weight criterion | R6_N1_hospital_alpha_distribution.csv; R6_N1_hospital_criteria.csv; R6_N1_hospital_pairdiff.csv; R6_N1_whole_criteria.csv; R6_N1_whole_pairdiff.csv | notebook 03 cell 87 |
| Tables S14 and S15: upper limit of the mixing weight and a fixed weight of 0.50 | R7_N1_hospital_arms.csv; R7_N1_hospital_pairdiff.csv; R7_N1_hospital_alpha_distribution.csv; R7_N1_whole_arms.csv; R7_N1_whole_pairdiff.csv; R7_single_bank_alpha_distribution.csv; R7_single_bank_alpha_summary.csv | notebook 03 cell 90, 91, 92 |
| Table S11: federated analyses under the original split and the three-fold split | R8_A_scheme_summary.csv; R8_B_method_summary.csv; R8_C_network_summary.csv; R8_C_prevalence_correlation.csv; R8_C_prevalence_scenarios.csv; R8_split_frame_effect.csv | notebook 03 cell 93, 94, 95, 96 |

## Results (`results_csv/`)

`results_csv/README.md` lists every file with its rows and columns, what one row is, and the tables it supports. `results_csv/table_map.csv` lists, for each table and figure, the files that hold its values. Hospital labels in the csv files (H001 to H067) are arbitrary labels assigned within `results_csv/`; a rerun of the notebooks labels hospitals differently and orders their rows differently, so hospital-level summaries agree, and the bootstrap limits of the hospital mean differ by resampling noise.
