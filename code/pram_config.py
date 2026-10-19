"""Shared settings of all notebooks and scripts: random seeds, bootstrap repetitions, thresholds and input folders.

Every value that must be identical across the analyses is defined here once.
Input folders can be set with environment variables before starting Jupyter (see README):
    PRAM_DATA_ROOT    folder with the prepared feature and outcome files (default ./prepared_data)
    PRAM_MIMIC4_ROOT  folder of the MIMIC-IV release with hosp/patients and icu/icustays (default ./mimic4_raw)
    PRAM_EICU_ROOT    folder of the eICU-CRD release with patient, vitalPeriodic, nurseCharting (default ./eicu_raw)
"""
import os
from pathlib import Path

# ---- random numbers ----------------------------------------------------------------------------------------------
SEED = 42                         # model fitting, data splitting and bank draws
BOOT_SEED = 20261003              # bootstrap random stream (hospital-level intervals and the within-hospital bootstrap)

# ---- bootstrap repetitions (one value for the whole analysis) -----------------------------------------------------
N_BOOT = 5000                     # hospital-level intervals (resampling of hospitals)
N_BOOT_WITHIN = N_BOOT            # within-hospital bootstrap of the variance diagnosis (Supplementary S8 Panel F)

# ---- decision-curve threshold range per outcome (the full 1 to 50% grid is always written to csv) --
THRESH_RANGE = {"AKI_168h": (0.02, 0.10), "mortality_168h": (0.10, 0.30)}

# ---- input folders --------------------------------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("PRAM_DATA_ROOT", "./prepared_data"))
MIMIC4_ROOT = Path(os.environ.get("PRAM_MIMIC4_ROOT", "./mimic4_raw"))
EICU_ROOT = Path(os.environ.get("PRAM_EICU_ROOT", "./eicu_raw"))
