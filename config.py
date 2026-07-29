from pathlib import Path
from typing import List
# ── Column definitions ────────────────────────────────────────────────────────

# NUMERIC_PREDICTORS: list[str] = [
#     # Demographics
#     "AGE", "PTEDUCAT",
#     # Genetics (ordinal: 0 / 1 / 2 ε4 alleles)
#     "APOE4",
#     # Time from baseline
#     # "Years_bl", "Month_bl", "Month", "M",
#     # Baseline cognitive scores
#     # "MMSE_bl", "CDRSB_bl","LDELTOTAL_BL",
#     "ADAS11_bl", "ADAS13_bl", "ADASQ4_bl",
#     "RAVLT_immediate_bl", "RAVLT_learning_bl", "RAVLT_forgetting_bl",
#     "RAVLT_perc_forgetting_bl", "DIGITSCOR_bl",
#     "TRABSCOR_bl", "FAQ_bl", "mPACCdigit_bl", "mPACCtrailsB_bl", "MOCA_bl",
#     # Baseline Ecog — patient self-report
#     "EcogPtMem_bl", "EcogPtLang_bl", "EcogPtVisspat_bl", "EcogPtPlan_bl",
#     "EcogPtOrgan_bl", "EcogPtDivatt_bl", "EcogPtTotal_bl",
#     # Baseline Ecog — study partner report
#     "EcogSPMem_bl", "EcogSPLang_bl", "EcogSPVisspat_bl", "EcogSPPlan_bl",
#     "EcogSPOrgan_bl", "EcogSPDivatt_bl", "EcogSPTotal_bl",
# ]

# CATEGORICAL_PREDICTORS: list[str] = [
#     "Sex", 
#     # "PTETHCAT", 
#     # "PTRACCAT", 
#     # "PTMARRY",
# ]

# new experimental predictors
NUMERIC_PREDICTORS: list[str] = [
    # Demographics
    "AGE", "PTEDUCAT",
    # Genetics
    "APOE4",
    # Neuropsychological / cognitive tests
    "ADAS11_bl", "ADAS13_bl", "ADASQ4_bl",
    "RAVLT_immediate_bl", "RAVLT_learning_bl", "RAVLT_forgetting_bl", "RAVLT_perc_forgetting_bl",
    "DIGITSCOR_bl", "TRABSCOR_bl",
    "MOCA_bl",
]

CATEGORICAL_PREDICTORS: list[str] = [
    "Sex",
]

TARGET: str = "Group"

BASE_DIR = "/home/ybrima/Data/ADNI_Data"
PROJECT_DIR = "/home/ybrima/dev"

DATA_DIR = Path(f"{BASE_DIR}")
RESULTS_DIR = Path(f"{PROJECT_DIR}/Scripts/results")
FIGURS_DIR = Path(f"{PROJECT_DIR}/Scripts/figures")
METADATA_DIR = Path(f"{PROJECT_DIR}/Scripts/metadata")
RAW_NIFTI_DIR = Path("/home/ybrima/Data/ADNI/")

EXPERIMENTS = {
    0: None,                  # Multiclass (default)
    1: ["CN", "AD"],
    2: ["CN", "MCI"],
    3: ["MCI", "AD"],
}

DATA_SPLITS = [0.8,0.1,0.1]