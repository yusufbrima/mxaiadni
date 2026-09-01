from pathlib import Path
from typing import List
# ── Column definitions ────────────────────────────────────────────────────────

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

BASE_DIR = "/path/to/your/base/directory"  # Replace with your actual base directory path
PROJECT_DIR = "/path/to/your/project/directory"  # Replace with your actual project directory path

DATA_DIR = Path(f"{BASE_DIR}")
RESULTS_DIR = Path(f"{PROJECT_DIR}/Scripts/results")
FIGURS_DIR = Path(f"{PROJECT_DIR}/Scripts/figures")
METADATA_DIR = Path(f"{PROJECT_DIR}/Scripts/metadata")

EXPERIMENTS = {
    0: None,                  # Multiclass (default)
    1: ["CN", "AD"],
    2: ["CN", "MCI"],
    3: ["MCI", "AD"],
}

DATA_SPLITS = [0.8,0.1,0.1]