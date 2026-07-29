from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.class_weight import compute_class_weight

from config import CATEGORICAL_PREDICTORS, NUMERIC_PREDICTORS, TARGET,DATA_SPLITS


# def install_if_missing(package_name, import_name=None):
#     """
#     package_name: name used in pip install
#     import_name: name used in import (if different)
#     """
#     if import_name is None:
#         import_name = package_name

#     try:
#         importlib.import_module(import_name)
#         print(f"{package_name} is already installed ✅")
#     except ImportError:
#         print(f"{package_name} not found. Installing...")
#         subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
#         print(f"{package_name} installed successfully ✅")


# # Install packages safely
# install_if_missing("pyreadstat") 
# install_if_missing("optuna") 
# install_if_missing("joblib") 
# install_if_missing("dcurves") 
# install_if_missing("python-dotenv") 
# install_if_missing("nibabel")
# install_if_missing("SimpleITK")
# install_if_missing("scipy")
# install_if_missing("deepbrain")
# install_if_missing("antspyx")
# install_if_missing("nilearn")
# install_if_missing("antspynet")
# install_if_missing("monai")
# install_if_missing("tqdm")
# install_if_missing("shap")
# install_if_missing("captum")
# install_if_missing("grad-cam")

import SimpleITK as sitk
import nibabel as nib
# from deepbrain import Extractor
def n4_correction(input_path):
    # why: sitkFloat32 is required for the math inside the N4 algorithm.
    image = sitk.ReadImage(input_path, sitk.sitkFloat32)
    
    # why: This filter calculates the "shading" and removes it.
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    
    # output: A corrected SimpleITK image object.
    corrected_img = corrector.Execute(image)
    
    # TO SAVE: sitk.WriteImage(corrected_img, "n4_output.nii")
    # TO PIPE: Return the object to the next function.
    return corrected_img


def preprocess_adni(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = TARGET,
    subject_col: str = "PTID",
    missing_threshold: float = 0.60,
    imputer_iters: int = 10,
    random_state: int = 42,
    numeric_predictors: Optional[list[str]] = None,
    categorical_predictors: Optional[list[str]] = None,
    keep_paths: bool = False,
    path_col: str = "processed_path",
) -> dict:
    """
    Preprocessing pipeline for the ADNI multimodal dataset.

    Accepts pre-split train / val / test DataFrames so that splitting
    strategy (e.g. GroupShuffleSplit, stratified, longitudinal) is fully
    controlled by the caller. All transformers are fit exclusively on
    train_df, then applied to val_df and test_df.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split (one row per visit).
    val_df : pd.DataFrame
        Validation split (one row per visit).
    test_df : pd.DataFrame
        Test split (one row per visit).
    target_col : str
        Diagnosis label column (default: "Group").
    subject_col : str
        Column used to verify no subject appears across splits (default: "PTID").
    missing_threshold : float
        Predictors missing more than this fraction in train_df are dropped
        before imputation (default: 0.60).
    imputer_iters : int
        Maximum MICE iterations for IterativeImputer (default: 10).
    random_state : int
        Seed for all stochastic steps (default: 42).
    numeric_predictors : list[str] | None
        Override the module-level NUMERIC_PREDICTORS list.
    categorical_predictors : list[str] | None
        Override the module-level CATEGORICAL_PREDICTORS list.
    keep_paths : bool
        If True, the returned dict includes 'train_paths', 'val_paths',
        and 'test_paths' — Series of processed_path values aligned row-by-row
        to the feature matrices. Pass directly to ADNIDataset. (default: False)
    path_col : str
        Name of the column containing NIfTI file paths (default: "processed_path").

    Returns
    -------
    dict with keys:
        X_train, X_val, X_test : np.ndarray   — scaled feature matrices
        y_train, y_val, y_test : np.ndarray   — integer-encoded labels
        label_encoder          : LabelEncoder  — maps int ↔ class name
        feature_names          : list[str]     — column names matching X_* columns
        fitted                 : dict          — fitted transformers for inference
        train_paths, val_paths, test_paths : pd.Series  — only if keep_paths=True
    """

    num_cols = (numeric_predictors or NUMERIC_PREDICTORS).copy()
    cat_cols = (categorical_predictors or CATEGORICAL_PREDICTORS).copy()

    # ── 0. Guard: no subject should appear across splits ──────────────────────
    train_subjects = set(train_df[subject_col])
    val_subjects   = set(val_df[subject_col])
    test_subjects  = set(test_df[subject_col])

    assert train_subjects & test_subjects == set(), \
        "Subject leakage: train ∩ test is non-empty"
    assert train_subjects & val_subjects == set(), \
        "Subject leakage: train ∩ val is non-empty"

    print(f"Splits — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── 1. Drop predictors with >threshold missingness (measured on train) ────
    missing_rate = train_df[num_cols + cat_cols].isnull().mean().sort_values(ascending=False)
    high_missing = missing_rate[missing_rate > missing_threshold].index.tolist()

    if high_missing:
        print(f"Dropping {len(high_missing)} column(s) exceeding "
              f"{missing_threshold:.0%} missingness: {high_missing}")

    num_cols = [c for c in num_cols if c not in high_missing]
    cat_cols = [c for c in cat_cols if c not in high_missing]

    # ── 2. Impute numerics — MICE via RandomForest (fit on train only) ────────
    num_imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50, random_state=random_state),
        max_iter=imputer_iters,
        random_state=random_state,
        skip_complete=True,
    )
    train_num = num_imputer.fit_transform(train_df[num_cols])
    val_num   = num_imputer.transform(val_df[num_cols])
    test_num  = num_imputer.transform(test_df[num_cols])

    # ── 3. Impute categoricals — mode (fit on train only) ────────────────────
    cat_imputer = SimpleImputer(strategy="most_frequent")
    train_cat = cat_imputer.fit_transform(train_df[cat_cols])
    val_cat   = cat_imputer.transform(val_df[cat_cols])
    test_cat  = cat_imputer.transform(test_df[cat_cols])

    # ── 4. Encode categoricals — OHE (fit on train only) ─────────────────────
    ohe = OneHotEncoder(
        handle_unknown="ignore",  # unseen SITE levels in test won't crash
        sparse_output=False,
        drop="if_binary",         # Sex → single column
    )
    train_cat_enc = ohe.fit_transform(train_cat)
    val_cat_enc   = ohe.transform(val_cat)
    test_cat_enc  = ohe.transform(test_cat)

    # ── 5. Scale numerics — RobustScaler (fit on train only) ─────────────────
    # RobustScaler uses median + IQR: resistant to extreme clinical outliers
    scaler = RobustScaler()
    train_num_sc = scaler.fit_transform(train_num)
    val_num_sc   = scaler.transform(val_num)
    test_num_sc  = scaler.transform(test_num)

    # ── 6. Assemble final feature matrices ────────────────────────────────────
    X_train = np.hstack([train_num_sc, train_cat_enc])
    X_val   = np.hstack([val_num_sc,   val_cat_enc])
    X_test  = np.hstack([test_num_sc,  test_cat_enc])

    # ── 7. Encode labels ──────────────────────────────────────────────────────
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[target_col].values)
    y_val   = le.transform(val_df[target_col].values)
    y_test  = le.transform(test_df[target_col].values)

    print(f"Classes: {le.classes_}")
    print(f"Label distribution — Train: {np.bincount(y_train)} | "
          f"Val: {np.bincount(y_val)} | Test: {np.bincount(y_test)}")

    # ── 8. Sanity checks ──────────────────────────────────────────────────────
    assert not np.isnan(X_train).any(), "NaNs remain in X_train after imputation"
    assert not np.isnan(X_val).any(),   "NaNs remain in X_val after imputation"
    assert not np.isnan(X_test).any(),  "NaNs remain in X_test after imputation"

    print(f"Feature matrix — Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    feature_names: list[str] = num_cols + ohe.get_feature_names_out(cat_cols).tolist()

    # ── 9. Assemble output ────────────────────────────────────────────────────
    output = {
        # Feature matrices
        "X_train": X_train,
        "X_val":   X_val,
        "X_test":  X_test,
        # Labels
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        # Metadata
        "label_encoder": le,
        "feature_names": feature_names,
        # Fitted transformers — save with joblib for inference on new patients
        "fitted": {
            "num_imputer": num_imputer,
            "cat_imputer": cat_imputer,
            "ohe":         ohe,
            "scaler":      scaler,
        },
    }

    # ── 10. Optionally carry paths through, row-aligned to feature matrices ───
    if keep_paths:
        assert path_col in train_df.columns, \
            f"'{path_col}' not found in dataframe columns"
        output["train_paths"] = train_df[path_col].reset_index(drop=True)
        output["val_paths"]   = val_df[path_col].reset_index(drop=True)
        output["test_paths"]  = test_df[path_col].reset_index(drop=True)

    return output

def compute_class_weights_effective_num(y, beta=0.9999):
    """
    Class-Balanced Loss Based on Effective Number of Samples (Cui et al. 2019)
    https://arxiv.org/abs/1901.05555
    Args:
        y (array-like): training labels
        beta (float): hyperparameter close to 1.0 (e.g., 0.9–0.9999).
                      Larger beta → smoother weights.
    Returns:
        torch.Tensor of weights aligned to CLASSES order
    """
    classes, counts = np.unique(y, return_counts=True)
    effective_num   = 1.0 - np.power(beta, counts)
    weights         = (1.0 - beta) / np.array(effective_num)
    weights         = weights / np.sum(weights) * len(classes)
    weight_dict     = dict(zip(classes, weights))
    return weight_dict


def create_or_load_splits(
    df,
    results_dir,
    target_col="Group",
    subject_col="PTID",
    date_col="EXAMDATE",
    selected_classes=None,
    test_size=DATA_SPLITS[1],
    val_size=DATA_SPLITS[2],
    random_seed=42,
    save=True,
):
    """
    Create or load patient-level train/validation/test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe.
    results_dir : str or Path
        Directory for saving/loading split CSVs.
    target_col : str
        Target label column.
    subject_col : str
        Subject identifier column.
    date_col : str
        Column used to determine baseline visit.
    selected_classes : list[str] or None
        Classes to keep (e.g. ["CN", "AD"]).
        None keeps all classes.
    test_size : float
        Fraction of patients used for testing.
    val_size : float
        Fraction of the remaining training patients used for validation.
    random_seed : int
        Random seed.
    save : bool
        Whether to save generated splits.

    Returns
    -------
    train_df, val_df, test_df
    """

    results_dir = Path(results_dir)

    # ------------------------------------------------------------------
    # Filter classes
    # ------------------------------------------------------------------
    if selected_classes is not None:
        df = df[df[target_col].isin(selected_classes)].copy()
        suffix = "_" + "_".join(selected_classes)
    else:
        suffix = ""

    train_file = results_dir / f"train_split{suffix}.csv"
    val_file   = results_dir / f"val_split{suffix}.csv"
    test_file  = results_dir / f"test_split{suffix}.csv"

    # ------------------------------------------------------------------
    # Load existing splits
    # ------------------------------------------------------------------
    if train_file.exists():
        print(f"Loading existing splits{suffix}...")

        train_df = pd.read_csv(train_file)
        val_df   = pd.read_csv(val_file)
        test_df  = pd.read_csv(test_file)

        return train_df, val_df, test_df,suffix

    # ------------------------------------------------------------------
    # Create patient-level split
    # ------------------------------------------------------------------
    patient_labels = (
        df.sort_values(date_col)
          .groupby(subject_col)[target_col]
          .first()
          .reset_index()
    )

    train_patients, test_patients = train_test_split(
        patient_labels,
        test_size=test_size,
        stratify=patient_labels[target_col],
        random_state=random_seed,
    )

    train_patients, val_patients = train_test_split(
        train_patients,
        test_size=val_size,
        stratify=train_patients[target_col],
        random_state=random_seed,
    )

    train_df = df[df[subject_col].isin(train_patients[subject_col])].copy()
    val_df   = df[df[subject_col].isin(val_patients[subject_col])].copy()
    test_df  = df[df[subject_col].isin(test_patients[subject_col])].copy()

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    assert set(train_df[subject_col]).isdisjoint(val_df[subject_col])
    assert set(train_df[subject_col]).isdisjoint(test_df[subject_col])
    assert set(val_df[subject_col]).isdisjoint(test_df[subject_col])

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if save:
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)

    print(f"\nCreated splits{suffix}:")
    print(f"Train: {len(train_df):,}")
    print(f"Val:   {len(val_df):,}")
    print(f"Test:  {len(test_df):,}")

    print("\nClass distributions")
    print(train_df[target_col].value_counts())
    print(val_df[target_col].value_counts())
    print(test_df[target_col].value_counts())

    return train_df, val_df, test_df,suffix