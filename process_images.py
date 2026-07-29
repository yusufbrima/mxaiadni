from config import DATA_DIR, RESULTS_DIR
import pandas as pd
from pathlib import Path
import util as util
import SimpleITK as sitk

from adni_preprocess import (
    n4_bias_field_correction,
    skull_strip,
    registration,
    normalize_zscore,
    resize_image
)


import tensorflow as tf
import tensorflow.compat.v1 as tf1

# Patch the tensorflow module so deepbrain sees v1 attributes
tf.Session = tf1.Session
tf.gfile = tf1.gfile
tf.GraphDef = tf1.GraphDef

tf1.disable_v2_behavior()

from deepbrain import Extractor

ext = Extractor()
# ----------------------------
# Setup
# ----------------------------
PROCESSED_DIR = Path(DATA_DIR) / "ADNI_PROCESSED"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

csv_path = RESULTS_DIR / "adni_file_mapping.csv"
output_csv = RESULTS_DIR / "processed_files_mapping.csv"

df = pd.read_csv(csv_path)

ref = Path(DATA_DIR, "mni152.nii.gz")

# ----------------------------
# Resume support
# ----------------------------
processed_data = []

if output_csv.exists():
    try:
        existing = pd.read_csv(output_csv)
        processed_data = existing.to_dict("records")
        print(f"Resuming from previous run: {len(processed_data)} files already logged.")
    except Exception:
        print("Warning: could not read existing CSV. Starting fresh.")

print(f"Starting pipeline for {len(df)} files...\n")

# ----------------------------
# Pipeline
# ----------------------------
for index, row in df.iterrows():

    subject_id = row['subject_id']
    input_path = Path(row['file_path'])
    output_path = PROCESSED_DIR / input_path.name

    # Skip if output file already exists in PROCESSED_DIR
    if output_path.exists():
        print(f"[{index+1}/{len(df)}] SKIPPING (already processed): {subject_id}")
        continue

    try:
        # ---------------- LOAD ----------------
        img = sitk.ReadImage(str(input_path), sitk.sitkFloat32)
        if img is None:
            raise ValueError("Failed to load image")

        # ---------------- PIPELINE ----------------
        corrected_img, _ = n4_bias_field_correction(str(input_path))
        if corrected_img is None:
            raise ValueError("N4 bias correction failed")

        skull_img = skull_strip(corrected_img, ext=ext)
        if skull_img is None:
            raise ValueError("Skull stripping failed")

        reg_img = registration(skull_img, str(ref))
        if reg_img is None:
            raise ValueError("Registration failed")

        norm_img = normalize_zscore(reg_img)
        if norm_img is None:
            raise ValueError("Normalization failed")

        res_img = resize_image(norm_img, (128, 128, 128))
        if res_img is None:
            raise ValueError("Resizing failed")

        if not isinstance(res_img, sitk.Image):
            raise TypeError(f"Invalid output type: {type(res_img)}")

        # ---------------- WRITE ----------------
        sitk.WriteImage(res_img, str(output_path))

        # ---------------- VERIFY WRITE ----------------
        if not output_path.exists():
            raise RuntimeError(f"File not written: {output_path}")

        # ---------------- SAFE CSV UPDATE ----------------
        new_entry = {
            "subject_id": subject_id,
            "file_path": str(input_path),
            "processed_path": str(output_path)
        }

        processed_data.append(new_entry)

        df_processed = pd.DataFrame(processed_data)
        df_processed.to_csv(output_csv, index=False)

        print(f"[{index+1}/{len(df)}] SUCCESS: {subject_id}")

    except Exception as e:
        print(f"[{index+1}/{len(df)}] FAILED: {subject_id} | {e}")

# ----------------------------
# DONE
# ----------------------------
print("\nProcessing Complete!")
print(f"Processed images saved to: {PROCESSED_DIR}")
print(f"Mapping CSV saved to: {output_csv}")
