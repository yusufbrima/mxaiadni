from config import DATA_DIR, RESULTS_DIR, METADATA_DIR
import pandas as pd
import numpy as np
from util import n4_correction
from adni_preprocess import n4_bias_field_correction, skull_strip, registration, normalize_zscore, resize_image
from pathlib import Path
import SimpleITK as sitk

# Setup Directories
# PROCESSED_DIR = Path(DATA_DIR) / "ADNI_PROCESSED"
PROCESSED_DIR = Path(DATA_DIR) / "OASIS3_PROCESSED"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

additional = ""

# csv_path = RESULTS_DIR / f"adni_file_mapping{additional}.csv"
csv_path = RESULTS_DIR / f"oasis3_file_mapping{additional}.csv"
df = pd.read_csv(csv_path)

# MAPPING_OUT = RESULTS_DIR / f"processed_files_mapping{additional}.csv"
MAPPING_OUT = RESULTS_DIR / f"processed_oasis3_file_mapping{additional}.csv"
ref = Path(DATA_DIR, "mni152.nii.gz")
ref2 = Path(DATA_DIR, "mni.nii.gz")


def is_valid_output(path: Path) -> bool:
    """Check that a previously written output file exists, is non-empty,
    and can actually be read by SimpleITK (guards against truncated/corrupt
    files left behind by an interrupted run)."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        sitk.ReadImage(str(path))
        return True
    except Exception:
        return False


# --- RESUME: load already-processed entries ---
if MAPPING_OUT.exists():
    df_done = pd.read_csv(MAPPING_OUT)
    done_files = set(df_done["processed_path"].apply(lambda p: Path(p).name))
    processed_data = df_done.to_dict("records")
    print(f"Resuming: {len(done_files)} already processed, {len(df) - len(done_files)} remaining.")
else:
    done_files = set()
    processed_data = []
    print(f"Starting fresh pipeline for {len(df)} files...")

for index, row in df.iterrows():
    subject_id = str(row["subject_id"])
    input_path = Path(row["file_path"])
    output_path = PROCESSED_DIR / input_path.name

    # Skip if already logged in mapping CSV
    if input_path.name in done_files:
        print(f"[{index+1}/{len(df)}] Skipping (already logged): {input_path.name}")
        continue

    # Skip if a valid output file already exists on disk (e.g. mapping got out of sync)
    if output_path.exists():
        if is_valid_output(output_path):
            print(f"[{index+1}/{len(df)}] Skipping (valid output on disk): {input_path.name}")
            # Backfill the mapping so it reflects reality
            processed_data.append({
                "subject_id": subject_id,
                "MRI_Day" : Path(row["MRI_Day"]),
                "processed_path": str(output_path)
            })
            done_files.add(input_path.name)
            pd.DataFrame(processed_data).to_csv(MAPPING_OUT, index=False)
            continue
        else:
            print(f"[{index+1}/{len(df)}] Found invalid/corrupt output, reprocessing: {input_path.name}")

    try:
        # A. Load Image
        img = sitk.ReadImage(str(input_path), sitk.sitkFloat32)
        # B. Bias Correction
        corrected_img, head_mask = n4_bias_field_correction(str(input_path))
        # C. Skull Stripping
        skull_img = skull_strip(corrected_img)
        # D. Registration
        reg_img = registration(skull_img, str(ref))
        # E. Intensity Normalization
        norm_img = normalize_zscore(reg_img)
        # F. Reshape/Downsample
        res_img = resize_image(norm_img, (128, 128, 128))

        sitk.WriteImage(res_img, str(output_path))

        # Log success and save progress immediately
        processed_data.append({
            "subject_id": subject_id,
            "processed_path": str(output_path)
        })
        done_files.add(input_path.name)

        # Incremental save after every success
        pd.DataFrame(processed_data).to_csv(MAPPING_OUT, index=False)

        print(f"[{index+1}/{len(df)}] Success: {subject_id} | {input_path.name}")

    except Exception as e:
        print(f"[{index+1}/{len(df)}] FAILED: {subject_id} | {input_path.name} | Error: {e}")

print("\nProcessing Complete!")
print(f"Processed: {len(processed_data)}/{len(df)} files")
print(f"Processed files saved to: {PROCESSED_DIR}")