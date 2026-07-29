import pandas as pd
from pathlib import Path
from config import DATA_DIR, RESULTS_DIR, FIGURS_DIR, METADATA_DIR
import os


NIFTI_DIR = Path(DATA_DIR) / "ADNI"

all_nifti_files = [
    Path(root) / f
    for root, _, files in os.walk(NIFTI_DIR)
    for f in files
    if f.endswith(".nii") or f.endswith(".nii.gz")
]

print(f"Total NIfTI files found: {len(all_nifti_files)}")

# 1. Initialize your list
data = []

# 2. Iterate through your globbed files
for file_path in all_nifti_files:
    # Get filename
    filename = file_path.name 
    
    # Extract subject_id using your split logic
    parts = filename.split('_')
    
    # Safety check: ensure the filename has enough parts before joining
    if len(parts) > 4:
        subject_id = "_".join(parts[1:4])
        
        # Append to our list
        data.append({
            "subject_id": subject_id,
            "file_path": str(file_path)
        })

# 3. Create the DataFrame
df = pd.DataFrame(data)

# Show the first few rows to verify
print(f"Total files indexed: {len(df)}")
df.head()


# 1. Ensure the directory exists
RESULTS_DIR = Path(RESULTS_DIR)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Define the path for the CSV
csv_path = RESULTS_DIR / "adni_file_mapping.csv"

# 3. Save the DataFrame
# index=False prevents pandas from adding an extra 'Unnamed: 0' column
df.to_csv(csv_path, index=False)

print(f"DataFrame saved successfully to: {csv_path}")
