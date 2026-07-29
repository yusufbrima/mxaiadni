from sklearn.model_selection import train_test_split
from config import DATA_DIR, RESULTS_DIR, FIGURS_DIR, METADATA_DIR
import pandas as pd
import numpy as np
from util import n4_correction,compute_class_weights_effective_num
from pathlib import Path
import SimpleITK as sitk
from monai.networks.nets import DenseNet121, resnet10, resnet18, SEResNet50
from torch.utils.data import DataLoader
from model import Small3DCNN,ImagingCNNClassifier
import torch.optim as optim
from adni_data import ADNIDataset,ADNIDatasetLite
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize
import torch.nn as nn
import torch
from tqdm import tqdm



# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
df_processed = pd.read_csv(Path(RESULTS_DIR, "ADNI_Combined_Multimodal_FIXED.csv"))

CLASSES = df_processed['Group'].unique().tolist()
# CLASSES = df_processed["label_encoder"].classes_


# ==============================================================================
# 2. TRAIN / VAL / TEST SPLIT  (grouped by subject to prevent data leakage)
# ==============================================================================
if Path(RESULTS_DIR, "train_split.csv").exists():
    print("Loading existing train/val/test splits...")
    train_df = pd.read_csv(Path(RESULTS_DIR, "train_split.csv"))
    val_df   = pd.read_csv(Path(RESULTS_DIR, "val_split.csv"))
    test_df  = pd.read_csv(Path(RESULTS_DIR, "test_split.csv"))
else:
    random_seed = 42

    # 1. Decide a single representative diagnosis per patient.
    #    Common choices: baseline diagnosis, most frequent, or "worst" (max severity) diagnosis.
    #    Baseline is usually most defensible for train/test splitting.
    patient_dx = (
        df_processed.sort_values("EXAMDATE")   # or VISCODE, whatever orders visits
        .groupby("PTID")["DX"]
        .first()                                # baseline diagnosis
        .reset_index()
    )

    # 2. Stratified split at the PATIENT level
    train_ptid, test_ptid = train_test_split(
        patient_dx, test_size=0.15, stratify=patient_dx["DX"], random_state=42
    )
    train_ptid, val_ptid = train_test_split(
        train_ptid, test_size=0.15, stratify=train_ptid["DX"], random_state=42
    )

    # 3. Expand back to full scan-level dataframe using PTID membership
    train_df = df_processed[df_processed["PTID"].isin(train_ptid["PTID"])]
    val_df   = df_processed[df_processed["PTID"].isin(val_ptid["PTID"])]
    test_df  = df_processed[df_processed["PTID"].isin(test_ptid["PTID"])]

    assert set(train_df["PTID"]) & set(val_df["PTID"]) == set()
    assert set(train_df["PTID"]) & set(test_df["PTID"]) == set()
    assert set(val_df["PTID"]) & set(test_df["PTID"]) == set()

    train_df.to_csv(Path(RESULTS_DIR, "train_split.csv"), index=False)
    val_df.to_csv(Path(RESULTS_DIR,   "val_split.csv"),   index=False)
    test_df.to_csv(Path(RESULTS_DIR,  "test_split.csv"),  index=False)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


# ==============================================================================
# 3. DATASETS & DATALOADERS
# ==============================================================================
train_ds = ADNIDataset(train_df, CLASSES=CLASSES)
val_ds   = ADNIDataset(val_df,   CLASSES=CLASSES)
test_ds  = ADNIDataset(test_df,  CLASSES=CLASSES)

DEBUG_SAMPLES = 200  # start with 100–200
n_bootstrap = 1
train_df = train_df.sample(n=DEBUG_SAMPLES, random_state=42)
# val_df   = val_df.sample(n=int(DEBUG_SAMPLES * 0.2), random_state=42)
# test_df  = test_df.sample(n=int(DEBUG_SAMPLES * 0.2), random_state=42)


# train_ds = ADNIDatasetLite(train_df, CLASSES=CLASSES)
# val_ds   = ADNIDatasetLite(val_df,   CLASSES=CLASSES)
# test_ds  = ADNIDatasetLite(test_df,  CLASSES=CLASSES)

batch_size = 8

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,           # Essential: prevents learning sample order
    num_workers=2,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# Sanity check
images, labels = next(iter(train_loader))
print(f"Batch Image Shape: {images.shape}")   # Expect: [B, 1, D, H, W]
print(f"Batch Label Shape: {labels.shape}")   # Expect: [B]
print(f"Sample Labels:     {labels}")


# ==============================================================================
# 4. MODEL
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model  = Small3DCNN(num_classes=len(CLASSES),drop3d=0.2, drop_fc=0.4).to(device)
# model  = ImagingCNNClassifier(num_classes=len(CLASSES)).to(device)
# Alternatives:
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=len(CLASSES),dropout_prob=0.1).to(device)
# model = resnet10(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=len(CLASSES)).to(device)


# ==============================================================================
# 5. LOSS, OPTIMIZER, SCHEDULER
# ==============================================================================
num_epochs    = 10
learning_rate = 1e-4

# Inverse-frequency class weighting to handle class imbalance
# class_counts = train_df['Group'].value_counts()
# weights = torch.tensor(
#     [1.0 / class_counts[cls] for cls in CLASSES], dtype=torch.float
# ).to(device)

# Map weights to the same class order used by the model
weight_dict = compute_class_weights_effective_num(train_df['Group'].values, beta=0.99)
weights = torch.tensor(
    [weight_dict[cls] for cls in CLASSES], dtype=torch.float
).to(device)

# weights[CLASSES.index("AD")] *= 1.5  # extra penalty for AD

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


# ==============================================================================
# 6. TRAINING LOOP
# ==============================================================================
for epoch in range(num_epochs):

    # --- Train ---
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train", leave=False)
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = torch.max(outputs.data, 1)
        total        += labels.size(0)
        correct      += (predicted == labels).sum().item()

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100 * correct / total:.2f}%"
        )

    train_acc = 100 * correct / total

    # --- Validate ---
    model.eval()
    val_correct = 0
    val_total   = 0

    val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Val", leave=False)
    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            _, predicted   = torch.max(outputs.data, 1)
            val_total     += labels.size(0)
            val_correct   += (predicted == labels).sum().item()

            val_bar.set_postfix(acc=f"{100 * val_correct / val_total:.2f}%")

    val_acc = 100 * val_correct / val_total

    scheduler.step()

    print(
        f"Epoch [{epoch+1}/{num_epochs}] | "
        f"Loss: {running_loss/len(train_loader):.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}%"
    )

print("Training Finished!")


# ==============================================================================
# 7. TEST EVALUATION WITH BOOTSTRAP CONFIDENCE INTERVALS
# ==============================================================================

# --- Collect predictions & probabilities ---
model.eval()
all_labels = []
all_preds  = []
all_probs  = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
        images, labels = images.to(device), labels.to(device)
        outputs        = model(images)
        probs          = torch.softmax(outputs, dim=1)
        _, predicted   = torch.max(outputs.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels     = np.array(all_labels)
all_preds      = np.array(all_preds)
all_probs      = np.array(all_probs)

# One-hot encode labels for multi-class AUROC (OvR)
all_labels_bin = label_binarize(all_labels, classes=list(range(len(CLASSES))))


def compute_metrics(labels, preds, probs, labels_bin):
    """Return (accuracy, macro-F1, macro-AUROC) for a given sample."""
    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        auroc = roc_auc_score(labels_bin, probs, multi_class="ovr", average="macro")
    except ValueError:
        # Edge case: bootstrap sample contains only one class
        auroc = float("nan")
    return acc, macro_f1, auroc


def bootstrap_ci(labels, preds, probs, labels_bin, n_bootstrap=n_bootstrap, ci=95, seed=42):
    """
    Percentile bootstrap confidence intervals for Accuracy, Macro F1, AUROC.

    Returns a dict: { metric_name: {"point": ..., "lower": ..., "upper": ...} }
    Note: point estimate is computed on the full test set (not the bootstrap mean).
    """
    rng = np.random.default_rng(seed)
    n   = len(labels)
    accs, f1s, aurocs = [], [], []

    for _ in range(n_bootstrap):
        idx          = rng.integers(0, n, size=n)
        b_labels     = labels[idx]
        b_preds      = preds[idx]
        b_probs      = probs[idx]
        b_labels_bin = labels_bin[idx]

        acc, f1, auroc = compute_metrics(b_labels, b_preds, b_probs, b_labels_bin)
        accs.append(acc)
        f1s.append(f1)
        aurocs.append(auroc)

    alpha   = (100 - ci) / 2
    results = {}
    for name, values in [("Accuracy", accs), ("Macro F1", f1s), ("AUROC", aurocs)]:
        arr = np.array(values)
        results[name] = {
            "lower": np.nanpercentile(arr, alpha),
            "upper": np.nanpercentile(arr, 100 - alpha),
        }
    return results


# --- Point estimates (full test set) ---
point_acc, point_f1, point_auroc = compute_metrics(
    all_labels, all_preds, all_probs, all_labels_bin
)

# --- Bootstrap CIs ---
ci_results = bootstrap_ci(
    all_labels, all_preds, all_probs, all_labels_bin,
    n_bootstrap=1000, ci=95,
)

# --- Print report ---
print("\n========== Test Set Results (95% Bootstrap CI) ==========")
print(f"{'Metric':<12} {'Point Est.':>12} {'95% CI':>25}")
print("-" * 52)
for metric, point in [("Accuracy", point_acc), ("Macro F1", point_f1), ("AUROC", point_auroc)]:
    lo = ci_results[metric]["lower"]
    hi = ci_results[metric]["upper"]
    print(f"{metric:<12} {point:>12.4f} {'[' + f'{lo:.4f}, {hi:.4f}' + ']':>25}")
print("=" * 52)

# --- Save results ---
results_rows = []
for metric, point in [("Accuracy", point_acc), ("Macro F1", point_f1), ("AUROC", point_auroc)]:
    results_rows.append({
        "Model":       "Small3DCNN",
        "Metric":      metric,
        "Point_Est":   round(point, 4),
        "CI_Lower":    round(ci_results[metric]["lower"], 4),
        "CI_Upper":    round(ci_results[metric]["upper"], 4),
        "CI_Level":    "95%",
        "N_Bootstrap": 1000,
    })

test_results_df = pd.DataFrame(results_rows)
test_results_df.to_csv(Path(RESULTS_DIR, "vision_test_results.csv"), index=False)
print(f"\nResults saved → {Path(RESULTS_DIR, 'vision_test_results.csv')}")


# ==============================================================================
# 8. SAVE MODEL
# ==============================================================================
torch.save(model.state_dict(), Path("./savedmodels", "small3dcnn.pth"))
print("Model saved → ./savedmodels/small3dcnn.pth")