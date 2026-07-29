from sklearn.model_selection import train_test_split
from config import DATA_DIR, RESULTS_DIR, FIGURS_DIR, METADATA_DIR,EXPERIMENTS
import pandas as pd
import numpy as np
from util import compute_class_weights_effective_num, create_or_load_splits
from pathlib import Path
import SimpleITK as sitk
from monai.networks.nets import DenseNet121, resnet10, resnet18, SEResNet50
from torch.utils.data import DataLoader
from model import Small3DCNN,ImagingCNNClassifier
import torch.optim as optim
from adni_data import ADNIDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize
import torch.nn as nn
import torch
from tqdm import tqdm
import os
import monai.transforms as mt
import argparse



# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
df_processed = pd.read_csv(Path(RESULTS_DIR, "ADNI_Combined_Multimodal_FIXED.csv"))

# CLASSES = df_processed['Group'].unique().tolist()
# CLASSES = df_processed["label_encoder"].classes_


# ==============================================================================
# 2. TRAIN / VAL / TEST SPLIT  (grouped by subject to prevent data leakage)
# ==============================================================================
# ==============================================================================
# CONFIGURATION
# ==============================================================================
# None -> use all classes
# ["CN", "AD"] -> binary experiment
# ["CN", "MCI"] -> binary experiment
# ["MCI", "AD"] -> binary experiment

parser = argparse.ArgumentParser(description="Train ADNI Tabular Classifier")

parser.add_argument(
    "--experiment",
    "-e",
    type=int,
    default=0,
    choices=EXPERIMENTS.keys(),
    help="""
0 = CN/MCI/AD (default)
1 = CN vs AD
2 = CN vs MCI
3 = MCI vs AD
""",
)


args = parser.parse_args()

selected_classes = EXPERIMENTS[args.experiment]

print(f"Running experiment {args.experiment}")
print(f"Selected classes: {selected_classes or 'All classes'}")

# Three-class experiment (default)
train_df, val_df, test_df, suffix = create_or_load_splits(
    df_processed,
    RESULTS_DIR,
    selected_classes=selected_classes,
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


# ==============================================================================
# 3. DATASETS & DATALOADERS
# ==============================================================================
CLASSES = train_df['Group'].unique().tolist()



# --- TRAINING TRANSFORMS (Purely Geometric & Safe for Z-Score) ---
train_transform = mt.Compose([
    # Combats orientation and viewpoint memorization
    mt.RandFlip(prob=0.5, spatial_axis=0),             # Left/Right flip
    mt.RandFlip(prob=0.5, spatial_axis=1),             # Anterior/Posterior flip
    
    # Combats positioning and size memorization
    mt.RandAffine(
        prob=0.5, 
        rotate_range=(0.26, 0.26, 0.26),               # Up to ~15 degrees
        scale_range=(0.1, 0.1, 0.1),                   # Scale change +/- 10%
        mode="bilinear"
    ),
    
    # Combats brain shape memorization via local warping
    # mt.Rand3DElastic(
    #     prob=0.2, 
    #     sigma_range=(5, 7), 
    #     magnitude_range=(50, 150), 
    #     mode="bilinear"
    # )
])


# --- VALIDATION TRANSFORMS ---
val_transform = None

train_ds = ADNIDataset(train_df, CLASSES=CLASSES,transform=train_transform)
val_ds   = ADNIDataset(val_df,   CLASSES=CLASSES,transform=val_transform)
test_ds  = ADNIDataset(test_df,  CLASSES=CLASSES, transform=val_transform)


n_bootstrap = 1000
# train_df = train_df.sample(n=DEBUG_SAMPLES, random_state=42)


batch_size = 16

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)


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
# growth_rate: int = 32
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=len(CLASSES),dropout_prob=0.2).to(device)
# model = resnet10(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=len(CLASSES)).to(device)


# ==============================================================================
# 5. LOSS, OPTIMIZER, SCHEDULER
# ==============================================================================
num_epochs    = 20
learning_rate = 1e-4

# Map weights to the same class order used by the model
weight_dict = compute_class_weights_effective_num(train_df['Group'].values, beta=0.99)
weights = torch.tensor(
    [weight_dict[cls] for cls in CLASSES], dtype=torch.float
).to(device)

# weights[CLASSES.index("AD")] *= 1.5  # extra penalty for AD

criterion = nn.CrossEntropyLoss(weight=weights)

# optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
optimizer =  optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


# ==============================================================================
# 6. CHECKPOINT / RESUME LOGIC
# ==============================================================================

if suffix != "":
    suffix = f"_{suffix}"

CHECKPOINT_DIR = Path("./savedmodels")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = CHECKPOINT_DIR / f"checkpoint{suffix}.pth"

start_epoch = 0  # <-- set this to the epoch you want to resume from

if CHECKPOINT_PATH.exists():
    print(f"Checkpoint found at {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    saved_epoch = checkpoint["epoch"]
    if saved_epoch + 1 != start_epoch:
        print(
            f"Warning: checkpoint was saved after epoch {saved_epoch} "
            f"(would naturally resume at epoch {saved_epoch + 1}), "
            f"but start_epoch={start_epoch} was requested. "
            f"Proceeding with start_epoch={start_epoch}."
        )
    print(f"Resuming training from epoch {start_epoch}/{num_epochs}...")
else:
    print("No checkpoint found — starting training from scratch.")
    start_epoch = 0

if start_epoch >= num_epochs:
    print(
        f"start_epoch ({start_epoch}) >= num_epochs ({num_epochs}); "
        f"nothing to train. Increase num_epochs if you want more training."
    )


# ==============================================================================
# 7. TRAINING LOOP
# ==============================================================================
for epoch in range(start_epoch, num_epochs):

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

    # --- Save checkpoint after every epoch ---
    torch.save(
        {
            "epoch": epoch,                      # last completed epoch (0-indexed)
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_acc": train_acc,
            "val_acc": val_acc,
        },
        CHECKPOINT_PATH,
    )

print("Training Finished!")


# ==============================================================================
# 8. TEST EVALUATION WITH BOOTSTRAP CONFIDENCE INTERVALS
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
    """Return (accuracy, micro-F1, AUROC) for a given sample."""
    acc      = accuracy_score(labels, preds)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    try:
        n_classes = probs.shape[1]
        if n_classes == 2:
            auroc = roc_auc_score(labels, probs[:, 1])
        else:
            auroc = roc_auc_score(labels_bin, probs, multi_class="ovr", average="micro")
    except ValueError:
        # Edge case: bootstrap sample contains only one class
        auroc = float("nan")
    return acc, micro_f1, auroc


def bootstrap_ci(labels, preds, probs, labels_bin, n_bootstrap=n_bootstrap, ci=95, seed=42):
    """
    Percentile bootstrap confidence intervals for Accuracy, Micro F1, AUROC.

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
    for name, values in [("Accuracy", accs), ("Micro F1", f1s), ("AUROC", aurocs)]:
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
for metric, point in [("Accuracy", point_acc), ("Micro F1", point_f1), ("AUROC", point_auroc)]:
    lo = ci_results[metric]["lower"]
    hi = ci_results[metric]["upper"]
    print(f"{metric:<12} {point:>12.4f} {'[' + f'{lo:.4f}, {hi:.4f}' + ']':>25}")
print("=" * 52)

# --- Save results ---
results_rows = []
for metric, point in [("Accuracy", point_acc), ("Micro F1", point_f1), ("AUROC", point_auroc)]:
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
test_results_df.to_csv(Path(RESULTS_DIR, f"vision_test_results{suffix}.csv"), index=False)
print(f"\nResults saved → {Path(RESULTS_DIR, f'vision_test_results{suffix}.csv')}")


# ==============================================================================
# 9. SAVE FINAL MODEL
# ==============================================================================
torch.save(model.state_dict(), Path("./savedmodels", f"small3dcnn{suffix}.pth"))
print(f"Model saved → ./savedmodels/small3dcnn{suffix}.pth")