import torch
import torch.nn as nn
import torch.optim as optim
from config import DATA_DIR, RESULTS_DIR, FIGURS_DIR, METADATA_DIR
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from model import TabularClassifier
from adni_data import ADNITabularDataset
from torch.utils.data import DataLoader
import argparse
from config import NUMERIC_PREDICTORS, CATEGORICAL_PREDICTORS,EXPERIMENTS
from util import preprocess_adni, compute_class_weights_effective_num, create_or_load_splits


# ==============================================================================
# 0. DEVICE
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
df_processed = pd.read_csv(Path(RESULTS_DIR, "ADNI_Combined_Multimodal_FIXED.csv"))

CLASSES = df_processed["Group"].unique().tolist()


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
data    = preprocess_adni(train_df, val_df, test_df, keep_paths=True)
CLASSES = data["label_encoder"].classes_

train_dataset = ADNITabularDataset(train_df, data["X_train"], CLASSES)
val_dataset   = ADNITabularDataset(val_df,   data["X_val"],   CLASSES)
test_dataset  = ADNITabularDataset(test_df,  data["X_test"],  CLASSES)

batch_size = 32

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True,
)

# Sanity check
X_sample, y_sample = next(iter(train_loader))
print(f"\nSample batch — X: {X_sample.shape} | y: {y_sample.shape}")
print(f"Unique labels in batch: {y_sample.unique().tolist()}")

# CLASSES = df_processed['Group'].unique().tolist()
# ==============================================================================
# 4. MODEL, LOSS, OPTIMIZER, SCHEDULER
# ==============================================================================
model = TabularClassifier(
    n_features=X_sample.shape[1],
    num_classes=len(CLASSES),
    dropout=0.6,
).to(DEVICE)
print(f"\n{model}\n")

num_epochs    = 50
learning_rate = 3e-4


# Map weights to the same class order used by the model
weight_dict = compute_class_weights_effective_num(train_df['Group'].values, beta=0.99)
# weights = torch.tensor(
#     [weight_dict[cls] for cls in CLASSES], dtype=torch.float
# ).to(DEVICE)

weights = torch.tensor(
    [weight_dict[cls] for cls in CLASSES], 
    dtype=torch.float
).to(DEVICE)

# Asymmetric loss — penalise AD misses more heavily than other errors:
# weights[CLASSES.tolist().index("AD")] *= 1.5  # extra penalty for AD

# Inverse-frequency class weights from the dataset helper
criterion = nn.CrossEntropyLoss(
    weight=weights, 
    label_smoothing=0.1,  # Optional: can help with generalization
)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
optimizer =  optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


# ==============================================================================
# 5. TRAINING LOOP
# ==============================================================================
best_val_acc  = 0.0
best_val_loss = float("inf")


# Delete stale checkpoint before training so you never evaluate a ghost model
if suffix != "":
    suffix = f"_{suffix}"
checkpoint_path = Path(RESULTS_DIR, f"best_tabular_model{suffix}.pth")
if checkpoint_path.exists():
    checkpoint_path.unlink()
    print("Removed stale checkpoint.")

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(num_epochs):

    # ── Train ─────────────────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = torch.max(logits, 1)
        total        += y_batch.size(0)
        correct      += (predicted == y_batch).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc  = 100 * correct / total

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    val_loss    = 0.0
    val_correct = 0
    val_total   = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits           = model(X_batch)
            loss             = criterion(logits, y_batch)
            val_loss        += loss.item()
            _, predicted     = torch.max(logits, 1)
            val_total       += y_batch.size(0)
            val_correct     += (predicted == y_batch).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc  = 100 * val_correct / val_total

    scheduler.step()

    # Track history
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # Save best checkpoint (by val loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc  = val_acc
        torch.save(model.state_dict(), Path(RESULTS_DIR, f"best_tabular_model{suffix}.pth"))

    print(
        f"Epoch [{epoch+1:>3}/{num_epochs}] | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
    )

print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.2f}%")


# ==============================================================================
# 6. TEST EVALUATION WITH BOOTSTRAP CONFIDENCE INTERVALS
# ==============================================================================

# Load best checkpoint before evaluating
model.load_state_dict(torch.load(Path(RESULTS_DIR, f"best_tabular_model{suffix}.pth"), map_location=DEVICE))
model.eval()

all_labels = []
all_preds  = []
all_probs  = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch  = X_batch.to(DEVICE)
        logits   = model(X_batch)
        probs    = torch.softmax(logits, dim=1)
        _, preds = torch.max(logits, 1)

        all_labels.extend(y_batch.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels     = np.array(all_labels)
all_preds      = np.array(all_preds)
all_probs      = np.array(all_probs)

# One-hot encode labels for multi-class AUROC (OvR)
all_labels_bin = label_binarize(all_labels, classes=list(range(len(CLASSES))))


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    labels_bin: np.ndarray,
) -> tuple[float, float, float]:
    """Return (accuracy, micro-F1, AUROC) for a given sample."""
    acc      = accuracy_score(labels, preds)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    try:
        n_classes = probs.shape[1]
        if n_classes == 2:
            # Standard binary AUROC — use prob of the positive class
            auroc = roc_auc_score(labels, probs[:, 1])
        else:
            auroc = roc_auc_score(labels_bin, probs, multi_class="ovr", average="micro")
    except ValueError:
        auroc = float("nan")   # e.g. only one class present in this bootstrap draw
    return acc, micro_f1, auroc

# def compute_metrics(
#     labels: np.ndarray,
#     preds: np.ndarray,
#     probs: np.ndarray,
#     labels_bin: np.ndarray,
# ) -> tuple[float, float, float]:
#     """Return (accuracy, micro-F1, micro-AUROC) for a given sample."""
#     acc      = accuracy_score(labels, preds)
#     micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
#     try:
#         auroc = roc_auc_score(labels_bin, probs, multi_class="ovr", average="micro")
#     except ValueError:
#         auroc = float("nan")   # Only one class present in this bootstrap draw
#     return acc, micro_f1, auroc


def bootstrap_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    labels_bin: np.ndarray,
    n_bootstrap: int = 1000,
    ci: int = 95,
    seed: int = 42,
) -> dict:
    """
    Percentile bootstrap confidence intervals for Accuracy, Micro F1, AUROC.

    Point estimates are computed on the full test set externally; this
    function returns only the CI bounds.
    """
    rng = np.random.default_rng(seed)
    n   = len(labels)
    accs, f1s, aurocs = [], [], []

    for _ in range(n_bootstrap):
        idx          = rng.integers(0, n, size=n)
        acc, f1, auroc = compute_metrics(
            labels[idx], preds[idx], probs[idx], labels_bin[idx]
        )
        accs.append(acc)
        f1s.append(f1)
        aurocs.append(auroc)

    alpha   = (100 - ci) / 2
    results = {}
    for name, values in [("Accuracy", accs), ("Micro F1", f1s), ("AUROC", aurocs)]:
        arr = np.array(values)
        results[name] = {
            "lower": float(np.nanpercentile(arr, alpha)),
            "upper": float(np.nanpercentile(arr, 100 - alpha)),
        }
    return results


# ── Point estimates (full test set) ──────────────────────────────────────────
point_acc, point_f1, point_auroc = compute_metrics(
    all_labels, all_preds, all_probs, all_labels_bin
)

# ── Bootstrap CIs ─────────────────────────────────────────────────────────────
ci_results = bootstrap_ci(
    all_labels, all_preds, all_probs, all_labels_bin,
    n_bootstrap=1000, ci=95,
)

# ── Print report ──────────────────────────────────────────────────────────────
print("\n========== Test Set Results (95% Bootstrap CI) ==========")
print(f"{'Metric':<12} {'Point Est.':>12} {'95% CI':>25}")
print("-" * 52)
for metric, point in [("Accuracy", point_acc), ("Micro F1", point_f1), ("AUROC", point_auroc)]:
    lo = ci_results[metric]["lower"]
    hi = ci_results[metric]["upper"]
    print(f"{metric:<12} {point:>12.4f} {'[' + f'{lo:.4f}, {hi:.4f}' + ']':>25}")
print("=" * 52)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0))


# ==============================================================================
# 7. SAVE RESULTS & PLOTS
# ==============================================================================

# ── Metrics CSV ───────────────────────────────────────────────────────────────
results_rows = []
for metric, point in [("Accuracy", point_acc), ("Micro F1", point_f1), ("AUROC", point_auroc)]:
    results_rows.append({
        "Model":       "TabularClassifier",
        "Metric":      metric,
        "Point_Est":   round(point, 4),
        "CI_Lower":    round(ci_results[metric]["lower"], 4),
        "CI_Upper":    round(ci_results[metric]["upper"], 4),
        "CI_Level":    "95%",
        "N_Bootstrap": 1000,
    })
pd.DataFrame(results_rows).to_csv(
    Path(RESULTS_DIR, f"tabular_test_results{suffix}.csv"), index=False
)
print(f"\nMetrics saved → {Path(RESULTS_DIR, f'tabular_test_results{suffix}.csv')}")

# ── Training curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"],   label="Val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].legend()

axes[1].plot(history["train_acc"], label="Train")
axes[1].plot(history["val_acc"],   label="Val")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()

plt.tight_layout()
plt.savefig(Path(FIGURS_DIR, f"tabular_training_curves{suffix}.pdf"), dpi=300)
plt.savefig(Path(FIGURS_DIR, f"tabular_training_curves{suffix}.png"), dpi=300)

plt.close()
print(f"Training curves saved → {Path(FIGURS_DIR, f'tabular_training_curves{suffix}.pdf')}")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix — TabularClassifier (Test Set)")
plt.tight_layout()
plt.savefig(Path(FIGURS_DIR, "tabular_confusion_matrix.pdf"), dpi=300)
plt.savefig(Path(FIGURS_DIR, "tabular_confusion_matrix.png"), dpi=300)

plt.close()
print(f"Confusion matrix saved → {Path(FIGURS_DIR, f'tabular_confusion_matrix{suffix}.pdf')}")

# ── Model checkpoint ──────────────────────────────────────────────────────────
torch.save(model.state_dict(), Path("./savedmodels", f"tabular_classifier_final{suffix}.pth"))
print(f"Model saved → {Path("./savedmodels", f'tabular_classifier_final{suffix}.pth')}")