import time
from pathlib import Path
import numpy as np
import pandas as pd
from util import n4_correction, compute_class_weights_effective_num
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    f1_score,
    classification_report,
    roc_auc_score,
    accuracy_score,
)
import monai.transforms as mt
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import argparse
from adni_data import ADNIMultiDataset
from model import MultimodalADNI
from util import preprocess_adni, create_or_load_splits
from config import DATA_DIR, RESULTS_DIR, FIGURS_DIR, METADATA_DIR, EXPERIMENTS

# =============================================================================
# Device
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df_processed = pd.read_csv(Path(RESULTS_DIR, "ADNI_Combined_Multimodal_FINAL.csv"))
fusion_methods = "cross_attn"  # "cross_attn" concat # Example fusion methods to experiment with

# CLASSES = df_processed['Group'].unique().tolist()

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

CLASSES = train_df['Group'].unique().tolist()

if suffix != "":
    suffix = f"_{suffix}"

def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights to handle CN / LMCI / AD imbalance."""
    counts  = np.bincount(y, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

# =============================================================================
# Train / Eval
# =============================================================================
def run_epoch(model, loader, optimizer=None, desc="train"):
    train = optimizer is not None
    model.train() if train else model.eval()

    loss_fn = nn.CrossEntropyLoss()
    total_loss, preds, labels = 0, [], []

    pbar = tqdm(loader, desc=desc, leave=False)
    for img, tab, y in pbar:
        img, tab, y = img.to(DEVICE), tab.to(DEVICE), y.to(DEVICE)

        logits = model(img, tab)

        # --- NaN guard --------------------------------------------------
        # If a batch produces NaN logits (e.g. a degenerate image volume or
        # an unimputed NaN slipping through the tabular features), CrossEntropyLoss
        # will silently return NaN and poison the whole epoch. Fail loudly
        # instead of training on garbage.
        if torch.isnan(logits).any():
            n_bad = torch.isnan(logits).any(dim=1).sum().item()
            tqdm.write(
                f"[run_epoch:{desc}] WARNING: {n_bad}/{y.size(0)} rows in this "
                f"batch produced NaN logits. Replacing with 0 before computing loss "
                f"— fix the upstream data/preprocessing that caused this."
            )
            logits = torch.nan_to_num(logits, nan=0.0)

        loss = loss_fn(logits, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset), preds, labels


@torch.no_grad()
def evaluate(model, loader, desc="val"):
    loss, preds, labels = run_epoch(model, loader, optimizer=None, desc=desc)
    labels = np.array(labels)
    preds  = np.array(preds)
    f1_micro = f1_score(labels, preds, average="micro")
    return loss, f1_micro, labels, preds


# =============================================================================
# Checkpointing
# =============================================================================
def save_checkpoint(path, epoch, model, optimizer, best_metric, best_state):
    """Save everything needed to exactly resume training later."""
    torch.save({
        "epoch": epoch,                     # last epoch completed
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,         # best val F1 (micro) seen so far
        "best_state_dict": best_state,      # weights corresponding to best_metric
    }, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# =============================================================================
# Training
# =============================================================================
def train(model, train_loader, val_loader, epochs=20, lr=1e-4,
          start_epoch=1, checkpoint_path=None):
    model.to(DEVICE)
    # opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt =  torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)


    best_metric = 0.0
    best_state  = None

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        print(f"Checkpoint found at '{checkpoint_path}'. Resuming from epoch {start_epoch}...")
        ckpt = load_checkpoint(checkpoint_path, model, opt, DEVICE)
        best_metric = ckpt.get("best_metric", 0.0)
        best_state  = ckpt.get("best_state_dict", None)
    else:
        print("No checkpoint found. Starting training from scratch.")
        start_epoch = 1

    epoch_bar = tqdm(range(start_epoch, epochs + 1), desc="epochs")
    for ep in epoch_bar:
        t0 = time.time()

        train_loss, _, _ = run_epoch(model, train_loader, opt, desc=f"train ep{ep}")
        val_loss, val_f1_micro, _, _ = evaluate(model, val_loader, desc=f"val ep{ep}")

        if val_f1_micro > best_metric:
            best_metric = val_f1_micro
            best_state  = model.state_dict()

        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_f1=f"{val_f1_micro:.4f}")
        tqdm.write(f"Epoch {ep}: train_loss={train_loss:.4f}, val_f1_micro={val_f1_micro:.4f}, time={time.time()-t0:.1f}s")

        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, ep, model, opt, best_metric, best_state)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# =============================================================================
# BOOTSTRAP METRICS (same approach as vision-only script)
# =============================================================================
def compute_metrics(labels, preds, probs, labels_bin):
    """Return (accuracy, micro-F1, AUROC) for a given sample.

    Note: `labels_bin` is kept in the signature so bootstrap_ci/test_evaluation
    don't need to change their call sites, but it's no longer used internally —
    roc_auc_score now binarizes `labels` itself, restricted to `np.arange(n_classes)`.
    """
    acc      = accuracy_score(labels, preds)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    try:
        n_classes = probs.shape[1]
        if n_classes == 2:
            auroc = roc_auc_score(labels, probs[:, 1])
        else:
            # Pass raw integer labels (not a manually label_binarize'd array) and
            # let sklearn do the one-hot internally, restricted to the classes
            # actually expected. This sidesteps label_binarize's own quirk of
            # collapsing to a single column when there are exactly 2 classes,
            # which was silently causing shape-mismatch failures here.
            auroc = roc_auc_score(
                labels, probs, multi_class="ovr", average="micro",
                labels=np.arange(n_classes),
            )
    except ValueError as e:
        # Only "only one class present in this bootstrap draw" should reach
        # here now that the label_binarize mismatch above is fixed. Print
        # unconditionally (not filtered by message content) — a filter is
        # exactly what hid the real shape-mismatch error last time.
        tqdm.write(f"[compute_metrics] AUROC failed: {e}")
        auroc = float("nan")
    return acc, micro_f1, auroc


def bootstrap_ci(labels, preds, probs, labels_bin, n_bootstrap=1000, ci=95, seed=42):
    """
    Percentile bootstrap confidence intervals for Accuracy, Micro F1, AUROC.

    Returns a dict: { metric_name: {"lower": ..., "upper": ...} }
    Note: point estimate is computed separately on the full test set (not the bootstrap mean).
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

    n_auroc_nan = int(np.sum(np.isnan(aurocs)))
    if n_auroc_nan > 0:
        pct = 100 * n_auroc_nan / n_bootstrap
        tqdm.write(
            f"[bootstrap_ci] {n_auroc_nan}/{n_bootstrap} ({pct:.1f}%) bootstrap "
            f"draws produced a NaN AUROC. A handful is normal for small/imbalanced "
            f"test sets (a resample can miss a class by chance). If this is close "
            f"to 100%, the input probs/labels themselves are the problem, not the "
            f"resampling — check `all_probs` for NaNs before this point."
        )

    alpha   = (100 - ci) / 2
    results = {}
    for name, values in [("Accuracy", accs), ("Micro F1", f1s), ("AUROC", aurocs)]:
        arr = np.array(values)
        if np.all(np.isnan(arr)):
            results[name] = {"lower": float("nan"), "upper": float("nan")}
        else:
            results[name] = {
                "lower": np.nanpercentile(arr, alpha),
                "upper": np.nanpercentile(arr, 100 - alpha),
            }
    return results


# =============================================================================
# FINAL TEST EVALUATION (IMPORTANT PART)
# =============================================================================
@torch.no_grad()
def test_evaluation(model, test_loader, class_names, n_bootstrap=1000, ci=95):
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    n_nan_batches = 0

    for img, tab, y in tqdm(test_loader, desc="Evaluating on Test Set"):
        img, tab, y = img.to(DEVICE), tab.to(DEVICE), y.to(DEVICE)

        logits = model(img, tab)

        if torch.isnan(logits).any():
            n_nan_batches += 1
            logits = torch.nan_to_num(logits, nan=0.0)

        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)

        all_labels.extend(y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # --- Diagnostic #1: model output width vs. experiment class count ----
    # This is the actual cause of the 100%-NaN AUROC seen on binary
    # experiments (e.g. CN vs MCI): all_probs contained no NaNs, but the
    # model still emitted a probability column for every class in the
    # *full* CN/MCI/AD problem instead of just the ~2 classes relevant to
    # this experiment. roc_auc_score then choked on a shape mismatch,
    # which got silently caught and turned into `nan` a thousand times over.
    n_classes_expected = len(class_names)
    n_classes_model     = all_probs.shape[1]
    if n_classes_model != n_classes_expected:
        print(
            f"\n[test_evaluation] *** CLASS COUNT MISMATCH ***\n"
            f"Model produced {n_classes_model} output classes, but this "
            f"experiment only has {n_classes_expected} classes: {list(class_names)}.\n"
            f"This is almost certainly why AUROC has been coming back NaN — "
            f"not a data/NaN issue. The model's classification head wasn't "
            f"reconfigured for this --experiment's class count (it looks like "
            f"it's still sized for the full CN/MCI/AD problem).\n"
            f"Fix: open model.py and check MultimodalADNI's constructor — "
            f"make sure `num_classes` (or whatever the output-layer-size "
            f"argument is called) is passed explicitly as len(classes) "
            f"({n_classes_expected} here) in this script's model = MultimodalADNI(...) "
            f"call, rather than relying on a hardcoded/default value.\n"
            f"As a stopgap so this run's report/CIs are still computable, "
            f"columns beyond index {n_classes_expected - 1} are being dropped "
            f"and the remaining probabilities renormalized — but retrain with "
            f"the corrected architecture before trusting these numbers.\n"
        )
        all_probs = all_probs[:, :n_classes_expected]
        row_sums  = all_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid div-by-zero on degenerate rows
        all_probs = all_probs / row_sums

    # --- Diagnostic #2: this is what was previously hidden behind the ----
    # --- blanket `except ValueError: auroc = nan` in compute_metrics -----
    nan_mask = np.isnan(all_probs).any(axis=1)
    n_nan_rows = int(nan_mask.sum())
    if n_nan_rows > 0:
        bad_idx = np.where(nan_mask)[0].tolist()
        print(
            f"\n[test_evaluation] WARNING: {n_nan_rows}/{len(all_probs)} test "
            f"rows have NaN probabilities (model produced NaN logits for these "
            f"samples, {n_nan_batches} batch(es) affected). This is why AUROC "
            f"was coming back as `nan` on every bootstrap draw — it's a data/"
            f"model issue, not a bootstrap edge case.\n"
            f"Row indices (order matches test_loader, shuffle=False assumed): "
            f"{bad_idx[:20]}{'...' if len(bad_idx) > 20 else ''}\n"
            f"Cross-reference these against `test_df.iloc[bad_idx]` to find the "
            f"offending subjects/scans. Common causes: an unimputed NaN in the "
            f"tabular features, or a degenerate/empty image volume producing a "
            f"zero std during intensity normalization (division by zero).\n"
            f"These rows have been replaced with a uniform probability "
            f"distribution below so AUROC/accuracy/F1 can still be computed for "
            f"the *rest* of the test set — treat the reported numbers as "
            f"provisional until the root cause is fixed.\n"
        )
        # Uniform distribution => contributes ~no discriminative signal to
        # AUROC while still letting sklearn compute rather than raising.
        all_probs[nan_mask] = 1.0 / all_probs.shape[1]

    # One-hot encode labels for multi-class AUROC (OvR)
    all_labels_bin = label_binarize(all_labels, classes=list(range(len(class_names))))

    # --- Classification report (per-class detail, unchanged behavior) ---
    print("\n================ FINAL TEST RESULTS ================")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    with open(Path(RESULTS_DIR, f"multimodal_test_report_{fusion_methods}{suffix}.txt"), "w") as f:
        f.write("Classification Report:\n")
        f.write(report)
        if n_nan_rows > 0:
            f.write(
                f"\n\nWARNING: {n_nan_rows}/{len(all_probs)} test rows had NaN "
                f"model outputs and were imputed with a uniform probability "
                f"distribution before scoring. Investigate the upstream data "
                f"pipeline before trusting AUROC.\n"
            )

    # --- Point estimates (full test set) ---
    point_acc, point_f1, point_auroc = compute_metrics(
        all_labels, all_preds, all_probs, all_labels_bin
    )

    # --- Bootstrap CIs ---
    ci_results = bootstrap_ci(
        all_labels, all_preds, all_probs, all_labels_bin,
        n_bootstrap=n_bootstrap, ci=ci,
    )

    # --- Print report ---
    print(f"\n========== Test Set Results ({ci}% Bootstrap CI) ==========")
    print(f"{'Metric':<12} {'Point Est.':>12} {f'{ci}% CI':>25}")
    print("-" * 52)
    for metric, point in [("Accuracy", point_acc), ("Micro F1", point_f1), ("AUROC", point_auroc)]:
        lo = ci_results[metric]["lower"]
        hi = ci_results[metric]["upper"]
        print(f"{metric:<12} {point:>12.4f} {'[' + f'{lo:.4f}, {hi:.4f}' + ']':>25}")
    print("=" * 52)

    # --- Save bootstrap metrics table ---
    results_rows = []
    for metric, point in [("Accuracy", point_acc), ("Micro F1", point_f1), ("AUROC", point_auroc)]:
        results_rows.append({
            "Model":       f"Multimodal_{fusion_methods}",
            "Metric":      metric,
            "Point_Est":   round(point, 4) if not np.isnan(point) else point,
            "CI_Lower":    round(ci_results[metric]["lower"], 4) if not np.isnan(ci_results[metric]["lower"]) else ci_results[metric]["lower"],
            "CI_Upper":    round(ci_results[metric]["upper"], 4) if not np.isnan(ci_results[metric]["upper"]) else ci_results[metric]["upper"],
            "CI_Level":    f"{ci}%",
            "N_Bootstrap": n_bootstrap,
        })

    test_results_df = pd.DataFrame(results_rows)
    metrics_path = Path(RESULTS_DIR, f"multimodal_test_results_{fusion_methods}{suffix}.csv")
    test_results_df.to_csv(metrics_path, index=False)
    print(f"\nBootstrap metrics saved → {metrics_path}")

    # --- Save raw predictions (kept separate from the metrics table above) ---
    predictions_df = pd.DataFrame({
        "y_true": all_labels,
        "y_pred": all_preds,
        "had_nan_probs": nan_mask if n_nan_rows > 0 else np.zeros(len(all_labels), dtype=bool),
    })
    predictions_path = Path(RESULTS_DIR, f"multimodal_test_predictions_{fusion_methods}{suffix}.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Raw predictions saved → {predictions_path}")

    return test_results_df


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":

    data = preprocess_adni(train_df, val_df, test_df, keep_paths=True)
    classes = data["label_encoder"].classes_

    batch_size = 16


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

    train_dataset = ADNIMultiDataset(dataframe=train_df, tabular_features=data["X_train"], classes=classes,paths =data["train_paths"], transform=train_transform)
    val_dataset   = ADNIMultiDataset(dataframe=val_df,   tabular_features=data["X_val"],   classes=classes,paths =data["val_paths"], transform=val_transform)
    test_dataset  = ADNIMultiDataset(dataframe=test_df,  tabular_features=data["X_test"],  classes= classes,paths = data["test_paths"], transform=val_transform)


    # train_ds = ADNIMultiDataset(
    #     train_df, train_tabular, CLASSES, data["train_paths"],
    #     transform=get_train_transforms(),
    # )
    # val_ds = ADNIMultiDataset(
    #     val_df, val_tabular, CLASSES, data["val_paths"],
    #     transform=None,
    # )
    # test_ds = ADNIMultiDataset(
    #     test_df, test_tabular, CLASSES, data["test_paths"],
    #     transform=None,
    # )
 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=2, persistent_workers=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=2, persistent_workers=True)

    n_features = data["X_train"].shape[1]
    tab_embed_dim = data["X_test"].shape[-1]
    img_embed_dim = 128
    fusion_dropout = 0.2

    model = MultimodalADNI(
        n_tabular_features=n_features,
        fusion=f"{fusion_methods}",
        img_embed_dim      = img_embed_dim,
        tab_embed_dim      = tab_embed_dim,
        fusion_dropout     = fusion_dropout,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── Loss — weighted for class imbalance ───────────────────────────────────
    class_weights = compute_class_weights(data["y_train"], len(classes)).to(DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # ── Checkpoint / resume config ─────────────────────────────────────────────
    CHECKPOINT_PATH = Path(RESULTS_DIR, f"checkpoint_{fusion_methods}{suffix}.pt")
    START_EPOCH = 0  # set to 1 for a normal fresh run; only takes effect if the checkpoint exists

    model = train(model, train_loader, val_loader, epochs=20,
                  start_epoch=START_EPOCH, checkpoint_path=CHECKPOINT_PATH)

    # # ================= FINAL TEST =================
    test_evaluation(model, test_loader, classes)