import time
from pathlib import Path
import numpy as np
import pandas as pd
from util import n4_correction,compute_class_weights_effective_num
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from tqdm import tqdm
from adni_data import ADNIMultiDataset
from model import MultimodalADNI
from util import preprocess_adni
from config import DATA_DIR, RESULTS_DIR, FIGURS_DIR, METADATA_DIR

# =============================================================================
# Device
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df_processed = pd.read_csv(Path(RESULTS_DIR, "ADNI_Combined_Multimodal_FINAL.csv"))
fusion_methods = "cross_attn"  # "cross_attn" concat # Example fusion methods to experiment with

CLASSES = df_processed['Group'].unique().tolist()

if Path(RESULTS_DIR, "train_split.csv").exists():

    print("Loading existing train/val/test splits...")
    train_df = pd.read_csv(Path(RESULTS_DIR, "train_split.csv"))
    val_df   = pd.read_csv(Path(RESULTS_DIR, "val_split.csv"))
    test_df  = pd.read_csv(Path(RESULTS_DIR, "test_split.csv"))
else:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)

    train_idx, test_idx = next(gss.split(df_processed, groups=df_processed["PTID"]))

    train_df = df_processed.iloc[train_idx]
    test_df  = df_processed.iloc[test_idx]

    # Further split train → train/val, again by subject
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, val_idx = next(gss_val.split(train_df, groups=train_df["PTID"]))

    val_df   = train_df.iloc[val_idx]
    train_df = train_df.iloc[tr_idx]

    # save the splits for future reference
    train_df.to_csv(Path(RESULTS_DIR, "train_split.csv"), index=False)
    val_df.to_csv(Path(RESULTS_DIR, "val_split.csv"), index=False)
    test_df.to_csv(Path(RESULTS_DIR, "test_split.csv"), index=False)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


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
    opt = torch.optim.Adam(model.parameters(), lr=lr)

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
# FINAL TEST EVALUATION (IMPORTANT PART)
# =============================================================================
@torch.no_grad()
def test_evaluation(model, test_loader, class_names):
    loss, f1_micro, y_true, y_pred = evaluate(model, test_loader)

    print("\n================ FINAL TEST RESULTS ================")
    print(f"F1 Score (micro): {f1_micro:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # save to text file
    with open(Path(RESULTS_DIR, f"multimodal_test_report_{fusion_methods}.txt"), "w") as f:
        f.write(f"F1 Score (micro): {f1_micro:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))

    # Optional ROC-AUC (multiclass one-vs-rest)
    try:
        probs = []
        model.eval()
        for img, tab, _ in tqdm(test_loader, desc="test probs"):
            img, tab = img.to(DEVICE), tab.to(DEVICE)
            probs.append(torch.softmax(model(img, tab), 1).cpu().numpy())
        probs = np.vstack(probs)

        print(f"ROC-AUC (macro): {roc_auc_score(y_true, probs, multi_class='ovr'):.4f}")
    except:
        print("ROC-AUC not computed.")

    # Save results to CSV
    results_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })
    results_df.to_csv(Path(RESULTS_DIR, f"multimodal_test_results_{fusion_methods}.csv"), index=False)


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":

    data = preprocess_adni(train_df, val_df, test_df, keep_paths=True)
    classes = data["label_encoder"].classes_

    batch_size = 16

    train_dataset = ADNIMultiDataset(train_df, data["X_train"], classes,
                                data["train_paths"])
    val_dataset   = ADNIMultiDataset(val_df,   data["X_val"],   classes,
                                data["val_paths"])
    test_dataset  = ADNIMultiDataset(test_df,  data["X_test"],  classes,
                                data["test_paths"])

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
    CHECKPOINT_PATH = Path(RESULTS_DIR, f"checkpoint_{fusion_methods}.pt")
    START_EPOCH = 20  # set to 1 for a normal fresh run; only takes effect if the checkpoint exists

    model = train(model, train_loader, val_loader, epochs=20,
                  start_epoch=START_EPOCH, checkpoint_path=CHECKPOINT_PATH)

    # # ================= FINAL TEST =================
    test_evaluation(model, test_loader, classes)