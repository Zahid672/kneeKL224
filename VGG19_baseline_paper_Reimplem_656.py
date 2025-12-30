"""
VGG-19 + Adjustable Ordinal Loss (PD-2) for KL grading on OAI knee joints (kneeKL224).

This follows the settings in:
"Fully automatic knee osteoarthritis severity grading using deep neural
networks with a novel ordinal loss" (Chen et al., CMIG 2019).

Key points:
- Backbone: VGG-19 (no batch norm), pretrained on ImageNet.
- Input: 224x224 knee joint crops (already detected & expanded in dataset).
- Loss: Adjustable ordinal loss with penalty distance 2 (PD-2), squared.
- Metrics: Accuracy and MAE, plus standard classification metrics.
- Splits: Uses existing train/val/test folders; train+val are used inside a
  5-fold stratified CV, with best-val fold evaluated on test.
"""

import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter


# =========================
# CONFIG
# =========================

DATA_ROOT = r"E:\Knee-Osteoarthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224"

NUM_CLASSES = 5          # KL 0..4
IMG_SIZE = 224
NUM_FOLDS = 5

BATCH_SIZE = 16
NUM_EPOCHS = 80
LR = 3e-5
WEIGHT_DECAY = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
EARLY_STOP_PATIENCE = 10
FEATURE_EXTRACT = False
SEED = 42

METRICS_CSV = "VGG19_Ordinal_PD2_kfold_results_kneeKL224.csv"
PREDICTIONS_CSV = "VGG19_Ordinal_PD2_bestFold_test_predictions_kneeKL224.csv"
BEST_MODEL_PATH = "VGG19_Ordinal_PD2_bestFold_kneeKL224.pth"


# =========================
# UTILITIES
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def specificity_score(y_true, y_pred, num_classes, average="macro"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    TN, FP = [], []
    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm[i, :], i))
        TN.append(tn)
        FP.append(fp)
    spec = np.array(TN) / (np.array(TN) + np.array(FP) + 1e-8)
    if average == "macro":
        return float(np.mean(spec))
    return spec


class PathLabelDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


def load_trainval_samples_and_labels(root: str):
    train_folder = datasets.ImageFolder(os.path.join(root, "train"))
    val_folder = datasets.ImageFolder(os.path.join(root, "val"))

    samples = train_folder.samples + val_folder.samples
    labels = np.array(train_folder.targets + val_folder.targets, dtype=np.int64)
    return samples, labels


def load_test_dataset(root: str):
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(os.path.join(root, "test"), transform=test_tf)
    return test_ds


def make_loader(dataset: Dataset, indices, shuffle: bool):
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return loader


# =========================
# MODEL (VGG-19 + Ordinal loss)
# =========================

class VGG19Ordinal(nn.Module):
    """
    VGG-19 backbone with a 5-class classifier head.
    Loss is handled outside via the adjustable ordinal loss.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        try:
            vgg = models.vgg19(pretrained=pretrained)
        except TypeError:
            vgg = models.vgg19(
                weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            )

        # Replace classifier last layer to output NUM_CLASSES logits
        in_features = vgg.classifier[-1].in_features
        vgg.classifier[-1] = nn.Linear(in_features, num_classes)

        self.backbone = vgg

    def forward(self, x):
        logits = self.backbone(x)
        return logits

    @torch.no_grad()
    def predict_from_logits(self, logits):
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds

    @torch.no_grad()
    def predict(self, x):
        logits = self.forward(x)
        return self.predict_from_logits(logits)


def build_pd2_matrix(num_classes: int, penalty_distance: int = 2) -> torch.Tensor:
    """
    Build adjustable ordinal matrix W for PD-2 as in Chen et al.
    w[i, j] = 1                   if i == j
            = 1 + d * |i - j|     otherwise
    (d = penalty_distance, here 2)

    We then use Eq.(1) from the paper:
        loss = sum_i w[i,m] * q_i
    where q_i = p_i (i != m), q_m = 1 - p_m,
    and finally use (loss)^2.
    """
    W = np.zeros((num_classes, num_classes), dtype=np.float32)
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                W[i, j] = 1.0
            else:
                W[i, j] = 1.0 + penalty_distance * abs(i - j)
    return torch.tensor(W, dtype=torch.float32)


def ordinal_loss_pd2(
    logits: torch.Tensor,
    labels: torch.Tensor,
    W: torch.Tensor,
) -> torch.Tensor:
    """
    Adjustable ordinal loss with PD-2, squared.

    logits: [B, C]
    labels: [B] (0..C-1)
    W: [C, C] penalty matrix as build_pd2_matrix

    Steps:
    - softmax to get probabilities p
    - build q: q = p, except q[b, label_b] = 1 - p[b, label_b]
    - for each sample b: L_b = sum_i W[i, label_b] * q[b, i]
    - final loss = mean(L_b^2)
    """
    B, C = logits.shape
    device = logits.device
    W = W.to(device)

    probs = torch.softmax(logits, dim=1)  # [B, C]

    # q = p except on the true label
    q = probs.clone()
    idx = torch.arange(B, device=device)
    q[idx, labels] = 1.0 - probs[idx, labels]

    # For each sample, get the column W[:, label_b]
    # W_cols: [C, B]
    W_cols = W[:, labels]            # [C, B]
    W_cols = W_cols.transpose(0, 1)  # [B, C]

    # Elementwise multiply and sum over classes
    L_vec = (W_cols * q).sum(dim=1)  # [B]

    loss = (L_vec ** 2).mean()
    return loss


def train_one_epoch(
    model,
    loader,
    optimizer,
    W_ord,
    device,
    writer=None,
    epoch_idx=None,
    fold_idx=None,
):
    model.train()
    running = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        logits = model(images)
        loss = ordinal_loss_pd2(logits, labels, W_ord)

        loss.backward()
        optimizer.step()

        running += loss.item() * labels.size(0)

    epoch_loss = running / len(loader.dataset)
    if writer is not None and epoch_idx is not None:
        writer.add_scalar(f"Fold{fold_idx}/Loss_train", epoch_loss, epoch_idx)
    return epoch_loss


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    W_ord,
    writer=None,
    epoch_idx=None,
    fold_idx=None,
    split_name="val",
):
    model.eval()

    total_loss = 0.0
    all_labels, all_preds = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        logits = model(images)
        loss = ordinal_loss_pd2(logits, labels, W_ord)
        total_loss += loss.item() * labels.size(0)

        preds = model.predict_from_logits(logits)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    avg_loss = total_loss / len(loader.dataset)
    acc = float(np.mean(y_true == y_pred))

    if writer is not None and epoch_idx is not None:
        writer.add_scalar(f"Fold{fold_idx}/Loss_{split_name}", avg_loss, epoch_idx)
        writer.add_scalar(f"Fold{fold_idx}/Acc_{split_name}", acc, epoch_idx)

    return avg_loss, acc, y_true, y_pred


def compute_metrics(y_true, y_pred, num_classes):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "mae": mae,
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "specificity_macro": specificity_score(y_true, y_pred, num_classes),
    }

    y_true_oh = np.eye(num_classes)[y_true]
    y_pred_oh = np.eye(num_classes)[y_pred]
    try:
        metrics["auroc_macro"] = float(
            roc_auc_score(y_true_oh, y_pred_oh, average="macro")
        )
    except Exception:
        metrics["auroc_macro"] = 0.0

    return metrics


# =========================
# MAIN TRAINING PIPELINE
# =========================

def run_vgg19_ordinal_training():
    set_seed(SEED)
    print("Using device:", DEVICE)

    if not os.path.isdir(DATA_ROOT):
        raise RuntimeError(f"DATA_ROOT not found: {DATA_ROOT}")

    samples, labels = load_trainval_samples_and_labels(DATA_ROOT)
    labels = np.asarray(labels, dtype=np.int64)

    test_ds = load_test_dataset(DATA_ROOT)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Ordinal penalty matrix PD-2
    W_ord = build_pd2_matrix(NUM_CLASSES, penalty_distance=2)
    print("Ordinal penalty matrix (PD-2):")
    print(W_ord.numpy())

    # Transforms (resize 224, normalization) – no fancy augmentation
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = PathLabelDataset(samples, transform=train_tf)
    val_ds = PathLabelDataset(samples, transform=val_tf)

    indices = np.arange(len(samples))
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    all_metrics_rows = []
    best_overall_val_acc = -1.0
    best_overall_state = None
    best_fold_idx = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels), start=1):
        print(f"\n========== FOLD {fold_idx}/{NUM_FOLDS} ==========")

        uniq_tr, cnt_tr = np.unique(labels[train_idx], return_counts=True)
        print("Train class counts:", dict(zip(uniq_tr.tolist(), cnt_tr.tolist())))
        uniq_val, cnt_val = np.unique(labels[val_idx], return_counts=True)
        print("Val class counts:", dict(zip(uniq_val.tolist(), cnt_val.tolist())))

        train_loader = make_loader(train_ds, train_idx, shuffle=True)
        val_loader = make_loader(val_ds, val_idx, shuffle=False)

        model = VGG19Ordinal(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)

        if FEATURE_EXTRACT:
            for p in model.backbone.features.parameters():
                p.requires_grad = False

        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params_to_update, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        writer = SummaryWriter(log_dir=f"runs/VGG19_Ordinal_PD2_kneeKL224/fold_{fold_idx}")

        best_val_acc = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"Fold {fold_idx} | Epoch {epoch}/{NUM_EPOCHS}")

            train_loss = train_one_epoch(
                model, train_loader, optimizer,
                W_ord, DEVICE, writer, epoch, fold_idx
            )
            val_loss, val_acc, y_val, y_val_pred = evaluate(
                model, val_loader, DEVICE, W_ord,
                writer, epoch, fold_idx, split_name="val"
            )

            print(f"  train_loss: {train_loss:.4f} | "
                  f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                best_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "fold": fold_idx,
                }
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping on fold {fold_idx} at epoch {epoch}")
                break

        writer.close()

        if best_state is None:
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "fold": fold_idx,
            }

        model.load_state_dict(best_state["model"])
        val_loss, val_acc, y_val, y_val_pred = evaluate(
            model, val_loader, DEVICE, W_ord
        )

        fold_metrics = compute_metrics(y_val, y_val_pred, NUM_CLASSES)
        fold_metrics["fold"] = fold_idx
        fold_metrics["split"] = "val"
        fold_metrics["val_loss"] = float(val_loss)
        all_metrics_rows.append(fold_metrics)

        print("\nFold", fold_idx, "validation confusion matrix:")
        print(confusion_matrix(y_val, y_val_pred))
        print("\nFold", fold_idx, "validation classification report:")
        print(classification_report(y_val, y_val_pred, digits=4, zero_division=0))

        if val_acc > best_overall_val_acc:
            best_overall_val_acc = val_acc
            best_overall_state = best_state
            best_fold_idx = fold_idx

    print(f"\nBest fold: {best_fold_idx} with val_acc={best_overall_val_acc:.4f}")
    torch.save(best_overall_state, BEST_MODEL_PATH)
    print(f"Best fold model saved to {BEST_MODEL_PATH}")

    # Final test with best fold model (single center crop, no TTA)
    best_model = VGG19Ordinal(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    best_model.load_state_dict(best_overall_state["model"])

    test_loss, test_acc, y_test, y_test_pred = evaluate(
        best_model, test_loader, DEVICE, W_ord,
        writer=None, epoch_idx=None, fold_idx=best_fold_idx, split_name="test"
    )
    test_metrics = compute_metrics(y_test, y_test_pred, NUM_CLASSES)
    test_metrics["fold"] = best_fold_idx
    test_metrics["split"] = "test"
    test_metrics["val_loss"] = float(test_loss)
    all_metrics_rows.append(test_metrics)

    print("\n===== TEST RESULTS (best fold model) =====")
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
    print("Test MAE:", test_metrics["mae"])
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

    df_metrics = pd.DataFrame(all_metrics_rows)
    df_metrics.to_csv(METRICS_CSV, index=False)
    print(f"\nAll fold + test metrics saved to {METRICS_CSV}")

    df_pred = pd.DataFrame({
        "true_label": y_test,
        "predicted_label": y_test_pred,
    })
    df_pred.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Test predictions saved to {PREDICTIONS_CSV}")


if __name__ == "__main__":
    run_vgg19_ordinal_training()
