"""
ViT-L/16 (softmax) baseline for KL grading (kneeKL224), single script.

What it does:
1) Load train+val as one pool and run StratifiedKFold (5 folds).
2) Train ViT-L/16 with CrossEntropyLoss (with optional class weights).
3) Save best checkpoint per fold (by val accuracy).
4) Evaluate:
   - best fold model on test
   - 5-fold ensemble on test (mean probs)

Folder structure:
DATA_ROOT/
    train/0..4
    val/0..4
    test/0..4
"""

import os
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# =========================
# CONFIG
# =========================

DATA_ROOT = r"E:\Knee-Osteoarthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224"

NUM_CLASSES = 5
IMG_SIZE = 224
NUM_FOLDS = 5

BATCH_SIZE = 8
NUM_EPOCHS = 60
LR = 3e-5
WEIGHT_DECAY = 0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
EARLY_STOP_PATIENCE = 10
SEED = 42

MODEL_TAG = "ViT_L16_Softmax"

METRICS_CSV = f"{MODEL_TAG}_kfold_metrics_kneeKL224.csv"
BEST_MODEL_PATH = f"{MODEL_TAG}_bestFold_kneeKL224.pth"
TEST_PRED_CSV = f"{MODEL_TAG}_bestFold_test_predictions_kneeKL224.csv"

ENSEMBLE_METRICS_CSV = f"{MODEL_TAG}_ensemble_test_metrics_kneeKL224.csv"
ENSEMBLE_PRED_CSV = f"{MODEL_TAG}_ensemble_test_predictions_kneeKL224.csv"

# Optional: class weights for CE (match your earlier weighting spirit)
# If you want "no weighting", set USE_CLASS_WEIGHTS = False
USE_CLASS_WEIGHTS = True
CLASS_WEIGHT_VEC = torch.tensor([1.0, 1.5, 1.5, 1.0, 1.0], dtype=torch.float32)

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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity_macro": specificity_score(y_true, y_pred, num_classes),
    }

    # AUROC (macro) computed using one-hot of predicted hard labels (simple + consistent).
    # If you want AUROC from probabilities, compute from softmax probs instead.
    y_true_oh = np.eye(num_classes)[y_true]
    y_pred_oh = np.eye(num_classes)[y_pred]
    try:
        metrics["auroc_macro"] = float(roc_auc_score(y_true_oh, y_pred_oh, average="macro"))
    except Exception:
        metrics["auroc_macro"] = 0.0

    return metrics


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
# MODEL (ViT-L/16 softmax)
# =========================

class ViT_Softmax(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        try:
            vit = models.vit_l_16(pretrained=pretrained)
        except TypeError:
            vit = models.vit_l_16(
                weights=models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            )

        in_dim = vit.heads.head.in_features
        vit.heads = nn.Identity()
        self.backbone = vit
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits

    @torch.no_grad()
    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


# =========================
# TRAIN / EVAL
# =========================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).long()

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running += loss.item() * labels.size(0)

    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total = 0.0
    all_labels, all_preds = [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).long()

        logits = model(images)
        loss = criterion(logits, labels)
        total += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    avg_loss = total / len(loader.dataset)
    acc = float(np.mean(y_true == y_pred))
    return avg_loss, acc, y_true, y_pred


@torch.no_grad()
def predict_proba(model, loader):
    """Return softmax probabilities and labels for a loader."""
    model.eval()
    probs_list, labels_list = [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        probs_list.append(probs.cpu().numpy())
        labels_list.append(labels.numpy())

    probs_all = np.concatenate(probs_list)
    labels_all = np.concatenate(labels_list)
    return probs_all, labels_all


# =========================
# MAIN
# =========================

def run_vit_softmax_kfold_and_ensemble():
    set_seed(SEED)
    print("Using device:", DEVICE)

    if not os.path.isdir(DATA_ROOT):
        raise RuntimeError(f"DATA_ROOT not found: {DATA_ROOT}")

    samples, labels_all = load_trainval_samples_and_labels(DATA_ROOT)
    labels_all = np.asarray(labels_all, dtype=np.int64)

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
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

    test_ds = load_test_dataset(DATA_ROOT)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # loss
    if USE_CLASS_WEIGHTS:
        class_w = CLASS_WEIGHT_VEC.to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_w)
        print("Using class weights:", class_w.tolist())
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using unweighted CrossEntropyLoss")

    indices = np.arange(len(samples))
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    all_rows = []
    best_overall_val_acc = -1.0
    best_overall_state = None
    best_fold_idx = None

    # ========== K-fold training ==========
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(indices, labels_all), start=1):
        print(f"\n========== FOLD {fold_idx}/{NUM_FOLDS} ==========")

        train_loader = make_loader(train_ds, tr_idx, shuffle=True)
        val_loader = make_loader(val_ds, va_idx, shuffle=False)

        model = ViT_Softmax(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_val_acc = -1.0
        best_state = None
        no_improve = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            va_loss, va_acc, y_va, y_va_pred = evaluate(model, val_loader, criterion)
            scheduler.step()

            print(f"Fold {fold_idx} | Epoch {epoch:03d} | "
                  f"train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | val_acc {va_acc:.4f}")

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                no_improve = 0
                best_state = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": va_acc,
                    "fold": fold_idx,
                }
            else:
                no_improve += 1

            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping fold {fold_idx} at epoch {epoch}")
                break

        if best_state is None:
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": best_val_acc,
                "fold": fold_idx,
            }

        # Save fold ckpt
        fold_ckpt_path = f"{MODEL_TAG}_fold{fold_idx}_best.pth"
        torch.save(best_state, fold_ckpt_path)
        print("Saved:", fold_ckpt_path)

        # final val metrics for this fold
        model.load_state_dict(best_state["model"])
        va_loss, va_acc, y_va, y_va_pred = evaluate(model, val_loader, criterion)

        row = compute_metrics(y_va, y_va_pred, NUM_CLASSES)
        row.update({
            "fold": fold_idx,
            "split": "val",
            "val_loss": float(va_loss),
            "val_acc": float(va_acc),
        })
        all_rows.append(row)

        print("Val confusion matrix:\n", confusion_matrix(y_va, y_va_pred))
        print(classification_report(y_va, y_va_pred, digits=4, zero_division=0))

        if va_acc > best_overall_val_acc:
            best_overall_val_acc = va_acc
            best_overall_state = best_state
            best_fold_idx = fold_idx

    print(f"\nBest fold: {best_fold_idx} with val_acc={best_overall_val_acc:.4f}")
    torch.save(best_overall_state, BEST_MODEL_PATH)
    print("Best fold model saved to:", BEST_MODEL_PATH)

    # Write kfold val metrics
    df = pd.DataFrame(all_rows)
    df.to_csv(METRICS_CSV, index=False)
    print("Saved CV metrics:", METRICS_CSV)

    # ========== TEST: Best fold ==========
    print("\n===== TEST (best fold model) =====")
    best_model = ViT_Softmax(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    best_model.load_state_dict(best_overall_state["model"])

    # evaluate on test
    # build a loader that returns labels (ImageFolder already does)
    test_loss, test_acc, y_te, y_te_pred = evaluate(best_model, test_loader, criterion)

    test_row = compute_metrics(y_te, y_te_pred, NUM_CLASSES)
    test_row.update({
        "fold": best_fold_idx,
        "split": "test_single_bestfold",
        "val_loss": float(test_loss),
        "val_acc": float(test_acc),
    })
    print("Test accuracy:", test_acc)
    print("Confusion matrix:\n", confusion_matrix(y_te, y_te_pred))
    print(classification_report(y_te, y_te_pred, digits=4, zero_division=0))

    # Save predictions
    pd.DataFrame({"true_label": y_te, "predicted_label": y_te_pred}).to_csv(TEST_PRED_CSV, index=False)
    print("Saved best-fold test predictions:", TEST_PRED_CSV)

    # Append test row to metrics CSV
    df2 = pd.read_csv(METRICS_CSV)
    df2 = pd.concat([df2, pd.DataFrame([test_row])], ignore_index=True)
    df2.to_csv(METRICS_CSV, index=False)
    print("Updated metrics CSV with test row:", METRICS_CSV)

    # ========== TEST: 5-fold ensemble ==========
    print("\n===== TEST (5-fold ensemble; mean softmax probs) =====")
    models_list = []
    for fold_idx in range(1, NUM_FOLDS + 1):
        ckpt_path = f"{MODEL_TAG}_fold{fold_idx}_best.pth"
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        m = ViT_Softmax(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models_list.append(m)

    # ensemble probabilities
    probs_sum = None
    y_true_all = None

    for m in models_list:
        probs, y_true = predict_proba(m, test_loader)
        probs_sum = probs if probs_sum is None else (probs_sum + probs)
        y_true_all = y_true  # same ordering for all models

    probs_ens = probs_sum / float(len(models_list))
    y_pred_ens = np.argmax(probs_ens, axis=1)

    ens_row = compute_metrics(y_true_all, y_pred_ens, NUM_CLASSES)
    ens_row.update({
        "fold": -1,
        "split": "test_ensemble_meanprobs",
        "val_loss": 0.0,
        "val_acc": float(ens_row["accuracy"]),
    })

    print("Ensemble accuracy:", ens_row["accuracy"])
    print("Ensemble confusion matrix:\n", confusion_matrix(y_true_all, y_pred_ens))
    print(classification_report(y_true_all, y_pred_ens, digits=4, zero_division=0))

    pd.DataFrame({"true_label": y_true_all, "predicted_label": y_pred_ens}).to_csv(ENSEMBLE_PRED_CSV, index=False)
    print("Saved ensemble predictions:", ENSEMBLE_PRED_CSV)

    pd.DataFrame([ens_row]).to_csv(ENSEMBLE_METRICS_CSV, index=False)
    print("Saved ensemble metrics:", ENSEMBLE_METRICS_CSV)


if __name__ == "__main__":
    run_vit_softmax_kfold_and_ensemble()
