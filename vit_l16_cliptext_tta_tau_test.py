"""
ViT-L/16 CORAL + CLIPtext (best fold)
TTA + tau tuning for KL grading on kneeKL224.

Uses the checkpoint produced by:
  vit_l16_coral_clipTextContrast_kfold.py

DATA_ROOT/
    train/0..4
    val/0..4
    test/0..4
"""

import os
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

# =========================
# CONFIG
# =========================

DATA_ROOT = r"E:\Knee-Osteoarthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224"

CHECKPOINT = "ViT_L16_CORAL_CLIPtext_kfold_bestFold.pth"
NUM_CLASSES = 5
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TAU_MIN = 0.2
TAU_MAX = 0.8
TAU_STEPS = 61  # step ~0.01

METRICS_CSV = "ViT_L16_CORAL_CLIPtext_TTA_tau_metrics.csv"
PREDICTIONS_CSV = "ViT_L16_CORAL_CLIPtext_TTA_tau_test_predictions.csv"


# =========================
# STUDENT MODEL
# =========================

class ViT_L16_CORAL(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        num_thresholds = num_classes - 1

        try:
            vit = models.vit_l_16(pretrained=pretrained)
        except TypeError:
            vit = models.vit_l_16(
                weights=models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            )

        embed_dim = vit.heads.head.in_features
        vit.heads = nn.Identity()
        self.backbone = vit
        self.classifier = nn.Linear(embed_dim, num_thresholds)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits, feats

    @torch.no_grad()
    def predict_from_logits(self, logits, tau: float = 0.5):
        probs = torch.sigmoid(logits)
        preds = torch.sum(probs >= tau, dim=1).long()
        return preds


# =========================
# DATA
# =========================

def build_loaders():
    base_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=base_tf)
    test_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=base_tf)

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    return val_loader, test_loader


# =========================
# TTA + TAU UTILITIES
# =========================

@torch.no_grad()
def logits_with_tta(model, images, n_augs: int = 4):
    """
    Simple TTA: original, hflip, +10 deg, -10 deg
    Returns mean logits.
    """
    model.eval()
    augs = [images]
    if n_augs > 1:
        augs.append(F.hflip(images))
    if n_augs > 2:
        augs.append(F.rotate(images, 10))
    if n_augs > 3:
        augs.append(F.rotate(images, -10))

    logits_sum = None
    for im in augs[:n_augs]:
        logits, _ = model(im)
        if logits_sum is None:
            logits_sum = logits
        else:
            logits_sum = logits_sum + logits

    return logits_sum / float(len(augs[:n_augs]))


@torch.no_grad()
def collect_probs_and_labels(model, loader):
    """
    Run model with TTA on loader and collect
    sigmoid(logits) and labels.
    """
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = logits_with_tta(model, images, n_augs=4)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return probs, labels


def tune_tau(probs, labels):
    best_tau = 0.5
    best_f1 = -1.0
    taus = np.linspace(TAU_MIN, TAU_MAX, TAU_STEPS)
    for tau in taus:
        preds = (probs >= tau).sum(axis=1)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    return best_tau, best_f1


@torch.no_grad()
def eval_with_tau(model, loader, tau: float):
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = logits_with_tta(model, images, n_augs=4)
        probs = torch.sigmoid(logits)
        preds = torch.sum(probs >= tau, dim=1).long()
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# =========================
# MAIN
# =========================

def main():
    print("Device:", DEVICE)

    model = ViT_L16_CORAL(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)

    state = torch.load(CHECKPOINT, map_location=DEVICE)
    if "model" in state:
        model.load_state_dict(state["model"])
        print(f"Loaded model from {CHECKPOINT} (fold={state.get('fold')}, val_acc={state.get('val_acc')})")
    else:
        model.load_state_dict(state)
        print(f"Loaded state_dict directly from {CHECKPOINT}")

    val_loader, test_loader = build_loaders()

    # ---- tau tuning on val ----
    print("\nCollecting validation probs with TTA for tau tuning...")
    val_probs, val_labels = collect_probs_and_labels(model, val_loader)
    best_tau, best_val_f1 = tune_tau(val_probs, val_labels)
    print(f"\nBest tau on validation: {best_tau:.4f} (macro F1 = {best_val_f1:.4f})")

    # ---- test with tuned tau ----
    print("\nEvaluating on test set with TTA + tuned tau...")
    y_test, y_test_pred = eval_with_tau(model, test_loader, tau=best_tau)
    metrics = compute_metrics(y_test, y_test_pred)

    print("\n===== TEST RESULTS (best fold, tuned tau, with TTA) =====")
    print("Accuracy:", metrics["accuracy"])
    print("Macro precision:", metrics["precision_macro"])
    print("Macro recall:", metrics["recall_macro"])
    print("Macro F1:", metrics["f1_macro"])

    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

    # save CSVs
    df_metrics = pd.DataFrame([{
        "setting": "ViT_L16_CORAL_CLIPtext_TTA_tau",
        "tau": best_tau,
        **metrics,
    }])
    df_metrics.to_csv(METRICS_CSV, index=False)
    print(f"\nMetrics saved to {METRICS_CSV}")

    df_pred = pd.DataFrame({
        "true_label": y_test,
        "predicted_label": y_test_pred,
    })
    df_pred.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Test predictions saved to {PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
