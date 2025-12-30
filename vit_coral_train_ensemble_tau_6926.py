"""
ViT-B/16 + CORAL for KL grading on OAI knee joints (kneeKL224).

This script does:
1) 5-fold training + per-fold checkpoints.
2) Threshold (tau) tuning on validation folds using TTA logits.
3) Test evaluation of:
   - best fold model + TTA + tuned tau
   - 5-fold ensemble + TTA + tuned tau.

Folder structure:

DATA_ROOT/
    train/0..4
    val/0..4
    test/0..4
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
import torchvision.transforms.functional as F

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

# single best-fold outputs
METRICS_CSV = "ViT_B16_CORAL_kfold_results_kneeKL224_TTA_tau.csv"
PREDICTIONS_CSV = "ViT_B16_CORAL_bestFold_test_predictions_kneeKL224_TTA_tau.csv"
BEST_MODEL_PATH = "ViT_B16_CORAL_bestFold_kneeKL224_TTA_tau.pth"

# ensemble outputs
ENSEMBLE_METRICS_CSV = "ViT_B16_CORAL_ensemble_test_metrics_kneeKL224_TTA_tau.csv"
ENSEMBLE_PREDS_CSV = "ViT_B16_CORAL_ensemble_test_predictions_kneeKL224_TTA_tau.csv"


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


def compute_pos_weight(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    For CORAL thresholds y > k, k = 0..3.
    pos_weight_k = (N_neg / N_pos) to rebalance thresholds.
    """
    pos_weights = []
    total = len(labels)
    for k in range(num_classes - 1):
        pos = int(np.sum(labels > k))
        neg = total - pos
        if pos == 0:
            pos_weight_k = 1.0
        else:
            pos_weight_k = neg / float(pos)
        pos_weights.append(pos_weight_k)
    return torch.tensor(pos_weights, dtype=torch.float32)


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
# MODEL (ViT-B/16 + CORAL)
# =========================

class ViT_CORAL(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        num_thresholds = num_classes - 1

        try:
            vit = models.vit_b_16(pretrained=pretrained)
        except TypeError:
            vit = models.vit_b_16(
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            )

        embed_dim = vit.heads.head.in_features
        vit.heads = nn.Identity()
        self.backbone = vit
        self.classifier = nn.Linear(embed_dim, num_thresholds)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits

    @torch.no_grad()
    def predict_from_logits(self, logits, tau: float = 0.5):
        probs = torch.sigmoid(logits)
        preds = torch.sum(probs >= tau, dim=1).long()
        return preds

    @torch.no_grad()
    def predict(self, x, tau: float = 0.5):
        logits = self.forward(x)
        return self.predict_from_logits(logits, tau=tau)


def labels_to_coral_targets(labels: torch.Tensor, num_classes: int):
    labels = labels.unsqueeze(1)                      # [B,1]
    thresholds = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)
    targets = (labels > thresholds).float()           # [B, K-1]
    return targets


# =========================
# TRAIN / EVAL (single model)
# =========================

def train_one_epoch(
    model,
    loader,
    optimizer,
    pos_weight,
    cls_weight_vec,
    device,
    num_classes,
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
        targets = labels_to_coral_targets(labels, num_classes)

        sample_w = cls_weight_vec[labels]  # [B]

        loss_per_sample = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )  # [B, K-1]
        loss_per_sample = loss_per_sample.mean(dim=1)  # [B]
        loss = (loss_per_sample * sample_w).mean()

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
    num_classes,
    pos_weight,
    writer=None,
    epoch_idx=None,
    fold_idx=None,
    split_name="val",
    tau: float = 0.5,
):
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")

    total = 0.0
    all_labels, all_preds = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        logits = model(images)
        targets = labels_to_coral_targets(labels, num_classes)
        loss = bce_loss(logits, targets)
        total += loss.item()

        preds = model.predict_from_logits(logits, tau=tau)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    avg_loss = total / len(loader.dataset)
    acc = float(np.mean(y_true == y_pred))

    if writer is not None and epoch_idx is not None:
        writer.add_scalar(f"Fold{fold_idx}/Loss_{split_name}", avg_loss, epoch_idx)
        writer.add_scalar(f"Fold{fold_idx}/Acc_{split_name}", acc, epoch_idx)

    return avg_loss, acc, y_true, y_pred


@torch.no_grad()
def collect_logits_tta(
    model,
    loader,
    device,
    n_augs: int = 4,
):
    """
    Collect TTA-averaged logits and labels (for validation sets).
    """
    model.eval()
    logits_list = []
    labels_list = []

    def apply_augs(x):
        augs = [
            x,
            F.hflip(x),
            F.rotate(x, 10),
            F.rotate(x, -10),
        ]
        return augs[:n_augs]

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        logits_sum = None
        for aug_imgs in apply_augs(images):
            logits_aug = model(aug_imgs)
            logits_sum = logits_aug if logits_sum is None else logits_sum + logits_aug
        logits_mean = logits_sum / float(n_augs)

        logits_list.append(logits_mean.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

    logits_all = np.concatenate(logits_list)
    labels_all = np.concatenate(labels_list)
    return logits_all, labels_all


@torch.no_grad()
def evaluate_tta_single_model(
    model,
    loader,
    device,
    num_classes,
    pos_weight,
    tau: float = 0.5,
    n_augs: int = 4,
):
    """
    TTA for test: original, flip, +10°, -10°.
    """
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")
    total = 0.0
    all_labels, all_preds = [], []

    def apply_augs(x):
        augs = [
            x,
            F.hflip(x),
            F.rotate(x, 10),
            F.rotate(x, -10),
        ]
        return augs[:n_augs]

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        logits_sum = None
        for aug_imgs in apply_augs(images):
            logits_aug = model(aug_imgs)
            logits_sum = logits_aug if logits_sum is None else logits_sum + logits_aug
        logits_mean = logits_sum / float(n_augs)

        targets = labels_to_coral_targets(labels, num_classes)
        loss = bce_loss(logits_mean, targets)
        total += loss.item()

        preds = model.predict_from_logits(logits_mean, tau=tau)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    avg_loss = total / len(loader.dataset)
    acc = float(np.mean(y_true == y_pred))
    return avg_loss, acc, y_true, y_pred


def compute_metrics(y_true, y_pred, num_classes):
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
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
# ENSEMBLE EVAL (5 folds)
# =========================

@torch.no_grad()
def ensemble_evaluate_tta(
    models_list,
    loader,
    device,
    num_classes,
    tau: float = 0.5,
    n_augs: int = 4,
):
    """
    For each batch:
      - For each model:
          - apply TTA augs, average logits per model
      - average logits across models
      - decode CORAL predictions with tuned tau.
    """
    for m in models_list:
        m.eval()

    all_labels, all_preds = [], []

    def apply_augs(x):
        augs = [
            x,
            F.hflip(x),
            F.rotate(x, 10),
            F.rotate(x, -10),
        ]
        return augs[:n_augs]

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        logits_ens_sum = None

        for model in models_list:
            logits_sum = None
            for aug_imgs in apply_augs(images):
                logits_aug = model(aug_imgs)
                logits_sum = logits_aug if logits_sum is None else logits_sum + logits_aug
            logits_mean_model = logits_sum / float(n_augs)

            logits_ens_sum = (
                logits_mean_model
                if logits_ens_sum is None
                else logits_ens_sum + logits_mean_model
            )

        logits_ens = logits_ens_sum / float(len(models_list))

        probs = torch.sigmoid(logits_ens)
        preds = torch.sum(probs >= tau, dim=1).long()

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    return y_true, y_pred


# =========================
# THRESHOLD (TAU) TUNING
# =========================

def tune_tau_from_val_logits(val_logits: np.ndarray, val_labels: np.ndarray):
    """
    Grid search a single global tau in [0.30, 0.70] maximizing macro F1.
    val_logits: [N, K-1]
    val_labels: [N]
    """
    best_tau = 0.5
    best_f1 = -1.0

    # sigmoid on logits
    probs = 1.0 / (1.0 + np.exp(-val_logits))  # shape [N, K-1]

    for tau in np.linspace(0.30, 0.70, 41):  # step 0.01
        preds = np.sum(probs >= tau, axis=1)
        f1 = f1_score(val_labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)

    return best_tau, best_f1


# =========================
# MAIN TRAINING + ENSEMBLE
# =========================

def run_vit_coral_training_and_ensemble():
    set_seed(SEED)
    print("Using device:", DEVICE)

    if not os.path.isdir(DATA_ROOT):
        raise RuntimeError(f"DATA_ROOT not found: {DATA_ROOT}")

    samples, labels_all = load_trainval_samples_and_labels(DATA_ROOT)
    labels_all = np.asarray(labels_all, dtype=np.int64)

    test_ds = load_test_dataset(DATA_ROOT)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    pos_weight = compute_pos_weight(labels_all, NUM_CLASSES).to(DEVICE)
    print("pos_weight thresholds (y>0..3):", pos_weight.tolist())

    # Upweight G1 and G2
    cls_weight_vec = torch.tensor(
        [1.0, 1.5, 1.5, 1.0, 1.0], dtype=torch.float32, device=DEVICE
    )

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

    indices = np.arange(len(samples))
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    all_metrics_rows = []
    best_overall_val_acc = -1.0
    best_overall_state = None
    best_fold_idx = None

    # ---------- K-fold training ----------
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels_all), start=1):
        print(f"\n========== FOLD {fold_idx}/{NUM_FOLDS} ==========")

        uniq_tr, cnt_tr = np.unique(labels_all[train_idx], return_counts=True)
        print("Train class counts:", dict(zip(uniq_tr.tolist(), cnt_tr.tolist())))
        uniq_val, cnt_val = np.unique(labels_all[val_idx], return_counts=True)
        print("Val class counts:", dict(zip(uniq_val.tolist(), cnt_val.tolist())))

        train_loader = make_loader(train_ds, train_idx, shuffle=True)
        val_loader = make_loader(val_ds, val_idx, shuffle=False)

        model = ViT_CORAL(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)

        if FEATURE_EXTRACT:
            for p in model.backbone.parameters():
                p.requires_grad = False

        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params_to_update, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        writer = SummaryWriter(log_dir=f"runs/ViT_B16_CORAL_kneeKL224/fold_{fold_idx}")

        best_val_acc = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"Fold {fold_idx} | Epoch {epoch}/{NUM_EPOCHS}")

            train_loss = train_one_epoch(
                model, train_loader, optimizer,
                pos_weight, cls_weight_vec, DEVICE, NUM_CLASSES,
                writer, epoch, fold_idx
            )
            val_loss, val_acc, y_val, y_val_pred = evaluate(
                model, val_loader, DEVICE, NUM_CLASSES,
                pos_weight, writer, epoch, fold_idx, split_name="val", tau=0.5
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

        # Load best state for this fold
        model.load_state_dict(best_state["model"])
        val_loss, val_acc, y_val, y_val_pred = evaluate(
            model, val_loader, DEVICE, NUM_CLASSES, pos_weight, tau=0.5
        )

        # Save fold checkpoint (for tuning + ensemble later)
        fold_ckpt_path = f"ViT_B16_CORAL_fold{fold_idx}_best.pth"
        torch.save(best_state, fold_ckpt_path)
        print(f"Saved best model for fold {fold_idx} to {fold_ckpt_path}")

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

    # ---------- Tau tuning using validation TTA logits ----------
    print("\nCollecting validation logits with TTA for tau tuning...")

    # rebuild skf to get same splits
    skf_val = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    val_logits_all_list = []
    val_labels_all_list = []

    for fold_idx, (_, val_idx) in enumerate(skf_val.split(indices, labels_all), start=1):
        ckpt_path = f"ViT_B16_CORAL_fold{fold_idx}_best.pth"
        print(f"Loading fold {fold_idx} model for tau tuning from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model = ViT_CORAL(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
        model.load_state_dict(ckpt["model"])

        val_loader = make_loader(val_ds, val_idx, shuffle=False)
        logits_fold, labels_fold = collect_logits_tta(model, val_loader, DEVICE, n_augs=4)
        val_logits_all_list.append(logits_fold)
        val_labels_all_list.append(labels_fold)

    val_logits_all = np.concatenate(val_logits_all_list)
    val_labels_all = np.concatenate(val_labels_all_list)

    best_tau, best_f1 = tune_tau_from_val_logits(val_logits_all, val_labels_all)
    print(f"\nBest tau from validation tuning: {best_tau:.3f} (macro F1={best_f1:.4f})")

    # ---------- Test with best fold model + TTA + tuned tau ----------
    best_model = ViT_CORAL(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    best_model.load_state_dict(best_overall_state["model"])

    test_loss, test_acc, y_test, y_test_pred = evaluate_tta_single_model(
        best_model, test_loader, DEVICE, NUM_CLASSES, pos_weight, tau=best_tau, n_augs=4
    )
    test_metrics = compute_metrics(y_test, y_test_pred, NUM_CLASSES)
    test_metrics["fold"] = best_fold_idx
    test_metrics["split"] = "test_TTA_single_tau"
    test_metrics["val_loss"] = float(test_loss)
    all_metrics_rows.append(test_metrics)

    print("\n===== TEST RESULTS (best fold model, with TTA, tuned tau) =====")
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

    df_metrics = pd.DataFrame(all_metrics_rows)
    df_metrics.to_csv(METRICS_CSV, index=False)
    print(f"\nAll fold + single-model test metrics saved to {METRICS_CSV}")

    df_pred = pd.DataFrame({
        "true_label": y_test,
        "predicted_label": y_test_pred,
    })
    df_pred.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Single best-fold test predictions saved to {PREDICTIONS_CSV}")

    # ---------- ENSEMBLE (5 folds) + TTA + tuned tau ----------
    print("\n===== ENSEMBLE EVALUATION (5-fold, with TTA, tuned tau) =====")

    models_list = []
    for fold_idx in range(1, NUM_FOLDS + 1):
        ckpt_path = f"ViT_B16_CORAL_fold{fold_idx}_best.pth"
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        m = ViT_CORAL(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
        m.load_state_dict(ckpt["model"])
        models_list.append(m)

    y_true_ens, y_pred_ens = ensemble_evaluate_tta(
        models_list, test_loader, DEVICE, NUM_CLASSES, tau=best_tau, n_augs=4
    )

    ens_metrics = compute_metrics(y_true_ens, y_pred_ens, NUM_CLASSES)
    ens_metrics["fold"] = -1           # indicates ensemble
    ens_metrics["split"] = "test_TTA_ensemble_tau"
    ens_metrics["val_loss"] = 0.0
    print(f"\nEnsemble test accuracy (tuned tau): {ens_metrics['accuracy']:.4f}")
    print(f"Ensemble macro F1 (tuned tau): {ens_metrics['f1_macro']:.4f}")
    print("\nEnsemble confusion matrix:")
    print(confusion_matrix(y_true_ens, y_pred_ens))
    print("\nEnsemble classification report:")
    print(classification_report(y_true_ens, y_pred_ens, digits=4, zero_division=0))

    # append ensemble row to metrics CSV
    df_metrics = pd.read_csv(METRICS_CSV)
    df_metrics = pd.concat([df_metrics, pd.DataFrame([ens_metrics])], ignore_index=True)
    df_metrics.to_csv(METRICS_CSV, index=False)
    print(f"\nUpdated metrics (including ensemble) saved to {METRICS_CSV}")

    # save ensemble predictions
    df_pred_ens = pd.DataFrame({
        "true_label": y_true_ens,
        "predicted_label": y_pred_ens,
    })
    df_pred_ens.to_csv(ENSEMBLE_PREDS_CSV, index=False)
    print(f"Ensemble test predictions saved to {ENSEMBLE_PREDS_CSV}")


if __name__ == "__main__":
    run_vit_coral_training_and_ensemble()
