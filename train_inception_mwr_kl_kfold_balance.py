"""
InceptionV3 + Moving Window Regression (MWR) for KL-grade (0..4)
WITHOUT any data balancing.

- Uses natural train/val distributions.
- Stratified K-Fold on train+val combined.
- Test set is kept separate and untouched.
- Early stopping, LR scheduler, TensorBoard logging.
- Saves metrics and test predictions to CSV.

Folder structure:

DATA_ROOT/
    train/
        0/
        1/
        2/
        3/
        4/
    val/
        0/
        1/
        2/
        3/
        4/
    test/
        0/
        1/
        2/
        3/
        4/
"""

import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

from torch.utils.tensorboard import SummaryWriter
from PIL import Image


# =========================
# CONFIGURATION
# =========================
DATA_ROOT = r"E:\Knee-Osteoarthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224"  # <-- change

NUM_CLASSES = 5
NUM_FOLDS = 5

BATCH_SIZE = 16
NUM_EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

EARLY_STOP_PATIENCE = 7          # epochs with no val improvement
FEATURE_EXTRACT = False          # if True: freeze backbone

SEED = 42

METRICS_CSV = "MWR_InceptionV3_kfold_results_NO_BALANCING.csv"
PREDICTIONS_CSV = "MWR_InceptionV3_bestFold_test_predictions_NO_BALANCING.csv"
BEST_MODEL_PATH = "MWR_InceptionV3_bestFold_NO_BALANCING.pth"


# =========================
# UTILS
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


# =========================
# DATASETS
# =========================
class PathLabelDataset(Dataset):
    """Dataset built from (path, label) pairs plus transform."""

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
    """Read file paths + labels for train and val and merge them."""
    train_folder = datasets.ImageFolder(os.path.join(root, "train"))
    val_folder = datasets.ImageFolder(os.path.join(root, "val"))

    train_samples = train_folder.samples
    val_samples = val_folder.samples
    samples = train_samples + val_samples

    train_targets = train_folder.targets
    val_targets = val_folder.targets
    labels = np.array(train_targets + val_targets, dtype=np.int64)

    return samples, labels


def load_test_dataset(root: str):
    test_tf = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(os.path.join(root, "test"),
                                   transform=test_tf)
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
# MODEL: MWR HEAD + InceptionV3
# =========================
class MWRHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.num_classes = num_classes
        num_windows = num_classes - 1

        self.global_regressor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )

        self.local_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_windows)
        ])

    def forward(self, feats):
        B = feats.size(0)
        g_raw = self.global_regressor(feats).view(B)
        g_norm = torch.sigmoid(g_raw)
        g_cont = g_norm * (self.num_classes - 1)

        local_logits_list = []
        for head in self.local_heads:
            ll = head(feats).view(B)
            local_logits_list.append(ll)
        local_logits = torch.stack(local_logits_list, dim=1)  # [B, C-1]
        return g_norm, g_cont, local_logits

    @torch.no_grad()
    def predict_from_feats(self, feats):
        g_norm, g_cont, local_logits = self.forward(feats)
        B = feats.size(0)
        num_windows = self.num_classes - 1

        w_pred = torch.floor(g_cont).long()
        w_pred = torch.clamp(w_pred, 0, num_windows - 1)

        local_probs = torch.sigmoid(local_logits)
        idx = torch.arange(B, device=feats.device)
        p = local_probs[idx, w_pred]

        side = (p >= 0.5).long()
        grades = w_pred + side
        return grades


class InceptionV3_MWR(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        # compatible with your older torchvision: use `pretrained`, then disable aux logits
        inception = models.inception_v3(pretrained=pretrained)
        inception.aux_logits = False

        in_features = inception.fc.in_features
        inception.fc = nn.Identity()

        self.backbone = inception
        self.head = MWRHead(in_dim=in_features, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    @torch.no_grad()
    def predict(self, x):
        feats = self.backbone(x)
        return self.head.predict_from_feats(feats)


def set_parameter_requires_grad(model, feature_extracting: bool):
    if feature_extracting:
        for p in model.backbone.parameters():
            p.requires_grad = False


# =========================
# TRAIN / EVAL
# =========================
def train_one_epoch(
    model,
    loader,
    optimizer,
    mse_loss,
    bce_loss,
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
        g_norm, g_cont, local_logits_all = model(images)

        # global loss
        target_global = labels.float() / (num_classes - 1)
        loss_global = mse_loss(g_norm, target_global)

        # local window loss
        num_windows = num_classes - 1
        w_gt = labels.clamp(0, num_windows - 1)
        local_target = (labels > w_gt).float()  # 0 or 1

        idx = torch.arange(labels.size(0), device=device)
        local_logits = local_logits_all[idx, w_gt]
        loss_local = bce_loss(local_logits, local_target)

        loss = loss_global + loss_local
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
    writer=None,
    epoch_idx=None,
    fold_idx=None,
    split_name="val",
):
    model.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    bce_loss = nn.BCEWithLogitsLoss(reduction="sum")

    total = 0.0
    all_labels, all_preds = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        g_norm, g_cont, local_logits_all = model(images)

        target_global = labels.float() / (num_classes - 1)
        lg = mse_loss(g_norm, target_global)

        num_windows = num_classes - 1
        w_gt = labels.clamp(0, num_windows - 1)
        local_target = (labels > w_gt).float()
        idx = torch.arange(labels.size(0), device=device)
        local_logits = local_logits_all[idx, w_gt]
        ll = bce_loss(local_logits, local_target)

        total += (lg + ll).item()

        preds = model.predict(images)
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
# MAIN
# =========================
def main():
    set_seed(SEED)
    print("Using device:", DEVICE)

    # 1) Load combined train+val samples and labels, plus test dataset
    samples, labels = load_trainval_samples_and_labels(DATA_ROOT)
    labels = np.asarray(labels, dtype=np.int64)

    test_ds = load_test_dataset(DATA_ROOT)

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = PathLabelDataset(samples, transform=train_tf)
    val_ds = PathLabelDataset(samples, transform=val_tf)

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # 2) Stratified K-Fold over natural (unbalanced) data
    indices = np.arange(len(samples))
    skf = StratifiedKFold(
        n_splits=NUM_FOLDS, shuffle=True, random_state=SEED
    )

    all_metrics_rows = []
    best_overall_val_acc = -1.0
    best_overall_state = None
    best_fold_idx = None

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(indices, labels), start=1
    ):
        print(f"\n========== FOLD {fold_idx}/{NUM_FOLDS} ==========")

        # sanity check: class distribution
        print("Train class counts (natural):")
        uniq_tr, cnt_tr = np.unique(labels[train_idx], return_counts=True)
        print(dict(zip(uniq_tr.tolist(), cnt_tr.tolist())))

        print("Val class counts (natural):")
        uniq_val, cnt_val = np.unique(labels[val_idx], return_counts=True)
        print(dict(zip(uniq_val.tolist(), cnt_val.tolist())))

        train_loader = make_loader(train_ds, train_idx, shuffle=True)
        val_loader = make_loader(val_ds, val_idx, shuffle=False)

        # model / optimizer / scheduler
        model = InceptionV3_MWR(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
        set_parameter_requires_grad(model, FEATURE_EXTRACT)

        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params_to_update, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
        )

        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()

        writer = SummaryWriter(
            log_dir=f"runs/MWR_InceptionV3_NO_BALANCING/fold_{fold_idx}"
        )

        best_val_acc = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"Fold {fold_idx} | Epoch {epoch}/{NUM_EPOCHS}")

            train_loss = train_one_epoch(
                model, train_loader, optimizer, mse_loss, bce_loss,
                DEVICE, NUM_CLASSES, writer, epoch, fold_idx
            )
            val_loss, val_acc, y_val, y_val_pred = evaluate(
                model, val_loader, DEVICE, NUM_CLASSES,
                writer, epoch, fold_idx, split_name="val"
            )

            print(f"  train_loss: {train_loss:.4f} | "
                  f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

            scheduler.step(val_acc)

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
            model, val_loader, DEVICE, NUM_CLASSES
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

        if best_val_acc > best_overall_val_acc:
            best_overall_val_acc = best_val_acc
            best_overall_state = best_state
            best_fold_idx = fold_idx

    print(f"\nBest fold: {best_fold_idx} with val_acc={best_overall_val_acc:.4f}")
    torch.save(best_overall_state, BEST_MODEL_PATH)
    print(f"Best fold model saved to {BEST_MODEL_PATH}")

    # 3) Final test evaluation using best fold model
    best_model = InceptionV3_MWR(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    best_model.load_state_dict(best_overall_state["model"])

    test_loss, test_acc, y_test, y_test_pred = evaluate(
        best_model, test_loader, DEVICE, NUM_CLASSES
    )
    test_metrics = compute_metrics(y_test, y_test_pred, NUM_CLASSES)
    test_metrics["fold"] = best_fold_idx
    test_metrics["split"] = "test"
    test_metrics["val_loss"] = float(test_loss)
    all_metrics_rows.append(test_metrics)

    print("\n===== TEST RESULTS (best fold model) =====")
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

    # 4) Save metrics & predictions
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
    main()
