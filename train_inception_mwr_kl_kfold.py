import os
import random
from typing import Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import KFold

from torch.utils.tensorboard import SummaryWriter


# =========================
# CONFIGURATION
# =========================
DATA_ROOT = r"E:\Knee-OsteoArthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL299"   # <<< CHANGE THIS

# Original KL labels on disk: 0,1,2,3,4
RAW_NUM_CLASSES = 5

# If True, merge KL-0 and KL-1 into one class:
#   0,1 -> 0; 2->1; 3->2; 4->3  (total 4 classes)
MERGE_KL01 = False  # set True if you want 0/1 merge

# K-fold CV
NUM_FOLDS = 5

BATCH_SIZE = 16
NUM_EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

# Early stopping
EARLY_STOP_PATIENCE = 7  # epochs without val improvement

# Feature extraction mode (freeze backbone if True)
FEATURE_EXTRACT = False

# Random seed
SEED = 42

# Output files
METRICS_CSV = "MWR_InceptionV3_kfold_results.csv"
PREDICTIONS_CSV = "MWR_InceptionV3_bestFold_test_predictions.csv"
BEST_MODEL_PATH = "MWR_InceptionV3_bestFold.pth"

# =========================
# DERIVED CONFIG
# =========================
if MERGE_KL01:
    NUM_CLASSES = 4
else:
    NUM_CLASSES = RAW_NUM_CLASSES


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


def get_label_map(merge_01: bool) -> Dict[int, int]:
    if not merge_01:
        return {i: i for i in range(RAW_NUM_CLASSES)}
    # 0,1 -> 0; 2->1; 3->2; 4->3
    return {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}


def specificity_score(y_true, y_pred, num_classes, average="macro"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    TN = []
    FP = []
    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm[i, :], i))
        TN.append(tn)
        FP.append(fp)
    spec = np.array(TN) / (np.array(TN) + np.array(FP) + 1e-8)
    if average == "macro":
        return float(np.mean(spec))
    else:
        return spec


class LabelMapDataset(Dataset):
    """Wrap another dataset and remap label indices."""
    def __init__(self, base_ds: Dataset, label_map: Dict[int, int]):
        self.base_ds = base_ds
        self.label_map = label_map

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        return img, self.label_map[int(label)]


# =========================
# DATA LOADERS
# =========================
def build_datasets(data_root: str):
    label_map = get_label_map(MERGE_KL01)

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    train_base = datasets.ImageFolder(train_dir, transform=train_tf)
    val_base = datasets.ImageFolder(val_dir, transform=train_tf)
    test_base = datasets.ImageFolder(test_dir, transform=eval_tf)

    train_ds = LabelMapDataset(train_base, label_map)
    val_ds = LabelMapDataset(val_base, label_map)
    test_ds = LabelMapDataset(test_base, label_map)

    trainval_ds = ConcatDataset([train_ds, val_ds])

    return trainval_ds, test_ds


def make_loader(ds: Dataset, indices, shuffle, batch_size=BATCH_SIZE):
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=NUM_WORKERS, pin_memory=True)
    return loader


# =========================
# MWR HEAD & MODEL
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
            nn.Linear(hidden_dim, 1)
        )

        self.local_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(num_windows)
        ])

    def forward(self, feats):
        B = feats.size(0)
        g_raw = self.global_regressor(feats).view(B)
        g_norm = torch.sigmoid(g_raw)
        g_cont = g_norm * (self.num_classes - 1)

        local_logits = []
        for head in self.local_heads:
            ll = head(feats).view(B)
            local_logits.append(ll)
        local_logits = torch.stack(local_logits, dim=1)  # [B, C-1]
        return g_norm, g_cont, local_logits

    @torch.no_grad()
    def predict_from_feats(self, feats):
        g_norm, g_cont, local_logits = self.forward(feats)
        B = feats.size(0)
        num_windows = self.num_classes - 1

        w_pred = torch.floor(g_cont).long()
        w_pred = torch.clamp(w_pred, 0, num_windows - 1)

        local_probs_all = torch.sigmoid(local_logits)
        idx = torch.arange(B, device=feats.device)
        local_prob = local_probs_all[idx, w_pred]

        side = (local_prob >= 0.5).long()
        grade_pred = w_pred + side
        return grade_pred


class InceptionV3_MWR(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()

        # Compatible with older torchvision versions
        inception = models.inception_v3(pretrained=pretrained)
        inception.aux_logits = False  # disable auxiliary classifier manually

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
def train_one_epoch(model, loader, optimizer, mse_loss, bce_loss,
                    device, num_classes, writer=None, epoch_idx=None):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        g_norm, g_cont, local_logits_all = model(images)

        target_global = labels.float() / (num_classes - 1)
        loss_global = mse_loss(g_norm, target_global)

        num_windows = num_classes - 1
        w_gt = labels.clamp(0, num_windows - 1)
        local_target = (labels > w_gt).float()

        idx = torch.arange(labels.size(0), device=device)
        local_logits = local_logits_all[idx, w_gt]
        loss_local = bce_loss(local_logits, local_target)

        loss = loss_global + loss_local
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    if writer is not None and epoch_idx is not None:
        writer.add_scalar("Loss/train", epoch_loss, epoch_idx)
    return epoch_loss


@torch.no_grad()
def evaluate(model, loader, device, num_classes, writer=None, epoch_idx=None,
             split_name="val"):
    model.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    bce_loss = nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    all_labels = []
    all_preds = []

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

        total_loss += (lg + ll).item()

        preds = model.predict(images)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    avg_loss = total_loss / len(loader.dataset)
    acc = float(np.mean(y_true == y_pred))

    if writer is not None and epoch_idx is not None:
        writer.add_scalar(f"Loss/{split_name}", avg_loss, epoch_idx)
        writer.add_scalar(f"Acc/{split_name}", acc, epoch_idx)

    return avg_loss, acc, y_true, y_pred


def compute_metrics(y_true, y_pred, num_classes):
    metrics = {}
    metrics["accuracy"] = float(np.mean(y_true == y_pred))
    metrics["precision_macro"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["recall_macro"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["f1_macro"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["specificity_macro"] = specificity_score(y_true, y_pred, num_classes)

    # One-hot encoding of labels & preds for simple macro AUROC
    y_true_ovr = np.eye(num_classes)[y_true]
    y_pred_ovr = np.eye(num_classes)[y_pred]
    try:
        metrics["auroc_macro"] = float(
            roc_auc_score(y_true_ovr, y_pred_ovr, average="macro")
        )
    except Exception:
        metrics["auroc_macro"] = 0.0

    return metrics


# =========================
# MAIN WITH K-FOLD + TEST
# =========================
def main():
    set_seed(SEED)
    print("Device:", DEVICE)
    print("NUM_CLASSES (after mapping):", NUM_CLASSES)

    # Datasets
    trainval_ds, test_ds = build_datasets(DATA_ROOT)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    all_metrics_rows = []
    best_overall_val_acc = -1.0
    best_overall_state = None
    best_fold_idx = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(trainval_ds))), start=1):
        print(f"\n========== FOLD {fold}/{NUM_FOLDS} ==========")

        train_loader = make_loader(trainval_ds, train_idx, shuffle=True)
        val_loader = make_loader(trainval_ds, val_idx, shuffle=False)

        model = InceptionV3_MWR(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
        set_parameter_requires_grad(model, FEATURE_EXTRACT)

        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params_to_update, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
        )
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()

        writer = SummaryWriter(log_dir=f"runs/MWR_InceptionV3_KL/fold_{fold}")

        best_val_acc = -1.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"Fold {fold} | Epoch {epoch}/{NUM_EPOCHS}")
            train_loss = train_one_epoch(
                model, train_loader, optimizer, mse_loss, bce_loss,
                DEVICE, NUM_CLASSES, writer, epoch
            )
            val_loss, val_acc, y_val, y_val_pred = evaluate(
                model, val_loader, DEVICE, NUM_CLASSES,
                writer, epoch, split_name="val"
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
                    "fold": fold,
                }
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping on fold {fold} at epoch {epoch}")
                break

        writer.close()

        # Evaluate best model of this fold on its validation set
        if best_state is None:
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "fold": fold,
            }

        model.load_state_dict(best_state["model"])
        val_loss, val_acc, y_val, y_val_pred = evaluate(
            model, val_loader, DEVICE, NUM_CLASSES
        )
        fold_metrics = compute_metrics(y_val, y_val_pred, NUM_CLASSES)
        fold_metrics["fold"] = fold
        fold_metrics["split"] = "val"
        fold_metrics["val_loss"] = float(val_loss)
        all_metrics_rows.append(fold_metrics)

        print("\nFold", fold, "validation confusion matrix:")
        print(confusion_matrix(y_val, y_val_pred))
        print("\nFold", fold, "validation classification report:")
        print(classification_report(y_val, y_val_pred, digits=4))

        # Track best fold across all folds
        if best_val_acc > best_overall_val_acc:
            best_overall_val_acc = best_val_acc
            best_overall_state = best_state
            best_fold_idx = fold

    print(f"\nBest fold: {best_fold_idx} with val_acc={best_overall_val_acc:.4f}")
    torch.save(best_overall_state, BEST_MODEL_PATH)
    print(f"Best fold model saved to {BEST_MODEL_PATH}")

    # =========================
    # FINAL TEST EVALUATION (best fold model)
    # =========================
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
    print(classification_report(y_test, y_test_pred, digits=4))

    # =========================
    # SAVE METRICS & PREDICTIONS
    # =========================
    df_metrics = pd.DataFrame(all_metrics_rows)
    df_metrics.to_csv(METRICS_CSV, index=False)
    print(f"\nAll fold + test metrics saved to {METRICS_CSV}")

    df_pred = pd.DataFrame({
        "true_label": y_test,
        "predicted_label": y_test_pred
    })
    df_pred.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Test predictions saved to {PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
