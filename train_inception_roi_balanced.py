"""
Reproduction-oriented script for:

  "Knee osteoarthritis severity detection using deep inception transfer learning"

Features:
  - Automatic knee ROI extraction from X-ray.
  - CLAHE + resize preprocessing.
  - Strong data augmentation for training (rotation, zoom, translation, flip, etc.).
  - Class balancing as in Table 1 (each class 0..4 -> 1652 samples in train).
  - Inception V3 transfer learning (IV3 or FT-IV3).
  - Full test evaluation: Accuracy, Precision, Sensitivity, F1, Specificity, AUROC.

Directory layout expected:

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

Balanced training set will be created at:

      train_balanced/
          0/ ... 1652 images
          1/ ... 1652 images
          2/ ... 1652 images
          3/ ... 1652 images
          4/ ... 1652 images
"""

import os
import random
import time
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

DATA_ROOT = r"E:\Knee-OsteoArthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224"

ORIG_TRAIN_DIR = os.path.join(DATA_ROOT, "train")
BAL_TRAIN_DIR  = os.path.join(DATA_ROOT, "train_balanced")
VAL_DIR        = os.path.join(DATA_ROOT, "val")
TEST_DIR       = os.path.join(DATA_ROOT, "test")

NUM_CLASSES = 5
TARGET_PER_CLASS = 1652    # from Table 1
BATCH_SIZE = 32
EPOCHS = 70

# MODE:
#   "IV3"    -> feature extraction (only final classifier layers trained), LR = 5e-4
#   "FT_IV3" -> full fine-tuning, LR = 1e-4
MODE = "IV3"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------------------------------------------------------
# UTILS: DIR HANDLING
# -----------------------------------------------------------------------------

def make_dir_clean(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# STEP 1: TRAIN BALANCING (AS IN TABLE 1)
# -----------------------------------------------------------------------------

def balance_train_set(orig_train_dir, bal_train_dir, target_per_class=1652):
    """
    Create a balanced training set:

        - If class has > target, randomly downsample.
        - If class has < target, copy all originals and create augmented images
          (horizontal flip + zoom + translation) to reach target.
    """
    orig_train_dir = Path(orig_train_dir)
    bal_train_dir  = Path(bal_train_dir)

    make_dir_clean(bal_train_dir)

    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
    ])

    print("\n=== Balancing training set (Table 1) ===")

    for class_idx in range(NUM_CLASSES):
        class_name = str(class_idx)
        src_class_dir = orig_train_dir / class_name
        dst_class_dir = bal_train_dir / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        img_paths = sorted([
            p for p in src_class_dir.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]
        ])
        orig_count = len(img_paths)
        target = target_per_class

        print(f"Class {class_name}: original = {orig_count}, target = {target}")

        if orig_count == 0:
            raise RuntimeError(f"No images found in {src_class_dir}")

        # DOWN-SAMPLE
        if orig_count > target:
            chosen = random.sample(img_paths, target)
            for p in chosen:
                shutil.copy2(p, dst_class_dir / p.name)
            print(f"  Down-sampled -> kept {target}, removed {orig_count - target}")

        # UP-SAMPLE
        elif orig_count < target:
            # copy originals
            for p in img_paths:
                shutil.copy2(p, dst_class_dir / p.name)

            needed = target - orig_count
            print(f"  Up-sampling -> need {needed} augmented images")

            for i in range(needed):
                src = img_paths[i % orig_count]
                img = Image.open(src).convert("RGB")
                aug_img = aug_transform(img)
                new_name = f"{src.stem}_aug{i:04d}{src.suffix}"
                aug_img.save(dst_class_dir / new_name)

            print(f"  Generated {needed} augmented images")

        # ALREADY EXACTLY TARGET
        else:
            for p in img_paths:
                shutil.copy2(p, dst_class_dir / p.name)
            print("  Already at target, just copied.")

    print("Balanced training set created at:", bal_train_dir)


# -----------------------------------------------------------------------------
# STEP 2: ROI EXTRACTION + PREPROCESSING
# -----------------------------------------------------------------------------

def extract_knee_roi(gray: np.ndarray) -> np.ndarray:
    """
    Automatic knee joint ROI extraction.

    Strategy:
    - Input: 2D grayscale image.
    - Smooth, detect edges via Canny.
    - Sum edges along rows, pick the row with highest edge density
      (joint line around tibiofemoral space).
    - Crop a window around that row + center 60% of width.
    - Fallback to central crop if something goes wrong.
    """
    h, w = gray.shape
    try:
        # smooth & edge detection
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny((blur * 255).astype(np.uint8), 50, 150)

        # row-wise edge density
        row_sum = edges.sum(axis=1)
        joint_row = int(np.argmax(row_sum))

        # vertical extent around joint
        roi_half_h = int(0.25 * h)
        top = max(joint_row - roi_half_h, 0)
        bottom = min(joint_row + roi_half_h, h)

        # horizontal center crop
        center_x = w // 2
        roi_half_w = int(0.3 * w)
        left = max(center_x - roi_half_w, 0)
        right = min(center_x + roi_half_w, w)

        roi = gray[top:bottom, left:right]

        # ensure reasonable size
        if roi.shape[0] < h // 4 or roi.shape[1] < w // 4:
            raise ValueError("ROI too small, fallback.")

    except Exception:
        # Fallback: central 60% x 60% crop
        crop_h = int(0.6 * h)
        crop_w = int(0.6 * w)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        roi = gray[top:top + crop_h, left:left + crop_w]

    return roi


def preprocess_xray(pil_img: Image.Image, out_size=(299, 299)) -> Image.Image:
    """
    Preprocessing used before torchvision transforms:
      - Convert to grayscale.
      - CLAHE contrast enhancement.
      - Automatic ROI extraction.
      - Resize to Inception input size.
      - Convert back to 3-channel PIL image.
    """
    img_np = np.array(pil_img)

    # to grayscale
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    gray = gray.astype(np.float32) / 255.0

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_uint8 = (gray * 255).astype(np.uint8)
    gray_eq = clahe.apply(gray_uint8)

    # ROI extraction
    roi = extract_knee_roi(gray_eq)

    # resize to target
    roi_resized = cv2.resize(roi, out_size, interpolation=cv2.INTER_CUBIC)

    # to 3-channel RGB
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(roi_rgb)


# -----------------------------------------------------------------------------
# STEP 3: CUSTOM DATASET USING PREPROCESSING + AUGMENTATION
# -----------------------------------------------------------------------------

class KneeOADataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        self.samples = []
        for class_idx in range(NUM_CLASSES):
            class_name = str(class_idx)
            class_dir = self.root_dir / class_name
            for p in class_dir.iterdir():
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
                    self.samples.append((str(p), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        # preprocessing: ROI + CLAHE + resize 299x299
        img = preprocess_xray(img, out_size=(299, 299))

        # augmentation / tensor / normalisation
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label).long()


# -----------------------------------------------------------------------------
# STEP 4: DATALOADERS WITH STRONG AUGMENTATION
# -----------------------------------------------------------------------------

def get_dataloaders():
    input_size = 299

    # train: strong augmentations
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # val / test: no augmentation, just tensor + norm
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = KneeOADataset(BAL_TRAIN_DIR, split="train",
                                  transform=train_transform)
    val_dataset   = KneeOADataset(VAL_DIR, split="val",
                                  transform=eval_transform)
    test_dataset  = KneeOADataset(TEST_DIR, split="test",
                                  transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    dataset_sizes = {
        "train": len(train_dataset),
        "val":   len(val_dataset),
    }

    print("\nDataset sizes:")
    print("  Balanced train:", dataset_sizes["train"])
    print("  Val:           ", dataset_sizes["val"])
    print("  Test:          ", len(test_dataset))

    return train_loader, val_loader, test_loader, dataset_sizes


# -----------------------------------------------------------------------------
# STEP 5: INCEPTION V3 (IV3 / FT-IV3)
# -----------------------------------------------------------------------------

def build_inception_v3(num_classes, mode="IV3"):
    model = models.inception_v3(pretrained=True, aux_logits=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if model.aux_logits:
        in_features_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)

    if mode == "IV3":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        if model.aux_logits:
            for p in model.AuxLogits.parameters():
                p.requires_grad = True
    elif mode == "FT_IV3":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError("mode must be 'IV3' or 'FT_IV3'")

    return model


# -----------------------------------------------------------------------------
# STEP 6: TRAINING LOOP
# -----------------------------------------------------------------------------

def train_model(model, dataloaders, dataset_sizes,
                criterion, optimizer, num_epochs=25, device="cpu"):

    since = time.time()
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    if phase == "train" and model.training and model.aux_logits:
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase:5s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed / 60:.1f}m")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


# -----------------------------------------------------------------------------
# STEP 7: TEST EVALUATION
# -----------------------------------------------------------------------------

def evaluate_on_test(model, test_loader, device="cpu"):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = softmax(outputs)
            _, preds = torch.max(probs, 1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_score = np.concatenate(all_probs)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    spec_per_class = []
    for i in range(NUM_CLASSES):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        denom = TN + FP
        spec = TN / denom if denom > 0 else 0.0
        spec_per_class.append(spec)
    specificity = float(np.mean(spec_per_class))

    try:
        auroc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")

    print("\n===== TEST METRICS =====")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Precision:    {precision:.4f} (macro)")
    print(f"Sensitivity:  {recall:.4f} (macro recall)")
    print(f"F1-score:     {f1:.4f} (macro)")
    print(f"Specificity:  {specificity:.4f} (macro)")
    print(f"AUROC (OVR):  {auroc:.4f}")
    print("========================\n")

    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "specificity_macro": specificity,
        "auroc_macro_ovr": auroc,
        "confusion_matrix": cm,
    }


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 1) Balance training set as in Table 1
    balance_train_set(ORIG_TRAIN_DIR, BAL_TRAIN_DIR, TARGET_PER_CLASS)

    # 2) Build dataloaders
    train_loader, val_loader, test_loader, dataset_sizes = get_dataloaders()
    loaders = {"train": train_loader, "val": val_loader}

    # 3) Build model + optimizer with paper hyperparameters
    model = build_inception_v3(NUM_CLASSES, mode=MODE).to(device)
    criterion = nn.CrossEntropyLoss()

    if MODE == "IV3":
        lr = 5e-4
    else:
        lr = 1e-4

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=lr)

    print(f"\nMODE: {MODE}, LR: {lr}, trainable params: {len(params_to_update)}")

    # 4) Train
    model = train_model(
        model,
        loaders,
        dataset_sizes,
        criterion,
        optimizer,
        num_epochs=EPOCHS,
        device=device,
    )

    # 5) Save best model
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"inceptionv3_{MODE}_roi_balanced.pth"
    torch.save(model.state_dict(), ckpt_path)
    print("Saved best model to:", ckpt_path)

    # 6) Test evaluation
    _ = evaluate_on_test(model, test_loader, device=device)
