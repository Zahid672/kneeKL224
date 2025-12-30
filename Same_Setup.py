"""
Reproduce baseline of:
  "Knee osteoarthritis severity detection using deep inception transfer learning"

Steps:
1) Balance the TRAIN set according to Table 1 (G0–G4→1652 each):
       Class  Original  Down  Up   Balanced
       G0(0)    3251    1599  -     1652
       G1(1)    1495      -   157   1652
       G2(2)    2175     523  -     1652
       G3(3)    1086      -   566   1652
       G4(4)     251      -   1401  1652

2) Train Inception V3 with transfer learning on the balanced train set.

3) Evaluate on the (unbalanced) test set: accuracy, precision, recall (sensitivity),
   F1, specificity, AUROC.
"""

import os
import random
import time
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)


# -------------------------------------------------------------------------
# CONFIG (edit these for your environment)
# -------------------------------------------------------------------------

DATA_ROOT = r"E:\\Knee-OsteoArthritis-severity-detection\\KneeXrayData\\KneeXrayData\\ClsKLData\\kneeKL224"

# Original train dir and new balanced train dir
ORIG_TRAIN_DIR = os.path.join(DATA_ROOT, "train")
BAL_TRAIN_DIR  = os.path.join(DATA_ROOT, "train_balanced")

VAL_DIR  = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")

NUM_CLASSES = 5
TARGET_PER_CLASS = 1652               # from Table 1
BATCH_SIZE = 32
EPOCHS = 70

# "IV3"    -> feature extraction (only classifier trained), LR = 5e-4
# "FT_IV3" -> full fine-tuning, LR = 1e-4
MODE = "IV3"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# -------------------------------------------------------------------------
# STEP 1: DATA BALANCING (Train only)
# -------------------------------------------------------------------------

def make_dir_clean(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def balance_train_set(orig_train_dir, bal_train_dir, target_per_class=1652):
    """
    Create a balanced training set according to Table 1.
    - For classes with > target: randomly downsample.
    - For classes with < target: keep all originals and generate
      (target - original) augmented images using H-flip + zoom + translation.
    """

    orig_train_dir = Path(orig_train_dir)
    bal_train_dir  = Path(bal_train_dir)
    make_dir_clean(bal_train_dir)

    # augmentation transforms (for up-sampling)
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),            # always flip
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),                          # up to 10% translation
            scale=(0.9, 1.1),                              # zoom 0.9–1.1
        ),
    ])

    print("Balancing training set according to Table 1...\n")

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

        # -------------------- DOWN-SAMPLING --------------------
        if orig_count > target:
            # random subset of original images
            chosen = random.sample(img_paths, target)
            for p in chosen:
                shutil.copy2(p, dst_class_dir / p.name)
            print(f"  Down-sampled: kept {target}, removed {orig_count - target}")

        # -------------------- UP-SAMPLING ----------------------
        elif orig_count < target:
            # copy all originals first
            for p in img_paths:
                shutil.copy2(p, dst_class_dir / p.name)

            needed = target - orig_count
            print(f"  Up-sampling: need {needed} augmented images")

            if orig_count == 0:
                raise RuntimeError(f"No images found in {src_class_dir}")

            # generate new augmented images
            for i in range(needed):
                src = img_paths[i % orig_count]
                img = Image.open(src).convert("RGB")
                aug_img = aug_transform(img)

                new_name = f"{src.stem}_aug{i:04d}{src.suffix}"
                aug_img.save(dst_class_dir / new_name)

            print(f"  Generated {needed} augmented images")

        # -------------------- ALREADY BALANCED -----------------
        else:
            # simply copy all
            for p in img_paths:
                shutil.copy2(p, dst_class_dir / p.name)
            print(f"  Already at target, copied {orig_count}")

    print("\nBalanced training set created at:", bal_train_dir)


# -------------------------------------------------------------------------
# STEP 2: DATALOADERS
# -------------------------------------------------------------------------

def get_dataloaders():
    # Inception v3 typical input: 299x299
    input_size = 299

    train_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=BAL_TRAIN_DIR, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(root=VAL_DIR,   transform=eval_transforms)
    test_dataset  = datasets.ImageFolder(root=TEST_DIR,  transform=eval_transforms)

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
    print(f"  Balanced train: {dataset_sizes['train']}")
    print(f"  Val:            {dataset_sizes['val']}")
    print(f"  Test:           {len(test_dataset)}")
    print(f"  Classes (train): {train_dataset.classes}")

    return train_loader, val_loader, test_loader, dataset_sizes


# -------------------------------------------------------------------------
# STEP 3: INCEPTION V3 MODEL (IV3 / FT-IV3)
# -------------------------------------------------------------------------

def build_inception_v3(num_classes, mode="IV3"):
    model = models.inception_v3(pretrained=True, aux_logits=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if model.aux_logits:
        in_features_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)

    if mode == "IV3":
        # feature extraction: freeze all except new classifiers
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        if model.aux_logits:
            for p in model.AuxLogits.parameters():
                p.requires_grad = True
    elif mode == "FT_IV3":
        # full fine-tuning
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError("mode must be 'IV3' or 'FT_IV3'")

    return model


# -------------------------------------------------------------------------
# STEP 4: TRAINING
# -------------------------------------------------------------------------

def train_model(model, dataloaders, dataset_sizes,
                criterion, optimizer, num_epochs=25, device="cpu"):

    since = time.time()
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
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
                    # Inception v3: (main_output, aux_output) in train mode
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

                running_loss   += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase:5s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed/60:.1f}m")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


# -------------------------------------------------------------------------
# STEP 5: TEST EVALUATION
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) build balanced training set as in Table 1
    balance_train_set(ORIG_TRAIN_DIR, BAL_TRAIN_DIR, TARGET_PER_CLASS)

    # 2) dataloaders
    train_loader, val_loader, test_loader, dataset_sizes = get_dataloaders()
    loaders = {"train": train_loader, "val": val_loader}

    # 3) model + optimizer with paper hyperparameters
    model = build_inception_v3(NUM_CLASSES, mode=MODE).to(device)
    criterion = nn.CrossEntropyLoss()
    if MODE == "IV3":
        lr = 5e-4
    else:
        lr = 1e-4

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=lr)

    print(f"\nMODE: {MODE}, LR: {lr}, trainable params: {len(params_to_update)}")

    # 4) train
    model = train_model(
        model,
        loaders,
        dataset_sizes,
        criterion,
        optimizer,
        num_epochs=EPOCHS,
        device=device,
    )

    # 5) save and test
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"inceptionv3_{MODE}_balanced.pth"
    torch.save(model.state_dict(), ckpt_path)
    print("Saved best model to:", ckpt_path)

    # 6) evaluation on test set
    _ = evaluate_on_test(model, test_loader, device=device)
