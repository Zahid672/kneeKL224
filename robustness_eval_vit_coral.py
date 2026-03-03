import os
import random
from typing import List, Tuple, Dict, Callable

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF

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
# CONFIG (EDIT THESE)
# =========================
DATA_ROOT = r"E:\Knee-Osteoarthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224"

NUM_CLASSES = 5
IMG_SIZE = 224
NUM_FOLDS = 5

BATCH_SIZE = 8
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

MODEL_TAG = "ViT_L16_CORAL"

# checkpoints produced by your training script:
#   f"{MODEL_TAG}_fold{fold}_best.pth"
CKPT_PATTERN = f"{MODEL_TAG}_fold{{fold}}_best.pth"

# If you saved the best fold state:
BEST_MODEL_PATH = f"{MODEL_TAG}_bestFold_kneeKL224_TTA_tau.pth"

# outputs
ROBUSTNESS_CSV = f"{MODEL_TAG}_robustness_stress_test.csv"


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
        pos_weight_k = 1.0 if pos == 0 else (neg / float(pos))
        pos_weights.append(pos_weight_k)
    return torch.tensor(pos_weights, dtype=torch.float32)


def specificity_score(y_true, y_pred, num_classes, average="macro"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    TN, FP = [], []
    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm[i, :], i))
        TN.append(tn)
        FP.append(fp)
    spec = np.array(TN) / (np.array(TN) + np.array(FP) + 1e-8)
    return float(np.mean(spec)) if average == "macro" else spec


def compute_metrics(y_true, y_pred, num_classes):
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity_macro": float(specificity_score(y_true, y_pred, num_classes)),
    }

    y_true_oh = np.eye(num_classes)[y_true]
    y_pred_oh = np.eye(num_classes)[y_pred]
    try:
        metrics["auroc_macro"] = float(roc_auc_score(y_true_oh, y_pred_oh, average="macro"))
    except Exception:
        metrics["auroc_macro"] = 0.0

    return metrics


def labels_to_coral_targets(labels: torch.Tensor, num_classes: int):
    labels = labels.unsqueeze(1)  # [B,1]
    thresholds = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)
    targets = (labels > thresholds).float()  # [B, K-1]
    return targets


# =========================
# MODEL (ViT-L/16 + CORAL)
# =========================
class ViT_CORAL(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
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
        return logits

    @torch.no_grad()
    def predict_from_logits(self, logits, tau: float = 0.5):
        probs = torch.sigmoid(logits)
        preds = torch.sum(probs >= tau, dim=1).long()
        return preds


# =========================
# DATA LOADING
# =========================
def load_trainval_samples_and_labels(root: str):
    train_folder = datasets.ImageFolder(os.path.join(root, "train"))
    val_folder = datasets.ImageFolder(os.path.join(root, "val"))

    samples = train_folder.samples + val_folder.samples
    labels = np.array(train_folder.targets + val_folder.targets, dtype=np.int64)
    return samples, labels


class ImageFolderWithCustomTransform(Dataset):
    """
    Wrap ImageFolder but allow us to control PIL-level perturbations.
    """
    def __init__(self, root: str, base_tf: transforms.Compose, perturb_fn: Callable[[Image.Image], Image.Image]):
        self.folder = datasets.ImageFolder(root)  # no transform here
        self.samples = self.folder.samples
        self.targets = self.folder.targets
        self.base_tf = base_tf
        self.perturb_fn = perturb_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        # base resize/crop first (PIL)
        img = self.base_tf(img)

        # apply stress perturbation (PIL)
        img = self.perturb_fn(img)

        return img, int(label)


# =========================
# TAU TUNING (from val logits)
# =========================
@torch.no_grad()
def collect_logits_tta_general(
    model,
    loader,
    device,
    aug_fns: List[Callable[[torch.Tensor], torch.Tensor]],
):
    model.eval()
    logits_list = []
    labels_list = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        logits_sum = None
        for fn in aug_fns:
            x = fn(images)
            logits = model(x)
            logits_sum = logits if logits_sum is None else logits_sum + logits

        logits_mean = logits_sum / float(len(aug_fns))
        logits_list.append(logits_mean.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

    return np.concatenate(logits_list), np.concatenate(labels_list)


def tune_tau_from_val_logits(val_logits: np.ndarray, val_labels: np.ndarray):
    """
    Grid search a single global tau in [0.30, 0.70] maximizing macro F1.
    val_logits: [N, K-1]
    """
    best_tau = 0.5
    best_f1 = -1.0

    probs = 1.0 / (1.0 + np.exp(-val_logits))  # sigmoid

    for tau in np.linspace(0.30, 0.70, 41):  # step 0.01
        preds = np.sum(probs >= tau, axis=1)
        f1 = f1_score(val_labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)

    return best_tau, best_f1


# =========================
# STRESS TEST AUGS (PIL + Tensor)
# =========================
def make_base_pil_tf(img_size: int):
    # Keep as PIL so we can apply PIL perturbations after resize/crop
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
    ])


def make_to_tensor_norm_tf():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# --- PIL perturbations (simulate vendor + quality) ---
def pil_identity(img: Image.Image) -> Image.Image:
    return img


def pil_rotate(deg: float) -> Callable[[Image.Image], Image.Image]:
    def fn(img: Image.Image) -> Image.Image:
        return TF.rotate(img, deg, interpolation=transforms.InterpolationMode.BILINEAR, expand=False)
    return fn


def pil_brightness(f: float) -> Callable[[Image.Image], Image.Image]:
    def fn(img: Image.Image) -> Image.Image:
        return TF.adjust_brightness(img, f)
    return fn


def pil_contrast(f: float) -> Callable[[Image.Image], Image.Image]:
    def fn(img: Image.Image) -> Image.Image:
        return TF.adjust_contrast(img, f)
    return fn


def pil_gamma(g: float) -> Callable[[Image.Image], Image.Image]:
    def fn(img: Image.Image) -> Image.Image:
        return TF.adjust_gamma(img, gamma=g)
    return fn


def pil_blur(kernel: int = 5, sigma: float = 1.0) -> Callable[[Image.Image], Image.Image]:
    def fn(img: Image.Image) -> Image.Image:
        return TF.gaussian_blur(img, kernel_size=[kernel, kernel], sigma=[sigma, sigma])
    return fn


# --- Tensor-level noise ---
def add_gaussian_noise(std: float = 0.03) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * std
        y = x + noise
        return torch.clamp(y, 0.0, 1.0)
    return fn


# =========================
# EVALUATION (single + ensemble) under stress
# =========================
@torch.no_grad()
def eval_single_model_stress(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
    pos_weight: torch.Tensor,
    tau: float,
    aug_fns_tensor: List[Callable[[torch.Tensor], torch.Tensor]],
):
    """
    Tensor-level multi-aug evaluation:
    - apply aug_fns_tensor to normalized tensors (or to pre-normalized tensors if you want)
    - average logits over augs
    """
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")

    total = 0.0
    all_labels, all_preds = [], []

    for images, labels in loader:
        images = images.to(device)  # already normalized tensors
        labels = labels.to(device).long()

        logits_sum = None
        for fn in aug_fns_tensor:
            x = fn(images)
            logits = model(x)
            logits_sum = logits if logits_sum is None else logits_sum + logits

        logits_mean = logits_sum / float(len(aug_fns_tensor))

        targets = labels_to_coral_targets(labels, num_classes)
        loss = bce_loss(logits_mean, targets)
        total += loss.item()

        preds = model.predict_from_logits(logits_mean, tau=tau)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    avg_loss = total / len(loader.dataset)
    return avg_loss, y_true, y_pred


@torch.no_grad()
def eval_ensemble_stress(
    models_list: List[nn.Module],
    loader: DataLoader,
    device: str,
    tau: float,
    aug_fns_tensor: List[Callable[[torch.Tensor], torch.Tensor]],
):
    for m in models_list:
        m.eval()

    all_labels, all_preds = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        logits_ens_sum = None

        for model in models_list:
            logits_sum = None
            for fn in aug_fns_tensor:
                x = fn(images)
                logits = model(x)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            logits_mean_model = logits_sum / float(len(aug_fns_tensor))

            logits_ens_sum = logits_mean_model if logits_ens_sum is None else (logits_ens_sum + logits_mean_model)

        logits_ens = logits_ens_sum / float(len(models_list))

        probs = torch.sigmoid(logits_ens)
        preds = torch.sum(probs >= tau, dim=1).long()

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def make_test_loader_with_pil_perturb(
    root_test: str,
    img_size: int,
    pil_perturb: Callable[[Image.Image], Image.Image],
    batch_size: int,
    num_workers: int,
):
    base_pil_tf = make_base_pil_tf(img_size)
    to_tensor_norm = make_to_tensor_norm_tf()

    class _Wrapped(Dataset):
        def __init__(self):
            self.ds = ImageFolderWithCustomTransform(root_test, base_pil_tf, pil_perturb)
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img_pil, label = self.ds[idx]
            img = to_tensor_norm(img_pil)
            return img, label

    ds = _Wrapped()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader, np.array(ds.ds.targets, dtype=np.int64)


# =========================
# MAIN
# =========================
def main():
    set_seed(SEED)

    if not os.path.isdir(DATA_ROOT):
        raise RuntimeError(f"DATA_ROOT not found: {DATA_ROOT}")

    print("Device:", DEVICE)

    # Load train+val labels (for pos_weight and for tau tuning splits)
    samples, labels_all = load_trainval_samples_and_labels(DATA_ROOT)
    labels_all = np.asarray(labels_all, dtype=np.int64)

    pos_weight = compute_pos_weight(labels_all, NUM_CLASSES).to(DEVICE)
    print("pos_weight thresholds:", pos_weight.tolist())

    # ---- Tau tuning (same logic as your script: TTA on val folds) ----
    print("\n[1] Tuning tau using validation folds with standard TTA (flip, +/-10°)...")

    # Build datasets for val logits collection (PIL -> tensor)
    # We'll reuse PathLabelDataset logic implicitly by making a simple dataset from samples list.
    # For simplicity, implement a small inline dataset here.
    class _SamplesDataset(Dataset):
        def __init__(self, samples_list, pil_tf):
            self.samples = samples_list
            self.pil_tf = pil_tf
            self.to_tensor_norm = make_to_tensor_norm_tf()
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            img = self.pil_tf(img)
            img = self.to_tensor_norm(img)
            return img, int(label)

    # val pipeline = resize+centercrop (like your val_tf)
    val_pil_tf = make_base_pil_tf(IMG_SIZE)
    full_val_ds = _SamplesDataset(samples, val_pil_tf)

    indices = np.arange(len(samples))
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # standard TTA augmentations on tensor (to match your original TTA)
    def t_identity(x): return x
    def t_hflip(x): return TF.hflip(x)
    def t_rot_p10(x): return TF.rotate(x, 10)
    def t_rot_m10(x): return TF.rotate(x, -10)

    standard_tta = [t_identity, t_hflip, t_rot_p10, t_rot_m10]

    val_logits_all_list, val_labels_all_list = [], []

    for fold_idx, (_, val_idx) in enumerate(skf.split(indices, labels_all), start=1):
        ckpt_path = CKPT_PATTERN.format(fold=fold_idx)
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model = ViT_CORAL(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
        model.load_state_dict(ckpt["model"])

        val_loader = DataLoader(
            torch.utils.data.Subset(full_val_ds, val_idx),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
        )

        logits_fold, labels_fold = collect_logits_tta_general(model, val_loader, DEVICE, standard_tta)
        val_logits_all_list.append(logits_fold)
        val_labels_all_list.append(labels_fold)

        print(f"  Fold {fold_idx}: collected {len(labels_fold)} val samples")

    val_logits_all = np.concatenate(val_logits_all_list)
    val_labels_all = np.concatenate(val_labels_all_list)

    best_tau, best_f1 = tune_tau_from_val_logits(val_logits_all, val_labels_all)
    print(f"Best tau = {best_tau:.3f} (val macro-F1={best_f1:.4f})")

    # ---- Load best fold (by val_acc stored in ckpt) and load all folds for ensemble ----
    print("\n[2] Loading models...")
    best_fold = None
    best_val_acc = -1.0
    fold_states = {}

    for fold_idx in range(1, NUM_FOLDS + 1):
        ckpt_path = CKPT_PATTERN.format(fold=fold_idx)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        fold_states[fold_idx] = ckpt
        va = float(ckpt.get("val_acc", -1.0))
        if va > best_val_acc:
            best_val_acc = va
            best_fold = fold_idx

    print(f"Best fold inferred from checkpoints: Fold {best_fold} (val_acc={best_val_acc:.4f})")

    best_model = ViT_CORAL(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    best_model.load_state_dict(fold_states[best_fold]["model"])

    models_list = []
    for fold_idx in range(1, NUM_FOLDS + 1):
        m = ViT_CORAL(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
        m.load_state_dict(fold_states[fold_idx]["model"])
        models_list.append(m)

    # ---- Define stress-test settings ----
    print("\n[3] Running stress tests on test set...")

    test_root = os.path.join(DATA_ROOT, "test")

    # Tensor-level augmentations for evaluation (like TTA, but stress settings)
    # These operate on normalized tensors.
    def make_tensor_aug_list(setting: str) -> List[Callable[[torch.Tensor], torch.Tensor]]:
        # Basic: identity always included
        if setting == "clean":
            return [t_identity]
        if setting == "tta_std":
            return standard_tta
        if setting == "rot15":
            return [t_identity, lambda x: TF.rotate(x, 15), lambda x: TF.rotate(x, -15)]
        if setting == "rot20":
            return [t_identity, lambda x: TF.rotate(x, 20), lambda x: TF.rotate(x, -20)]
        # For "vendor-like" effects, apply PIL perturbation, not tensor-level
        # So keep tensor aug as identity for those (handled by loader).
        return [t_identity]

    # PIL perturbations (simulate manufacturer / processing shifts)
    pil_settings: Dict[str, Callable[[Image.Image], Image.Image]] = {
        "clean": pil_identity,
        # vendor/post-processing simulation (fixed factors, deterministic)
        "bright_0.8": pil_brightness(0.8),
        "bright_1.2": pil_brightness(1.2),
        "contrast_0.8": pil_contrast(0.8),
        "contrast_1.2": pil_contrast(1.2),
        "gamma_0.8": pil_gamma(0.8),
        "gamma_1.2": pil_gamma(1.2),
        # quality shifts
        "blur": pil_blur(kernel=5, sigma=1.0),
        "blur_strong": pil_blur(kernel=7, sigma=1.5),
        # combined (mild)
        "combo_mild": lambda img: pil_gamma(1.2)(pil_contrast(0.8)(pil_brightness(0.9)(img))),
        # combined (hard)
        "combo_hard": lambda img: pil_gamma(0.8)(pil_contrast(1.2)(pil_brightness(1.2)(pil_blur(7, 1.5)(img)))),
    }

    # Tensor noise settings (applied after normalize is not ideal, so we apply noise BEFORE normalize)
    # To keep it simple and stable, we apply noise in tensor space assuming values roughly standardized.
    # If you want more realism, apply noise before normalization. This version still works as a stress test.
    def noise_aug(std: float):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x + torch.randn_like(x) * std
        return fn

    stress_jobs = [
        # (name, pil_perturb_key, tensor_aug_setting, extra_tensor_aug)
        ("clean", "clean", "clean", None),
        ("std_TTA", "clean", "tta_std", None),

        ("rot15", "clean", "rot15", None),
        ("rot20", "clean", "rot20", None),

        ("bright_0.8", "bright_0.8", "clean", None),
        ("bright_1.2", "bright_1.2", "clean", None),
        ("contrast_0.8", "contrast_0.8", "clean", None),
        ("contrast_1.2", "contrast_1.2", "clean", None),
        ("gamma_0.8", "gamma_0.8", "clean", None),
        ("gamma_1.2", "gamma_1.2", "clean", None),

        ("blur", "blur", "clean", None),
        ("blur_strong", "blur_strong", "clean", None),

        ("noise_0.03", "clean", "clean", noise_aug(0.03)),
        ("noise_0.05", "clean", "clean", noise_aug(0.05)),

        ("combo_mild", "combo_mild", "clean", None),
        ("combo_hard", "combo_hard", "clean", None),
    ]

    rows = []

    for job_name, pil_key, tensor_setting, extra_aug in stress_jobs:
        pil_perturb = pil_settings[pil_key]
        test_loader, test_labels = make_test_loader_with_pil_perturb(
            root_test=test_root,
            img_size=IMG_SIZE,
            pil_perturb=pil_perturb,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

        tensor_augs = make_tensor_aug_list(tensor_setting)
        if extra_aug is not None:
            # include extra aug along with identity to keep averaging meaningful
            tensor_augs = tensor_augs + [extra_aug]

        # single best model
        loss_single, y_true_s, y_pred_s = eval_single_model_stress(
            best_model, test_loader, DEVICE, NUM_CLASSES, pos_weight, best_tau, tensor_augs
        )
        met_s = compute_metrics(y_true_s, y_pred_s, NUM_CLASSES)
        met_s.update({
            "setting": job_name,
            "model": f"single_best_fold_{best_fold}",
            "tau": best_tau,
            "loss": float(loss_single),
        })
        rows.append(met_s)

        # ensemble
        y_true_e, y_pred_e = eval_ensemble_stress(
            models_list, test_loader, DEVICE, best_tau, tensor_augs
        )
        met_e = compute_metrics(y_true_e, y_pred_e, NUM_CLASSES)
        met_e.update({
            "setting": job_name,
            "model": "ensemble_5fold",
            "tau": best_tau,
            "loss": 0.0,
        })
        rows.append(met_e)

        print(f"[{job_name}] single_acc={met_s['accuracy']:.4f}, single_f1={met_s['f1_macro']:.4f} | "
              f"ens_acc={met_e['accuracy']:.4f}, ens_f1={met_e['f1_macro']:.4f}")

    df = pd.DataFrame(rows)

    # Add deltas relative to clean for each model (helps rebuttal)
    df_out = []
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name].copy()
        base = sub[sub["setting"] == "clean"].iloc[0]
        sub["delta_acc_vs_clean"] = sub["accuracy"] - float(base["accuracy"])
        sub["delta_f1_vs_clean"] = sub["f1_macro"] - float(base["f1_macro"])
        df_out.append(sub)

    df_final = pd.concat(df_out, ignore_index=True)
    df_final.to_csv(ROBUSTNESS_CSV, index=False)

    print("\nSaved robustness table to:", ROBUSTNESS_CSV)
    print("\nTip: In the paper, report clean + a few key stress settings (rot20, gamma_0.8/1.2, combo_hard).")


if __name__ == "__main__":
    main()