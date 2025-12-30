import os
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

CHECKPOINT_PATH = "ViT_L16_CORAL_fold4_best.pth"  # change if needed

DATA_ROOT = r"E:\Knee-Osteoarthritis-severity-detection\KneeXrayData\KneeXrayData\ClsKLData\kneeKL224"

# Five representative images (KL0..KL4)
IMAGE_PATHS: List[str] = [
    os.path.join(DATA_ROOT, r"train\0\9001897L.png"),  # KL0
    os.path.join(DATA_ROOT, r"train\1\9000622R.png"),  # KL1
    os.path.join(DATA_ROOT, r"train\2\9000296R.png"),  # KL2
    os.path.join(DATA_ROOT, r"train\3\9000296L.png"),  # KL3
    os.path.join(DATA_ROOT, r"train\4\9031426R.png"),  # KL4
]
GRADES = ["KL0", "KL1", "KL2", "KL3", "KL4"]

# Output
SINGLE_OUT_TEMPLATE = "vit_l16_attn_rollout_{grade}.png"
GRID_OUT = "vit_l16_attention_rollout_all.png"


# ---------------------------------------------------------
# MODEL (ViT-L/16 + CORAL head)
# ---------------------------------------------------------
class ViT_CORAL_L16(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        num_thresholds = num_classes - 1

        try:
            vit = models.vit_l_16(
                weights=models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except TypeError:
            vit = models.vit_l_16(pretrained=pretrained)

        embed_dim = vit.heads.head.in_features
        vit.heads = nn.Identity()
        self.backbone = vit
        self.classifier = nn.Linear(embed_dim, num_thresholds)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits


# ---------------------------------------------------------
# WRAPPER TO SAVE ATTENTION WEIGHTS
# ---------------------------------------------------------
class MHAWithSave(nn.Module):
    """
    Wrap nn.MultiheadAttention so we can grab attention weights
    without changing the external interface.
    """
    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.mha = mha
        self.last_attn_weights = None  # [B, T, T] or [B, heads, T, T]

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
    ):
        # Always ask for weights, regardless of need_weights flag
        out, attn_weights = self.mha(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
        )
        self.last_attn_weights = attn_weights
        # Return the same tuple type as the original MHA
        return out, attn_weights


def attach_mha_savers(model: ViT_CORAL_L16):
    """
    After loading the checkpoint, call this once so each EncoderBlock
    has an MHAWithSave wrapper around its self_attention.
    """
    for blk in model.backbone.encoder.layers:
        if not isinstance(blk.self_attention, MHAWithSave):
            blk.self_attention = MHAWithSave(blk.self_attention)


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------
transform_input = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_image(path: str):
    """
    Returns:
      - tensor: [1,3,H,W] for model
      - img_np: [H,W,3] in [0,1] for visualization
    """
    img = Image.open(path).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_np = np.asarray(img_resized).astype(np.float32) / 255.0
    tensor = transform_input(img_resized).unsqueeze(0)
    return tensor, img_np


# ---------------------------------------------------------
# ATTENTION ROLLOUT
# ---------------------------------------------------------
def attention_rollout(
    model: ViT_CORAL_L16,
    img_tensor: torch.Tensor,
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """
    Compute attention rollout map (14x14) for a single image.
    No per-image min-max normalization here; we'll normalize globally later.
    """
    model.eval()
    img_tensor = img_tensor.to(DEVICE)

    # Forward through backbone to fill last_attn_weights
    with torch.no_grad():
        _ = model.backbone(img_tensor)

    attn_mats = []
    for blk in model.backbone.encoder.layers:
        w = blk.self_attention.last_attn_weights
        if w is None:
            raise RuntimeError("Attention weights not captured; check MHAWithSave.")

        # w: [B, heads, T, T] or [B, T, T]
        if w.dim() == 4:
            # [B, heads, T, T] -> [T, T] (first batch, mean heads)
            w = w[0].mean(dim=0)
        else:
            # [B, T, T] -> [T, T]
            w = w[0]

        attn_mats.append(w)  # [T, T]

    T = attn_mats[0].shape[-1]
    result = torch.eye(T, device=attn_mats[0].device)

    for attn in attn_mats:
        attn = attn.clone()

        # 1) zero out a fraction of the smallest attention entries
        if discard_ratio > 0:
            flat = attn.reshape(-1)
            numel = flat.numel()
            k = int(numel * discard_ratio)
            if k > 0:
                _, idx = flat.topk(k, largest=False)
                flat[idx] = 0.0
                attn = flat.view(T, T)

        # 2) add identity and normalize rows
        attn = attn + torch.eye(T, device=attn.device)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        # 3) aggregate
        result = attn @ result

    # CLS -> patch tokens (skip CLS token)
    mask = result[0, 1:]  # [T-1]
    side = int(math.sqrt(mask.numel()))
    mask = mask.view(side, side)  # [14, 14]
    return mask.detach().cpu().numpy()


def upsample_cam(cam: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Upsample 14x14 map to size x size using bilinear interpolation."""
    cam_t = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    cam_up = F.interpolate(cam_t, size=(size, size), mode="bilinear", align_corners=False)
    cam_up = cam_up[0, 0].cpu().numpy()
    return cam_up


# ---------------------------------------------------------
# VISUALIZATION HELPERS
# ---------------------------------------------------------
def overlay_heatmap(img_np: np.ndarray, cam_up: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend jet heatmap with original image.
    img_np: [H,W,3] in [0,1]
    cam_up: [H,W] in [0,1]
    """
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam_up)[..., :3]

    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def save_single_figure(img_np: np.ndarray,
                       cam_up: np.ndarray,
                       grade: str,
                       out_path: str):
    """Save 1×3 per-grade figure with its own colorbar."""
    ov = overlay_heatmap(img_np, cam_up, alpha=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    im = axes[1].imshow(cam_up, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Attention rollout")
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Attention", fontsize=9)

    axes[2].imshow(ov)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(f"Attention rollout – {grade}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def save_grid_figure(imgs_np: List[np.ndarray],
                     cams_up_norm: List[np.ndarray],
                     grades: List[str],
                     out_path: str):
    """
    Save 3×5 grid:
      row1: input images
      row2: normalized attention maps (shared colorbar)
      row3: overlays
    """
    fig, axes = plt.subplots(
        3, 5, figsize=(15, 8),
        gridspec_kw={"wspace": 0.05, "hspace": 0.15}
    )

    im_for_cbar = None

    for col in range(5):
        img_np = imgs_np[col]
        cam_up = cams_up_norm[col]

        # Row 1: original
        axes[0, col].imshow(img_np, cmap="gray")
        axes[0, col].set_title(grades[col], fontsize=14)
        axes[0, col].axis("off")

        # Row 2: heatmap (no individual colorbars)
        im_for_cbar = axes[1, col].imshow(cam_up, cmap="jet", vmin=0.0, vmax=1.0)
        axes[1, col].axis("off")

        # Row 3: overlay
        ov = overlay_heatmap(img_np, cam_up, alpha=0.5)
        axes[2, col].imshow(ov)
        axes[2, col].axis("off")

    # Row labels
    axes[0, 0].set_ylabel("Input", fontsize=12)
    axes[1, 0].set_ylabel("Attention", fontsize=12)
    axes[2, 0].set_ylabel("Overlay", fontsize=12)

    # Shared colorbar for middle row
    if im_for_cbar is not None:
        cbar = fig.colorbar(
            im_for_cbar,
            ax=axes[1, :],
            location="right",
            fraction=0.02,
            pad=0.02,
        )
        cbar.ax.set_ylabel("Attention", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("Device:", DEVICE)

    # 1) Build model and load checkpoint
    model = ViT_CORAL_L16(num_classes=5, pretrained=False).to(DEVICE)
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded.")

    # 2) Wrap MHA modules so we can read attention
    attach_mha_savers(model)

    cams_14 = []
    imgs_np = []

    # 3) Compute rollout maps for all 5 images
    for path in IMAGE_PATHS:
        print(f"Processing {path}")
        tensor, img_np = load_image(path)
        cam_14 = attention_rollout(model, tensor, discard_ratio=0.9)
        cams_14.append(cam_14)
        imgs_np.append(img_np)

    # 4) Global min–max normalization
    cams_stack = np.stack(cams_14, axis=0)
    global_min = cams_stack.min()
    global_max = cams_stack.max()
    print(f"Global attention range: min={global_min:.4f}, max={global_max:.4f}")

    cams_up_norm: List[np.ndarray] = []
    for cam in cams_14:
        # normalize with global min/max
        cam_norm = (cam - global_min) / (global_max - global_min + 1e-8)
        cam_up = upsample_cam(cam_norm, size=IMG_SIZE)  # 224×224
        cams_up_norm.append(cam_up)

    # 5) Save per-grade figures
    for img_np, cam_up, grade in zip(imgs_np, cams_up_norm, GRADES):
        out_single = SINGLE_OUT_TEMPLATE.format(grade=grade)
        save_single_figure(img_np, cam_up, grade, out_single)
        print(f"Saved {out_single}")

    # 6) Save combined 3×5 grid
    save_grid_figure(imgs_np, cams_up_norm, GRADES, GRID_OUT)
    print(f"Saved combined grid to {GRID_OUT}")


if __name__ == "__main__":
    main()
