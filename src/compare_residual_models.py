import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from model_residual import ResidualSRNet
from dataloader import validation_loader
from detail_loss import DetailAwareLoss


# ============================================================
# CONFIGURATION
# ============================================================

RESIDUAL_MODEL_PATH = (
    "checkpoints/residual_sr/best_model.pth"
)

DETAIL_MODEL_PATH = (
    "checkpoints/residual_detail/best_model.pth"
)

OUTPUT_DIR = "results/model_comparison"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("RESIDUAL MODEL VISUAL COMPARISON")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD RESIDUAL MODEL
# ============================================================

residual_model = ResidualSRNet().to(device)

residual_model.load_state_dict(
    torch.load(
        RESIDUAL_MODEL_PATH,
        map_location=device
    )
)

residual_model.eval()


# ============================================================
# LOAD DETAIL-AWARE MODEL
# ============================================================

detail_model = ResidualSRNet().to(device)

detail_model.load_state_dict(
    torch.load(
        DETAIL_MODEL_PATH,
        map_location=device
    )
)

detail_model.eval()


print()
print("Both models loaded successfully.")


# ============================================================
# GET ONE VALIDATION BATCH
# ============================================================

noisy, gt = next(iter(validation_loader))

noisy = noisy.to(device)
gt = gt.to(device)


# ============================================================
# PREDICTIONS
# ============================================================

with torch.no_grad():

    residual_prediction = residual_model(
        noisy
    )

    detail_prediction = detail_model(
        noisy
    )


# ============================================================
# CLAMP OUTPUTS
# ============================================================

residual_prediction = torch.clamp(
    residual_prediction,
    0.0,
    1.0
)

detail_prediction = torch.clamp(
    detail_prediction,
    0.0,
    1.0
)


# ============================================================
# SAVE COMPARISONS
# ============================================================

num_images = min(
    5,
    noisy.shape[0]
)


for i in range(num_images):

    # --------------------------------------------------------
    # Convert tensors to numpy
    # --------------------------------------------------------

    input_image = (
        noisy[i, 0]
        .detach()
        .cpu()
        .numpy()
    )

    residual_image = (
        residual_prediction[i, 0]
        .detach()
        .cpu()
        .numpy()
    )

    detail_image = (
        detail_prediction[i, 0]
        .detach()
        .cpu()
        .numpy()
    )

    gt_image = (
        gt[i, 0]
        .detach()
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )


    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    axes[0].imshow(
        input_image,
        cmap="gray"
    )

    axes[0].set_title(
        "Noisy LR"
    )

    axes[0].axis("off")


    # --------------------------------------------------------
    # Residual-SR
    # --------------------------------------------------------

    axes[1].imshow(
        residual_image,
        cmap="gray"
    )

    axes[1].set_title(
        "Residual-SR\n27.8233 dB"
    )

    axes[1].axis("off")


    # --------------------------------------------------------
    # Detail-aware
    # --------------------------------------------------------

    axes[2].imshow(
        detail_image,
        cmap="gray"
    )

    axes[2].set_title(
        "Residual + Detail\n27.8148 dB"
    )

    axes[2].axis("off")


    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    axes[3].imshow(
        gt_image,
        cmap="gray"
    )

    axes[3].set_title(
        "Ground Truth"
    )

    axes[3].axis("off")


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"comparison_{i:02d}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


print()
print("=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)

print(
    "Saved comparisons to:",
    OUTPUT_DIR
)

for i in range(num_images):

    print(
        f"comparison_{i:02d}.png"
    )