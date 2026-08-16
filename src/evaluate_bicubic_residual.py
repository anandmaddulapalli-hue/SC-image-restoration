import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataloader import validation_loader
from model_bicubic_residual import BicubicResidualSRNet


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = (
    "checkpoints/bicubic_residual/best_model.pth"
)

RESULT_DIR = (
    "results/bicubic_residual"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("BICUBIC RESIDUAL SUPER-RESOLUTION EVALUATION")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# MODEL
# ============================================================

model = BicubicResidualSRNet(
    channels=64,
    num_blocks=8
).to(device)


model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=device
    )
)

model.eval()

print()
print("Model loaded successfully.")

print(
    "Checkpoint:",
    CHECKPOINT
)


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    prediction,
    target
):

    mse = F.mse_loss(
        prediction,
        target
    ).item()

    if mse == 0:

        return float("inf")

    return 10.0 * np.log10(
        1.0 / mse
    )


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(
    prediction,
    target
):

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # --------------------------------------------------------
    # Local means
    # --------------------------------------------------------

    mu_x = F.avg_pool2d(
        prediction,
        kernel_size=7,
        stride=1,
        padding=3
    )

    mu_y = F.avg_pool2d(
        target,
        kernel_size=7,
        stride=1,
        padding=3
    )

    # --------------------------------------------------------
    # Variance
    # --------------------------------------------------------

    sigma_x_sq = (
        F.avg_pool2d(
            prediction ** 2,
            kernel_size=7,
            stride=1,
            padding=3
        )
        - mu_x ** 2
    )

    sigma_y_sq = (
        F.avg_pool2d(
            target ** 2,
            kernel_size=7,
            stride=1,
            padding=3
        )
        - mu_y ** 2
    )

    # --------------------------------------------------------
    # Covariance
    # --------------------------------------------------------

    sigma_xy = (
        F.avg_pool2d(
            prediction * target,
            kernel_size=7,
            stride=1,
            padding=3
        )
        - mu_x * mu_y
    )

    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

    numerator = (
        (2 * mu_x * mu_y + C1)
        *
        (2 * sigma_xy + C2)
    )

    denominator = (
        (mu_x ** 2 + mu_y ** 2 + C1)
        *
        (
            sigma_x_sq
            + sigma_y_sq
            + C2
        )
    )

    ssim_map = (
        numerator
        /
        (denominator + 1e-8)
    )

    return ssim_map.mean().item()


# ============================================================
# EVALUATION
# ============================================================

total_psnr = 0.0
total_ssim = 0.0

total_images = 0
total_time = 0.0

sample_images = []


print()
print("Evaluating validation dataset...")
print()


with torch.no_grad():

    for noisy, gt in validation_loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        # ----------------------------------------------------
        # Synchronize GPU before timing
        # ----------------------------------------------------

        if device.type == "cuda":

            torch.cuda.synchronize()

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model(
            noisy
        )

        # ----------------------------------------------------
        # Synchronize GPU after inference
        # ----------------------------------------------------

        if device.type == "cuda":

            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        total_time += elapsed

        # ----------------------------------------------------
        # Calculate metrics image-by-image
        # ----------------------------------------------------

        for i in range(
            noisy.size(0)
        ):

            pred_i = prediction[
                i:i + 1
            ]

            gt_i = gt[
                i:i + 1
            ]

            psnr = calculate_psnr(
                pred_i,
                gt_i
            )

            ssim = calculate_ssim(
                pred_i,
                gt_i
            )

            total_psnr += psnr
            total_ssim += ssim

            total_images += 1

            # ------------------------------------------------
            # Save first 6 visual samples
            # ------------------------------------------------

            if len(sample_images) < 6:

                sample_images.append(
                    (
                        noisy[i]
                        .detach()
                        .cpu(),

                        prediction[i]
                        .detach()
                        .cpu(),

                        gt[i]
                        .detach()
                        .cpu()
                    )
                )


# ============================================================
# FINAL METRICS
# ============================================================

average_psnr = (
    total_psnr
    /
    total_images
)

average_ssim = (
    total_ssim
    /
    total_images
)

average_inference_time = (
    total_time
    /
    total_images
)

images_per_second = (
    1.0
    /
    average_inference_time
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("BICUBIC RESIDUAL SUPER-RESOLUTION RESULTS")
print("=" * 70)

print(
    f"Images evaluated       : "
    f"{total_images}"
)

print(
    f"Average PSNR           : "
    f"{average_psnr:.4f} dB"
)

print(
    f"Average SSIM           : "
    f"{average_ssim:.6f}"
)

print(
    f"Average inference time : "
    f"{average_inference_time * 1000:.2f} ms/image"
)

print(
    f"Inference speed       : "
    f"{images_per_second:.2f} images/second"
)


# ============================================================
# SAVE VISUAL RESULTS
# ============================================================

for index, (
    noisy,
    prediction,
    gt
) in enumerate(sample_images):

    noisy = noisy.squeeze().numpy()

    prediction = (
        prediction
        .squeeze()
        .numpy()
    )

    gt = (
        gt
        .squeeze()
        .numpy()
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    axes[0].imshow(
        noisy,
        cmap="gray"
    )

    axes[0].set_title(
        "Noisy LR"
    )

    axes[1].imshow(
        prediction,
        cmap="gray"
    )

    axes[1].set_title(
        "Bicubic Residual"
    )

    axes[2].imshow(
        gt,
        cmap="gray"
    )

    axes[2].set_title(
        "Ground Truth"
    )

    for ax in axes:

        ax.axis("off")

    plt.tight_layout()

    filename = os.path.join(
        RESULT_DIR,
        f"sample_{index + 1}.png"
    )

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


print()
print(
    "Visual results saved to:"
)

print(
    RESULT_DIR
)

print()
print("Evaluation complete!")