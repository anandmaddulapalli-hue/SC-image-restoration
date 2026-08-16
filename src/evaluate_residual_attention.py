import os
import time

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model_residual_attention import ResidualAttentionSRNet
from dataloader import validation_loader


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "checkpoints/residual_attention/best_model.pth"

RESULTS_DIR = "results/residual_attention"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("RESIDUAL SUPER-RESOLUTION EVALUATION")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD MODEL
# ============================================================

model = ResidualAttentionSRNet().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print()
print("Model loaded successfully.")
print("Model:", MODEL_PATH)


# ============================================================
# METRIC VARIABLES
# ============================================================

total_psnr = 0.0
total_ssim = 0.0

total_images = 0

total_inference_time = 0.0


# ============================================================
# EVALUATION
# ============================================================

print()
print("Evaluating validation dataset...")


with torch.no_grad():

    for noisy, gt in validation_loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()

        prediction = model(noisy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        total_inference_time += (
            time.time() - start
        )

        prediction = torch.clamp(
            prediction,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        for i in range(noisy.shape[0]):

            pred = prediction[i, 0].cpu().numpy()

            target = gt[i, 0].cpu().numpy()

            psnr = peak_signal_noise_ratio(
                target,
                pred,
                data_range=1.0
            )

            ssim = structural_similarity(
                target,
                pred,
                data_range=1.0
            )

            total_psnr += psnr
            total_ssim += ssim

            total_images += 1


# ============================================================
# FINAL RESULTS
# ============================================================

average_psnr = (
    total_psnr /
    total_images
)

average_ssim = (
    total_ssim /
    total_images
)

average_time = (
    total_inference_time /
    total_images
) * 1000

inference_speed = (
    total_images /
    total_inference_time
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("RESIDUAL + CHANNEL ATTENTION EVALUATION")
print("=" * 70)

print(
    "Images evaluated       :",
    total_images
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
    f"{average_time:.2f} ms/image"
)

print(
    f"Inference speed       : "
    f"{inference_speed:.2f} images/second"
)

print()
print("Evaluation complete!")