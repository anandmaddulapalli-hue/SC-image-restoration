import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from dataloader import validation_loader


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = "results/bicubic"

NUM_VISUAL_SAMPLES = 10

os.makedirs(
    RESULTS_DIR,
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
print("BICUBIC BASELINE EVALUATION")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# EVALUATION VARIABLES
# ============================================================

total_psnr = 0.0
total_ssim = 0.0

total_images = 0

total_time = 0.0


# ============================================================
# VISUAL SAMPLE COUNTER
# ============================================================

visual_count = 0


# ============================================================
# EVALUATION
# ============================================================

print()
print("Evaluating validation dataset...")

with torch.no_grad():

    for noisy, gt in validation_loader:

        # ----------------------------------------------------
        # Move input to device
        # ----------------------------------------------------

        noisy = noisy.to(device)

        gt = gt.to(device)


        # ----------------------------------------------------
        # Bicubic interpolation
        # ----------------------------------------------------

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        prediction = F.interpolate(
            noisy,
            size=(256, 256),
            mode="bicubic",
            align_corners=False
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        total_time += (
            time.time() -
            start_time
        )


        # ----------------------------------------------------
        # Clamp output
        # ----------------------------------------------------

        prediction = torch.clamp(
            prediction,
            0.0,
            1.0
        )


        # ----------------------------------------------------
        # Calculate metrics image by image
        # ----------------------------------------------------

        batch_size = noisy.shape[0]

        for i in range(batch_size):

            pred = prediction[i, 0].cpu().numpy()

            target = gt[i, 0].cpu().numpy()


            # ------------------------------------------------
            # PSNR
            # ------------------------------------------------

            psnr = peak_signal_noise_ratio(
                target,
                pred,
                data_range=1.0
            )


            # ------------------------------------------------
            # SSIM
            # ------------------------------------------------

            ssim = structural_similarity(
                target,
                pred,
                data_range=1.0
            )


            total_psnr += psnr

            total_ssim += ssim

            total_images += 1


            # ------------------------------------------------
            # Save visual samples
            # ------------------------------------------------

            if visual_count < NUM_VISUAL_SAMPLES:

                sample_dir = os.path.join(
                    RESULTS_DIR,
                    f"sample_{visual_count:03d}"
                )

                os.makedirs(
                    sample_dir,
                    exist_ok=True
                )


                np.save(
                    os.path.join(
                        sample_dir,
                        "GT.npy"
                    ),
                    target
                )

                np.save(
                    os.path.join(
                        sample_dir,
                        "Noisy.npy"
                    ),
                    noisy[
                        i,
                        0
                    ].cpu().numpy()
                )

                np.save(
                    os.path.join(
                        sample_dir,
                        "Bicubic.npy"
                    ),
                    pred
                )

                visual_count += 1


# ============================================================
# FINAL METRICS
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
    total_time /
    total_images
) * 1000

inference_speed = (
    total_images /
    total_time
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("BICUBIC RESULTS")
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

print(
    "Visual results saved to:"
)

print(
    RESULTS_DIR
)

print()

print("Evaluation complete!")