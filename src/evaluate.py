import os
import time

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from PIL import Image

from dataloader import validation_loader
from model import RestorationNet


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"checkpoints\edge_aware_fft\best_model.pth"

RESULTS_DIR = r"results\edge_aware_fft"

NUM_VISUAL_SAMPLES = 10


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("BASELINE MODEL EVALUATION")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

model = RestorationNet()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("\nModel loaded successfully.")


# ============================================================
# EVALUATION VARIABLES
# ============================================================

total_psnr = 0.0
total_ssim = 0.0

total_inference_time = 0.0

num_images = 0


# ============================================================
# EVALUATE
# ============================================================

print("\nEvaluating validation dataset...\n")


with torch.no_grad():

    for batch_index, (noisy, gt) in enumerate(validation_loader):

        noisy = noisy.to(device)
        gt = gt.to(device)

        # ----------------------------------------------------
        # Measure inference time
        # ----------------------------------------------------

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        prediction = model(noisy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        batch_time = end_time - start_time

        total_inference_time += batch_time


        # ----------------------------------------------------
        # Move tensors to CPU
        # ----------------------------------------------------

        prediction_cpu = prediction.cpu().numpy()
        gt_cpu = gt.cpu().numpy()
        noisy_cpu = noisy.cpu().numpy()


        # ----------------------------------------------------
        # Process each image
        # ----------------------------------------------------

        batch_size = prediction_cpu.shape[0]

        for i in range(batch_size):

            pred_img = prediction_cpu[i, 0]
            gt_img = gt_cpu[i, 0]
            noisy_img = noisy_cpu[i, 0]

            # ------------------------------------------------
            # PSNR
            # ------------------------------------------------

            psnr = peak_signal_noise_ratio(
                gt_img,
                pred_img,
                data_range=1.0
            )

            # ------------------------------------------------
            # SSIM
            # ------------------------------------------------

            ssim = structural_similarity(
                gt_img,
                pred_img,
                data_range=1.0
            )

            total_psnr += psnr
            total_ssim += ssim

            num_images += 1


            # ------------------------------------------------
            # Save visual examples
            # ------------------------------------------------

            if num_images <= NUM_VISUAL_SAMPLES:

                # Clip only for visualization
                noisy_display = np.clip(
                    noisy_img,
                    0.0,
                    1.0
                )

                pred_display = np.clip(
                    pred_img,
                    0.0,
                    1.0
                )

                gt_display = np.clip(
                    gt_img,
                    0.0,
                    1.0
                )


                # Convert to uint8
                noisy_uint8 = (
                    noisy_display * 255
                ).astype(np.uint8)

                pred_uint8 = (
                    pred_display * 255
                ).astype(np.uint8)

                gt_uint8 = (
                    gt_display * 255
                ).astype(np.uint8)


                # Save individual images
                Image.fromarray(
                    noisy_uint8
                ).resize(
                    (256, 256)
                ).save(
                    os.path.join(
                        RESULTS_DIR,
                        f"sample_{num_images:03d}_noisy.png"
                    )
                )


                Image.fromarray(
                    pred_uint8
                ).save(
                    os.path.join(
                        RESULTS_DIR,
                        f"sample_{num_images:03d}_restored.png"
                    )
                )


                Image.fromarray(
                    gt_uint8
                ).save(
                    os.path.join(
                        RESULTS_DIR,
                        f"sample_{num_images:03d}_gt.png"
                    )
                )


# ============================================================
# FINAL METRICS
# ============================================================

average_psnr = total_psnr / num_images

average_ssim = total_ssim / num_images

average_inference_time = (
    total_inference_time / num_images
)

images_per_second = (
    1.0 / average_inference_time
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 70)
print("BASELINE RESULTS")
print("=" * 70)

print(f"Images evaluated       : {num_images}")

print(f"Average PSNR           : {average_psnr:.4f} dB")

print(f"Average SSIM           : {average_ssim:.6f}")

print(
    f"Average inference time : "
    f"{average_inference_time * 1000:.2f} ms/image"
)

print(
    f"Inference speed       : "
    f"{images_per_second:.2f} images/second"
)

print("\nVisual results saved to:")

print(RESULTS_DIR)

print("\nEvaluation complete!")