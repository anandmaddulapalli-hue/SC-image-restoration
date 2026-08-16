import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from dataloader import validation_loader
from model_hf_residual import HFResidualSRNet


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = "checkpoints/hf_residual/best_model.pth"

RESULT_DIR = "results/frequency_aware"

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("FREQUENCY-AWARE SELF-ENSEMBLE")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# MODEL
# ============================================================

model = HFResidualSRNet(
    channels=96,
    num_blocks=8
).to(device)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# AUGMENTATION FUNCTIONS
# ============================================================

def transform(x, mode):

    if mode == 0:
        return x

    if mode == 1:
        return torch.flip(x, dims=[-1])

    if mode == 2:
        return torch.flip(x, dims=[-2])

    if mode == 3:
        return torch.flip(x, dims=[-2, -1])

    if mode == 4:
        return torch.rot90(x, 1, dims=[-2, -1])

    if mode == 5:
        return torch.rot90(x, 2, dims=[-2, -1])

    if mode == 6:
        return torch.rot90(x, 3, dims=[-2, -1])

    if mode == 7:
        return torch.flip(
            torch.rot90(x, 1, dims=[-2, -1]),
            dims=[-1]
        )

    return x


def inverse_transform(x, mode):

    if mode == 0:
        return x

    if mode == 1:
        return torch.flip(x, dims=[-1])

    if mode == 2:
        return torch.flip(x, dims=[-2])

    if mode == 3:
        return torch.flip(x, dims=[-2, -1])

    if mode == 4:
        return torch.rot90(
            x, 3,
            dims=[-2, -1]
        )

    if mode == 5:
        return torch.rot90(
            x, 2,
            dims=[-2, -1]
        )

    if mode == 6:
        return torch.rot90(
            x, 1,
            dims=[-2, -1]
        )

    if mode == 7:
        x = torch.flip(
            x,
            dims=[-1]
        )

        return torch.rot90(
            x, 3,
            dims=[-2, -1]
        )

    return x


# ============================================================
# PSNR
# ============================================================

def psnr(pred, target):

    mse = F.mse_loss(
        pred,
        target
    ).item()

    if mse == 0:
        return float("inf")

    return 10 * np.log10(
        1.0 / mse
    )


# ============================================================
# SSIM
# ============================================================

def ssim(pred, target):

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(
        pred,
        7,
        stride=1,
        padding=3
    )

    mu_y = F.avg_pool2d(
        target,
        7,
        stride=1,
        padding=3
    )

    sigma_x = (
        F.avg_pool2d(
            pred ** 2,
            7,
            stride=1,
            padding=3
        )
        - mu_x ** 2
    )

    sigma_y = (
        F.avg_pool2d(
            target ** 2,
            7,
            stride=1,
            padding=3
        )
        - mu_y ** 2
    )

    sigma_xy = (
        F.avg_pool2d(
            pred * target,
            7,
            stride=1,
            padding=3
        )
        - mu_x * mu_y
    )

    numerator = (
        (2 * mu_x * mu_y + C1)
        *
        (2 * sigma_xy + C2)
    )

    denominator = (
        (mu_x ** 2 + mu_y ** 2 + C1)
        *
        (sigma_x + sigma_y + C2)
    )

    return (
        numerator /
        (denominator + 1e-8)
    ).mean().item()


# ============================================================
# FREQUENCY DECOMPOSITION
# ============================================================

def low_frequency(x):

    return F.avg_pool2d(
        x,
        kernel_size=5,
        stride=1,
        padding=2
    )


# ============================================================
# EVALUATION
# ============================================================

total_psnr = 0.0
total_ssim = 0.0
total_time = 0.0
total_images = 0

print()
print("Evaluating validation dataset...")
print()


with torch.no_grad():

    for noisy, gt in validation_loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        predictions = []

        # ----------------------------------------------------
        # Generate 8 transformed predictions
        # ----------------------------------------------------

        for mode in range(8):

            transformed = transform(
                noisy,
                mode
            )

            prediction = model(
                transformed
            )

            prediction = inverse_transform(
                prediction,
                mode
            )

            prediction = torch.clamp(
                prediction,
                0.0,
                1.0
            )

            predictions.append(
                prediction
            )

        # ----------------------------------------------------
        # Stack predictions
        # ----------------------------------------------------

        stack = torch.stack(
            predictions,
            dim=0
        )

        # ----------------------------------------------------
        # Standard ensemble
        # ----------------------------------------------------

        standard = stack.mean(
            dim=0
        )

        # ----------------------------------------------------
        # Low-frequency components
        # ----------------------------------------------------

        low_components = torch.stack(
            [
                low_frequency(p)
                for p in predictions
            ],
            dim=0
        )

        low_average = (
            low_components.mean(dim=0)
        )

        # ----------------------------------------------------
        # High-frequency components
        # ----------------------------------------------------

        high_components = (
            stack
            -
            low_components
        )

        high_average = (
            high_components.mean(dim=0)
        )

        # ----------------------------------------------------
        # Frequency-aware reconstruction
        # ----------------------------------------------------

        prediction = (
            low_average
            +
            high_average
        )

        prediction = torch.clamp(
            prediction,
            0.0,
            1.0
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        total_time += elapsed

        # ----------------------------------------------------
        # Metrics
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

            total_psnr += psnr(
                pred_i,
                gt_i
            )

            total_ssim += ssim(
                pred_i,
                gt_i
            )

            total_images += 1


# ============================================================
# RESULTS
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
)

speed = 1.0 / average_time


print()
print("=" * 70)
print("FREQUENCY-AWARE RESULTS")
print("=" * 70)

print(
    f"Images evaluated       : {total_images}"
)

print(
    f"Average PSNR           : {average_psnr:.4f} dB"
)

print(
    f"Average SSIM           : {average_ssim:.6f}"
)

print(
    f"Average inference time : "
    f"{average_time * 1000:.2f} ms/image"
)

print(
    f"Inference speed       : "
    f"{speed:.2f} images/second"
)

print()
print("Evaluation complete!")