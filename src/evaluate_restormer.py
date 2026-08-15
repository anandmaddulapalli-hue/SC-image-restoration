import time
import torch
import torch.nn.functional as F

from model_restormer import HybridRestormerNet
from dataloader import validation_loader


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("RESTORMER-INSPIRED MODEL EVALUATION")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# MODEL
# ============================================================

model = HybridRestormerNet().to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/restormer/best_model.pth",
        map_location=device
    )
)

model.eval()

print("\nModel loaded successfully.")


# ============================================================
# METRICS
# ============================================================

def calculate_psnr(prediction, target):

    mse = F.mse_loss(
        prediction,
        target
    )

    if mse == 0:
        return float("inf")

    return (
        10 *
        torch.log10(
            1.0 / mse
        )
    ).item()


def calculate_ssim(prediction, target):

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(
        prediction,
        7,
        1,
        3
    )

    mu_y = F.avg_pool2d(
        target,
        7,
        1,
        3
    )

    sigma_x = F.avg_pool2d(
        prediction ** 2,
        7,
        1,
        3
    ) - mu_x ** 2

    sigma_y = F.avg_pool2d(
        target ** 2,
        7,
        1,
        3
    ) - mu_y ** 2

    sigma_xy = F.avg_pool2d(
        prediction * target,
        7,
        1,
        3
    ) - mu_x * mu_y

    ssim = (
        (2 * mu_x * mu_y + C1) *
        (2 * sigma_xy + C2)
    ) / (
        (mu_x ** 2 + mu_y ** 2 + C1) *
        (sigma_x + sigma_y + C2)
    )

    return ssim.mean().item()


# ============================================================
# EVALUATION
# ============================================================

total_psnr = 0.0
total_ssim = 0.0
total_images = 0

total_time = 0.0

print("\nEvaluating validation dataset...")

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

        total_time += time.time() - start

        batch_size = noisy.size(0)

        for i in range(batch_size):

            pred = prediction[i:i+1]
            target = gt[i:i+1]

            total_psnr += calculate_psnr(
                pred,
                target
            )

            total_ssim += calculate_ssim(
                pred,
                target
            )

            total_images += 1


# ============================================================
# RESULTS
# ============================================================

average_psnr = total_psnr / total_images

average_ssim = total_ssim / total_images

average_time = (
    total_time /
    total_images
) * 1000

speed = (
    total_images /
    total_time
)


print()
print("=" * 70)
print("RESTORMER-INSPIRED RESULTS")
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
    f"Average inference time : {average_time:.2f} ms/image"
)

print(
    f"Inference speed       : {speed:.2f} images/second"
)

print("=" * 70)