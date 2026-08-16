import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import validation_loader


# ============================================================
# FRIEND'S MODEL
# ============================================================

class SRModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                1, 32,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32, 32,
                3,
                padding=1
            ),

            nn.ReLU()
        )

        self.up = nn.ConvTranspose2d(
            32,
            32,
            kernel_size=2,
            stride=2
        )

        self.out = nn.Sequential(

            nn.Conv2d(
                32,
                32,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                1,
                3,
                padding=1
            )
        )

    def forward(self, x):

        return self.out(
            self.up(
                self.net(x)
            )
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
print("FRIEND'S MODEL EVALUATION")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = SRModel().to(device)

model.load_state_dict(
    torch.load(
        r"C:\Users\Dell\Downloads\semiconductor_project\semiconductor_project\model.pth",
        map_location=device
    )
)

model.eval()

print()
print("Model loaded successfully.")


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

    return 10 * np.log10(
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

    mu_x = F.avg_pool2d(
        prediction,
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

    sigma_x_sq = (
        F.avg_pool2d(
            prediction ** 2,
            7,
            stride=1,
            padding=3
        )
        - mu_x ** 2
    )

    sigma_y_sq = (
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
            prediction * target,
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


print()
print("Evaluating validation dataset...")
print()


with torch.no_grad():

    for noisy, gt in validation_loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        # ----------------------------------------------------
        # GPU synchronization for accurate timing
        # ----------------------------------------------------

        if device.type == "cuda":

            torch.cuda.synchronize()

        start = time.perf_counter()

        prediction = model(
            noisy
        )

        if device.type == "cuda":

            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        total_time += elapsed

        # ----------------------------------------------------
        # Metrics for every image
        # ----------------------------------------------------

        for i in range(
            noisy.size(0)
        ):

            pred = prediction[
                i:i + 1
            ]

            target = gt[
                i:i + 1
            ]

            # ------------------------------------------------
            # IMPORTANT
            # ------------------------------------------------
            # Your friend's model does NOT use sigmoid.
            #
            # Clamp output to valid [0,1] image range
            # before calculating image-quality metrics.
            # ------------------------------------------------

            pred = torch.clamp(
                pred,
                0.0,
                1.0
            )

            psnr = calculate_psnr(
                pred,
                target
            )

            ssim = calculate_ssim(
                pred,
                target
            )

            total_psnr += psnr
            total_ssim += ssim

            total_images += 1


# ============================================================
# FINAL RESULTS
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

average_time = (
    total_time
    /
    total_images
)

images_per_second = (
    1.0
    /
    average_time
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("FRIEND'S MODEL RESULTS")
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
    f"{average_time * 1000:.2f} ms/image"
)

print(
    f"Inference speed       : "
    f"{images_per_second:.2f} images/second"
)

print()
print("Evaluation complete!")