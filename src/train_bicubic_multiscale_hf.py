import os
import time

import torch

from dataloader import train_loader, validation_loader
from model_bicubic_residual import BicubicResidualSRNet
from loss_multiscale_hf import MultiScaleHFLoss


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 10

LEARNING_RATE = 1e-4

CHECKPOINT_DIR = (
    "checkpoints/bicubic_multiscale_hf"
)

os.makedirs(
    CHECKPOINT_DIR,
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
print("BICUBIC RESIDUAL + MULTI-SCALE + HIGH-FREQUENCY TRAINING")
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


print()
print("Model parameters:")

print(
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = MultiScaleHFLoss(
    alpha=1.0,
    beta=0.15,
    gamma=0.10,
    delta=0.05
).to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_validation_loss = float("inf")


for epoch in range(1, EPOCHS + 1):

    start_time = time.time()

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_train_loss = 0.0

    for noisy, gt in train_loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        prediction = model(
            noisy
        )

        loss = criterion(
            prediction,
            gt
        )

        loss.backward()

        optimizer.step()

        running_train_loss += (
            loss.item()
        )

    average_train_loss = (
        running_train_loss
        /
        len(train_loader)
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    running_validation_loss = 0.0

    with torch.no_grad():

        for noisy, gt in validation_loader:

            noisy = noisy.to(device)
            gt = gt.to(device)

            prediction = model(
                noisy
            )

            loss = criterion(
                prediction,
                gt
            )

            running_validation_loss += (
                loss.item()
            )

    average_validation_loss = (
        running_validation_loss
        /
        len(validation_loader)
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    is_best = (
        average_validation_loss
        <
        best_validation_loss
    )

    if is_best:

        best_validation_loss = (
            average_validation_loss
        )

        torch.save(
            model.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )
        )

    # ========================================================
    # EPOCH REPORT
    # ========================================================

    elapsed = (
        time.time()
        -
        start_time
    )

    print(
        f"Epoch [{epoch}/{EPOCHS}] "
        f"| Train Loss: "
        f"{average_train_loss:.6f} "
        f"| Val Loss: "
        f"{average_validation_loss:.6f} "
        f"| Time: "
        f"{elapsed:.1f}s"
        +
        (
            " <- BEST MODEL"
            if is_best
            else ""
        )
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

torch.save(
    model.state_dict(),
    os.path.join(
        CHECKPOINT_DIR,
        "final_model.pth"
    )
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Best validation loss:",
    best_validation_loss
)

print(
    "Best model:",
    os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )
)

print(
    "Final model:",
    os.path.join(
        CHECKPOINT_DIR,
        "final_model.pth"
    )
)