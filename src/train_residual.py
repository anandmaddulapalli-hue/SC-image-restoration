import os
import time

import torch
import torch.nn as nn

from model_residual import ResidualSRNet
from dataloader import (
    train_loader,
    validation_loader
)


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 10

LEARNING_RATE = 1e-4

CHECKPOINT_DIR = "checkpoints/residual_sr"

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
print("RESIDUAL SUPER-RESOLUTION TRAINING")
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

model = ResidualSRNet().to(device)


# ============================================================
# LOSS
# ============================================================

criterion = nn.L1Loss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# BEST MODEL
# ============================================================

best_validation_loss = float("inf")


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    start_time = time.time()

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_train_loss = 0.0

    for noisy, gt in train_loader:

        noisy = noisy.to(device)

        gt = gt.to(device)

        optimizer.zero_grad()

        prediction = model(noisy)

        loss = criterion(
            prediction,
            gt
        )

        loss.backward()

        optimizer.step()

        running_train_loss += loss.item()


    average_train_loss = (
        running_train_loss /
        len(train_loader)
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    running_validation_loss = 0.0

    with torch.no_grad():

        for noisy, gt in validation_loader:

            noisy = noisy.to(device)

            gt = gt.to(device)

            prediction = model(noisy)

            loss = criterion(
                prediction,
                gt
            )

            running_validation_loss += loss.item()


    average_validation_loss = (
        running_validation_loss /
        len(validation_loader)
    )


    # --------------------------------------------------------
    # SAVE BEST
    # --------------------------------------------------------

    marker = ""

    if average_validation_loss < best_validation_loss:

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

        marker = " <- BEST MODEL"


    elapsed = time.time() - start_time


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Train Loss: {average_train_loss:.6f} "
        f"| Val Loss: {average_validation_loss:.6f} "
        f"| Time: {elapsed:.1f}s"
        f"{marker}"
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