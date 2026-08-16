import os
import time

import torch
import torch.optim as optim

from frequency_loss import FrequencyAwareLoss
from dataloader import (
    train_loader,
    validation_loader
)

from model_hf_residual import (
    HFResidualSRNet
)

from hf_loss import (
    HFRestorationLoss
)


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 10

LEARNING_RATE = 1e-4

CHECKPOINT_DIR = (
    "checkpoints/hf_frequency"
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
print("HIGH-FREQUENCY RESIDUAL SUPER-RESOLUTION")
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

model = HFResidualSRNet(
    channels=96,
    num_blocks=8
).to(device)


# ============================================================
# LOSS
# ============================================================

criterion = FrequencyAwareLoss(
    alpha=1.0,
    beta=0.15,
    gamma=0.05,
    delta=0.05
).to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# BEST MODEL
# ============================================================

best_validation_loss = float("inf")


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    start_time = time.time()

    # --------------------------------------------------------
    # TRAIN
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

        # Prevent unstable gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_train_loss += loss.item()


    average_train_loss = (
        running_train_loss
        /
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


    # --------------------------------------------------------
    # LEARNING RATE
    # --------------------------------------------------------

    scheduler.step()


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    best_marker = ""

    if (
        average_validation_loss
        <
        best_validation_loss
    ):

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

        best_marker = " <- BEST MODEL"


    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    if epoch == EPOCHS - 1:

        torch.save(
            model.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                "final_model.pth"
            )
        )


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    elapsed = time.time() - start_time


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Train Loss: "
        f"{average_train_loss:.6f} "
        f"| Val Loss: "
        f"{average_validation_loss:.6f} "
        f"| Time: "
        f"{elapsed:.1f}s"
        f"{best_marker}"
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