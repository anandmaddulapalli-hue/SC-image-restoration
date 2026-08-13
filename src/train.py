import os
import time

import torch
import torch.optim as optim

from dataloader import train_loader, validation_loader
from model import RestorationNet
from loss import EdgeAwareLoss
from fft_loss import FFTLoss
from self_ensemble import self_ensemble_predict

# ============================================================
# CONFIGURATION
# ============================================================

NUM_EPOCHS = 10

LEARNING_RATE = 1e-4

CHECKPOINT_DIR = "checkpoints/edge_aware_fft"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# CREATE CHECKPOINT DIRECTORY
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# CREATE MODEL
# ============================================================

model = RestorationNet()

model = model.to(device)


# ============================================================
# LOSS FUNCTION
# ============================================================


edge_aware_criterion = EdgeAwareLoss(
    alpha=1.0,
    beta=0.2,
    gamma=0.1
)

fft_criterion = FFTLoss(high_freq_weight=2.0)

FFT_LOSS_WEIGHT = 0.005   # start small — FFT loss magnitudes can be large


def criterion(prediction, gt):
    return edge_aware_criterion(prediction, gt) + FFT_LOSS_WEIGHT * fft_criterion(prediction, gt)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_validation_loss = float("inf")


for epoch in range(NUM_EPOCHS):

    epoch_start = time.time()

    # --------------------------------------------------------
    # TRAINING MODE
    # --------------------------------------------------------

    model.train()

    running_train_loss = 0.0

    for batch_index, (noisy, gt) in enumerate(train_loader):

        # Move data to GPU
        noisy = noisy.to(device)
        gt = gt.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        prediction = model(noisy)

        # Calculate loss
        loss = criterion(
            prediction,
            gt
        )

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate loss
        running_train_loss += loss.item()

    # Average training loss
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

            prediction = self_ensemble_predict(model, noisy, device)

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
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if average_validation_loss < best_validation_loss:

        best_validation_loss = average_validation_loss

        best_model_path = os.path.join(
            CHECKPOINT_DIR,
            "best_model.pth"
        )

        torch.save(
            model.state_dict(),
            best_model_path
        )

        saved_message = " <- BEST MODEL"

    else:

        saved_message = ""


    # --------------------------------------------------------
    # EPOCH TIME
    # --------------------------------------------------------

    epoch_time = time.time() - epoch_start


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"| Train Loss: {average_train_loss:.6f} "
        f"| Val Loss: {average_validation_loss:.6f} "
        f"| Time: {epoch_time:.1f}s"
        f"{saved_message}"
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    CHECKPOINT_DIR,
    "final_model.pth"
)

torch.save(
    model.state_dict(),
    final_model_path
)


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("Best validation loss:", best_validation_loss)

print("Best model:")
print(
    os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )
)

print("\nFinal model:")
print(
    os.path.join(
        CHECKPOINT_DIR,
        "final_model.pth"
    )
)