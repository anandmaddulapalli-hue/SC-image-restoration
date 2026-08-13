import torch

from loss import EdgeAwareLoss


# ------------------------------------------------------------
# Create random test images
# ------------------------------------------------------------

prediction = torch.rand(
    2, 1, 256, 256
)

target = torch.rand(
    2, 1, 256, 256
)


# ------------------------------------------------------------
# Create loss function
# ------------------------------------------------------------

criterion = EdgeAwareLoss()


# ------------------------------------------------------------
# Calculate loss
# ------------------------------------------------------------

loss = criterion(
    prediction,
    target
)


print("Prediction shape:", prediction.shape)
print("Target shape:", target.shape)

print("Combined loss:", loss.item())

print("Loss is finite:", torch.isfinite(loss).item())