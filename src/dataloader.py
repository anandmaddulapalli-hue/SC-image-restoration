import torch
from torch.utils.data import DataLoader, random_split

from dataset import SemiconductorDataset


# ============================================================
# DATASET PATHS
# ============================================================

NOISY_DIR = r"D:\SEMICON\dataset\train\NoisyLR"
GT_DIR = r"D:\SEMICON\dataset\train\GT"


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 8
TRAIN_RATIO = 0.8

RANDOM_SEED = 42


# ============================================================
# CREATE FULL DATASET
# ============================================================

full_dataset = SemiconductorDataset(
    noisy_dir=NOISY_DIR,
    gt_dir=GT_DIR
)


# ============================================================
# CALCULATE SPLIT SIZES
# ============================================================

total_size = len(full_dataset)

train_size = int(TRAIN_RATIO * total_size)

validation_size = total_size - train_size


print("\nDataset split:")
print("Total      :", total_size)
print("Training   :", train_size)
print("Validation :", validation_size)


# ============================================================
# REPRODUCIBLE SPLIT
# ============================================================

generator = torch.Generator().manual_seed(RANDOM_SEED)


train_dataset, validation_dataset = random_split(
    full_dataset,
    [train_size, validation_size],
    generator=generator
)


# ============================================================
# CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CHECK ONE BATCH
# ============================================================

noisy_batch, gt_batch = next(iter(train_loader))


print("\nTraining batch:")
print("Noisy batch shape :", noisy_batch.shape)
print("GT batch shape    :", gt_batch.shape)

print("\nValidation batches:")
print("Number of training batches   :", len(train_loader))
print("Number of validation batches :", len(validation_loader))