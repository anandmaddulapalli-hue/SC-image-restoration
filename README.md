# AI-Based Restoration of Degraded Images for Semiconductor Inspection

An AI-based image restoration project designed to reconstruct high-resolution semiconductor inspection images from degraded low-resolution and noisy inputs.

---

## KLA Final Submission

### Final Model

The final submission uses:

**HF Residual Super-Resolution Network + 4-Way Self-Ensemble**

Model configuration:

- Input: grayscale `.npy` image
- Input resolution: 128 x 128
- Output resolution: 256 x 256
- Model: `HFResidualSRNet`
- Feature channels: 96
- Residual blocks: 8
- Upscaling factor: 2x
- Self-ensemble:
  - Original input
  - Horizontal flip
  - Vertical flip
  - Horizontal + vertical flip
- Final prediction is obtained by averaging the four restored outputs.

### Submission Folder Structure

```text
SC-image-restoration/
├── run.py
├── requirements.txt
├── README.md
├── models/
│   └── hf_residual_best.pth
└── src/
    └── model_hf_residual.py
```

### Setup

Clone the repository:

```bash
git clone https://github.com/anandmaddulapalli-hue/SC-image-restoration.git
cd SC-image-restoration
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Run Inference

The required execution command is:

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py test/NoisyLR restored_output
```

The input directory must contain grayscale `.npy` files.

For every input `.npy` file, `run.py`:

1. Loads the degraded grayscale image.
2. Runs `HFResidualSRNet`.
3. Applies 4-way self-ensemble inference.
4. Produces a 2x restored image.
5. Clips output values to `[0,1]`.
6. Checks for NaN and Inf values.
7. Saves the restored result as `float32` `.npy`.
8. Preserves the original input filename.

### Output Format

For an input file:

```text
000123.npy
```

the corresponding restored output is:

```text
000123.npy
```

Expected output properties:

- Grayscale image
- Shape: `(H, W)`
- For 128 x 128 input: 256 x 256 output
- Data type: `float32`
- Value range: `[0,1]`
- No NaN values
- No Inf values

### Hardware and Offline Execution

The script automatically uses an NVIDIA CUDA GPU when available and falls back to CPU otherwise.

All required model weights are included locally in:

```text
models/hf_residual_best.pth
```

Inference does not require:

- Internet access
- API keys
- Additional model downloads
- User interaction
- Manual model configuration

---

## Project Overview

Semiconductor inspection systems require high-quality images to detect small defects and structural details. Captured images can suffer from:

- Low spatial resolution
- Noise and artifacts
- Loss of high-frequency details
- Blurring and degradation

This project explores deep-learning-based restoration and super-resolution approaches for reconstructing high-resolution semiconductor inspection images from degraded low-resolution inputs.

Multiple restoration architectures, loss functions, and frequency/detail-aware approaches were explored before selecting the final HF Residual model.

---

## Dataset

The dataset contains paired ground-truth and degraded images.

### Training Data

```text
data/
└── train/
    ├── GT/
    │   ├── 000000.npy
    │   ├── 000001.npy
    │   └── ...
    └── NoisyLR/
        ├── 000000.npy
        ├── 000001.npy
        └── ...
```

- Training samples: 3200
- Ground-truth image size: 256 x 256
- Noisy low-resolution image size: 128 x 128
- Images are stored as NumPy `.npy` arrays

### Test Data

```text
data/
└── test/
    └── NoisyLR/
        ├── 000000.npy
        ├── 000001.npy
        └── ...
```

- Test samples: 400
- Input size: 128 x 128

The dataset is not included in this repository.

---

## Final Model Architecture

The final model is a High-Frequency Residual Super-Resolution Network.

The restoration pipeline is:

```text
Noisy Low-Resolution Input
        |
        v
Bicubic 2x Upsampling
        |
        +----------------------+
        |                      |
        v                      |
Feature Extraction            |
        |                      |
        v                      |
8 Residual Blocks             |
        |                      |
        v                      |
Feature Fusion                |
        |                      |
        v                      |
PixelShuffle 2x Upsampling    |
        |                      |
        v                      |
High-Frequency Detail Head    |
        |                      |
        +----------+-----------+
                   |
                   v
      Bicubic + Learned Detail
                   |
                   v
          Restored Output
```

The network learns a high-frequency correction and adds it to a bicubic-upsampled baseline.

---

## 4-Way Self-Ensemble

The final submission improves prediction robustness using geometric self-ensemble inference.

The same trained model is evaluated on:

1. Original input
2. Horizontally flipped input
3. Vertically flipped input
4. Horizontally and vertically flipped input

Each transformed prediction is converted back to the original orientation.

The four restored outputs are then averaged:

```text
Final Prediction =
(
    Original Prediction
    + Horizontal Flip Prediction
    + Vertical Flip Prediction
    + Horizontal + Vertical Flip Prediction
) / 4
```

This is a self-ensemble of one trained model, not an ensemble of four separately trained models.

---

## Validation Result of Final Model

The selected HF Residual + 4-Way Self-Ensemble model achieved:

- Average PSNR: **28.0318 dB**
- Average SSIM: **0.756524**
- Validation images: **640**

These metrics were measured on the validation dataset.

---

## Experimental Approaches

Several restoration approaches were explored during development.

### Baseline

A convolutional restoration network was used as the initial reference model.

### Residual Learning

Residual architectures were tested to improve feature reconstruction.

### Bicubic + Residual Restoration

The low-resolution image was first upscaled using bicubic interpolation, after which a neural network learned the remaining restoration details.

### Attention-Based Restoration

Channel and spatial attention mechanisms were explored to help the network focus on important features.

### High-Frequency / Detail-Aware Restoration

Additional experiments focused on recovering fine image information using:

- High-frequency-aware losses
- Frequency-domain losses
- Detail-aware losses
- Multi-scale high-frequency losses
- High-frequency residual learning

---

## Project Structure

```text
SC-image-restoration/
│
├── run.py
│
├── models/
│   └── hf_residual_best.pth
│
├── src/
│   ├── model_baseline.py
│   ├── model_residual.py
│   ├── model_bicubic_residual.py
│   ├── model_residual_attention.py
│   ├── model_hf_residual.py
│   ├── model_hf_deep.py
│   │
│   ├── detail_loss.py
│   ├── frequency_loss.py
│   ├── hf_loss.py
│   ├── loss_multiscale_hf.py
│   │
│   ├── train.py
│   ├── train_residual.py
│   ├── train_bicubic_residual.py
│   ├── train_residual_attention.py
│   ├── train_residual_detail.py
│   ├── train_hf_residual.py
│   ├── train_hf_deep.py
│   ├── train_bicubic_multiscale_hf.py
│   │
│   ├── evaluate.py
│   ├── evaluate_residual.py
│   ├── evaluate_bicubic.py
│   ├── evaluate_bicubic_residual.py
│   ├── evaluate_residual_attention.py
│   ├── evaluate_residual_detail.py
│   ├── evaluate_hf_residual.py
│   ├── evaluate_hf_deep.py
│   ├── evaluate_frequency_aware.py
│   ├── evaluate_bicubic_multiscale_hf.py
│   ├── evaluate_hf_ensemble.py
│   │
│   ├── compare_residual_models.py
│   ├── evaluation_friend.py
│   ├── preview_test_results.py
│   ├── generate_final_submission.py
│   └── generate_final_submission_ensemble.py
│
├── checkpoints/          # Local experimental checkpoints, ignored by Git
├── results/              # Generated experimental results, ignored by Git
├── data/                 # Dataset, ignored by Git
├── .venv/                # Virtual environment, ignored by Git
│
├── .gitignore
├── .gitattributes
├── requirements.txt
└── README.md
```

---

## Training

Training scripts are located in the `src/` directory.

Examples:

```bash
python src/train.py
python src/train_residual.py
python src/train_bicubic_residual.py
python src/train_residual_attention.py
python src/train_hf_residual.py
python src/train_hf_deep.py
python src/train_bicubic_multiscale_hf.py
```

Experimental checkpoints are saved locally under:

```text
checkpoints/
```

The experimental checkpoint directory is ignored by Git.

The final submission checkpoint is separately included under:

```text
models/hf_residual_best.pth
```

---

## Evaluation

Evaluation scripts are provided for the different model variants.

Examples:

```bash
python src/evaluate.py
python src/evaluate_residual.py
python src/evaluate_bicubic_residual.py
python src/evaluate_hf_residual.py
python src/evaluate_hf_deep.py
python src/evaluate_hf_ensemble.py
```

The selected final validation method is:

```bash
python src/evaluate_hf_ensemble.py
```

---

## Technologies Used

- Python
- PyTorch
- NumPy
- SciPy
- scikit-image
- Matplotlib
- ImageIO
- CUDA
- Deep Learning
- Image Restoration
- Super-Resolution
- Residual Learning
- High-Frequency Restoration

---

## Hardware

Development and testing were performed using a CUDA-enabled NVIDIA GPU.

Example development hardware:

- NVIDIA GeForce RTX 3050 Laptop GPU
- 6 GB VRAM
- CUDA-enabled PyTorch

The final inference script automatically selects CUDA when available.

---

## Objective

The primary objective is to develop an AI-based restoration pipeline capable of reconstructing high-resolution semiconductor inspection images while preserving fine structural and high-frequency information.

The final solution combines high-frequency residual learning with 4-way self-ensemble inference to improve restoration quality while maintaining the required `.npy` input/output format.

---

## Repository

GitHub:

```text
https://github.com/anandmaddulapalli-hue/SC-image-restoration
```