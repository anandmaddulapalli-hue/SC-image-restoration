# AI-Based Restoration of Degraded Images for Semiconductor Inspection

An AI-based image restoration project designed to reconstruct high-resolution semiconductor inspection images from degraded low-resolution and noisy inputs.

## 📌 Project Overview

Semiconductor inspection systems require high-quality images to detect small defects and structural details. However, captured images can suffer from:

* Low spatial resolution
* Noise and artifacts
* Loss of high-frequency details
* Blurring and degradation

This project explores deep-learning-based image restoration and super-resolution techniques to recover a high-resolution image from a degraded low-resolution input.

The project compares multiple restoration architectures, loss functions, and frequency/detail-aware approaches to identify an effective restoration model.

---

## 📂 Dataset

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

* Training samples: **3200**
* Ground-truth image size: **256 × 256**
* Noisy low-resolution image size: **128 × 128**
* Images are stored as NumPy `.npy` arrays.

### Test Data

```text
data/
└── test/
    └── NoisyLR/
        ├── 000000.npy
        ├── 000001.npy
        └── ...
```

* Test samples: **400**
* Input size: **128 × 128**

> The dataset is not included in this repository.

---

## 🧠 Approach

The project progressively experiments with several restoration strategies.

### Baseline

A convolutional restoration network is used as the initial baseline.

### Residual Learning

Residual architectures are explored to learn the difference between the degraded input and the desired high-resolution image.

### Bicubic + Residual Restoration

Bicubic interpolation is used to first upscale the low-resolution image to 256 × 256, after which a neural network learns to restore the remaining details.

### Attention-Based Restoration

Residual attention mechanisms are investigated to help the model focus on important image features.

### High-Frequency / Detail-Aware Restoration

Additional experiments focus on recovering high-frequency information that is commonly lost during image degradation.

These include:

* High-frequency-aware losses
* Frequency-domain losses
* Detail-aware losses
* Multi-scale high-frequency losses
* High-frequency residual learning

### Model Comparison

Multiple trained models are evaluated and compared using the validation dataset to identify the strongest restoration approach.

---

## 🏗️ Project Structure

```text
SC-image-restoration/
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
├── checkpoints/              # Ignored - trained model weights
├── results/                  # Ignored - generated results
├── data/                     # Ignored - dataset
├── .venv/                    # Ignored - Python virtual environment
├── .gitignore
├── .gitattributes
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/anandmaddulapalli-hue/SC-image-restoration.git
cd SC-image-restoration
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The project was developed and tested using a CUDA-enabled PyTorch environment where GPU acceleration was available.

---

## 🚀 Training

Training scripts are located inside the `src/` directory.

For example:

```bash
python src/train.py
```

Other experiments can be trained using their corresponding scripts:

```bash
python src/train_residual.py
python src/train_bicubic_residual.py
python src/train_residual_attention.py
python src/train_hf_residual.py
python src/train_hf_deep.py
python src/train_bicubic_multiscale_hf.py
```

Trained model checkpoints are saved under:

```text
checkpoints/
```

This directory is intentionally excluded from Git because model files can be large.

---

## 📊 Evaluation

Evaluation scripts are provided for the different model variants.

Examples:

```bash
python src/evaluate.py
python src/evaluate_residual.py
python src/evaluate_bicubic_residual.py
python src/evaluate_hf_residual.py
python src/evaluate_hf_deep.py
```

Model comparison can be performed using:

```bash
python src/compare_residual_models.py
```

---

## 📦 Generate Final Submission

The final restored test images can be generated using:

```bash
python src/generate_final_submission.py
```

An ensemble-based submission can also be generated using:

```bash
python src/generate_final_submission_ensemble.py
```

Generated submission files are stored under:

```text
results/
```

The generated results are intentionally excluded from Git.

---

## 🔬 Experiments

The project investigates the effect of different restoration strategies, including:

| Experiment         | Main Idea                                |
| ------------------ | ---------------------------------------- |
| Baseline           | Basic convolutional restoration          |
| Residual           | Residual learning                        |
| Bicubic Residual   | Bicubic upsampling + residual refinement |
| Residual Attention | Attention-enhanced residual restoration  |
| Residual Detail    | Detail-aware restoration                 |
| HF Residual        | High-frequency residual learning         |
| HF Deep            | Deeper high-frequency restoration        |
| Frequency Aware    | Frequency-domain information             |
| Multi-scale HF     | Multi-scale high-frequency information   |
| Ensemble           | Combining multiple restoration models    |

The goal is to determine which approach best preserves fine semiconductor image structures while suppressing degradation and noise.

---

## 🖥️ Hardware

The project was trained using a CUDA-enabled NVIDIA GPU.

Example development environment:

* NVIDIA GeForce RTX 3050 Laptop GPU
* 6 GB VRAM
* CUDA-enabled PyTorch

CPU execution is also possible, although training is significantly slower.

---

## 🛠️ Technologies Used

* Python
* PyTorch
* NumPy
* SciPy
* scikit-image
* Matplotlib
* ImageIO
* Deep Learning
* Image Restoration
* Super-Resolution
* Frequency-Domain Processing

---

## 🎯 Objective

The primary objective is to develop an effective AI-based restoration pipeline capable of reconstructing high-resolution semiconductor inspection images while preserving important fine-scale and high-frequency details.

The experiments focus not only on reducing pixel-level reconstruction error, but also on improving the visual and structural quality of restored images.

---

## 👥 Project

This repository contains the implementation, experiments, evaluation scripts, and submission-generation pipeline developed for the semiconductor image restoration project.
