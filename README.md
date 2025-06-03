# ACVTT
**Anisotropic Cross-View Texture Transfer with Multi-Reference Non-Local Attention for CT Slice Interpolation**  
Official implementation of our IEEE TMI submission.

---

## 🔧 Overview
ACVTT is designed for CT slice interpolation under anisotropic volume conditions. It introduces a Multi-Reference Non-Local Attention (MRNLA) mechanism that adaptively aggregates information from multiple in-plane reference slices, enabling effective interpolation across sparsely sampled slices.

This repository contains:
- Full implementation of the ACVTT architecture  
- Data preprocessing scripts for CT and MRI datasets  
- Training & inference pipelines with configurable upscaling factors  
- MRI generalization experiments (IXI dataset)  
- Statistical significance testing scripts for reproducibility

---

## 📁 Repository Structure

| Path                | Description                                                |
|---------------------|------------------------------------------------------------|
| `models/ACVTT.py`   | Main model architecture with transfer & fusion modules     |
| `scripts/`          | Training and inference scripts                             |
| `others/`           | Dataset-specific preprocessing scripts                     |
| `stats/`            | Paired t-test script for reproducibility analysis          |
| `results/`          | Directory to store output reconstructions and metrics      |

---

## ⚙️ Installation

Requires Python 3.8+, PyTorch ≥1.10.

We recommend using a virtual environment (e.g., conda):

```bash
conda create -n acvtt python=3.8
conda activate acvtt
pip install -r requirements.txt
```

## 🧹 Data Preprocessing

Each preprocessing script supports the `--scale_factor` argument, which controls the target interpolation factor (e.g., ×2, ×3, ×4, ×5).  
Make sure to set it according to your experimental setting.

### 📌 Example: Preprocess for ×4 interpolation
```bash
# RPLHR-CT
python others/rplhr_nii_to_npy.py --scale_factor 4

# MSD
python preprocess_msd.py --scale_factor 4

# KiTS23
python downsample_image.py --scale_factor 4

# IXI (MRI)
python preprocess_ixi.py --scale_factor 4
```

## 🏋️ Training

We adopt a two-stage training process:

1. **Transfer module only** (`L_trans`)
2. **Full model with fusion** (`L_fuse`)

---

### 🔧 Configurable Upscaling Factor

Use the `--scale_factor` argument to specify the desired upscaling factor:  
Available values: `2`, `3`, `4`, `5`.

---

### 📌 Example: RPLHR-CT Dataset, ×3 Upscaling

```bash
# Stage 1: Train transfer module
python scripts/train.py --dataset RPLHRCT --model ACVTT --num_ref_slices 3 --scale_factor 3

# Stage 2: Train with fusion
python scripts/train.py --dataset RPLHRCT --do_fusion True --long_skip True --scale_factor 3
```

Repeat similarly for MSD, KiTS23, or IXI:
```bash
python scripts/train.py --dataset MSD --model ACVTT --num_ref_slices 3 --scale_factor 3
python scripts/train.py --dataset KiTS23 --model ACVTT --num_ref_slices 3 --scale_factor 3
python scripts/train.py --dataset IXI --model ACVTT --num_ref_slices 3 --scale_factor 3
```

## 🧪 Inference

After training, you can run inference using the following command.

### 📌 Example: IXI (MRI), ×4 Upscaling

```bash
python scripts/train.py --test_only True --dataset IXI --do_fusion True --long_skip True --scale_factor 4
```
Use the --dataset and --scale_factor arguments accordingly for other datasets and scaling settings.

## 📊 Statistical Significance Testing
We provide a script to conduct paired t-tests between model outputs to evaluate statistical significance.

### 🧪 Example Usage
```bash
python stats/paired_ttest.py --metric ssim --dir_a ./results/ours/ --dir_b ./results/baseline/ --output_file ./stats/ttest_results.txt
```

### 🔍 Arguments

- `--metric`: Metric to compare (`ssim`, `psnr`, etc.)
- `--dir_a`: Directory containing result files of Method A
- `--dir_b`: Directory containing result files of Method B
- `--output_file`: Path to save the paired t-test results

The script will compute paired t-statistics and p-values across all test subjects, highlighting statistically significant differences (*p* < 0.05).


## ✅ Complete Workflow Example (MRI, ×4)
```bash
python preprocess_ixi.py

python scripts/train.py --dataset IXI --model ACVTT --num_ref_slices 3 --scale_factor 4
python scripts/train.py --dataset IXI --do_fusion True --long_skip True --scale_factor 4

python scripts/train.py --test_only True --dataset IXI --do_fusion True --long_skip True --scale_factor 4

python stats/paired_ttest.py --metric ssim --dir_a ./results/ours_x4/ --dir_b ./results/baseline_x4/
```

## 📌 Citation
Coming soon after publication.


