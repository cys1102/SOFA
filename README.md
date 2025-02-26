# SOFA: Deep Learning Framework for Simulating and Optimizing Atrial Fibrillation Ablation

This repository contains the implementation of SOFA, a unified deep learning framework designed to simulate post-ablation outcomes, predict atrial fibrillation recurrence, and optimize ablative parameters using multi-modal and multi-view data.

## Requirements

- Python 3.7+
- PyTorch (1.7+)
- torchvision
- scikit-image
- scikit-learn
- pandas
- numpy
- matplotlib
- Pillow

Install dependencies via pip:

```bash
pip install torch torchvision scikit-image scikit-learn pandas numpy matplotlib pillow
```

## Dataset

SOFA is evaluated on the DECAAF-II dataset, a randomized multicenter study assessing the efficacy of fibrosis-targeted ablation in patients with persistent AF. The dataset includes:
- Pre-ablation and post-ablation MRI
- Detailed procedural data (ablation points, duration, temperature, power, impedance)

In DECAAF-II, commercial software (Merisight) was used for image segmentation, processing, quantification of left atrial fibrosis, and 3D MRI renderings. We preprocess the 3D models by applying rigid registration to the left atrial (LA) models, and then extract six view images per patient. All images are resized to 256×256 pixels.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/anonymous/SOFA.git
cd SOFA
```

2. Install the required packages (see above).

## Usage

SOFA supports three experimental phases:
- Phase 1: Train the simulation network to generate post-ablation images and scar maps.
- Phase 2: Train the outcome prediction network using pre-procedural multi-view data.
- Phase 3: Optimize ablative parameters (duration, force, temperature, power) to reduce predicted recurrence risk.

You can choose the phase via the --phase command-line argument.

## Training (Phases 1 & 2)
To run training with 5-fold cross-validation:

Phase 1 (Image Generation Training):
```bash
python train.py --phase 1 --csv_path dataset/dataset_filtered.csv --out output --epochs 100 --batch_size 2 --lr 1e-4 --loss l1
```
Phase 2 (Outcome Prediction):
```bash
python train.py --phase 2 --csv_path dataset/dataset_filtered.csv --out output --epochs 50 --batch_size 2 --phase2_lr 1e-5 --num_views 6 --loss l1
```
Checkpoints and validation logs are saved in the specified output directory.

After training (Phase 2), run Phase 3 to optimize ablative parameters:
```bash
python optimize.py --csv_path dataset/dataset_filtered.csv --output_dir output --num_views 6 --lr 0.01 --max_steps 100 --lambda_reg 0.1 --gaussian_kernel_size 5 --gaussian_sigma 3.0
```
This process loads the best Phase 2 checkpoint, refines the ablative parameters for each patient, and saves visualizations (including original, optimized, and difference maps) to the output directory.


### Command-Line Arguments
- --phase: Phase to run (1 or 2 for training) [required]
- --csv_path: Path to the CSV file with data information.
- --out / --output_dir: Directory to save checkpoints, logs, and visualizations.
- --epochs: Number of training epochs (Phases 1 & 2).
- --batch_size: Batch size for training.
- --lr: Learning rate for Phase 1.
- --phase2_lr: Learning rate for Phase 2.
- --num_views: Number of views per patient.
- --loss: Loss function for simulation (“l1” or “mse”).

For phase 3:
- --max_steps: Number of optimization steps.
- --lambda_reg: Regularization weight for preserving original ablative features.
- --gaussian_kernel_size: Kernel size for Gaussian blur (must be odd).
- --gaussian_sigma: Sigma for Gaussian blur.

### License
MIT License.

