# IEEE_ACCESS_PD-Gait-16Channel-GAF
Deep learning framework for Parkinson's disease classification using 16-channel VGRF-derived Gramian Angular Fields with multi-resolution analysis.
# Multi-Resolution 16-Channel GAF for Parkinson's Disease Classification

This repository contains the source code used in the study:

"Deep Learning-Based Parkinson's Disease Classification Using 16-Channel Gramian Angular Field Representations of Plantar Pressure Signals: A Comparative Study of Input Image Resolutions and CNN–Transformer Architectures"

## Overview

This study investigates Parkinson's disease classification using
16-channel Gramian Angular Field (GAF) representations derived from
plantar vertical ground reaction force (VGRF) signals.

Five GAF resolutions were evaluated:

- 64 × 64
- 128 × 128
- 224 × 224
- 512 × 512
- 1024 × 1024

Five deep learning architectures were evaluated:

- ResNet50
- EfficientNet-B0
- ConvNeXt-Tiny
- ViT-Tiny
- Swin-Tiny

Additional experiments include:

- Channel ablation analysis
- Raw VGRF 1D-CNN baseline
- Grad-CAM visualization
- Subject-level statistical evaluation

## Dataset

The study uses the publicly available PhysioNet
Gait in Parkinson's Disease dataset (version 1.0.0).

The original dataset is not redistributed in this repository.

After downloading the dataset, place it in:

gait-in-parkinsons-disease-1.0.0/

## Experimental Design

The analysis uses:

- 80% development cohort
- 20% independent holdout test cohort
- Stratified subject-level splitting
- Five-fold subject-wise cross-validation within the development cohort
- Random seed: 40
- Subject-level prediction by averaging recording-level probabilities
- 2,000 stratified bootstrap iterations for 95% confidence intervals

## Repository Structure

src/
    01_main_experiment.py
    02_ablation_analysis.py
    03_raw_vgrf_1dcnn_baseline.py
    04_gradcam_analysis.py
    05_roc_confusion_matrix.py

results/
    Selected numerical results reported in the manuscript

docs/
    Experimental configuration and protocol

## Installation

pip install -r requirements.txt

## Running the Main Experiment

python src/01_main_experiment.py

## Ablation Analysis

python src/02_ablation_analysis.py

## Raw VGRF Baseline

python src/03_raw_vgrf_1dcnn_baseline.py

## Grad-CAM

python src/04_gradcam_analysis.py

## Reproducibility

All experiments use subject-level partitioning to prevent recordings
from the same participant from appearing across development and
independent test cohorts.

The independent test set is not used during cross-validation or
model selection.

## License

See LICENSE.
