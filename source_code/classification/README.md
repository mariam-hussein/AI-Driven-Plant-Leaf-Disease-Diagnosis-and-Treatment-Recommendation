# Classification Module

This directory contains the source code for the proposed hybrid CNN–Transformer plant leaf disease classification framework.

## Model Architecture

The proposed classification model combines EfficientNet-B0 and Swin-Tiny Transformer through feature fusion. EfficientNet-B0 extracts fine-grained local features, while the Swin-Tiny Transformer captures global contextual representations. The extracted features are concatenated and passed to a fully connected classifier for final disease prediction.

## Training Configuration

- Image size: 224 × 224
- Batch size: 32
- Optimizer: AdamW
- Loss function: Weighted Cross-Entropy Loss
- Learning rate: 3e-4
- Learning rate scheduler: OneCycleLR
- Maximum epochs: 50
- Early stopping patience: 15
- Random seed: 42

The complete preprocessing and augmentation pipeline is implemented in `train_classification.py`.

## Directory Contents

- `train_classification.py` – Training and evaluation script for the proposed classification model.

## Pre-trained Models

Due to GitHub storage limitations, the model files can be downloaded from the following Google Drive folder:

**Google Drive:**

https://drive.google.com/drive/folders/1fMPwmNEaPOq8P6NWyboxHJ9dtDS0HnNh?usp=sharing

The folder contains:

- `classification_best_weight_model.pth` – Best model weights selected according to the highest validation accuracy.
- `classification_model_inference.pth` – Deployment-ready inference model containing the trained weights together with the class labels and inference configuration.

## Outputs

The training script generates:

- Best trained model
- Training history
- Classification report
- Confusion matrices
- Accuracy and loss curves
- Precision, Recall, and F1-score curves





