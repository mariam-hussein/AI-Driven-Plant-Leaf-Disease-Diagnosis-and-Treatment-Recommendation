# Leaf Segmentation Module

This directory contains the training and evaluation scripts for the RF-DETRSegNano leaf segmentation model.

## Leaf Segmentation Model

- RF-DETRSegNano

## Directory Contents

- `train_leaf_segmentation.py` – Training script for the leaf segmentation model.
- `evaluate_leaf_segmentation.py` – Evaluation script used to compute segmentation performance, including mAP, Precision, Recall, F1-score, and confusion matrix.

## leaf segmentation best Model

Due to GitHub storage limitations, the best-trained model weights (leaf_segmentation_best.pth) can be downloaded from the following Google Drive folder:

**Google Drive:**
https://drive.google.com/drive/folders/1fMPwmNEaPOq8P6NWyboxHJ9dtDS0HnNh?usp=sharing

The folder contains:

- `leaf_segmentation_best.pth` – Best trained RF-DETRSegNano leaf segmentation model.

## Evaluation Metrics

The evaluation script reports:

- mAP@50:95
- mAP@50
- mAP@75
- Precision
- Recall
- F1-score
- Confusion Matrix