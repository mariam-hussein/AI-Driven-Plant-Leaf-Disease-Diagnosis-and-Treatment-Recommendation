# Lesion Segmentation Module

This directory contains the training and evaluation scripts for the RF-DETRSegNano lesion segmentation model.

## Lesion Segmentation Model

- RF-DETRSegNano

## Contents

- `train_lesion_segmentation.py` – Training script for the lesion segmentation model.
- `evaluate_lesion_segmentation.py` – Evaluation script used to compute segmentation performance, including mAP, Precision, Recall, F1-score, and confusion matrix.

## lesion_segmentation_best_model
Due to GitHub storage limitations, the best-trained model weights (lesion_segmentation_best_model.pth) can be downloaded from the following Google Drive folder:

**Google Drive:**

https://drive.google.com/drive/folders/1fMPwmNEaPOq8P6NWyboxHJ9dtDS0HnNh?usp=sharing



## Evaluation Metrics

The evaluation script reports:

- mAP@50:95
- mAP@50
- mAP@75
- Precision
- Recall
- F1-score
- Confusion Matrix
