# Classification Models


## Description

This directory provides access to the classification component of the proposed framework.


To support the ablation study requested during the review process, three classification models were implemented and evaluated using the same training, validation, and testing split:
- EfficientNet-B0
- Swin-Tiny Transformer
- Proposed CNN–Transformer Fusion (EfficientNet-B0 + Swin-Tiny)
  
The fusion model combines local features extracted by EfficientNet-B0 with global contextual features extracted by Swin-Tiny through feature concatenation, followed by a fully connected classifier for final disease prediction.


## Common Training Configuration
All models were trained using identical experimental settings:
- Batch size: 32
- Optimizer: AdamW
- Loss function: Weighted Cross-Entropy Loss
- Learning rate: 3e-4
- Learning rate scheduler: OneCycleLR
- Maximum epochs: 50
- Early stopping patience: 15
- Random seed: 42


## Classification Models Repository

Due to GitHub repository size limitations, the complete implementation, trained models, and experimental results are available in the following Google Drive repository:

https://drive.google.com/drive/folders/1gZbtnshBHn9MFxzIOqdI0-LFSZbjVx9M?usp=sharing


The Google Drive repository is organized as follows:

```
classification_model_for_reviewer/
│
├── classification_fusion_model/
│   ├── classification_fusion_model_v2 (Google Colab Notebook)
│   └── Result_fusion_model_v2.zip
│
├── classification_EfficientNetB0_model/
│   ├── classification_EffB0_model (Google Colab Notebook)
│   └── Result_EfficientNetB0_model.zip
│
└── classification_SwinTiny_model/
    ├── classification_SwinTiny_model (Google Colab Notebook)
    └── Result_SwinTiny_model.zip
```
    

Each model directory contains:
- Google Colab notebook implementing the corresponding model.
- Compressed experimental results, including:
  - Best trained model.
  - Inference model.
  - Training history.
  - Classification report.
  - 95% confidence intervals.
  - Confusion matrices.
  - Accuracy, Loss, Precision, Recall, and F1-score curves.
  - Failure case analysis.
  - Representative failure cases.
