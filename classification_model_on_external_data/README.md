# Classification Model on External Dataset

## Description

This directory contains the implementation and experimental results of evaluating the proposed CNN–Transformer fusion classification model on an independent external dataset consisting of real-world plant leaf images.

The proposed CNN–Transformer fusion classification model was retrained on the external dataset using the same preprocessing, training protocol, and hyperparameters as those used for the PlantVillage dataset.

Unlike the PlantVillage dataset, the external dataset contains images captured under real-world conditions, including variations in illumination, complex backgrounds, occlusions, and other environmental factors, making the classification task more challenging.

## External Dataset

The external dataset originally contained **5,271** plant leaf images distributed across **19 classes**. Two classes (**Potato leaf** and **Tomato two-spotted spider mites leaf**) containing fewer than 20 images were excluded to ensure sufficient samples for training and evaluation.

The final dataset consists of:

- **5,258 images**
- **17 classes**

The dataset was divided into:

- Training: **70%**
- Validation: **20%**
- Testing: **10%**

## Classification Performance

The retrained CNN–Transformer fusion classification model achieved **Accuracy:** 92.21%

## External Evaluation Repository

Due to GitHub repository size limitations, the complete implementation, external dataset split, and experimental results are available in the following Google Drive repository:

https://drive.google.com/drive/folders/1LiGeCKC9wpa0-48B6JLYDaM2cNW2NEtL?usp=sharing

The Google Drive repository is organized as follows:

```
classificationModel_evaluation_external_dataset_for_reviewer/
│
├── classificationModel_on_external_dataset (Google Colab Notebook)
├── Result_classificationModel_on_external_dataset.zip
           - Classification report.
           - Confusion matrix.
           - Training history.
           - Accuracy and loss curves.
           - Precision, Recall, and F1-score.
           - Representative failure cases.
           - Confidence interval
└── externalData_split.zip
```

## Purpose

This experiment was conducted to evaluate the generalization capability of the proposed CNN–Transformer fusion classification model under real-world conditions and to complement the evaluation performed on the PlantVillage dataset.






