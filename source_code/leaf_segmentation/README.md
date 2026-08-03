# Leaf and Lesion Segmentation Models

## Description


This directory provides access to the segmentation models (RF-DETRSegNano leaf and RF-DETRSegNano lesion) developed in this study.

Two independent segmentation models were implemented for different purposes within the proposed framework:

- **RF-DETRSegNano Leaf Segmentation Model**, used to extract the leaf region from the original image.

- **RF-DETRSegNano Lesion Segmentation Model**, used to segment disease lesions.

## Segmentation Models Repository

Due to GitHub repository size limitations, the trained models and experimental results are available in the following Google Drive repository:

https://drive.google.com/drive/folders/1anYgPvBEezlmd5cBRr4NhY5E5DXaWZrv?usp=sharing

The Google Drive repository is organized as follows:

```
leaf_and_lesion_segmentation_model_for_reviewer/
│
├── leaf_segmentation/
│   ├── leaves_segmentation_model.ipynb
│   └── Result_Leaves_segmentation_model.zip
│
└── lesion_segmentation/
    ├── Lesion_segmentation_model.ipynb
    ├── Evaluate_Lesion_segmentation_model.ipynb
    └── Result_Lesion_segmentation_model.zip
```

## RF-DETRSegNano Leaf Segmentation Model

Contains:

- Google Colab notebook for training the leaf segmentation model.
- Compressed experimental results, including:
  - Trained model checkpoints.
  - Evaluation outputs.
  - Prediction results.
  - Training logs.
  - Performance metrics.

## RF-DETRSegNano Lesion Segmentation Model

Contains:

- Google Colab notebook for training the lesion segmentation model.
- Google Colab notebook for model evaluation.
- Compressed experimental results, including:
  - Trained model checkpoints.
  - Evaluation outputs.
  - Prediction results.
  - Training logs.
  - Performance metrics.

## Segmentation Datasets

The datasets used to train both RF-DETRSegNano segmentation models are provided separately in the **segmentation_dataset** directory of this repository.
