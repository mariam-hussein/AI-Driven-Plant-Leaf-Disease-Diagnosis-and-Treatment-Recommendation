# AI-Driven Plant Leaf Disease Diagnosis and Treatment Recommendation

This repository contains the datasets, source code, trained models, and supplementary materials associated with the proposed framework for plant leaf disease diagnosis, severity estimation, and treatment recommendation.

## Repository Structure

```
AI-Driven-Plant-Leaf-Disease-Diagnosis-and-Treatment-Recommendation/
│
├── datasets/
│   ├── classification_dataset/
│   ├── segmentation_dataset/
│   └── recommendation_dataset/
│
├── source_code/
│   ├── classification/
│   ├── leaf_and_lesion_segmentation_models/
│   ├── severity_estimation/
│   └── recommendation/
│
└── classification_model_on_external_data/
```

## Repository Contents

### datasets

Contains all datasets used in this study, including:

- Classification dataset
- Segmentation datasets
- PLDK-TR treatment recommendation dataset

### source_code

Contains the implementation of all stages of the proposed framework, including:

- EfficientNetB0, Swin-Tiny Transformer, and CNN–Transformer classification models
- RF-DETRSegNano leaf and lesion segmentation models
- Prototype-based severity estimation
- Segmentation-based severity estimation
- Treatment recommendation module

### classification_model_on_external_data

Contains the implementation and experimental results of evaluating the proposed CNN–Transformer fusion classification model on an independent external dataset consisting of real-world plant leaf images.

---

Please refer to the **README.md** file inside each directory for detailed descriptions and usage instructions.
