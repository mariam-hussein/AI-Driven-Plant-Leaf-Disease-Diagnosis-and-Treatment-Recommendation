# Severity Estimation

## Description

This directory contains the two disease severity estimation approaches proposed in this study:

- **Segmentation-Based Severity Estimation**
- **Prototype-Based Severity Estimation**

The segmentation-based approach estimates disease severity from the lesion-to-leaf area ratio obtained using the RF-DETRSegNano leaf and lesion segmentation models.

The prototype-based approach estimates disease severity by comparing the feature representation extracted by the trained CNN–Transformer fusion classification model with the corresponding healthy prototype using cosine similarity.

## Directory Structure
```
severity_estimation/
│
├── prototype_based_severity_estimation/
│   ├── with_leaf_segmentation/
│   │   ├── build_prototypes.py
│   │   ├── prototype_severity_estimation.py
│   │   └── README.md
│   │
│   ├── without_leaf_segmentation/
│   │   ├── build_prototypes.py
│   │   ├── prototype_severity_estimation.py
│   │   └── README.md
│   │
│   └── README.md
│
└── segmentation_based_severity_estimation/
    ├── segmentation_based_severity_estimation.py
    └── README.md
```
