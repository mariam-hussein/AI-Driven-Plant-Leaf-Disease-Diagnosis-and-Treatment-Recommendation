# Prototype-Based Severity Estimation (With Leaf Segmentation)

This directory contains the implementation of the prototype-based disease severity estimation method after applying leaf segmentation.

## Method

Healthy leaf images are first processed using the leaf segmentation model to remove the image background while preserving only the leaf region. The segmented healthy images are then passed through the trained CNN–Transformer fusion classification model (EfficientNet-B0 + Swin-Tiny Transformer) to extract feature vectors. These feature vectors are averaged to construct a representative prototype for each plant species.

During inference, disease severity is estimated by comparing the extracted feature representation of the segmented leaf image with the corresponding healthy prototype using cosine similarity. The resulting disease score is then mapped to a severity level according to the predefined severity thresholds.

## Contents

- `build_prototypes.py` – Constructs healthy prototypes from segmented healthy leaf images.
- `prototype_severity_estimation.py` – Performs prototype-based disease severity estimation using the generated healthy prototypes.

For each plant species, the prototype construction script generates:

- `*_prototype.npy` – Prototype feature vector obtained by averaging the feature vectors of all segmented healthy leaf images.


