# Prototype-Based Severity Estimation (Without Leaf Segmentation)

## Description

This directory contains the implementation of the prototype-based disease severity estimation method without applying leaf segmentation.

## Method

Healthy leaf images are directly processed using the trained CNN–Transformer fusion classification model to extract feature vectors. These feature vectors are averaged to construct a representative healthy prototype for each plant species.

During inference, disease severity is estimated by comparing the extracted feature representation of the input image with the corresponding healthy prototype using cosine similarity. The resulting disease score is then mapped to a severity level according to the predefined severity thresholds.

## Contents

- `build_prototypes.py` – Constructs healthy prototypes from the original healthy leaf images.
- `prototype_severity_estimation.py` – Performs prototype-based disease severity estimation using the generated healthy prototypes.

## Output

The prototype construction script generates:

- `*_prototype.npy` – Prototype feature vector obtained by averaging the feature vectors of all healthy leaf images for each plant species.


## Related Classification Models

The classification models used for feature extraction are provided separately in the **source_code/classification** directory of this repository.
