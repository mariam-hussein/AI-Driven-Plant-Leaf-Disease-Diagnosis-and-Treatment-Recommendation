# Prototype-Based Severity Estimation (With Leaf Segmentation)

This directory contains the implementation of the prototype-based disease severity estimation method after applying leaf segmentation.

## Method

Healthy leaf images are first processed using the leaf segmentation model to remove the image background while preserving only the leaf region. The segmented healthy images are then passed through the trained hybrid CNN–Transformer classification model (EfficientNet-B0 + Swin-Tiny Transformer) to extract feature vectors. These feature vectors are averaged to construct a representative prototype for each plant species.

During inference, the test image is first segmented to remove the background, classified using the hybrid classification model, and then compared with the corresponding healthy prototype using cosine similarity. The resulting disease score is mapped to a severity level according to the predefined severity thresholds.

## Contents

- `build_prototypes.py` – Constructs healthy prototypes from segmented healthy leaf images.
- `prototype_severity_estimation.py` – Performs prototype-based disease severity estimation using the generated healthy prototypes.


## Output

For each plant species, the prototype construction script generates:

- `*_prototype.npy` – Prototype feature vector obtained by averaging the feature vectors of all segmented healthy leaf images.


The severity estimation script outputs:

- Predicted disease class.
- Health score.
- Disease score.
- Disease severity level (Healthy, Early, Moderate, or Severe).