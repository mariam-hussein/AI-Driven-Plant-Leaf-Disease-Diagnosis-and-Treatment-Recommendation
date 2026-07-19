# Prototype-Based Severity Estimation (Without Leaf Segmentation)

This directory contains the implementation of the prototype-based disease severity estimation method without applying leaf segmentation.

## Method

Healthy leaf images are directly processed using the trained hybrid CNN–Transformer classification model (EfficientNet-B0 + Swin-Tiny Transformer) to extract feature vectors. These feature vectors are averaged to construct a representative healthy prototype for each plant species.

During inference, the input image is classified using the hybrid CNN–Transformer classification model. If the predicted class is healthy, the image is directly assigned to the **Healthy** severity level. Otherwise, the extracted feature vector is compared with the corresponding healthy prototype using cosine similarity. The resulting disease score is then mapped to a severity level according to the predefined severity thresholds.

## Contents

- `build_prototypes.py` – Constructs healthy prototypes from the original healthy leaf images.
- `prototype_severity_estimation.py` – Performs prototype-based disease severity estimation using the generated healthy prototypes.


## Output

The prototype construction script generates:

- `*_prototype.npy` – Prototype feature vector obtained by averaging the feature vectors of all healthy leaf images for each plant species.


The severity estimation script outputs:

- Predicted disease class.
- Health score.
- Disease score.
- Disease severity level (Healthy, Early, Moderate, or Severe).