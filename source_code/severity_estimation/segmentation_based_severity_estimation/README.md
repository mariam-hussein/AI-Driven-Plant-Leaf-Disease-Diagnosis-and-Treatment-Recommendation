# Segmentation-Based Severity Estimation

This directory contains the implementation of the segmentation-based disease severity estimation method.

## Method

Disease severity is estimated by calculating the ratio between the segmented disease lesion area and the segmented leaf area. Leaf and lesion regions are first segmented using two RF-DETRSegNano segmentation models. The disease severity percentage is then computed as:

Severity (%) = (Lesion Area / Leaf Area) × 100

The estimated severity percentage is subsequently assigned to one of the predefined severity levels:

- Early: 1–20%
- Moderate: 21–40%
- Severe: >40%

## Contents

- `segmentation_based_severity_estimation.py` – Performs disease classification, leaf segmentation, lesion segmentation, lesion-to-leaf area calculation, disease severity estimation, and treatment recommendation.

## Output

The script provides:

- Predicted disease class.
- Leaf segmentation result.
- Lesion segmentation result.
- Leaf area.
- Lesion area.
- Disease severity percentage.
- Disease severity level (Healthy, Early, Moderate, or Severe).
- Treatment recommendations based on the predicted disease and severity level.