# Segmentation-Based Severity Estimation

## Description

This directory contains the implementation of the segmentation-based disease severity estimation method.

## Method

Disease severity is estimated by calculating the ratio between the segmented disease lesion area and the segmented leaf area. The proposed RF-DETRSegNano leaf and lesion segmentation models are used to obtain the leaf and lesion regions required for severity estimation.

The disease severity percentage is computed as:

Severity (%) = (Lesion Area / Leaf Area) × 100

The estimated severity percentage is subsequently assigned to one of the predefined severity levels:

- Early: 1–20%
- Moderate: 21–40%
- Severe: >40%

## Contents

- `segmentation_based_severity_estimation.py` – Implements the segmentation-based disease severity estimation method using leaf and lesion segmentation results.


## Related Segmentation Models

The RF-DETRSegNano leaf and lesion segmentation models are provided separately in the **source_code/leaf_and_lesion_segmentation_models** directory of this repository.

## Related Classification Models

The CNN–Transformer fusion classification model used to identify the predicted disease class is provided separately in the **source_code/classification** directory of this repository.
