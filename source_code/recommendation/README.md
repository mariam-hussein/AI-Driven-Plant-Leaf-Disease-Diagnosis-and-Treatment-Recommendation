# Treatment Recommendation

## Description

This directory contains the implementation of the treatment recommendation module proposed in this study.

The recommendation module is based on **TF-IDF** and **Cosine Similarity** retrieval over the proposed **PLDK-TR (Plant Leaf Disease Knowledge and Treatment Recommendation)** dataset  to retrieve treatment recommendations according to the predicted disease class and the estimated disease severity.



## Contents

- **treatment_manager.py**
  - Implements the TF-IDF and Cosine Similarity-based treatment recommendation module.

## Input

The recommendation module requires:

- Predicted disease class.
- Estimated disease severity level (Early, Moderate, or Severe).

## Output

The module returns:

- Recommended treatment.
- Treatment type (Biological or Chemical).
- Mode of action (Contact or Systemic).
- Treatment description.
- Precautionary guidelines.

## Related Dataset

The treatment recommendation knowledge base (**PLDK-TR_Treatment_Recommendation.csv**) is provided separately in the **datasets/recommendation_dataset** directory of this repository.
