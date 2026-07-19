# Treatment Recommendation

This directory contains the implementation of the treatment recommendation module proposed in this work.

## Contents

- `treatment_manager.py` – Implements the TF-IDF and cosine similarity-based treatment recommendation module.

## Input

The module requires:

- Predicted disease class.
- Disease severity level (Early, Moderate, or Severe).

## Output

The module returns:

- Recommended treatment(s).
- Treatment type.
- Mode of action.
- Treatment description.
- Precautions.
