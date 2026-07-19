# Classification Dataset

This directory provides access to the classification dataset used to train, validate, and test the proposed plant leaf disease classification model.

# Dataset

The complete classification dataset is publicly available on Kaggle:

https://www.kaggle.com/datasets/memohussein/plant-leaf-disease-classification-dataset-split/data


** The repository contains only the dataset splitting script because the complete dataset is hosted on Kaggle due to repository size limitations.

The dataset was constructed using a filtered subset of the PlantVillage dataset. Plant categories containing only healthy samples (i.e., Blueberry healthy, Raspberry healthy, and Soybean healthy) were excluded. For Orange and Squash, which originally contain only diseased samples in PlantVillage, additional healthy leaf images were collected from external sources to ensure that every included plant species contained both healthy and diseased samples.

# Dataset Split

- Training set: 70%
- Validation set: 20%
- Test set: 10%

## Reproducibility

The dataset split was generated using a fixed random seed (42). The corresponding data splitting script (`create_dataset_split.py`) is included in this directory.