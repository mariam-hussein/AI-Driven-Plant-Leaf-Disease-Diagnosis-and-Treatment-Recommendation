# Classification Dataset

This directory provides access to the classification dataset used in this study.

# 1. Original Classification Dataset

The original classification dataset was constructed from a filtered subset of the PlantVillage dataset.

Plant categories containing only healthy samples (Blueberry healthy, Raspberry healthy, and Soybean healthy) were excluded. For Orange and Squash, which originally contain only diseased samples in PlantVillage, additional healthy leaf images were collected from external sources to ensure that every included plant species contained both healthy and diseased classes.

The complete original classification dataset is available on Kaggle:

https://www.kaggle.com/datasets/memohussein/plant-leaf-healthy-and-disease-dataset/data

# 2. Classification Dataset split (Train/Validation/Test)

The original classification dataset was divided into fixed training, validation, and testing subsets using the provided Python script (generate_train_val_test_split.py), which automatically generates the train/validation/test directory structure and reports the final dataset statistics.

The complete dataset after splitting is publicly available on Kaggle:

https://www.kaggle.com/datasets/memohussein/plant-leaf-healthy-and-disease-dataset-split/data


This is the exact dataset split used for all classification experiments reported in the manuscript, including EfficientNet-B0, Swin-Tiny, and the proposed CNN–Transformer fusion model.

** The repository includes only the dataset splitting script (generate_train_val_test_split.py), while the complete datasets are hosted on Kaggle due to GitHub repository size limitations.


# Dataset Split

- Training set: 70%
- Validation set: 20%
- Test set: 10%

  
  
