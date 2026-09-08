# Data

This project uses the Breast Cancer Wisconsin (Diagnostic) dataset, which ships directly with scikit-learn via `sklearn.datasets.load_breast_cancer`. No manual download or external files are required.

- Samples: 569
- Features: 30 numeric features computed from digitized images of breast mass fine needle aspirates
- Target: binary classification, malignant (0) vs benign (1)
- Source: UCI Machine Learning Repository, bundled with scikit-learn

The dataset is loaded programmatically inside `src/data_loader.py`, so this folder contains no raw data files.
