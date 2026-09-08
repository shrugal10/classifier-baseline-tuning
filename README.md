# Tune a Classifier to Beat a Baseline by 5 Points

## Overview

This project builds a rigorous, leakage-free machine learning pipeline on a small public classification dataset. It establishes a simple baseline, compares three candidate models under identical cross-validation, tunes the best candidate, and shows that the tuned model improves test ROC-AUC over the baseline by well more than 5 percentage points.

## Objective

Achieve at least a 5 percentage point improvement in ROC-AUC on a held-out test set, comparing a tuned model against a properly defined baseline, without any data leakage between training and test data.

## Dataset

The Breast Cancer Wisconsin (Diagnostic) dataset, loaded directly from scikit-learn (`sklearn.datasets.load_breast_cancer`). 569 samples, 30 numeric features, binary target (malignant vs benign). No manual download required.

## Models

- **Baseline:** Logistic Regression on three weak, minimally engineered features, no scaling.
- **Candidates (full feature set, scaled):** Logistic Regression, Random Forest, SVM.
- **Tuned model:** Logistic Regression, hyperparameters selected with RandomizedSearchCV.

## Evaluation Methodology

- 80/20 stratified train/test split.
- 5-fold stratified cross-validation for model comparison and hyperparameter tuning, using only the training set.
- Single metric: ROC-AUC.
- All preprocessing wrapped inside scikit-learn Pipelines to prevent data leakage.
- Test set evaluated exactly once, after model selection and tuning were finalized.

## Repository Structure

```
classifier-baseline-tuning/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── classifier_tuning.ipynb
│
├── src/
│   ├── __init__.py
│   └── data_loader.py
│
├── RESULTS.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation Instructions

### 1. Clone or download this repository

```
git clone https://github.com/<your-username>/classifier-baseline-tuning.git
cd classifier-baseline-tuning
```

### 2. Create and activate a virtual environment (Windows)

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

## How to Run the Notebook

```
jupyter notebook
```

Then open `notebooks/classifier_tuning.ipynb` in the browser tab that opens, and run all cells from top to bottom (Kernel > Restart & Run All).

## How to Reproduce Results

Running the notebook top to bottom regenerates every number in `RESULTS.md` from scratch: the data load, train/test split, cross-validation scores, hyperparameter search, and final test ROC-AUC values. No results are hard-coded; a fixed random seed (`RANDOM_STATE = 42`) is used throughout for reproducibility.

## Final Result

- Baseline test ROC-AUC: **0.6591**
- Tuned model test ROC-AUC: **0.9960**
- Improvement: **33.70 percentage points**
- Target of ≥5 percentage points: **achieved**

See `RESULTS.md` for the full breakdown, including the model comparison table and an explanation of the improvement.

## Technologies Used

- Python
- scikit-learn
- pandas / numpy
- matplotlib / seaborn
- Jupyter Notebook
- VS Code
- Git / GitHub
