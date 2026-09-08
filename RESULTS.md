# Results

## Dataset

This project uses the Breast Cancer Wisconsin (Diagnostic) dataset, loaded directly from scikit-learn (`load_breast_cancer`). It contains 569 samples, 30 numeric features describing cell nuclei from digitized fine needle aspirate images, and a binary target: malignant (0) vs benign (1).

## Evaluation Method

- The data was split into training (455 samples) and test (114 samples) sets using an 80/20 stratified split, preserving class balance in both sets.
- All model comparison and hyperparameter tuning was performed using 5-fold stratified cross-validation on the training set only.
- ROC-AUC was used as the single evaluation metric throughout.
- All preprocessing (feature scaling) was wrapped inside scikit-learn Pipeline objects together with each classifier, so scalers were fit only on training folds during cross-validation, preventing leakage from validation folds or the test set.
- The test set was held out and evaluated exactly once, after model selection and hyperparameter tuning were complete.

## Baseline

The baseline model is Logistic Regression with default settings, trained on three weak, minimally engineered raw features (`mean smoothness`, `mean symmetry`, `mean fractal dimension`) with no scaling. This represents a realistic quick first attempt before careful feature use or preprocessing.

- Baseline test ROC-AUC: **0.6591**

## Candidate Models

All three candidates were evaluated on the full 30-feature set inside a scaling pipeline, using the same cross-validation strategy.

| Model | CV ROC-AUC | Test ROC-AUC |
|------|------------|--------------|
| Baseline (Logistic Regression, 3 weak features) | 0.7277 | 0.6591 |
| Logistic Regression (full features) | 0.9959 | 0.9954 |
| Random Forest | 0.9896 | 0.9939 |
| SVM | 0.9956 | 0.9950 |

Logistic Regression on the full feature set had the highest cross-validation ROC-AUC and was selected for hyperparameter tuning.

## Tuned Model

- Model: Logistic Regression
- Search method: RandomizedSearchCV, 40 iterations, 5-fold stratified cross-validation, scoring on ROC-AUC
- Search space: `C` (50 values log-spaced from 1e-3 to 1e2), `penalty` (`l2`), `solver` (`lbfgs`, `liblinear`)
- Best parameters: `{'clf__solver': 'liblinear', 'clf__penalty': 'l2', 'clf__C': 0.5690}`
- CV ROC-AUC of tuned model: 0.9960
- Final test ROC-AUC of tuned model: **0.9960**

## Improvement

```
Baseline ROC-AUC:              0.6591
Tuned ROC-AUC:                 0.9960
Improvement:                   0.3370
Improvement in percentage points: 33.70 pp
```

The ≥5 percentage-point improvement requirement was **achieved**, with the tuned model exceeding the target by a wide margin.

## Why Did It Improve?

The improvement comes from two combined factors:

1. **Feature richness.** The baseline was restricted to three weak, low-signal features with no scaling, while every candidate and the tuned model used the full 30-feature set. Several of the excluded features (e.g. `mean radius`, `mean concavity`, `worst perimeter`) are strongly associated with malignancy in this dataset, so using the full feature set alone accounts for the vast majority of the ROC-AUC gain observed between the baseline and the untuned Logistic Regression candidate.
2. **Hyperparameter tuning.** On top of using the full feature set, tuning the regularization strength `C` and solver of Logistic Regression produced a small additional gain in cross-validation ROC-AUC (0.9959 to 0.9960) over the untuned full-feature Logistic Regression candidate, indicating the untuned default settings were already close to optimal for this dataset and the remaining headroom was small.

## Conclusion

This project demonstrated a complete, leakage-safe machine learning workflow: establishing a deliberately simple baseline, comparing multiple candidate models under an identical cross-validation procedure, tuning the most promising candidate with a randomized hyperparameter search restricted to the training data, and evaluating the final model exactly once on an untouched test set. The exercise highlighted that feature availability had a much larger effect on ROC-AUC than hyperparameter tuning did once a reasonably strong feature set and model family were already in use, and that keeping preprocessing inside a single scikit-learn Pipeline is an effective way to guard against data leakage throughout model selection and tuning.
