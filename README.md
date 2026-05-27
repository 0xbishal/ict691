

# ICT619 Artificial Intelligence — Machine Learning-Based Predictive Maintenance System

> **Client:** WestMine Resources  
> **Course:** ICT619 Artificial Intelligence  
> **Team:**
> - Bishal Neopaney
> - Ugyen Dorji
> - Binisha Sherchan

---

## Overview

This project builds a predictive maintenance system for industrial machinery using supervised machine learning. Two ensemble classifiers — **Random Forest** and **XGBoost** — are trained and compared to detect machine failures before they occur, helping reduce unplanned downtime and maintenance costs.

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**

- Loaded directly from a public GitHub repository (`ai4i2020.csv`)
- Contains sensor readings from industrial machines along with a binary failure label
- Features: `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
- Target: `Machine failure` (0 = No failure, 1 = Failure)
- Exhibits significant class imbalance (~3.4% failure rate)

---

## Project Structure

| Cell | Description |
|------|-------------|
| 1 | Install and import all required libraries |
| 2 | Load dataset from GitHub |
| 3 | Exploratory Data Analysis (EDA) — class distribution, feature histograms, correlation heatmap |
| 4 | Preprocessing — drop non-informative columns, encode categorical features, normalise with `StandardScaler` |
| 5 | Train/test split (80/20) and SMOTE oversampling to handle class imbalance |
| 6 | Train Random Forest classifier with 5-fold cross-validation |
| 7 | Train XGBoost classifier with 5-fold cross-validation |
| 8 | Evaluate both models — accuracy, precision, recall, F1, ROC-AUC |
| 9 | Plot confusion matrices |
| 10 | Plot ROC curves |
| 11 | Plot feature importance for both models |
| 12 | Side-by-side model comparison summary and bar chart |
| 13 | Inference example — predict failure probability on a new sensor reading |

---

## Methodology

### Preprocessing
- Dropped `UDI`, `Product ID`, and individual failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`)
- Label-encoded the `Type` column (L / M / H → 0 / 1 / 2)
- Standardised all numeric features using `StandardScaler`

### Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) on the training set to balance classes before model training

### Models

**Random Forest**
```
n_estimators=200, max_depth=10, min_samples_split=5,
min_samples_leaf=2, class_weight='balanced'
```

**XGBoost**
```
n_estimators=200, learning_rate=0.1, max_depth=6,
subsample=0.8, colsample_bytree=0.8, scale_pos_weight=<computed>
```

Both models are evaluated with **5-fold Stratified Cross-Validation**.

---

## Evaluation Metrics

Each model is assessed on the held-out test set using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Outputs include a full `classification_report`, confusion matrix heatmaps, and overlaid ROC curves for direct comparison.

---

## Requirements

```
xgboost
scikit-learn
imbalanced-learn
matplotlib
seaborn
pandas
numpy
```

Install all dependencies with:

```bash
pip install xgboost scikit-learn imbalanced-learn matplotlib seaborn
```

---

## Usage

1. Clone or download the notebook `ICT619_Predictive_Maintenance.ipynb`
2. Run all cells in order (top to bottom)
3. The dataset is fetched automatically from GitHub — no manual download required
4. To predict on new data, modify the `new_data` dictionary in **Cell 13** with your sensor readings

---

## Inference Example

The final cell demonstrates real-time inference. Given a new set of sensor readings, both models output a prediction and a failure probability:

```
=== Maintenance Alert System ===
  Random Forest → FAILURE PREDICTED (probability: XX.XX%)
  XGBoost       → FAILURE PREDICTED (probability: XX.XX%)
```
