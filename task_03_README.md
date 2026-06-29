# PRODIGY_DS_03: Decision Tree Classifier

## Overview

Build a decision tree classifier to predict whether customers will purchase a product or service based on demographic and behavioral data.

## Dataset

**Bank Marketing Dataset** - UCI Machine Learning Repository
- Target: Customer purchase decision (yes/no)
- Features: Age, job, education, balance, housing loan, contact type, etc.
- 45,211 instances, 17 features

## Requirements

- Python 3.x
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.2
- seaborn >= 0.11.1
- scikit-learn >= 1.0.0

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python task_03_decision_tree.py
```

## Workflow

1. **Load Data** - Import Bank Marketing dataset
2. **Preprocess** - Encode categorical variables, handle missing values
3. **Split Data** - Train/test split (80/20)
4. **Build Model** - Train decision tree classifier
5. **Evaluate** - Calculate accuracy, precision, recall, F1-score
6. **Visualize** - Create performance charts and confusion matrix
7. **Extract Insights** - Identify important features

## Model Performance

- Accuracy, Precision, Recall, F1-Score on test set
- Confusion matrix visualization
- Feature importance ranking
- Performance metrics comparison

## Key Outputs

- task_03_decision_tree.png - 4-panel visualization
- Console metrics (accuracy, precision, recall, F1)
- Feature importance scores
- Classification report

## Decision Tree Parameters

- max_depth: 5 (prevent overfitting)
- min_samples_split: 10 (minimum samples to split)
- min_samples_leaf: 5 (minimum leaf samples)

## Author

Saransh Lives - Prodigy Infotech Data Science Internship
