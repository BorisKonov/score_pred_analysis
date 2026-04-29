# Student Performance Prediction
A regression analysis exploring how well student exam scores can be predicted - first only from demographic data, then by utilising other exam results. Built as a structured, three-approach comparison to understand the limits and power of different sets.

---
## Project Overview
| | |
|---|---|
| **Dataset** | [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) |
| **Records** | 1,000 students |
| **Features** | 5 demographic + up to 2 exam scores |
| **Targets** | Average score (A1, A2) · Math score (A3) |
| **Models** | Linear Regression · Ridge Regression |
| **Best R²** | 0.88 (Approach 3) |
---
## The Three Approaches Being Compared
All three approaches use the same dataset and amd seed (```random_state = 42```), making results directly comparable

### Approach 1 — Linear Regression · Demographics → Average Score

**Question:** Can demographics alone predict academic performance?

Uses only the 5 categorical features in the dataset (gender, race/ethinicty, parental education, lunch type, test preparation) to predict the average of all three exam scores via plain Linear Regression.

**Results:**

| Metric | Value |
|--------|-------|
| R²     | 0.1622 |
| MAE    | 10.49 |

**Findings:** Demographics alone explain about 16% of score variance. The two strongest predictors are lunch type(related to socioeconomic status) and test preparation completion. The results do not show a failure of the model, but the true ceiling of what these features can predict

---
### Approach 2 - Ridge Regression · Demographics → Average Score
**Question:** Would regularisation improve on Approach 1?

Identical feature set and targets as Approach 1, but uses Ridge Regression to test whether L2 regularisation adds value on low-dimensional binary features.

| Metric | Value |
|--------|-------|
| R²     | 0.1621 |
| MAE    | 10.49 |

**Findings:** Ridge and Linear Regression perform identically here. With only around 12 one-hot columns and no multicollinearity, regularisation has nothing to do. The result confirms Approach 1 is already at the demographic ceiling.

---

### Approach 3 — Ridge Regression · Reading + Writing + Demographics → Math Score

**Question:** If we know the literacy scores of a student, how well can we predict their math score?

Adds reading score and writing score as continuous features alongside demographics. Predicts math scores specifically, using a Ridge + StandardScaler pipeline. 

| Metric | Value |
|--------|-------|
| R²     | 0.8805 |
| MAE    | 4.21  |

**Finding:** R2 jumps from 0.16 to **0.88** - a 5X improvement, mainly driven by the strong cross-subject correlation (reading-writing r = 0.955, math-reading r = 0.818). The model predicts math scores to within ±4.2 points on a 0–100 scale.

---
## Results Summary
| Approach | Model | Features | Target | R² | MAE |
|----------|-------|----------|--------|----|-----|
| A1 | Linear Regression | Demographics | Avg score | 0.162 | 10.49 |
| A2 | Ridge Regression | Demographics | Avg score | 0.162 | 10.49 |
| A3 | Ridge Regression | Reading + Writing + Demographics | Math score | 0.881 | 4.21 |

---

## Key Conclusions
**Demographics are weak but not useless.** A school can use Approaches 1 & 2 *before* any exams to identify students at risk of underperforming, based on enrolment data. Socioeconomic factors (free/reduced lunch, test prep absence) are the most actionable levers.

**Exam scores are the dominant signal.** Once reading and writing results are available, math scores can be predicted with high accuracy. This makes Approach 3 useful for midterm interventions - flagging students whose math performance is unexpectedly low relative to their literacy.

**Regularisation is not always the answer:** Ridge did not outperform plain Linear Regression on this dataset. Choosing the right *features* mattered far more than choosing the right *model*.

---

## ! A Note on Data Leakage ! 
An earlier version of this code contained a subtle but very crucial bug: `average score` was left inside the feature matrix `X` while simultaneously being the target `y`. This caused artificially perfect R2 scores (1.00 in Approach 2, 0.999 in Approach 3). The bug was fixed and all three approaches in this repository use correctly separated feature and target sets.

---

## Requirements
```
pandas>=1.3
numpy>=1.21
scikit-learn>=1.0
matplotlib>=3.4
seaborn>=0.11
jupyter>=1.0
```

---

## About the Author
I am Boris, and I am passionate about AI, Machine Learning, and optimising processes. This is my first machine learning project and I wanted to showcase what I have learned so far and touch upon feature engineering which has been one of the more interesting parts of Machine Learning for me. 
