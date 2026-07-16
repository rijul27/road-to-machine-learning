# Logistic Regression 

## Table of Contents

1. [What is Logistic Regression?](#1-what-is-logistic-regression)
2. [Why "Regression"? Regression vs Classification](#2-why-regression-regression-vs-classification)
3. [Real-World Applications](#3-real-world-applications)
4. [Advantages & Limitations](#4-advantages--limitations)
5. [Mathematical Foundations](#5-mathematical-foundations)
6. [End-to-End Code Walkthrough](#6-end-to-end-code-walkthrough)
7. [Evaluation Metrics In Depth](#7-evaluation-metrics-in-depth)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Handling Imbalanced Datasets](#9-handling-imbalanced-datasets)
10. [Multiclass Logistic Regression](#10-multiclass-logistic-regression)
11. [Regularization Deep Dive](#11-regularization-deep-dive)
12. [Assumptions of Logistic Regression](#12-assumptions-of-logistic-regression)
13. [Common Beginner Mistakes](#13-common-beginner-mistakes)
14. [Best Practices Checklist](#14-best-practices-checklist)
15. [Interview Questions & Answers](#15-interview-questions--answers)
16. [Cheat Sheet](#16-cheat-sheet)
17. [Glossary](#17-glossary)

---

## 1. What is Logistic Regression?

Logistic Regression is a **supervised machine learning algorithm used for classification** — predicting which category something belongs to, not a numeric value.

It answers yes/no (or category A/B/C) questions, while also telling you *how confident* it is:

- Will this email be **spam** or **not spam**?
- Is this tumor **malignant** or **benign**?
- Will this customer **churn** or **stay**?

**How it works, in one sentence:** it computes a weighted sum of the input features (like Linear Regression does), then squashes that sum through a **Sigmoid function** so the output is always a probability between 0 and 1. If that probability is ≥ 0.5, predict class 1; otherwise predict class 0.

```
Inputs (x1, x2, ...) → weighted sum (z = w·x + b) → Sigmoid(z) → probability (0 to 1) → threshold → class label
```

---

## 2. Why "Regression"? Regression vs Classification

This is the single most common point of confusion for beginners.

**Why the name "regression":** Logistic Regression fits a **linear equation** to the **log-odds** of the outcome (explained in Section 5.3). Fitting a linear equation to a target value is exactly what "regression" means in statistics — the name describes the internal *method*, not the final *output*. The classification step (turning a probability into a label) is a small addition on top of a fundamentally regression-based technique.

| | Linear Regression | Logistic Regression |
|---|---|---|
| **Predicts** | A continuous number (price, temperature, salary) | A category / class label |
| **Output range** | Any real number (−∞ to +∞) | Probability between 0 and 1 |
| **Example question** | "What will the house sell for?" | "Will the house sell within 30 days?" |
| **Core function** | Straight line: `y = wx + b` | Sigmoid-squashed line: `sigmoid(wx + b)` |
| **Loss function** | Mean Squared Error | Binary Cross-Entropy |
| **Evaluation** | R², MAE, RMSE | Accuracy, Precision, Recall, ROC-AUC |

---

## 3. Real-World Applications

- **Healthcare** — disease diagnosis from test results, predicting patient readmission
- **Finance** — credit scoring, loan default prediction, fraud detection
- **Marketing** — click-through prediction, customer churn
- **Tech** — spam filtering, sentiment classification (positive/negative)
- **HR** — predicting employee attrition
- **Insurance** — claim approval / risk classification

---

## 4. Advantages & Limitations

### Advantages
- Simple to understand, implement, and explain to non-technical people
- Fast to train, even on large datasets
- Outputs actual **probabilities**, not just hard labels — useful for ranking/risk scoring
- Coefficients are interpretable — you can see which features push predictions up or down
- A strong, cheap baseline before trying more complex models

### Limitations
- Assumes a roughly **linear relationship** between features and the log-odds of the outcome
- Sensitive to **outliers** and unscaled features
- Struggles with **imbalanced datasets** without extra techniques (Section 9)
- Assumes features aren't too strongly correlated with each other (multicollinearity)
- Native support is for **binary** outcomes — multiple classes need extensions (Section 10)

---

## 5. Mathematical Foundations

### 5.1 The Linear Part

Just like Linear Regression, we first compute a weighted sum of the inputs:

$$z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$

`z` can be any real number — but we need a probability (0 to 1). That's what the next 3 steps solve.

### 5.2 Odds

**Odds** compare the probability of an event happening to it *not* happening:

$$\text{Odds} = \frac{p}{1-p}$$

Example: a 75% chance of rain gives odds of `0.75 / 0.25 = 3` ("3 to 1 in favor of rain").

### 5.3 Log-Odds (the "Logit")

Odds are always positive and not symmetric, which makes them awkward for a straight line to model. Taking the natural log fixes this — this is the **logit function**:

$$\text{logit}(p) = \ln\left(\frac{p}{1-p}\right)$$

Logistic Regression fits its linear equation to *this* value:

$$\ln\left(\frac{p}{1-p}\right) = z = w_1x_1 + \dots + b$$

### 5.4 The Sigmoid Function

To convert `z` back into a usable probability, apply the inverse of the logit — the **Sigmoid function**:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

![Sigmoid / Logistic curve](https://commons.wikimedia.org/wiki/Special:FilePath/Logistic-curve.svg)

*Any real number `z` (x-axis) is squashed into a value between 0 and 1 (y-axis). Large positive `z` → near 1. Large negative `z` → near 0. `z = 0` → exactly 0.5. (Image: Wikimedia Commons)*

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))

plt.plot(z, sigmoid)
plt.axhline(0.5, color="red", linestyle="--")
plt.xlabel("z"); plt.ylabel("Probability"); plt.title("Sigmoid Function")
plt.show()
```

### 5.5 Decision Boundary

Once we have a probability, we need a rule to turn it into a label:

- `sigmoid(z) ≥ 0.5` → predict **Class 1**
- `sigmoid(z) < 0.5` → predict **Class 0**

The line (or curve, with more features) where the model is exactly 50/50 is the **decision boundary**. 0.5 is just the *default* threshold — Section 7.5 covers picking a better one.

### 5.6 Cost Function: Binary Cross-Entropy (BCE)

While training, the model needs a way to measure "how wrong" it is, so it can improve. We use **Binary Cross-Entropy Loss** (a.k.a. Log Loss):

$$\text{BCE} = -\frac{1}{n}\sum_{i=1}^{n} \Big[ y_i \ln(\hat{y}_i) + (1-y_i)\ln(1-\hat{y}_i) \Big]$$

In plain words: it **heavily penalizes confident wrong predictions** (e.g. predicting 0.99 for a sample that's actually class 0), and barely penalizes predictions that are right or only slightly off.

**Why not Mean Squared Error (used in Linear Regression)?** MSE combined with the Sigmoid function produces a "bumpy" (non-convex) loss surface with multiple local minima, making it hard for gradient descent to reliably find the best weights. BCE stays smooth (convex) for this problem, so training is more reliable.

### 5.7 How Training Actually Works: Gradient Descent

The model starts with random weights, then repeats this loop:

1. Compute predictions for the training data using current weights.
2. Compute the BCE loss (how wrong those predictions are).
3. Compute the **gradient** — the direction that would make the loss worse.
4. Update each weight a small step in the *opposite* direction (this is what "learning" means here).
5. Repeat until the loss stops improving much.

scikit-learn handles all of this internally when you call `.fit()` — you don't need to implement it by hand, but it helps to know what's happening under the hood.

---

## 6. End-to-End Code Walkthrough

A complete binary classification pipeline, following the standard workflow:

```
Import Libraries → Generate/Load Data → Explore → Visualize → Select X, y →
Train-Test Split → Feature Scaling → Create Model → Fit → Predict →
Predict Probabilities → Evaluate
```

```python
# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Generate a practice dataset (in real projects: pd.read_csv(...) instead)
X, y = make_classification(n_samples=1000, n_features=4, n_informative=3,
                            n_redundant=1, random_state=42)

# 3. Explore
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["target"] = y
print(df.shape)
print(df.info())
print(df["target"].value_counts())   # check class balance

# 4. Visualize class balance
sns.countplot(x="target", data=df)
plt.title("Class Distribution")
plt.show()

# 5. Select X and y
X = df.drop("target", axis=1)
y = df["target"]

# 6. Train-test split (stratify keeps class ratio consistent in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 7. Feature scaling — fit ONLY on training data, then transform both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Create and train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 9. Predict
y_pred = model.predict(X_test_scaled)                 # hard labels
y_prob = model.predict_proba(X_test_scaled)[:, 1]      # probability of class 1

# 10. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

---

## 7. Evaluation Metrics In Depth

### 7.1 Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | True Negative (TN) | False Positive (FP) |
| **Actual 1** | False Negative (FN) | True Positive (TP) |

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0","Pred 1"], yticklabels=["Actual 0","Actual 1"])
plt.show()
```

### 7.2 Accuracy, Precision, Recall, F1

| Metric | Formula | What it tells you |
|---|---|---|
| Accuracy | `(TP+TN) / Total` | Overall correctness — **misleading on imbalanced data** |
| Precision | `TP / (TP+FP)` | Of everything predicted positive, how much was actually positive |
| Recall (Sensitivity / TPR) | `TP / (TP+FN)` | Of everything actually positive, how much did we catch |
| F1-score | `2 × (Precision×Recall)/(Precision+Recall)` | Harmonic mean — balances both |
| False Positive Rate (FPR) | `FP / (FP+TN)` | How often we falsely raise an alarm |

With **3+ classes**, `classification_report` also shows:
- **macro avg** — averages the metric across classes equally (good for checking fairness to small classes)
- **weighted avg** — averages weighted by class size (closer to overall accuracy)

### 7.3 ROC Curve & AUC Score

The ROC curve plots **TPR vs FPR** across every possible threshold at once, instead of being locked into 0.5.

```python
from sklearn.metrics import roc_curve, roc_auc_score

auc = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, marker='.', label=f'Model (AUC = {auc:.2f})')
plt.plot([0,1], [0,1], linestyle='--', label='Random guess (AUC = 0.5)')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(); plt.show()
```

- **AUC = 0.5** → no better than random guessing
- **AUC = 1.0** → perfect separation between classes
- Best suited to roughly **balanced** datasets — see below for why.

### 7.4 Precision-Recall Curve (better for imbalanced data)

ROC can look deceptively good on imbalanced data, because huge numbers of true negatives keep FPR low. The Precision-Recall curve avoids this by only looking at the positive class.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)
no_skill = (y_test == 1).sum() / len(y_test)   # baseline = fraction of positives

plt.plot(recall, precision, marker='.', label=f'Model (AP={avg_precision:.2f})')
plt.axhline(no_skill, linestyle='--', label=f'No-skill baseline ({no_skill:.2f})')
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.show()
```

### 7.5 Choosing the Best Threshold (not always 0.5)

Different thresholds trade TPR against FPR. Which is "correct" depends on real-world costs:
- Missing a sick patient (low TPR) is often worse than a false alarm → prefer a **lower** threshold.
- Wrongly blocking an important email is worse than missing one spam email → prefer a **higher** threshold.

One systematic method — **Youden's J statistic** — picks the threshold that maximizes `TPR - FPR`:

```python
J = tpr - fpr
best_index = np.argmax(J)
best_threshold = thresholds[best_index]
print("Best threshold:", round(best_threshold, 3))
```

---

## 8. Hyperparameter Tuning

Instead of guessing settings by hand, `GridSearchCV` and `RandomizedSearchCV` test combinations automatically using cross-validation.

| | GridSearchCV | RandomizedSearchCV |
|---|---|---|
| **Strategy** | Tries **every** combination in the grid | Tries a **random sample** of combinations |
| **Speed** | Slower (exhaustive) | Faster (skips most combinations) |
| **Best for** | Small search spaces | Large search spaces |
| **Guarantee** | Best combo *within the grid given* | May miss the absolute best, but close, much faster |

### Key hyperparameters

- **`C`** — inverse of regularization strength. Small `C` → stronger regularization (simpler model). Large `C` → weaker regularization (fits training data more closely).
- **`penalty`** — `l1` (can zero out coefficients), `l2` (shrinks smoothly, default), `elasticnet` (mix of both).
- **`solver`** — optimization algorithm. Not every solver supports every penalty (`liblinear`/`saga` support `l1`; most others only support `l2`).

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"]
}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train_scaled, y_train)
print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

# RandomizedSearchCV — same idea, but samples n_iter random combinations
random_search = RandomizedSearchCV(LogisticRegression(), param_grid, n_iter=5, cv=5, random_state=42)
random_search.fit(X_train_scaled, y_train)
```

---

## 9. Handling Imbalanced Datasets

Many real problems (fraud, rare disease, churn) have far more of one class than the other, e.g. 99% "not fraud," 1% "fraud." A model can hit 99% accuracy just by always predicting "not fraud" — while being completely useless.

```python
# Proof: a "dummy" classifier that always predicts 0 still scores high accuracy
dummy_pred = np.zeros_like(y_test)
print("Dummy accuracy:", accuracy_score(y_test, dummy_pred))  # deceptively high
```

### `class_weight`

Makes mistakes on the minority class more "costly" during training:

```python
model = LogisticRegression(class_weight={0: 1, 1: 50})
# or let sklearn compute it automatically:
model = LogisticRegression(class_weight="balanced")
```

### `stratify=y` and `StratifiedKFold`

Keeps the same class ratio in every train/test split and every cross-validation fold, so no fold accidentally ends up with almost no minority-class examples:

```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Other techniques
- **SMOTE** (Synthetic Minority Oversampling) — generates synthetic minority-class samples instead of just reweighting.
- **Undersampling** the majority class.
- Always evaluate with **Precision, Recall, F1 per class**, and prefer the **Precision-Recall curve** over ROC (Section 7.4).

---

## 10. Multiclass Logistic Regression

For 3+ classes, Logistic Regression needs a strategy:

- **One-vs-Rest (OVR)** — trains one binary classifier **per class** ("Class 0 vs Rest", "Class 1 vs Rest", ...). Whichever is most confident wins. Required by solvers like `liblinear`.
- **Multinomial (Softmax)** — trains a single unified model that outputs a probability for every class directly, all summing to 1. Generally the statistically preferred approach when the solver supports it (e.g. `lbfgs`, `saga`).

```python
# One-vs-Rest
model_ovr = LogisticRegression(multi_class="ovr", solver="liblinear")

# Multinomial / Softmax
model_softmax = LogisticRegression(multi_class="multinomial", solver="lbfgs")

model_softmax.fit(X_train_scaled, y_train)
proba = model_softmax.predict_proba(X_test_scaled)   # one probability column per class, rows sum to 1
```

With multiclass, remember to check `macro avg` vs `weighted avg` in the classification report (Section 7.2) — the distinction matters much more once there are more than 2 classes.

---

## 11. Regularization Deep Dive

Regularization prevents **overfitting** by discouraging overly large coefficients — a model with huge weights tends to fit training-set noise rather than the true underlying pattern.

| | L1 (Lasso) | L2 (Ridge) | ElasticNet |
|---|---|---|---|
| **Penalty term** | Sum of `|weight|` | Sum of `weight²` | Mix of both |
| **Effect** | Can shrink weights to exactly **0** | Shrinks smoothly, rarely exactly 0 | Combination |
| **Use case** | Feature selection, many irrelevant features | Default, general overfitting control | Correlated features + some sparsity |

**Effect of `C`:** smaller `C` = stronger regularization = simpler model = higher bias, lower variance. Larger `C` = weaker regularization = more complex model = lower bias, higher variance (more overfitting risk). This is the classic **bias-variance tradeoff**.

---

## 12. Assumptions of Logistic Regression

1. The target is **categorical** (binary or multiclass).
2. A roughly **linear relationship** between the features and the **log-odds** of the outcome (not the raw probability).
3. Observations are **independent** of each other.
4. Little to no **multicollinearity** among features.
5. Reasonably **large sample size** — small imbalanced samples make coefficient estimates unstable.

---

## 13. Common Beginner Mistakes

- Forgetting to **scale features** before training.
- Fitting the scaler on the **full dataset** instead of only the training set (data leakage).
- Judging an imbalanced-data model using **accuracy alone**.
- Using `multi_class="ovr"` without realizing `multinomial`/softmax might perform better.
- Assuming the default **0.5 threshold** is always correct.
- Confusing `predict()` (hard label) with `predict_proba()` (probability) — using the wrong one for ROC/PR curves.
- Not using `stratify=y` when splitting imbalanced data.
- Reading `classification_report`'s `macro avg` and `weighted avg` as interchangeable.

---

## 14. Best Practices Checklist

- [ ] Check class balance before choosing an evaluation metric
- [ ] Scale features (`StandardScaler`) before training
- [ ] Use `stratify=y` in `train_test_split` for imbalanced data
- [ ] Look at Precision, Recall, F1, and the Confusion Matrix together — not accuracy alone
- [ ] Use cross-validation (`StratifiedKFold`) when tuning hyperparameters
- [ ] Start simple — Logistic Regression is a strong, cheap baseline
- [ ] Check for multicollinearity among features
- [ ] For imbalanced data, prefer the Precision-Recall curve over ROC

---

## 15. Interview Questions & Answers

**Q: Why is Logistic Regression called "regression" if it's used for classification?**
A: It fits a linear equation to the log-odds of the outcome — a regression technique — before converting the result into a class label via the Sigmoid function.

**Q: Why not just use Linear Regression for classification?**
A: Linear Regression can output any real number, including values below 0 or above 1, which doesn't make sense as a probability, and it's very sensitive to outliers. Logistic Regression's Sigmoid keeps outputs bounded between 0 and 1.

**Q: What loss function does Logistic Regression minimize?**
A: Binary Cross-Entropy Loss (Log Loss).

**Q: Why not use Mean Squared Error instead of BCE?**
A: MSE combined with the Sigmoid produces a non-convex loss surface with multiple local minima, making optimization unreliable. BCE stays convex for this problem.

**Q: What does the `C` hyperparameter control?**
A: The inverse of regularization strength — smaller `C` means stronger regularization (simpler model).

**Q: What's the difference between L1 and L2 regularization?**
A: L1 can shrink coefficients all the way to zero (feature selection); L2 shrinks them smoothly toward zero but rarely to exactly zero.

**Q: Why is accuracy a poor metric for imbalanced datasets?**
A: A model that always predicts the majority class can score very high accuracy while never correctly identifying the minority class — often the class you actually care about.

**Q: What's the difference between `predict()` and `predict_proba()`?**
A: `predict()` returns the final hard class label using the default 0.5 threshold. `predict_proba()` returns the underlying probability for each class before thresholding.

**Q: How does Logistic Regression handle more than 2 classes?**
A: Via One-vs-Rest (a separate binary model per class) or Multinomial/Softmax (one unified model outputting a probability per class directly).

**Q: What is the ROC curve, and when is it not the best choice?**
A: It plots True Positive Rate vs False Positive Rate across every threshold. On heavily imbalanced data it can look deceptively good, so a Precision-Recall curve is often more informative there.

**Q: What does `class_weight="balanced"` do?**
A: Automatically adjusts weights inversely proportional to class frequencies, so the minority class isn't overwhelmed during training.

**Q: Why do we scale features before Logistic Regression?**
A: It uses gradient-based optimization; features on very different scales can slow convergence and distort the effect of regularization.

**Q: What is multicollinearity, and why does it matter here?**
A: When features are highly correlated with each other, coefficient estimates become unstable and hard to interpret, even if overall predictions stay reasonable.

**Q: What's the difference between `macro avg` and `weighted avg` in a classification report?**
A: `macro avg` treats every class equally regardless of size; `weighted avg` weights each class's contribution by how many samples it has.

---

## 16. Cheat Sheet

- **Model:** `sigmoid(w·x + b)` → probability → threshold (default 0.5) → class label
- **Loss function:** Binary Cross-Entropy (Log Loss)
- **Key formulas:**
  - Odds = `p / (1-p)`
  - Logit = `ln(p / (1-p))`
  - Sigmoid = `1 / (1 + e^-z)`
  - Precision = `TP / (TP+FP)`  |  Recall = `TP / (TP+FN)`  |  FPR = `FP / (FP+TN)`
  - F1 = `2 × (Precision × Recall) / (Precision + Recall)`
- **Always scale features** (`StandardScaler`) before training
- **Imbalanced data** → `class_weight`, `stratify=y`, `StratifiedKFold`, prefer Precision-Recall curve
- **Tune with** `GridSearchCV` (exhaustive) or `RandomizedSearchCV` (sampled) over `C`, `penalty`, `solver`
- **Multiclass** → One-vs-Rest or Multinomial/Softmax
- **Regularization:** L1 = sparse/feature-selection, L2 = smooth shrinkage (default), smaller `C` = stronger regularization

---

## 17. Glossary

| Term | Meaning |
|---|---|
| **Feature** | An input column/variable used to make a prediction |
| **Target / Label** | The value being predicted (`y`) |
| **Weight (coefficient)** | A learned number that scales a feature's influence on the prediction |
| **Bias (intercept)** | A learned constant added to the weighted sum |
| **Odds** | `p / (1-p)` |
| **Logit** | `ln(odds)` — what the linear equation actually predicts |
| **Sigmoid** | Function that squashes any real number into (0, 1) |
| **Decision boundary** | The point/line/curve where predicted probability = the chosen threshold |
| **Regularization** | Technique to discourage overly large weights, reducing overfitting |
| **Overfitting** | Model fits training data (incl. noise) too closely, performs worse on new data |
| **Cross-validation** | Splitting data into multiple folds to get a more reliable performance estimate |
| **Class imbalance** | One class has far more samples than another |
| **AUC** | Area Under the (ROC) Curve — single-number summary of ranking quality |
