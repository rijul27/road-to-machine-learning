

# Lasso Regression

## What is Lasso Regression?

Lasso Regression (**Least Absolute Shrinkage and Selection Operator**) is a regularized version of Linear Regression. It adds an **L1 Regularization** penalty to the loss function to reduce model complexity and prevent overfitting.

Unlike Linear Regression, Lasso not only reduces the size of the coefficients but can also shrink some coefficients **exactly to zero**. As a result, it automatically removes less important features from the model, making it an embedded feature selection technique.

---

# Why do we need Lasso Regression?

Suppose we have a dataset with **100 features**, but only **15 features** are actually important.

Linear Regression:

* Uses all 100 features.
* May overfit the training data.
* May assign non-zero coefficients to irrelevant features.

Lasso Regression:

* Reduces the coefficients of less important features.
* Makes some coefficients exactly zero.
* Automatically removes unnecessary features.
* Produces a simpler and more interpretable model.

---

# What is Overfitting?

Overfitting occurs when a model learns the training data too well, including its noise and random patterns.

Characteristics:

* Very high training accuracy.
* Poor performance on unseen data.
* Poor generalization.

Lasso helps reduce overfitting by penalizing large coefficients.

---

# What is Regularization?

Regularization is a technique used to prevent overfitting by adding a penalty to the model's loss function.

Instead of only minimizing prediction error, the model also minimizes the size of its coefficients.

General idea:

```
Loss Function = Prediction Error + Penalty
```

Regularization encourages simpler models that generalize better.

---

# What is L1 Regularization?

Lasso uses **L1 Regularization**.

The penalty is calculated as the **sum of the absolute values** of all coefficients.

### Formula

λ(∣w1​∣+∣w2​∣+∣w3​∣+⋯+∣wn​∣)


where

* (w) = model coefficients (weights)
* (\lambda) (represented as `alpha` in Scikit-learn) = regularization strength

---

## Why absolute values?

Absolute values increase the penalty for larger coefficients.

Example:

Suppose

```
w1 = 6
w2 = -3
w3 = 2
```

L1 penalty becomes

```
|6| + |-3| + |2|

= 6 + 3 + 2

= 11
```

The model tries to reduce this value.

---

# Lasso Loss Function

Linear Regression minimizes

[
MSE
]

Lasso minimizes

[ MSE+α ∑ ∣w∣ ]

This means the model has two goals:

* Minimize prediction error.
* Keep coefficients as small as possible.

---

# What happens during training?

### Step 1

Model initializes random coefficients.

```
Feature A = 5.2
Feature B = -1.8
Feature C = 4.3
```

---

### Step 2

Calculate prediction error.

---

### Step 3

Calculate L1 penalty.

```
|5.2| + |−1.8| + |4.3|

= 11.3
```

---

### Step 4

Update coefficients.

Some coefficients become smaller.

```
5.2 → 3.8

−1.8 → −0.5

4.3 → 2.1
```

---

### Step 5

Repeat until convergence.

Eventually,

```
Feature A = 2.6

Feature B = 0

Feature C = 1.4
```

Feature B becomes zero.

This feature is automatically removed.

---

# Why does Lasso perform Feature Selection?

Because L1 Regularization can make coefficients exactly zero.

Example

| Feature     | Coefficient |
| ----------- | ----------: |
| Temperature |        2.41 |
| Humidity    |       -1.17 |
| Rain        |           0 |
| Wind Speed  |        0.82 |
| Region      |           0 |

Features with coefficient **0** no longer contribute to predictions.

---

# Advantages of Lasso

* Prevents overfitting.
* Performs automatic feature selection.
* Produces simpler models.
* Reduces model complexity.
* Works well when many irrelevant features exist.

---

# Disadvantages

* Can remove useful correlated features.
* May underfit if alpha is too large.
* Performance decreases when all features are important.

---

# What is Alpha?

Alpha controls the strength of regularization.

It determines how strongly Lasso penalizes coefficients.

---

## Small Alpha

Example

```
alpha = 0.0001
```

Penalty is very small.

Result

* Coefficients remain large.
* Most features stay.
* Model behaves like Linear Regression.
* Higher chance of overfitting.

---

## Large Alpha

Example

```
alpha = 100
```

Penalty is very large.

Result

* Coefficients become much smaller.
* Many become zero.
* Simpler model.
* Can underfit.

---

## Medium Alpha

Provides a balance.

* Good prediction.
* Reduced overfitting.
* Better generalization.

---

# Why is choosing Alpha important?

If alpha is

Too Small

→ Overfitting

Too Large

→ Underfitting

Correct Alpha

→ Best performance

---

# How do we choose Alpha?

Option 1

Manually

```
alpha = 0.1

alpha = 1

alpha = 10
```

Test every value.

Time consuming.

---

Option 2

Use LassoCV

Automatically finds the best alpha.

---

# What is LassoCV?

LassoCV is an improved version of Lasso Regression.

Instead of manually choosing alpha, it uses **Cross Validation** to find the best alpha automatically.

---

# Why use LassoCV?

Manual tuning

```
alpha = 0.1

alpha = 0.5

alpha = 1

alpha = 2

alpha = 5
```

Many experiments.

LassoCV

Automatically checks many alpha values and selects the one that gives the best validation performance.

---

# How LassoCV Works

Suppose

```
cv = 5
```

Training data is divided into

```
Fold 1

Fold 2

Fold 3

Fold 4

Fold 5
```

Suppose candidate alpha values are

```
0.001

0.01

0.1

1

10
```

For every alpha

Train the model

↓

Calculate validation error

↓

Repeat for all 5 folds

↓

Compute average error

↓

Repeat for every alpha

↓

Select the alpha with the lowest average error.

---

# Important Attributes

## alpha_

```
lasso_cv.alpha_
```

Returns

```
Best Alpha Selected
```

Example

```
0.0047
```

This is the alpha giving the lowest validation error.

---

## alphas_

```
lasso_cv.alphas_
```

Returns every alpha tested.

Example

```
[0.001

0.01

0.1

1

10]
```

---

## mse_path_

```
lasso_cv.mse_path_
```

Stores the Mean Squared Error for every alpha across every fold.

Rows

→ Different alpha values

Columns

→ Different cross-validation folds

Useful for plotting validation error against alpha.

---

# Lasso vs LassoCV

| Feature                 | Lasso     | LassoCV |
| ----------------------- | --------- | ------- |
| Select alpha manually   | ✅ Yes     | ❌ No    |
| Uses Cross Validation   | ❌ No      | ✅ Yes   |
| Finds best alpha        | ❌ No      | ✅ Yes   |
| Feature selection       | ✅ Yes     | ✅ Yes   |
| Better generalization   | Good      | Better  |
| Recommended in practice | Sometimes | ✅ Yes   |

---

# Workflow of Lasso Regression

1. Import libraries.
2. Load the dataset.
3. Perform data preprocessing.
4. Split features (`X`) and target (`y`).
5. Split the dataset into training and testing sets.
6. Apply `StandardScaler`.
7. Train the Lasso Regression model.
8. Predict on the test set.
9. Evaluate the model using MAE, MSE, RMSE, and R² Score.
10. Analyze feature coefficients and identify coefficients shrunk to zero.
11. Visualize Actual vs Predicted values.

---

# Workflow of LassoCV

1. Import `LassoCV`.
2. Create the model with `cv=5`.
3. Train the model on the scaled training data.
4. LassoCV automatically tests multiple alpha values.
5. Perform Cross Validation for each alpha.
6. Select the alpha with the lowest average validation error.
7. Predict on the test data.
8. Evaluate the model.
9. Display `alpha_`, `alphas_`, and `mse_path_`.
10. Compare LassoCV with standard Lasso Regression.

---

## Notes

* **Lasso Regression** uses **L1 Regularization** to reduce overfitting and perform automatic feature selection by shrinking some coefficients exactly to zero.
* **Alpha** controls the strength of the regularization penalty. A small alpha may lead to overfitting, while a large alpha may lead to underfitting.
* **LassoCV** automatically finds the optimal alpha using Cross Validation, making it more reliable than manually selecting alpha.
* In real-world machine learning projects, **LassoCV is generally preferred over standard Lasso Regression** because it typically provides better generalization and requires less manual tuning.
