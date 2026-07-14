
# Ridge Regression

## What is Ridge Regression?

Ridge Regression is a **regularized version of Linear Regression** that helps reduce **overfitting** by adding a penalty to large coefficient values.

Instead of minimizing only the prediction error, Ridge Regression also tries to keep the model coefficients small.

It uses **L2 Regularization**, which penalizes the **square of the coefficients**.

---

# Why do we need Ridge Regression?

Suppose we build a Linear Regression model.

It tries to find the best equation:

y=b0 ​+ b1​x1​ + b2​x2​ +...+ bn​xn​

Sometimes the model learns very large coefficient values like

```
Weight of Feature 1 = 350
Weight of Feature 2 = -420
Weight of Feature 3 = 510
```

Large coefficients usually mean

* Model is too sensitive to small changes
* High variance
* Overfitting
* Poor performance on unseen data

Ridge Regression solves this problem.

Instead of allowing coefficients to become very large, it adds a penalty.

---

# What is Regularization?

Regularization is a technique used to **prevent overfitting** by restricting the size of model coefficients.

Instead of only minimizing prediction error,

the model minimizes

```
Prediction Error
        +
Regularization Penalty
```

This makes the model simpler and more stable.

---

# L2 Regularization

Ridge Regression uses **L2 Regularization**.

Penalty Formula

$$
\lambda \sum_{j=1}^{n} w_j^2
$$

where

* λ (lambda or alpha in scikit-learn) = Regularization strength
* (w_j) = coefficient

Notice

Every coefficient is squared.

That is why it is called **L2**.

---

# Ridge Regression Cost Function

### Linear Regression minimizes

$$
RSS = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

### Ridge Regression minimizes

$$
RSS + \lambda \sum_{j=1}^{p} w_j^2
$$

This means the algorithm tries to

* reduce prediction error
* keep coefficients small

at the same time.

---

# What happens to coefficients?

Suppose Linear Regression learns

| Feature    | Coefficient |
| ---------- | ----------: |
| Age        |          35 |
| Salary     |         210 |
| Experience |          95 |

After Ridge

| Feature    | Coefficient |
| ---------- | ----------: |
| Age        |          20 |
| Salary     |         105 |
| Experience |          50 |

Notice

The coefficients become **smaller**, but they **do not become zero**.

This is called **Coefficient Shrinkage**.

---

# Why doesn't Ridge make coefficients zero?

Because Ridge squares every coefficient.

When a coefficient becomes smaller,

its squared value also becomes smaller,

so the optimizer prefers reducing coefficients rather than eliminating them.

Therefore,

Ridge

✅ Shrinks coefficients

❌ Does not perform feature selection

---

# Why Feature Scaling is Necessary

Suppose

```
Age = 25

Salary = 800000
```

Salary has much larger values.

Without scaling,

Salary's coefficient receives a much larger penalty.

The model becomes biased.

After StandardScaler

```
Age

↓

Mean = 0

Std = 1

Salary

↓

Mean = 0

Std = 1
```

Now every feature contributes equally.

That is why Ridge should almost always be used after feature scaling.

---

# What is Alpha?

Alpha controls

**how much regularization is applied**.

It is the most important parameter in Ridge Regression.

---

## Alpha = 0

```
No Regularization
```

The model behaves exactly like Linear Regression.

Large coefficients are allowed.

Higher chance of overfitting.

---

## Small Alpha

Example

```
alpha = 0.01
```

Very little regularization.

Coefficients shrink only a little.

Model behaves similarly to Linear Regression.

---

## Medium Alpha

Example

```
alpha = 1
```

Balanced regularization.

Usually a good starting point.

Most commonly used.

---

## Large Alpha

Example

```
alpha = 100
```

Strong regularization.

Coefficients become much smaller.

Model becomes simpler.

May underfit if alpha is too large.

---

# Effect of Alpha

| Alpha | Effect                     |
| ----- | -------------------------- |
| 0     | Linear Regression          |
| 0.01  | Very little regularization |
| 0.1   | Small shrinkage            |
| 1     | Moderate shrinkage         |
| 10    | Strong regularization      |
| 100   | Very strong regularization |
| 1000  | Underfitting may occur     |

---

# Ridge Regression Workflow

```
Load Dataset

↓

Data Cleaning

↓

EDA

↓

Feature Selection

↓

Train-Test Split

↓

Feature Scaling

↓

Train Ridge Model

↓

Predict

↓

Evaluate

↓

Interpret Coefficients
```

---

# Ridge Regression Syntax

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)

ridge.fit(X_train_scaled, y_train)

ridge_pred = ridge.predict(X_test_scaled)
```

---

# Ridge Parameters

---

## 1. alpha 

Default

```python
alpha=1.0
```

Controls regularization strength.

Higher alpha

↓

More shrinkage

↓

Simpler model

Lower alpha

↓

Behaves like Linear Regression

Example

```python
Ridge(alpha=10)
```

---

## 2. fit_intercept

Default

```python
fit_intercept=True
```

Whether to calculate the intercept.

Normally keep

```python
True
```

---

## 3. max_iter

Default

```python
None
```

Maximum number of optimization iterations.

Useful for very large datasets.

Example

```python
max_iter=10000
```

---

## 4. tol

Default

```python
tol=1e-4
```

Tolerance for convergence.

Smaller value

↓

More accurate

↓

Longer training

---

## 5. solver

Specifies the optimization algorithm.

Options

```
auto

svd

cholesky

lsqr

sparse_cg

sag

saga

lbfgs
```

Usually

```python
solver="auto"
```

is sufficient.

---

## 6. random_state

Used by stochastic solvers (such as `sag` or `saga`) to make results reproducible.

```python
random_state=42
```

---

# What is RidgeCV?

Choosing alpha manually is difficult.

Should we use

```
0.01

0.1

1

10

100
```

Which one is best?

Instead of guessing,

we let the model decide.

That is RidgeCV.

---

# What does RidgeCV do?

RidgeCV automatically

* tries many alpha values
* performs Cross Validation
* compares their performance
* selects the alpha with the lowest validation error
* trains the final model using that best alpha

---

# RidgeCV Workflow

```
Candidate Alphas

↓

0.001

↓

0.01

↓

0.1

↓

1

↓

10

↓

100

↓

Cross Validation

↓

Calculate Validation Error

↓

Choose Best Alpha

↓

Train Final Ridge Model

↓

Prediction
```

---

# Why use Cross Validation?

Instead of relying on one train-test split,

the data is divided into multiple folds.

Example

```
Fold 1

Train

Validate

↓

Fold 2

Train

Validate

↓

Fold 3

Train

Validate

↓

Fold 4

Train

Validate

↓

Fold 5

Train

Validate
```

Average validation performance is calculated.

This gives a more reliable estimate of model performance.

---

# RidgeCV Syntax

```python
from sklearn.linear_model import RidgeCV

alphas = np.logspace(-3, 3, 100)

ridgecv = RidgeCV(
    alphas=alphas,
    cv=5
)

ridgecv.fit(X_train_scaled, y_train)
```

---

# RidgeCV Parameters

## 1. alphas 

A list or array of alpha values to test.

Example

```python
alphas=[0.01,0.1,1,10,100]
```

or

```python
alphas=np.logspace(-3,3,100)
```

RidgeCV tests every alpha and chooses the best one.

---

## 2. cv 

Number of folds used for cross-validation.

```python
cv=5
```

means the data is split into 5 folds.

Common values:

* `cv=3`
* `cv=5` (most common)
* `cv=10`

Larger `cv` generally gives a more reliable estimate but increases training time.

---

## 3. scoring

Evaluation metric used to compare alpha values.

Examples

```python
scoring="r2"
```

```python
scoring="neg_mean_squared_error"
```

If omitted, RidgeCV uses its default scoring strategy.

---

## 4. fit_intercept

Default

```python
True
```

Whether to calculate the intercept.

---

## 5. store_cv_results (or `store_cv_values` in older versions)

Stores cross-validation results for later inspection.

Useful for analysis and visualization.

---

# Important Attributes after Training

## Best Alpha

```python
ridgecv.alpha_
```

Returns the best alpha selected.

---

## Coefficients

```python
ridgecv.coef_
```

Returns the learned coefficients.

---

## Intercept

```python
ridgecv.intercept_
```

Returns the intercept.

---

## Predictions

```python
ridgecv.predict(X_test_scaled)
```

Predicts target values for new data.

---

# Ridge vs RidgeCV

| Ridge                            | RidgeCV                            |
| -------------------------------- | ---------------------------------- |
| User chooses alpha manually      | Model selects alpha automatically  |
| No cross-validation              | Uses cross-validation              |
| Faster                           | Slightly slower                    |
| Requires experimentation         | More convenient and reliable       |
| Best when alpha is already known | Best when optimal alpha is unknown |

---

# Ridge vs Lasso vs ElasticNet

| Algorithm  | Regularization | Coefficients Become Zero? | Feature Selection |
| ---------- | -------------- | ------------------------- | ----------------- |
| Ridge      | L2             | ❌ No                      | ❌ No              |
| Lasso      | L1             | ✅ Yes                     | ✅ Yes             |
| ElasticNet | L1 + L2        | Some may become zero      | Partial           |

