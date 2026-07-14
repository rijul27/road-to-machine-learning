# ElasticNet Regression

## What is ElasticNet Regression?

ElasticNet Regression is a **regularized linear regression algorithm** that combines the strengths of **Lasso Regression (L1 Regularization)** and **Ridge Regression (L2 Regularization)**.

It is mainly used to:

* Reduce overfitting
* Improve model generalization
* Handle multicollinearity
* Perform feature selection while shrinking coefficients

ElasticNet works well when the dataset contains **many correlated features**, where using only Lasso or only Ridge may not give the best results.

---

# Why do we need ElasticNet?

Suppose we have three situations.

### Linear Regression

* No regularization
* May overfit the training data
* Performs poorly when features are highly correlated

---

### Ridge Regression

* Uses L2 Regularization
* Reduces coefficient values
* Keeps all features in the model

---

### Lasso Regression

* Uses L1 Regularization
* Removes less important features by making some coefficients exactly zero

---

### ElasticNet Regression

ElasticNet combines both techniques.

It:

* Shrinks coefficients like Ridge.
* Removes unnecessary features like Lasso.
* Performs better when predictors are highly correlated.

---

# Regularization

Regularization is a technique used to prevent overfitting by adding a penalty to the model.

Instead of minimizing only the prediction error, ElasticNet minimizes

**Prediction Error + Regularization Penalty**

This helps the model become simpler and more generalizable.

---

# L1 Regularization (Lasso)

L1 Regularization adds the absolute value of coefficients as a penalty.

### Characteristics

* Shrinks coefficients.
* Can make some coefficients exactly zero.
* Performs feature selection.
* Produces a simpler model.

### Example

Suppose we have five features.

```
Feature A = 2.1
Feature B = 1.4
Feature C = 0
Feature D = 0
Feature E = 0.8
```

Here,

Feature C and D are removed from the model.

---

# L2 Regularization (Ridge)

L2 Regularization adds the squared value of coefficients as a penalty.

### Characteristics

* Shrinks coefficients.
* Never makes coefficients exactly zero.
* Keeps all features.
* Reduces multicollinearity.

Example

```
Before

A = 4.2
B = 5.6
C = 3.1

After Ridge

A = 2.3
B = 2.8
C = 1.7
```

All features remain.

---

# ElasticNet Formula

ElasticNet combines both penalties.

Total Loss

```
Loss = Prediction Error
     + L1 Penalty
     + L2 Penalty
```

This allows ElasticNet to benefit from both feature selection and coefficient shrinkage.

---

# Parameters of ElasticNet

## 1. alpha

### Definition

`alpha` controls the overall strength of regularization.

### Effect

Small alpha

* Less regularization
* Model behaves like Linear Regression

Large alpha

* Strong regularization
* Coefficients become much smaller
* May underfit if too large

### Default

```
alpha = 1.0
```

---

## 2. l1_ratio

### Definition

`l1_ratio` controls how much L1 and L2 regularization should be used.

### Range

```
0 ≤ l1_ratio ≤ 1
```

| l1_ratio | Meaning         |
| -------- | --------------- |
| 0        | Only Ridge (L2) |
| 0.25     | Mostly Ridge    |
| 0.5      | Equal L1 and L2 |
| 0.75     | Mostly Lasso    |
| 1        | Only Lasso (L1) |

Example

```
l1_ratio = 0.7
```

means

70% Lasso

30% Ridge

---

# Steps of ElasticNet Regression

### Step 1

Import required libraries.

---

### Step 2

Load the dataset.

---

### Step 3

Perform data preprocessing.

* Handle missing values
* Encode categorical variables
* Remove unnecessary columns

---

### Step 4

Split the dataset into training and testing sets.

---

### Step 5

Apply StandardScaler.

Feature scaling is important because ElasticNet applies regularization based on feature coefficients.

---

### Step 6

Create the ElasticNet model.

```
ElasticNet(alpha=1.0,
           l1_ratio=0.5)
```

---

### Step 7

Train the model.

```
fit()
```

---

### Step 8

Predict test data.

```
predict()
```

---

### Step 9

Evaluate

* MAE
* MSE
* RMSE
* R² Score

---

### Step 10

Visualize

* Scatter Plot
* Residual Plot
* Coefficient Plot

---

# What is ElasticNetCV?

ElasticNetCV is an advanced version of ElasticNet.

Instead of manually selecting

* alpha
* l1_ratio

ElasticNetCV automatically finds the best values using Cross-Validation.

This usually produces a better-performing model.

---

# Why use ElasticNetCV?

Suppose you don't know

```
alpha = ?
```

or

```
l1_ratio = ?
```

You would need to train many ElasticNet models manually.

ElasticNetCV automates this process by:

* Trying different alpha values.
* Trying different l1_ratio values.
* Performing Cross-Validation.
* Selecting the combination with the lowest validation error.

---

# How ElasticNetCV Works

```
Train Data
      │
      ▼
Try Alpha = 0.001
      │
      ▼
Try Alpha = 0.01
      │
      ▼
Try Alpha = 0.1
      │
      ▼
Try Alpha = 1
      │
      ▼
Try Different l1_ratio Values
      │
      ▼
Perform Cross Validation
      │
      ▼
Compare Validation Errors
      │
      ▼
Select Best Alpha
      │
      ▼
Select Best l1_ratio
      │
      ▼
Train Final Model
```

---

# Parameters of ElasticNetCV

## 1. cv

Specifies the number of folds for Cross-Validation.

Example

```
cv = 5
```

Meaning

* Dataset is divided into five folds.
* Model trains five times.
* Average validation error is calculated.

---

## 2. alphas

List of alpha values to test.

Example

```
alphas=np.logspace(-3,3,100)
```

ElasticNetCV tests all these values and selects the best one.

---

## 3. l1_ratio

Specifies one or more L1/L2 mixing ratios to evaluate.

Example

```
l1_ratio=[0.1,0.3,0.5,0.7,0.9,1]
```

The model evaluates each ratio and selects the one with the lowest validation error.

---

## 4. max_iter

Maximum number of iterations for the optimization algorithm.

Higher values provide more time for convergence.

Default

```
1000
```

---

## 5. random_state

Controls randomness.

Using

```
random_state=42
```

ensures reproducible results every time the notebook is executed.

---

# Steps of ElasticNetCV

### Step 1

Import ElasticNetCV.

---

### Step 2

Create the model.

```
ElasticNetCV(
    cv=5,
    alphas=np.logspace(-3,3,100),
    l1_ratio=[0.1,0.3,0.5,0.7,0.9,1],
    max_iter=1000,
    random_state=42
)
```

---

### Step 3

Train the model.

During training, ElasticNetCV automatically tests multiple combinations of alpha and l1_ratio using Cross-Validation.

---

### Step 4

Retrieve the best hyperparameters.

```
elastic_cv.alpha_

elastic_cv.l1_ratio_
```

---

### Step 5

Predict on the test dataset.

---

### Step 6

Evaluate the model using MAE, MSE, RMSE, and R² Score.

---

### Step 7

Compare ElasticNetCV with ElasticNet and Linear Regression.

---

# ElasticNet vs ElasticNetCV

| Feature               | ElasticNet                   | ElasticNetCV                         |
| --------------------- | ---------------------------- | ------------------------------------ |
| Hyperparameter tuning | Manual                       | Automatic                            |
| Alpha selection       | User-defined                 | Selected using Cross-Validation      |
| l1_ratio selection    | User-defined                 | Selected using Cross-Validation      |
| Cross-Validation      | Not included                 | Built-in                             |
| Ease of use           | Requires experimentation     | More convenient                      |
| Model performance     | Depends on chosen parameters | Often achieves better generalization |

# Notes

* **ElasticNet** combines the strengths of Lasso (L1) and Ridge (L2) regularization.
* **L1 Regularization** performs feature selection by shrinking some coefficients to exactly zero.
* **L2 Regularization** shrinks coefficients while keeping all features in the model.
* **alpha** controls the overall strength of regularization.
* **l1_ratio** determines the balance between L1 and L2 penalties.
* **ElasticNetCV** automatically finds the optimal `alpha` and `l1_ratio` using Cross-Validation.
* Feature scaling is essential before applying ElasticNet or ElasticNetCV because regularization is sensitive to feature scales.
* ElasticNetCV is generally preferred when the best hyperparameters are unknown, as it typically produces a more robust and generalizable model.
