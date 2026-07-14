

# **📚 Polynomial Regression Steps**

# Step 1: Import Required Libraries

### Why?

Before building any Machine Learning model, we need to import the required libraries.

* **NumPy** → Numerical operations
* **Pandas** → Data manipulation
* **Matplotlib** → Data visualization
* **Scikit-learn** → Machine Learning algorithms

### Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
```

---

# Step 2: Create or Load the Dataset

### Why?

Polynomial Regression is useful when the relationship between the independent variable (X) and dependent variable (y) is **non-linear**.

For learning purposes, we create our own synthetic dataset.

### Example

Suppose we have the following data:

| Hours Studied | Exam Score |
| ------------- | ---------: |
| 1             |         10 |
| 2             |         18 |
| 3             |         32 |
| 4             |         55 |
| 5             |         82 |

The relationship is **curved**, not linear.

### Code

```python
np.random.seed(10)

X = 6 * np.random.rand(100,1) - 3

y = (
    0.5 * X**2
    + X
    + 2
    + np.random.randn(100,1)
)

y = y.reshape(100)
```

---

# Step 3: Visualize the Dataset

### Why?

Before training a model, it is important to understand the relationship between the variables.

A scatter plot helps us determine whether the relationship is linear or nonlinear.

### Code

```python
plt.figure(figsize=(8,5))

plt.scatter(X, y)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Non-Linear Dataset")

plt.show()
```

### Output

Notice that the points form a **curve** instead of a straight line.

---

# Step 4: Split the Dataset

### Why?

Machine Learning models should be evaluated on unseen data.

We divide the dataset into:

* Training Data → Used to train the model.
* Testing Data → Used to evaluate the model.

### Code

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

---

# Step 5: Train a Simple Linear Regression Model

### Why?

Before applying Polynomial Regression, we first train a Linear Regression model.

This acts as our **baseline model**.

### Code

```python
regression = LinearRegression()

regression.fit(X_train, y_train)
```

---

# Step 6: Evaluate Linear Regression

### Why?

We evaluate how well the straight-line model performs.

### Code

```python
y_pred = regression.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
```

### Observation

Since the data is nonlinear, the straight line cannot capture the curved pattern.

The result is:

* Lower R² Score
* Higher Mean Squared Error

---

# Step 7: Create Polynomial Features

### Why?

Linear Regression can only fit a straight line.

Polynomial Regression creates additional features like:

```
X²
X³
X⁴
```

These extra features allow the model to learn curved relationships.

### Example

Original feature

```
X

2
4
6
```

Degree = 2

```
1   X   X²

1   2    4
1   4   16
1   6   36
```

Degree = 3

```
1   X   X²   X³

1   2    4     8
1   4   16    64
1   6   36   216
```

### Code

```python
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)
```

---

# Step 8: Train the Polynomial Regression Model

### Why?

Even though it is called **Polynomial Regression**, the actual model is still **Linear Regression**.

The only difference is that the input features have been transformed.

### Code

```python
poly_model = LinearRegression()

poly_model.fit(
    X_train_poly,
    y_train
)
```

---

# Step 9: Predict Using Polynomial Regression

### Why?

Before prediction, the testing data must be transformed using the **same PolynomialFeatures object**.

### Code

```python
y_pred_poly = poly_model.predict(
    X_test_poly
)
```

> **Important:** Never call `fit_transform()` on the test data. Use only `transform()` to avoid data leakage.

---

# Step 10: Evaluate the Polynomial Regression Model

### Why?

We compare the Polynomial Regression model with the Linear Regression model.

### Code

```python
print(
    "R2 Score:",
    r2_score(y_test, y_pred_poly)
)

print(
    "MSE:",
    mean_squared_error(
        y_test,
        y_pred_poly
    )
)
```

### Observation

Compared to Linear Regression:

* Higher R² Score ✅
* Lower MSE ✅

This indicates a better fit for nonlinear data.

---

# Step 11: Visualize the Polynomial Curve

### Why?

The prediction curve appears smooth only if we generate many evenly spaced points.

### Code

```python
X_new = np.linspace(
    X.min(),
    X.max(),
    200
).reshape(-1,1)

X_new_poly = poly.transform(X_new)

y_new = poly_model.predict(X_new_poly)
```

Plot the graph:

```python
plt.figure(figsize=(8,5))

plt.scatter(
    X_train,
    y_train,
    label="Training Data"
)

plt.plot(
    X_new,
    y_new,
    color="red",
    linewidth=3,
    label="Polynomial Curve"
)

plt.legend()

plt.show()
```

---

# Step 12: Try Different Polynomial Degrees

### Why?

Increasing the polynomial degree changes the complexity of the model.

### Example

| Degree | Result                        |
| ------ | ----------------------------- |
| 1      | Straight line (Underfitting)  |
| 2      | Captures the curve well       |
| 3      | More flexible curve           |
| 10     | Very complex curve            |
| 15     | May overfit the training data |

Example:

```python
poly3 = PolynomialFeatures(degree=3)
poly5 = PolynomialFeatures(degree=5)
poly10 = PolynomialFeatures(degree=10)
```

---


# Step 13: Final Workflow

```text
Import Libraries
       │
       ▼
Create / Load Dataset
       │
       ▼
Visualize Dataset
       │
       ▼
Train-Test Split
       │
       ▼
Train Linear Regression
       │
       ▼
Evaluate Linear Regression
       │
       ▼
Create Polynomial Features
       │
       ▼
Transform Training Data
       │
       ▼
Train Polynomial Regression
       │
       ▼
Transform Test Data
       │
       ▼
Predict Test Data
       │
       ▼
Evaluate Model
       │
       ▼
Visualize Polynomial Curve
       │
       ▼
Repeat for Different Degrees
       │
       ▼
Compare Results
       │
       ▼
Select the Best Degree
```
