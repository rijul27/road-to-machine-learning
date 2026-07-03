## Steps of Performing Simple Linear Regression

### Step 1: Collect the Dataset

Gather a dataset with:

* **One independent variable (X)** (feature)
* **One dependent variable (Y)** (target)

Example:

| Hours Studied (X) | Marks (Y) |
| ----------------- | --------- |
| 1                 | 35        |
| 2                 | 45        |
| 3                 | 50        |
| 4                 | 60        |
| 5                 | 70        |

---

### Step 2: Visualize the Data

Plot the data using a scatter plot.

Purpose:

* Check if a linear relationship exists.
* Identify outliers.
* Understand the trend.

```python
import matplotlib.pyplot as plt

plt.scatter(X, Y)
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.show()
```

---

### Step 3: Split the Dataset

Split the data into:

* **Training Set** (usually 80%)
* **Testing Set** (usually 20%)

Purpose:

* Train the model on one set.
* Evaluate it on unseen data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

### Step 4: Create the Linear Regression Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

---

### Step 5: Train (Fit) the Model

The model learns the best values of:

* **Intercept (θ₀)**
* **Slope (θ₁)**

such that the prediction error is minimized.

```python
model.fit(X_train, y_train)
```

Mathematically, the learned equation is: 
y hat = theta_0 + theta_1 * x



---

### Step 6: Make Predictions

Use the trained model to predict values for the test data.

```python
y_pred = model.predict(X_test)
```

---

### Step 7: Plot the Best-Fit Line

Compare the original data with the learned regression line.

```python
plt.scatter(X_train, y_train)

plt.plot(
    X_train,
    model.predict(X_train),
    color="red"
)

plt.show()
```

---

### Step 8: Evaluate the Model

Common evaluation metrics include:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mse)
print(r2)
```

---

### Step 9: Predict New Values

Once the model is trained, use it for new inputs.

```python
hours = [[6]]

prediction = model.predict(hours)

print(prediction)
```

---

## Complete Workflow

```text
1. Import Libraries
        │
        ▼
2. Load Dataset
        │
        ▼
3. Explore Dataset
        │
        ▼
4. Visualize Data
        │
        ▼
5. Select X and y
        │
        ▼
6. Train-Test Split
        │
        ▼
7. Feature Scaling
        │
        ▼
8. Train Linear Regression Model
        │
        ▼
9. Plot Best-Fit Line
        │
        ▼
10. Predict on Test Data
        │
        ▼
11. Evaluate the Model
        │
        ▼
12. Compare Actual vs Predicted
        │
        ▼
13. Predict for New Data
        │
        ▼
14. Interpret Results
```

### What the algorithm learns

At the core of Simple Linear Regression, the algorithm tries to find the best values of **θ₀ (intercept)** and **θ₁ (slope)** in the equation: y hat = theta_0 + theta_1 * x


These parameters define the **best-fit line**—the line that minimizes the prediction error across all training examples.

This sequence is exactly the workflow you'll implement in scikit-learn and is the standard pipeline used in most machine learning projects involving simple linear regression.
