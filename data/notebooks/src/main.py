import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_csv("../../../data/students.csv")

# DATA EXPLORATION
print(df.head())
print("\nInfo:"); print(df.info())
print("\nDescription:"); print(df.describe())
print("\nMissing Values:"); print(df.isnull().sum())

# Now we will test different approaches to see their accuracy

# APPROACH 1: Predicting the average score of students using only demographics and a Linear Regression model

# Creating target variable for APPROACH 1 and 2
df["average score"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3
X = df.drop(["math score", "reading score", "writing score", "average score"], axis=1)

# Target
y = df["average score"]

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Training the model for APPROACH 1
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print(f"\nR²_1:  {r2_score(y_test, y_pred):.4f}")
print(f"MAE_1: {mean_absolute_error(y_test, y_pred):.4f}")

# Scatter plot
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([0, 100], [0, 100], 'r--', label='Perfect prediction')
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("APPROACH 1 - Linear Regression: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()

# APPROACH 2: Using Ridge Regression to predict the same target
X = df.drop(["math score", "reading score", "writing score", "average score"], axis=1)

# Target
y = df["average score"]

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Training the model for APPROACH 2(without a scaler)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

# Evaluation
print(f"\nR²_2:  {r2_score(y_test, y_pred):.4f}")
print(f"MAE_2: {mean_absolute_error(y_test, y_pred):.4f}")

# Scatter plot
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([0, 100], [0, 100], 'r--', label='Perfect prediction')
plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.title("APPROACH 2 - Ridge Regression: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()

# APPROACH 3: Predicting the students math score from reading + writing + demographics with Ridge Regression
X = df.drop(["math score", "average score"], axis=1)
y = df["math score"]

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline: we are going to use a scaler first at it gives better results when mixing score and binary results
model = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print(f"\nR²_3:  {r2_score(y_test, y_pred):.4f}")
print(f"MAE_3: {mean_absolute_error(y_test, y_pred):.4f}")

# Scatter plot
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([0, 100], [0, 100], 'r--', label='Perfect prediction')
plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.title("APPROACH 3 - Ridge Regression: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()