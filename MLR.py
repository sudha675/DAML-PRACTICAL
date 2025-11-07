# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the dataset
iris = pd.read_csv(r"C:\Users\91729\OneDrive\Documents\isir\second\iris.csv")
print("First 5 rows of the dataset:")
print(iris.head())

# Select features and target variable
x = iris[['sepal.length', 'sepal.width']]
y = iris['petal.length']
                # Dependent variable

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict using the trained model
y_pred = model.predict(x_test)

# Print predicted values
print("\nPredicted values:")
print(y_pred)

# Print model coefficients and intercept
print("\nModel coefficients:", model.coef_)
print("Model intercept:", model.intercept_)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("\nR² Score:", r2)
print("Mean Squared Error:", mse)

# Plot Actual vs Predicted values
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Petal Length")
plt.ylabel("Predicted Petal Length")
plt.title("Actual vs Predicted Petal Length")
plt.grid(True)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Line of perfect prediction
plt.show()
