import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset
iris = pd.read_csv(r"C:\Users\91729\OneDrive\Documents\isir\second\iris.csv")
print("First 5 rows of the dataset:")
print(iris.head())

# Feature and target selection
x = iris[['sepal.length', 'sepal.width']]
y = iris['petal.length']


# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
LR = LinearRegression()
LR.fit(x_train, y_train)

# Predict
y_pred = LR.predict(x_test)
print("\nPredicted Petal Length values:")
print(y_pred)

# Evaluate
MAE = metrics.mean_absolute_error(y_test, y_pred)
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)

print("\nModel Evaluation Metrics:")
print("Mean Absolute Error (MAE):", MAE)
print("Mean Squared Error (MSE):", MSE)
print("Root Mean Squared Error (RMSE):", RMSE)


import numpy as np

x = iris[['sepal.length']]
y = iris['petal.length']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
LR = LinearRegression()

training = LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)
print(y_pred, "\n")

#pip install numpy
import numpy as np
from sklearn import metrics

MAE = metrics.mean_absolute_error(y_test, y_pred)
print("Mean Absoute Error: ", MAE)
MSE = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squeared Error: ", MSE)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print("RMSE:",RMSE)
