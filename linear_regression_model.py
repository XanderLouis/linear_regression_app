import pandas as pd

#Import dataset
df = pd.read_excel(r'C:\Users\Louisalex\Desktop\Projects\Development\linear_regression_model\data\data.xlsx')

# Independent, dependent features
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Creating training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Building the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

#Training the model
model.fit(X_train, y_train)

#Making the predictions of the data points in the test set
y_pred = model.predict(X_test)
print(y_pred)

model.predict([[15, 40, 1000, 75]])

#R-Squared
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)

#Adjusted R-Squared
k = X_test.shape[1]
n = X_test.shape[0]

adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

adj_r2

#Scatter Plot of Actual vs. Predicted Values
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5) # Plot actaul vs. predicted
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

#Save the trained model to a .pkl file
import joblib

joblib.dump(model, "model/model.pkl")