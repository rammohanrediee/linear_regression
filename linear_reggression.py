# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate or load dataset
# Example dataset
data = {
    'Feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Target': [2.3, 4.1, 5.8, 8.0, 10.2, 12.3, 14.1, 16.2, 18.1, 20.3]
}
df = pd.DataFrame(data)

# Define features (X) and target (y)
X = df[['Feature']]
y = df['Target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the results
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Example usage: Predict a new value
new_value = np.array([[11]])  # Example feature value
predicted_target = model.predict(new_value)
print("Predicted target for {}: {}".format(new_value[0][0], predicted_target[0]))
