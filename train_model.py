import os
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
print("Loading the California Housing dataset...")
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Split the data
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training the Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete!")

# Evaluate the model
print("Evaluating the model...")
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Save the model
os.makedirs("Model", exist_ok=True)
model_path = os.path.join("Model", "house_price_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved successfully at {model_path}!")
