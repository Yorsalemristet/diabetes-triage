from pathlib import Path

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load diabetes dataset
data = load_diabetes(as_frame=True)
X = data.frame.drop(columns=["target"])
y = data.frame["target"]

# 2. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create machine learning pipeline
model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Predict on test data
predictions = model.predict(X_test)

# 6. Calculate RMSE
rmse = mean_squared_error(y_test, predictions) ** 0.5

print(f"RMSE: {rmse:.4f}")

# 7. Save trained model
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

joblib.dump(model, model_dir / "model.joblib")

print("Model saved successfully!")