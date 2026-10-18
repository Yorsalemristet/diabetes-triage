
from pathlib import Path

import joblib
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load diabetes dataset
data = load_diabetes(as_frame=True)
X = data.frame.drop(columns=["target"])
y = data.frame["target"]

# Keep the same held-out test split as v0.1
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split development data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=42
)

models = {
    "Ridge": make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        n_jobs=1,
    ),
}

results = {}

# Select model using validation data
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, predictions)

    results[name] = rmse
    print(f"{name} Validation RMSE: {rmse:.4f}")

best_name = min(results, key=results.get)
best_model = models[best_name]

print(f"Selected model: {best_name}")

# Retrain selected model on the complete development set
best_model.fit(X_dev, y_dev)

# Final evaluation on the held-out test set
test_predictions = best_model.predict(X_test)
test_rmse = root_mean_squared_error(y_test, test_predictions)

print(f"v0.2 Test RMSE: {test_rmse:.4f}")
print("v0.1 Test RMSE: 53.8534")

# Save the selected model
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

joblib.dump(best_model, model_dir / "model_v02.joblib")

print("v0.2 model saved successfully!")
