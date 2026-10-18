from pathlib import Path

import joblib
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load diabetes dataset
data = load_diabetes(as_frame=True)
X = data.frame.drop(columns=["target"])
y = data.frame["target"]

# Same split as v0.1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Candidate models
models = {
    "Ridge": make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0)
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        n_jobs=1
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = mean_squared_error(y_test, predictions) ** 0.5

    results[name] = rmse
    print(f"{name} RMSE: {rmse:.4f}")

# Select the best candidate
best_name = min(results, key=results.get)
best_model = models[best_name]

print(f"Best candidate: {best_name}")
print(f"Best candidate RMSE: {results[best_name]:.4f}")

# Save v0.2 model separately
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

joblib.dump(best_model, model_dir / "model_v02.joblib")

print("v0.2 candidate model saved successfully!")