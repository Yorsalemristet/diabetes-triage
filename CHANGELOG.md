
# Changelog

All notable changes to the Diabetes Triage MLOps project are documented here.

## [v0.2] - Ridge Regression

### Changed
- Added a new training script: `training/train_v02.py`.
- Compared Ridge Regression and Random Forest Regression.
- Added a separate validation dataset for model selection.
- Selected Ridge Regression with `alpha=10.0` based on validation RMSE.
- Retrained the selected model on the development data.
- Saved the trained model as `models/model_v02.joblib`.

### Evaluation results

| Model | Dataset | RMSE |
|---|---|---:|
| Ridge | Validation | 51.1605 |
| Random Forest | Validation | 54.6608 |
| Ridge (v0.2) | Test | 53.6263 |
| Linear Regression (v0.1) | Test | 53.8534 |

### Rationale
Ridge Regression was selected because it achieved a lower validation RMSE than Random Forest. Ridge also applies regularization, which can help reduce overfitting.

The v0.2 test RMSE was 53.6263 compared with 53.8534 for v0.1, an observed reduction of 0.2271 (approximately 0.42%).

This improvement is small. The held-out test set was also inspected during earlier experiments, so these results should not be interpreted as an independent confirmation of generalization performance.

## [v0.1] - Initial Baseline

### Added
- Initial diabetes progression regression model using StandardScaler and LinearRegression.
- Training script: `training/train.py`.
- Model artifact: `models/model.joblib`.
- FastAPI endpoints: `GET /health` and `POST /predict`.
- Input validation for all ten diabetes features.
- Docker support.
- Automated API tests and GitHub Actions CI.

### Evaluation results
- Test RMSE: 53.8534.

### Notes
Predictions are continuous diabetes progression scores, not medical diagnoses or probabilities.
  