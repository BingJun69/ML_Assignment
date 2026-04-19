# Smart Meter Electricity Prediction

This project compares three regression models for predicting `Electricity_Consumed` from a smart meter dataset:

- `KNeighborsRegressor`
- `RandomForestRegressor`
- `XGBRegressor`

The best-performing model in the current setup is `XGBoost`, evaluated with `R2 Score`, `MAE`, `MSE`, and `RMSE`.

## Dataset

Source file:

- `smart_meter_data.csv`

Main columns:

- `Timestamp`
- `Electricity_Consumed`
- `Temperature`
- `Humidity`
- `Wind_Speed`
- `Avg_Past_Consumption`
- `Anomaly_Label`

## Approach

The script:

1. sorts the readings by `Timestamp`
2. creates time-based and historical features
3. removes rows with missing values created by lagging and rolling windows
4. keeps the strongest engineered predictors
5. splits the data chronologically into training and testing sets
6. trains and evaluates the three regression models

The selected features are:

- `Past_Ratio_1`
- `Past_Diff_1`
- `RollingMean_6`
- `RollingMax_12`
- `RollingStd_12`
- `Past_RollingMean_12`
- `Avg_Past_Consumption`
- `RollingMin_12`
- `RollingMin_6`
- `EWM_12`

## Results

| Model | R2 Score | MAE | MSE | RMSE |
|---|---:|---:|---:|---:|
| XGBoost | 0.594886 | 0.081731 | 0.010660 | 0.103246 |
| Random Forest | 0.581963 | 0.082663 | 0.011000 | 0.104879 |
| KNN | 0.515218 | 0.090458 | 0.012756 | 0.112942 |

## Output Files

Each run writes:

- `model_outputs/model_comparison.csv`
- `model_outputs/actual_predicted_scatter.png`
- `saved_models/best_model.joblib`
- `saved_models/best_model_info.txt`

## Run

Install dependencies:

```powershell
python -m pip install pandas scikit-learn xgboost matplotlib seaborn joblib
```

Train and evaluate:

```powershell
python mlass.py
```

Predict on another CSV with the saved model:

```powershell
python mlass.py --predict-only new_smart_meter_data.csv
```

The new CSV should use the same column structure as the training data.

## Visual

![Actual vs Predicted Scatter Plot](model_outputs/actual_predicted_scatter.png)
