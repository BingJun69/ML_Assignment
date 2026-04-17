# Machine Learning Assignment

This project predicts `Electricity_Consumed` from the smart meter dataset using three regression models:

- `KNN`
- `Random Forest`
- `XGBoost`

The current best model is `XGBoost`, evaluated using `R2`, `MAE`, `MSE`, and `RMSE`.

## Latest Results

| Model | R2 Score | MAE | MSE | RMSE |
|---|---:|---:|---:|---:|
| XGBoost | 0.588573 | 0.082720 | 0.010826 | 0.104047 |
| Random Forest | 0.573985 | 0.084224 | 0.011210 | 0.105875 |
| KNN | 0.175227 | 0.117820 | 0.021702 | 0.147316 |

## Dataset

Source file:

- `smart_meter_data.csv`

Main columns in the dataset:

- `Timestamp`: date and time of the reading
- `Electricity_Consumed`: target variable to predict
- `Temperature`: normalized temperature value
- `Humidity`: normalized humidity value
- `Wind_Speed`: normalized wind speed value
- `Avg_Past_Consumption`: historical average consumption
- `Anomaly_Label`: anomaly tag from the original dataset, not used as the regression target

Example rows:

| Timestamp | Electricity_Consumed | Temperature | Humidity | Wind_Speed | Avg_Past_Consumption | Anomaly_Label |
|---|---:|---:|---:|---:|---:|---|
| 2024-01-01 00:00:00 | 0.457786 | 0.469524 | 0.396368 | 0.445441 | 0.692057 | Normal |
| 2024-01-01 00:30:00 | 0.351956 | 0.465545 | 0.451184 | 0.458729 | 0.539874 | Normal |
| 2024-01-01 01:00:00 | 0.482948 | 0.285415 | 0.408289 | 0.470360 | 0.614724 | Normal |
| 2024-01-01 01:30:00 | 0.628838 | 0.482095 | 0.512308 | 0.576241 | 0.757044 | Normal |
| 2024-01-01 02:00:00 | 0.335974 | 0.624741 | 0.672021 | 0.373004 | 0.673981 | Normal |

## Features Used

The script builds extra features from the original data, including:

- time-based features from `Timestamp`
- lag features from past electricity consumption
- rolling statistics
- exponential weighted moving averages
- interaction features between weather variables

These engineered features help the models learn consumption patterns over time.

## Models

The project compares three regression models:

1. `KNeighborsRegressor`
2. `RandomForestRegressor`
3. `XGBRegressor`

The script uses the tuned settings already found for each model.

## Evaluation

Main evaluation metric:

- `R2 Score`

Additional metrics:

- `MAE`
- `MSE`
- `RMSE`

The script also saves:

- `model_outputs/model_comparison.csv`
- `model_outputs/actual_predicted_scatter.png`

## Run The Project

Install dependencies:

```powershell
python -m pip install pandas scikit-learn xgboost matplotlib seaborn
```

Run the script:

```powershell
python mlass.py
```

## Output

After running the script, the results are written to:

- `model_outputs/model_comparison.csv`
- `model_outputs/actual_predicted_scatter.png`

## Model Comparison Plot

![Actual vs Predicted Scatter Plot](model_outputs/actual_predicted_scatter.png)

## Notes

- The target is `Electricity_Consumed`, so this is a regression problem, not a classification problem.
- `Timestamp` is used through engineered time features rather than as a raw string.
- Only leakage-safe historical features are used, meaning the current target value is not directly fed back into the model.

## Conclusion

Among the three models, `XGBoost` produced the best overall performance, followed closely by `Random Forest`. `KNN` performed much worse, which suggests this problem benefits more from tree-based nonlinear learning than distance-based regression. The results also show that time-based feature engineering and historical lag information are important for predicting electricity consumption.
