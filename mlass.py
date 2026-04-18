from pathlib import Path
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is not installed. Install the required packages with:\n"
        "python -m pip install pandas scikit-learn xgboost matplotlib seaborn"
    ) from exc


DATA_PATH = Path("smart_meter_data.csv")
OUTPUT_DIR = Path("model_outputs")
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Electricity_Consumed"
TIME_COLUMN = "Timestamp"
SHOW_PLOTS = False
ENABLE_TUNING = False
CV_SPLITS = 3
KEEP_OUTPUT_FILES = {
    "model_comparison.csv",
    "actual_predicted_scatter.png",
}
SELECTED_FEATURES = [
    "Past_Ratio_1",
    "Past_Diff_1",
    "RollingMean_6",
    "RollingMax_12",
    "RollingStd_12",
    "Past_RollingMean_12",
    "Avg_Past_Consumption",
    "RollingMin_12",
    "RollingMin_6",
    "EWM_12",
]


def load_and_prepare_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    df["TimeStep"] = np.arange(len(df))
    df["Hour"] = df[TIME_COLUMN].dt.hour
    df["Minute"] = df[TIME_COLUMN].dt.minute
    df["Day"] = df[TIME_COLUMN].dt.day
    df["Month"] = df[TIME_COLUMN].dt.month
    df["DayOfWeek"] = df[TIME_COLUMN].dt.dayofweek
    df["DayOfYear"] = df[TIME_COLUMN].dt.dayofyear
    df["WeekOfYear"] = df[TIME_COLUMN].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["IsMonthStart"] = df[TIME_COLUMN].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df[TIME_COLUMN].dt.is_month_end.astype(int)
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Minute_Sin"] = np.sin(2 * np.pi * df["Minute"] / 60)
    df["Minute_Cos"] = np.cos(2 * np.pi * df["Minute"] / 60)
    df["DayOfWeek_Sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_Cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfYear_Sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 366)
    df["DayOfYear_Cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 366)
    df["Temp_X_Humidity"] = df["Temperature"] * df["Humidity"]
    df["Temp_X_Wind"] = df["Temperature"] * df["Wind_Speed"]
    df["Humidity_X_Wind"] = df["Humidity"] * df["Wind_Speed"]
    df["Past_Diff_1"] = df["Avg_Past_Consumption"] - df["Avg_Past_Consumption"].shift(1)
    df["Past_Ratio_1"] = df["Avg_Past_Consumption"] / (
        df["Avg_Past_Consumption"].shift(1) + 1e-6
    )
    df["Past_RollingMean_12"] = df["Avg_Past_Consumption"].shift(1).rolling(12).mean()
    df["Past_RollingStd_12"] = df["Avg_Past_Consumption"].shift(1).rolling(12).std()

    for lag in [1, 2, 3, 4, 6, 12, 24, 48, 96, 144, 168, 336]:
        df[f"Lag_{lag}"] = df[TARGET_COLUMN].shift(lag)

    shifted = df[TARGET_COLUMN].shift(1)
    for window in [3, 6, 12, 24, 48, 96, 168]:
        df[f"RollingMean_{window}"] = shifted.rolling(window).mean()
        df[f"RollingStd_{window}"] = shifted.rolling(window).std()
        df[f"RollingMin_{window}"] = shifted.rolling(window).min()
        df[f"RollingMax_{window}"] = shifted.rolling(window).max()

    df["EWM_12"] = shifted.ewm(span=12, adjust=False).mean()
    df["EWM_48"] = shifted.ewm(span=48, adjust=False).mean()
    df["EWM_168"] = shifted.ewm(span=168, adjust=False).mean()
    df = df.dropna().reset_index(drop=True)

    features = df.drop(columns=[TARGET_COLUMN, TIME_COLUMN, "Anomaly_Label"])
    features = features[SELECTED_FEATURES].copy()
    target = df[TARGET_COLUMN]
    return features, target


def chronological_train_test_split(
    features: pd.DataFrame, target: pd.Series, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_index = int(len(features) * (1 - test_size))
    x_train = features.iloc[:split_index].copy()
    x_test = features.iloc[split_index:].copy()
    y_train = target.iloc[:split_index].copy()
    y_test = target.iloc[split_index:].copy()
    return x_train, x_test, y_train, y_test


def get_split_index(total_rows: int, test_size: float) -> int:
    return int(total_rows * (1 - test_size))


def clear_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for path in OUTPUT_DIR.iterdir():
        if path.is_file():
            path.unlink()


def save_and_display_plot(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def build_models() -> dict[str, Pipeline]:
    knn_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                slice(0, None),
            )
        ]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                slice(0, None),
            )
        ]
    )

    return {
        "KNN": Pipeline(
            steps=[
                ("preprocessor", knn_preprocessor),
                ("regressor", KNeighborsRegressor(n_neighbors=31, weights="distance", p=1)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=900,
                        max_depth=22,
                        min_samples_split=4,
                        min_samples_leaf=1,
                        max_features=0.7,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "regressor",
                    XGBRegressor(
                        n_estimators=1000,
                        learning_rate=0.015,
                        max_depth=4,
                        min_child_weight=2,
                        subsample=0.9,
                        colsample_bytree=1.0,
                        reg_alpha=0.0,
                        reg_lambda=3.0,
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def tune_models(
    models: dict[str, Pipeline], x_train: pd.DataFrame, y_train: pd.Series
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    if not ENABLE_TUNING:
        return models, pd.DataFrame(columns=["Model", "Best CV R2", "Best Parameters"])

    cv = TimeSeriesSplit(n_splits=CV_SPLITS)
    tuned_models: dict[str, Pipeline] = {}
    tuning_rows = []

    searchers = {
        "KNN": GridSearchCV(
            estimator=models["KNN"],
            param_grid={
                "regressor__n_neighbors": [15, 21, 31],
                "regressor__weights": ["uniform", "distance"],
                "regressor__p": [1],
            },
            scoring="r2",
            cv=cv,
            n_jobs=1,
        ),
        "Random Forest": RandomizedSearchCV(
            estimator=models["Random Forest"],
            param_distributions={
                "regressor__n_estimators": [900, 1200],
                "regressor__max_depth": [22, 28, None],
                "regressor__max_features": [0.4, 0.5, 0.7],
                "regressor__min_samples_split": [2, 4],
                "regressor__min_samples_leaf": [1, 2],
            },
            n_iter=8,
            scoring="r2",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "XGBoost": RandomizedSearchCV(
            estimator=models["XGBoost"],
            param_distributions={
                "regressor__n_estimators": [1000, 1200, 1400],
                "regressor__learning_rate": [0.01, 0.015, 0.02],
                "regressor__max_depth": [4, 5, 6],
                "regressor__min_child_weight": [1, 2, 3],
                "regressor__subsample": [0.85, 0.9, 1.0],
                "regressor__colsample_bytree": [0.8, 0.9, 1.0],
                "regressor__reg_alpha": [0.0, 0.1, 0.2],
                "regressor__reg_lambda": [2.0, 2.5, 3.0],
            },
            n_iter=10,
            scoring="r2",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    print(f"\n{'=' * 60}")
    print("Hyperparameter Tuning")
    print(f"{'=' * 60}")

    for model_name, searcher in searchers.items():
        print(f"Tuning {model_name}...")
        searcher.fit(x_train, y_train)
        tuned_models[model_name] = searcher.best_estimator_
        tuning_rows.append(
            {
                "Model": model_name,
                "Best CV R2": searcher.best_score_,
                "Best Parameters": str(searcher.best_params_),
            }
        )
        print(f"Best CV R2    : {searcher.best_score_:.4f}")
        print(f"Best Params   : {searcher.best_params_}")

    tuning_df = pd.DataFrame(tuning_rows).sort_values(by="Best CV R2", ascending=False)
    tuning_df.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)
    return tuned_models, tuning_df


def evaluate_model(
    model_name: str,
    model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.fit(x_train, y_train)
    predictions = pd.Series(model.predict(x_test), index=y_test.index, name="Predicted")

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5
    r2 = r2_score(y_test, predictions)

    comparison_df = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": predictions,
        }
    ).sort_index()
    comparison_df["Residual"] = comparison_df["Actual"] - comparison_df["Predicted"]
    comparison_df["Absolute_Error"] = comparison_df["Residual"].abs()

    print(f"\n{'=' * 60}")
    print(f"{model_name} Regression Results")
    print(f"{'=' * 60}")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"RMSE     : {rmse:.4f}")

    return (
        {
            "Model": model_name,
            "R2 Score": r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
        },
        comparison_df,
    )


def save_score_plot(results_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=results_df,
        x="Model",
        y="R2 Score",
        hue="Model",
        palette="crest",
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title("R2 Score Comparison")
    ax.set_ylim(min(0, results_df["R2 Score"].min() - 0.05), 1.0)
    ax.set_ylabel("R2 Score")
    ax.set_xlabel("")
    for idx, score in enumerate(results_df["R2 Score"]):
        ax.text(idx, score + 0.01, f"{score:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    save_and_display_plot(fig, "r2_comparison.png")


def save_metric_plot(results_df: pd.DataFrame) -> None:
    plot_df = results_df.set_index("Model")[["MAE", "RMSE"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Prediction Error Comparison")
    ax.set_ylabel("Error")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    save_and_display_plot(fig, "error_comparison.png")


def save_actual_vs_predicted_plot(predictions_by_model: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for ax, (model_name, comparison_df) in zip(axes, predictions_by_model.items()):
        subset = comparison_df.reset_index(drop=True).head(150)
        ax.plot(subset.index, subset["Actual"], label="Actual", linewidth=2, color="#1b4965")
        ax.plot(
            subset.index,
            subset["Predicted"],
            label=f"{model_name} Predicted",
            linewidth=1.8,
            linestyle="--",
            color="#e76f51",
        )
        ax.set_title(f"{model_name}: Actual vs Predicted")
        ax.set_ylabel("Electricity Consumed")
        ax.legend()

    axes[-1].set_xlabel("Test Sample Index")
    fig.tight_layout()
    save_and_display_plot(fig, "actual_vs_predicted.png")


def save_residual_plot(predictions_by_model: dict[str, pd.DataFrame]) -> None:
    residual_frames = []
    for model_name, comparison_df in predictions_by_model.items():
        residual_frame = comparison_df[["Residual"]].copy()
        residual_frame["Model"] = model_name
        residual_frames.append(residual_frame)

    residual_df = pd.concat(residual_frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=residual_df,
        x="Model",
        y="Residual",
        hue="Model",
        palette="Set2",
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Residual Difference from Actual Values")
    ax.set_xlabel("")
    ax.set_ylabel("Actual - Predicted")
    fig.tight_layout()
    save_and_display_plot(fig, "residual_comparison.png")


def save_scatter_plot(predictions_by_model: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)

    actual_min = min(df["Actual"].min() for df in predictions_by_model.values())
    actual_max = max(df["Actual"].max() for df in predictions_by_model.values())

    for ax, (model_name, comparison_df) in zip(axes, predictions_by_model.items()):
        ax.scatter(
            comparison_df["Actual"],
            comparison_df["Predicted"],
            alpha=0.65,
            s=25,
            color="#2a9d8f",
        )
        ax.plot([actual_min, actual_max], [actual_min, actual_max], "r--", linewidth=1.5)
        ax.set_title(model_name)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    fig.suptitle("Actual vs Predicted Scatter by Model", y=1.02)
    fig.tight_layout()
    save_and_display_plot(fig, "actual_predicted_scatter.png")


def save_feature_signal_plot(features: pd.DataFrame, target: pd.Series) -> None:
    correlation_df = features.copy()
    correlation_df[TARGET_COLUMN] = target.values
    corr = correlation_df.corr(numeric_only=True)[TARGET_COLUMN].drop(TARGET_COLUMN)
    corr = corr.sort_values(key=lambda series: series.abs(), ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        x=corr.values,
        y=corr.index,
        hue=corr.index,
        palette="viridis",
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title("Top Feature Correlations with Electricity Consumed")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("")
    fig.tight_layout()
    save_and_display_plot(fig, "feature_signal.png")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

    clear_output_dir()
    sns.set_theme(style="whitegrid")

    x, y = load_and_prepare_data(DATA_PATH)
    x_train, x_test, y_train, y_test = chronological_train_test_split(
        features=x,
        target=y,
        test_size=TEST_SIZE,
    )

    print("Dataset Summary")
    print("-" * 60)
    print(f"Samples          : {len(x)}")
    print(f"Features         : {x.shape[1]}")
    print(f"Training samples : {len(x_train)}")
    print(f"Testing samples  : {len(x_test)}")
    print(f"Target           : {TARGET_COLUMN}")
    print("Split method     : chronological 80/20")

    models = build_models()
    models, tuning_df = tune_models(models=models, x_train=x_train, y_train=y_train)

    results = []
    predictions_by_model: dict[str, pd.DataFrame] = {}

    for model_name, model in models.items():
        metrics, comparison_df = evaluate_model(
            model_name=model_name,
            model=model,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )
        results.append(metrics)
        predictions_by_model[model_name] = comparison_df

    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)

    print(f"\n{'=' * 60}")
    print("Final Model Comparison")
    print(f"{'=' * 60}")
    print(results_df.to_string(index=False))

    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    save_scatter_plot(predictions_by_model)

    best_model = results_df.iloc[0]["Model"]
    best_r2 = results_df.iloc[0]["R2 Score"]
    print(f"\nBest model based on R2 Score: {best_model} ({best_r2:.4f})")
    if best_r2 < 0.2:
        print(
            "Diagnostic note: even after adding lag and rolling features, the dataset "
            "still has weak predictive signal for Electricity_Consumed. This suggests "
            "the source data is better suited for anomaly classification than high-R2 regression."
        )
    print(f"Saved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
