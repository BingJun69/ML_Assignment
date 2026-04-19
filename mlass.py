import argparse
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is not installed. Install it with:\n"
        "python -m pip install pandas scikit-learn xgboost matplotlib seaborn joblib"
    ) from exc


matplotlib.use("Agg")


DATA_PATH = Path("smart_meter_data.csv")
NEW_DATA_PATH = Path("new_smart_meter_data.csv")
OUTPUT_DIR = Path("model_outputs")
MODEL_DIR = Path("saved_models")
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
BEST_MODEL_INFO_PATH = MODEL_DIR / "best_model_info.txt"

TARGET_COLUMN = "Electricity_Consumed"
TIME_COLUMN = "Timestamp"
TEST_SIZE = 0.2
RANDOM_STATE = 42
SHOW_PLOTS = False

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the smart meter models or use a saved model on a new CSV."
    )
    parser.add_argument(
        "--predict-only",
        metavar="CSV",
        help="Load the saved best model and predict on a different CSV file.",
    )
    parser.add_argument(
        "--train-data",
        default=str(DATA_PATH),
        help="CSV file used for training and internal evaluation.",
    )
    return parser.parse_args()


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)


def clear_output_dir() -> None:
    ensure_output_dirs()
    for path in OUTPUT_DIR.iterdir():
        if path.is_file():
            path.unlink()


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    df["Hour"] = df[TIME_COLUMN].dt.hour
    df["Minute"] = df[TIME_COLUMN].dt.minute
    df["DayOfWeek"] = df[TIME_COLUMN].dt.dayofweek
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["DayOfWeek_Sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_Cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    df["Past_Diff_1"] = df["Avg_Past_Consumption"] - df["Avg_Past_Consumption"].shift(1)
    df["Past_Ratio_1"] = df["Avg_Past_Consumption"] / (df["Avg_Past_Consumption"].shift(1) + 1e-6)
    df["Past_RollingMean_12"] = df["Avg_Past_Consumption"].shift(1).rolling(12).mean()

    shifted = df[TARGET_COLUMN].shift(1)
    for lag in [1, 2, 3, 6, 12, 24, 48, 96]:
        df[f"Lag_{lag}"] = df[TARGET_COLUMN].shift(lag)

    df["RollingMean_6"] = shifted.rolling(6).mean()
    df["RollingMin_6"] = shifted.rolling(6).min()
    df["RollingMin_12"] = shifted.rolling(12).min()
    df["RollingMax_12"] = shifted.rolling(12).max()
    df["RollingStd_12"] = shifted.rolling(12).std()
    df["EWM_12"] = shifted.ewm(span=12, adjust=False).mean()
    df = df.dropna().reset_index(drop=True)

    features = df[SELECTED_FEATURES].copy()
    target = df[TARGET_COLUMN]
    return features, target


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path.resolve()}")
    return prepare_features(pd.read_csv(csv_path))


def chronological_split(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_index = int(len(features) * (1 - TEST_SIZE))
    return (
        features.iloc[:split_index].copy(),
        features.iloc[split_index:].copy(),
        target.iloc[:split_index].copy(),
        target.iloc[split_index:].copy(),
    )


def build_models() -> dict[str, Pipeline]:
    tree_preprocessor = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), slice(0, None))]
    )
    knn_preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                slice(0, None),
            )
        ]
    )

    return {
        "KNN": Pipeline(
            [
                ("preprocessor", knn_preprocessor),
                ("regressor", KNeighborsRegressor(n_neighbors=31, weights="distance", p=1)),
            ]
        ),
        "Random Forest": Pipeline(
            [
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
            [
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


def evaluate_model(
    name: str, model: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple[dict[str, float], pd.DataFrame]:
    model.fit(x_train, y_train)
    preds = pd.Series(model.predict(x_test), index=y_test.index, name="Predicted")

    results = {
        "Model": name,
        "R2 Score": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds) ** 0.5,
    }

    comparison = pd.DataFrame({"Actual": y_test, "Predicted": preds}).sort_index()
    return results, comparison


def save_results_plot(results_by_model: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, (name, df) in zip(axes, results_by_model.items()):
        ax.scatter(df["Actual"], df["Predicted"], s=25, alpha=0.65, color="#2a9d8f")
        lo = min(df["Actual"].min(), df["Predicted"].min())
        hi = max(df["Actual"].max(), df["Predicted"].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
        ax.set_title(name)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "actual_predicted_scatter.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def save_best_model(model_name: str, model: Pipeline, metrics: dict[str, float]) -> None:
    joblib.dump(model, BEST_MODEL_PATH)
    BEST_MODEL_INFO_PATH.write_text(
        "\n".join(
            [
                f"Best model: {model_name}",
                f"R2 Score: {metrics['R2 Score']:.6f}",
                f"MAE: {metrics['MAE']:.6f}",
                f"MSE: {metrics['MSE']:.6f}",
                f"RMSE: {metrics['RMSE']:.6f}",
            ]
        ),
        encoding="utf-8",
    )


def load_saved_model() -> Pipeline:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found at {BEST_MODEL_PATH.resolve()}. Run training first."
        )
    return joblib.load(BEST_MODEL_PATH)


def predict_only(csv_path: Path) -> None:
    model = load_saved_model()
    features, target = load_dataset(csv_path)
    predictions = pd.DataFrame({"Predicted": model.predict(features)})
    if target is not None:
        predictions.insert(0, "Actual", target.reset_index(drop=True))
    predictions.to_csv(OUTPUT_DIR / "new_data_predictions.csv", index=False)
    print(f"New dataset predictions saved to: {OUTPUT_DIR / 'new_data_predictions.csv'}")


def train_and_eval(train_csv: Path) -> None:
    clear_output_dir()
    sns.set_theme(style="whitegrid")

    features, target = load_dataset(train_csv)
    x_train, x_test, y_train, y_test = chronological_split(features, target)

    print("Dataset Summary")
    print("-" * 60)
    print(f"Samples          : {len(features)}")
    print(f"Features         : {features.shape[1]}")
    print(f"Training samples : {len(x_train)}")
    print(f"Testing samples  : {len(x_test)}")
    print(f"Target           : {TARGET_COLUMN}")
    print("Split method     : chronological 80/20")

    models = build_models()
    scores = []
    comparisons: dict[str, pd.DataFrame] = {}

    for name, model in models.items():
        metrics, comparison = evaluate_model(name, model, x_train, x_test, y_train, y_test)
        scores.append(metrics)
        comparisons[name] = comparison
        print(f"\n{name} Regression Results")
        print("-" * 60)
        print(f"R2 Score : {metrics['R2 Score']:.4f}")
        print(f"MAE      : {metrics['MAE']:.4f}")
        print(f"MSE      : {metrics['MSE']:.4f}")
        print(f"RMSE     : {metrics['RMSE']:.4f}")

    results_df = pd.DataFrame(scores).sort_values(by="R2 Score", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    save_results_plot(comparisons)

    best = results_df.iloc[0]
    best_model = models[best["Model"]]
    best_model.fit(x_train, y_train)
    save_best_model(best["Model"], best_model, best.to_dict())

    print(f"\nBest model based on R2 Score: {best['Model']} ({best['R2 Score']:.4f})")
    print(f"Saved outputs to: {OUTPUT_DIR.resolve()}")
    print(f"Saved trained model to: {BEST_MODEL_PATH.resolve()}")


def main() -> None:
    args = parse_args()
    if args.predict_only:
        ensure_output_dirs()
        predict_only(Path(args.predict_only))
        return

    train_and_eval(Path(args.train_data))


if __name__ == "__main__":
    main()
