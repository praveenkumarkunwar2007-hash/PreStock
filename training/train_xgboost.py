
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss
)


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = [
    BASE_DIR / "datasets/india/indian_features.csv",
    BASE_DIR / "datasets/usa/us_features.csv"
]

MODEL_DIR = BASE_DIR / "backend/models"

MODEL_PATH = MODEL_DIR / "xgboost_5d_v2.joblib"

METADATA_PATH = MODEL_DIR / "xgboost_5d_v2_metadata.json"

TEST_RATIO = 0.20

FUTURE_DAYS = 5

RANDOM_STATE = 42


# ==========================================
# 2. FEATURE SELECTION
# ==========================================

# Only use information available on prediction day.
# Never include future returns or target columns.

FEATURES = [
    "Daily_Return",
    "Return_5D",
    "Return_10D",
    "Return_20D",
    "Price_vs_SMA20",
    "Price_vs_SMA50",
    "RSI_14",
    "BB_Width",
    "BB_Position",
    "ATR_Percent",
    "Volatility_20D",
    "Volume_Ratio"
]

# Original MACD values depend on stock price scale.
# Normalize them to make cross-stock learning easier.

EXTRA_FEATURES = [
    "MACD_pct",
    "MACD_Signal_pct",
    "MACD_Histogram_pct"
]

# Original labels:
# -1 = Loss
#  0 = Neutral
#  1 = Profit
#
# XGBoost labels:
#  0 = Loss
#  1 = Neutral
#  2 = Profit

LABEL_MAP = {
    -1: 0,
    0: 1,
    1: 2
}


# ==========================================
# 3. LOAD DATASETS
# ==========================================

def load_datasets():

    all_data = []

    for file_path in DATASETS:

        print("\nLoading:", file_path)

        if not file_path.exists():
            print("File not found:", file_path)
            continue

        df = pd.read_csv(file_path)

        if df.empty:
            continue

        df["Date"] = pd.to_datetime(df["Date"])

        # Normalize MACD indicators by closing price.

        df["MACD_pct"] = (
            df["MACD"] / df["Close"]
        )

        df["MACD_Signal_pct"] = (
            df["MACD_Signal"] / df["Close"]
        )

        df["MACD_Histogram_pct"] = (
            df["MACD_Histogram"] / df["Close"]
        )

        all_data.append(df)

        print("Rows loaded:", len(df))
        print("Stocks:", df["Symbol"].nunique())

    if not all_data:
        raise ValueError(
            "No feature datasets found. Run preprocess.py first."
        )

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    print("\nTotal records:", len(combined))
    print("Total stocks:", combined["Symbol"].nunique())

    return combined


# ==========================================
# 4. PREPARE FEATURES AND LABELS
# ==========================================

def prepare_data(df):

    required = (
        FEATURES
        + EXTRA_FEATURES
        + ["Symbol", "Date", "Target_5D"]
    )

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.copy()

    df["Target_5D"] = pd.to_numeric(
        df["Target_5D"],
        errors="coerce"
    )

    # Remove observations whose future outcome
    # has not yet become available.

    df = df.dropna(
        subset=["Target_5D"]
    )

    # Map original classes to XGBoost classes.

    df["Label"] = df["Target_5D"].map(LABEL_MAP)

    # Clean numerical data.

    model_features = FEATURES + EXTRA_FEATURES

    df[model_features] = df[
        model_features
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=model_features + ["Label"]
    )

    df["Label"] = df["Label"].astype(int)

    return df


# ==========================================
# 5. CHRONOLOGICAL TRAIN/TEST SPLIT
# ==========================================


def split_data(df):

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Use one common date cutoff for every stock.
    all_dates = np.sort(df["Date"].dt.normalize().unique())

    split_index = int(len(all_dates) * (1 - TEST_RATIO))
    cutoff_date = pd.Timestamp(all_dates[split_index])

    print("\nGlobal chronological cutoff:", cutoff_date.date())

    train_parts = []
    test_parts = []

    for symbol, stock in df.groupby("Symbol"):

        stock = stock.sort_values("Date").copy()

        # Training observations must be before cutoff.
        train_stock = stock[stock["Date"] < cutoff_date].copy()

        # Testing observations must be on or after cutoff.
        test_stock = stock[stock["Date"] >= cutoff_date].copy()

        # Remove the final five training observations
        # to reduce overlap between future labels.
        if len(train_stock) > FUTURE_DAYS:
            train_stock = train_stock.iloc[:-FUTURE_DAYS]
        else:
            continue

        if len(train_stock) < 50 or len(test_stock) == 0:
            continue

        train_parts.append(train_stock)
        test_parts.append(test_stock)

    if not train_parts or not test_parts:
        raise ValueError("Unable to create chronological datasets.")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    print("\nTraining rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Latest training date:", train_df["Date"].max().date())
    print("Earliest testing date:", test_df["Date"].min().date())

    return train_df, test_df


# ==========================================
# 6. TRAIN XGBOOST MODEL
# ==========================================


def train_model(train_df):

    X = train_df[FEATURES + EXTRA_FEATURES]
    y = train_df["Label"]

    # Sort by date before creating validation data.
    dates = np.sort(
        pd.to_datetime(train_df["Date"]).dt.normalize().unique()
    )

    validation_index = int(len(dates) * 0.80)
    validation_date = pd.Timestamp(dates[validation_index])

    # Keep a five-trading-day gap before validation.
    earlier_dates = dates[:validation_index]
    training_cutoff = pd.Timestamp(
        earlier_dates[-FUTURE_DAYS]
    )

    fit_mask = pd.to_datetime(train_df["Date"]).dt.normalize() <= training_cutoff
    val_mask = pd.to_datetime(train_df["Date"]).dt.normalize() >= validation_date

    X_fit = X.loc[fit_mask]
    y_fit = y.loc[fit_mask]

    X_val = X.loc[val_mask]
    y_val = y.loc[val_mask]

    if len(X_fit) == 0 or len(X_val) == 0:
        raise ValueError("Insufficient data for validation.")

    print("\nTraining samples:", len(X_fit))
    print("Validation samples:", len(X_val))

    # Test different configurations.
    configurations = [
        {
            "n_estimators": 250,
            "max_depth": 2,
            "learning_rate": 0.03,
            "min_child_weight": 5
        },
        {
            "n_estimators": 400,
            "max_depth": 2,
            "learning_rate": 0.03,
            "min_child_weight": 7
        },
        {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 5
        },
        {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.02,
            "min_child_weight": 7
        },
        {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 7
        },
        {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.05,
            "min_child_weight": 5
        }
    ]

    best_score = -1
    best_loss = float("inf")
    best_params = None

    print("\nSTARTING XGBOOST OPTIMIZATION")

    for number, params in enumerate(configurations, start=1):

        print(f"\nTesting configuration {number}/{len(configurations)}")
        print("Parameters:", params)

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=2,
            **params
        )

        model.fit(X_fit, y_fit, verbose=False)

        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)

        score = balanced_accuracy_score(y_val, predictions)

        loss = log_loss(
            y_val,
            probabilities,
            labels=[0, 1, 2]
        )

        print("Validation balanced accuracy:", round(score, 4))
        print("Validation log loss:", round(loss, 4))

        if (
            score > best_score
            or (score == best_score and loss < best_loss)
        ):
            best_score = score
            best_loss = loss
            best_params = params.copy()

    print("\nBEST CONFIGURATION:", best_params)
    print("Best validation balanced accuracy:", round(best_score, 4))

    # Train final model using all available outer training data.
    print("\nTraining final optimized model...")

    final_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=2,
        **best_params
    )

    final_model.fit(X, y, verbose=False)

    print("Optimized model training completed!")

    return final_model


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

def evaluate_model(model, test_df):

    X_test = test_df[
        FEATURES + EXTRA_FEATURES
    ]

    y_test = test_df["Label"]

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    print("\n" + "=" * 55)
    print("MODEL EVALUATION")
    print("=" * 55)

    print(
        "\nAccuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )

    print(
        "Balanced Accuracy:",
        round(balanced_accuracy_score(y_test, predictions), 4)
    )

    print(
        "Log Loss:",
        round(log_loss(
            y_test,
            probabilities,
            labels=[0, 1, 2]
        ), 4)
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=[
                "Loss",
                "Neutral",
                "Profit"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1, 2]
        )
    )

    return probabilities


# ==========================================
# 8. FEATURE IMPORTANCE
# ==========================================

def show_feature_importance(model):

    importance = pd.DataFrame({

        "Feature": FEATURES + EXTRA_FEATURES,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    print("\nFEATURE IMPORTANCE")

    print(importance.to_string(index=False))

    return importance


# ==========================================
# 9. SAVE TRAINED MODEL
# ==========================================

def save_model(model, importance):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    metadata = {

        "model_name": "XGBoost Stock Classifier",

        "prediction_horizon": "5 trading days",

        "classes": {
            "0": "Loss",
            "1": "Neutral",
            "2": "Profit"
        },

        "original_class_mapping": {
            "-1": "Loss",
            "0": "Neutral",
            "1": "Profit"
        },

        "features": FEATURES + EXTRA_FEATURES,

        "profit_threshold": 0.01,

        "loss_threshold": -0.01,

        "future_days": FUTURE_DAYS,

        "model_file": MODEL_PATH.name,

        "feature_importance": importance.to_dict(
            orient="records"
        )

    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\nModel saved:", MODEL_PATH)

    print("Metadata saved:", METADATA_PATH)


# ==========================================
# 10. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    print("=" * 55)
    print("AI STOCK PREDICTOR - XGBOOST TRAINING")
    print("=" * 55)

    raw_data = load_datasets()

    prepared_data = prepare_data(raw_data)

    train_df, test_df = split_data(prepared_data)

    model = train_model(train_df)

    evaluate_model(model, test_df)

    importance = show_feature_importance(model)

    save_model(model, importance)

    print("\nXGBOOST TRAINING COMPLETED!")
