import joblib
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss
)

try:
    from training.train_xgboost import (
        load_datasets,
        prepare_data,
        split_data,
        FEATURES,
        EXTRA_FEATURES,
        MODEL_DIR,
        RANDOM_STATE
    )
except ImportError:
    from train_xgboost import (
        load_datasets,
        prepare_data,
        split_data,
        FEATURES,
        EXTRA_FEATURES,
        MODEL_DIR,
        RANDOM_STATE
    )


MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "random_forest_5d.joblib"

print("=" * 55)
print("AI STOCK PREDICTOR - RANDOM FOREST")
print("=" * 55)

# 1. Load and prepare the same datasets.
raw_data = load_datasets()
data = prepare_data(raw_data)

# 2. Use the same chronological split.
train_df, test_df = split_data(data)

X_train = train_df[FEATURES + EXTRA_FEATURES]
y_train = train_df["Label"]

X_test = test_df[FEATURES + EXTRA_FEATURES]
y_test = test_df["Label"]

print("\nTraining Random Forest...")

# 3. Train model.
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=2
)

model.fit(X_train, y_train)

print("Training completed!")

# 4. Predictions.
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 5. Evaluation.
accuracy = accuracy_score(y_test, predictions)
balanced = balanced_accuracy_score(y_test, predictions)

loss = log_loss(
    y_test,
    probabilities,
    labels=[0, 1, 2]
)

print("\n" + "=" * 55)
print("RANDOM FOREST EVALUATION")
print("=" * 55)

print("Accuracy:", round(accuracy, 4))
print("Balanced Accuracy:", round(balanced, 4))
print("Log Loss:", round(loss, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=["Loss", "Neutral", "Profit"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2]
    )
)

# 6. Save model separately.
joblib.dump(model, MODEL_PATH)

metadata = {
    "model_name": "Random Forest Classifier",
    "prediction_horizon": "5 trading days",
    "features": FEATURES + EXTRA_FEATURES,
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced),
    "log_loss": float(loss)
}

with open(
    MODEL_DIR / "random_forest_5d_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)

print("\nModel saved:", MODEL_PATH)
print("\nRANDOM FOREST TRAINING COMPLETED!")
