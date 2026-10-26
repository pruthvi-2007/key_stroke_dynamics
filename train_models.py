
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = "keystroke_features.csv"
OUTPUT_DIR = "results"
RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 65)
print("KEYSTROKE DYNAMICS - USER IDENTIFICATION")
print("=" * 65)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Cannot find {DATA_FILE}. "
        "Place it in the same folder as this Python script."
    )

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)
print("\nAttempts per user:")
print(df["user_id"].value_counts().sort_index())


# ============================================================
# 3. VALIDATE DATA
# ============================================================

required_columns = {"user_id", "attempt_id"}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df["user_id"].isna().any():
    raise ValueError("The target column user_id contains missing values.")

# Remove rows without a valid user label, if any exist.
df = df.dropna(subset=["user_id"]).copy()

# user_id is the target. attempt_id is an identifier, not a feature.
X = df.drop(columns=["user_id", "attempt_id"], errors="ignore")
y = df["user_id"].astype(str)

# All model inputs must be numeric.
non_numeric_columns = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()

if non_numeric_columns:
    raise ValueError(
        "Non-numeric feature columns found: "
        f"{non_numeric_columns}. Convert or remove them first."
    )

# Reject missing or infinite feature values rather than silently
# changing the collected data.
X = X.astype(float)

if not np.isfinite(X.to_numpy()).all():
    raise ValueError(
        "Features contain NaN or infinite values. "
        "Clean the feature dataset before training."
    )

if y.nunique() < 2:
    raise ValueError("At least two different users are required.")

print("\nNumber of attempts:", len(df))
print("Number of users:", y.nunique())
print("Number of input features:", X.shape[1])


# ============================================================
# 4. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining attempts:", len(X_train))
print("Testing attempts:", len(X_test))


# ============================================================
# 5. DEFINE FOUR MACHINE LEARNING ALGORITHMS
# ============================================================

models = {
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE
        ))
    ])
}


# ============================================================
# 6. TRAIN, PREDICT AND EVALUATE EACH MODEL
# ============================================================

results = []
trained_models = {}
all_labels = sorted(y.unique())

for model_name, model in models.items():

    print("\n" + "=" * 65)
    print(f"TRAINING: {model_name}")
    print("=" * 65)

    # Train the model using only the training set.
    model.fit(X_train, y_train)

    # Predict users for unseen test attempts.
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics.
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Macro F1:  {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=all_labels,
            zero_division=0
        )
    )

    # Save confusion matrix.
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=all_labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=all_labels
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    matrix_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}_confusion_matrix.png"
    )

    fig.savefig(matrix_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save the trained model.
    model_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}_model.joblib"
    )

    joblib.dump(model, model_path)
    trained_models[model_name] = model

    # Record metrics for comparison.
    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Precision_Macro": precision,
        "Recall_Macro": recall,
        "F1_Macro": f1
    })

    print(f"\nSaved model: {model_path}")
    print(f"Saved confusion matrix: {matrix_path}")


# ============================================================
# 7. COMPARE ALL FOUR ALGORITHMS
# ============================================================

results_df = pd.DataFrame(results)

# Macro F1 is the primary selection metric; accuracy is also reported.
results_df = results_df.sort_values(
    by=["F1_Macro", "Accuracy"],
    ascending=False
).reset_index(drop=True)

comparison_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(comparison_path, index=False)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision_Macro": "{:.4f}".format,
            "Recall_Macro": "{:.4f}".format,
            "F1_Macro": "{:.4f}".format
        }
    )
)


# ============================================================
# 8. SELECT AND SAVE THE BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Algorithm"]
best_model = trained_models[best_model_name]

best_model_path = os.path.join(
    OUTPUT_DIR,
    "best_keystroke_model.joblib"
)

joblib.dump(best_model, best_model_path)

# Save feature names and labels for later use.
metadata = {
    "best_model": best_model_name,
    "feature_columns": X.columns.tolist(),
    "user_labels": all_labels,
    "selection_metric": "Macro F1-score",
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE
}

metadata_path = os.path.join(
    OUTPUT_DIR,
    "model_metadata.json"
)

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\n" + "=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)

print("Best model:", best_model_name)
print(
    "Best macro F1-score:",
    f"{results_df.iloc[0]['F1_Macro']:.4f}"
)
print(
    "Best model accuracy:",
    f"{results_df.iloc[0]['Accuracy'] * 100:.2f}%"
)
print("Comparison saved to:", comparison_path)
print("Best model saved to:", best_model_path)
print("Metadata saved to:", metadata_path)
print("\nAll four algorithms have been trained and evaluated.")
