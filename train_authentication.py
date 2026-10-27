
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# ------------------------------------------
# 1. Load the feature dataset
# ------------------------------------------

DATA_FILE = "keystroke_features.csv"
OUTPUT_DIR = "authentication_models"

df = pd.read_csv(DATA_FILE)

if "user_id" not in df.columns:
    raise ValueError("Dataset must contain a 'user_id' column.")

# Remove rows without a valid user label
df = df.dropna(subset=["user_id"]).copy()
df["user_id"] = df["user_id"].astype(str)

# Keep metadata out of the model features
metadata_columns = ["user_id", "attempt_id"]
feature_columns = [
    col for col in df.columns
    if col not in metadata_columns
]

# Keep only numeric feature columns
X = df[feature_columns].select_dtypes(include=np.number)
feature_columns = X.columns.tolist()

if X.shape[1] == 0:
    raise ValueError("No numeric feature columns were found.")

y = df["user_id"]

users = sorted(y.unique())

if len(users) != 4:
    raise ValueError(
        f"Expected 4 users, but found {len(users)}: {users}"
    )

print("Dataset shape:", df.shape)
print("Features used:", len(feature_columns))
print("\nAttempts per user:")
print(y.value_counts().sort_index())

# ------------------------------------------
# 2. Split BEFORE creating binary labels
#    60% train, 20% validation, 20% test
# ------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("\nSplit sizes:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# ------------------------------------------
# 3. Train a separate verifier for each user
# ------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata = {
    "feature_columns": feature_columns,
    "users": users,
    "split": {
        "train": 0.60,
        "validation": 0.20,
        "test": 0.20
    },
    "random_state": 42,
    "verifiers": {}
}

for user in users:
    print("\n" + "=" * 55)
    print(f"AUTHENTICATION MODEL: {user}")
    print("=" * 55)

    # Genuine = 1; other users = impostors = 0
    y_train_bin = (y_train == user).astype(int)
    y_val_bin = (y_val == user).astype(int)
    y_test_bin = (y_test == user).astype(int)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train_bin)

    # --------------------------------------
    # 4. Select threshold on validation data
    # --------------------------------------

    val_scores = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.50
    best_balanced_accuracy = -1

    # Search candidate probability thresholds
    for threshold in np.linspace(0.05, 0.95, 181):
        val_predictions = (
            val_scores >= threshold
        ).astype(int)

        score = balanced_accuracy_score(
            y_val_bin, val_predictions
        )

        if score > best_balanced_accuracy:
            best_balanced_accuracy = score
            best_threshold = float(threshold)

    # --------------------------------------
    # 5. Evaluate on untouched test data
    # --------------------------------------

    test_scores = model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_scores >= best_threshold
    ).astype(int)

    genuine_mask = (y_test_bin == 1)
    impostor_mask = (y_test_bin == 0)

    genuine_count = int(genuine_mask.sum())
    impostor_count = int(impostor_mask.sum())

    false_rejects = int(
        ((test_predictions == 0) & genuine_mask).sum()
    )

    false_accepts = int(
        ((test_predictions == 1) & impostor_mask).sum()
    )

    frr = (
        false_rejects / genuine_count
        if genuine_count else float("nan")
    )

    far = (
        false_accepts / impostor_count
        if impostor_count else float("nan")
    )

    accuracy = accuracy_score(
        y_test_bin, test_predictions
    )

    print(f"Selected threshold: {best_threshold:.3f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"False Acceptance Rate (FAR): {far:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")
    print(f"False accepts: {false_accepts}/{impostor_count}")
    print(f"False rejects: {false_rejects}/{genuine_count}")

    # Save this user's verifier
    model_path = os.path.join(
        OUTPUT_DIR, f"{user}_verifier.joblib"
    )
    joblib.dump(model, model_path)

    metadata["verifiers"][user] = {
        "model_file": f"{user}_verifier.joblib",
        "threshold": best_threshold,
        "validation_balanced_accuracy": float(
            best_balanced_accuracy
        ),
        "test_accuracy": float(accuracy),
        "test_far": float(far),
        "test_frr": float(frr),
        "test_false_accepts": false_accepts,
        "test_impostor_attempts": impostor_count,
        "test_false_rejects": false_rejects,
        "test_genuine_attempts": genuine_count
    }

# ------------------------------------------
# 6. Save thresholds and feature information
# ------------------------------------------

metadata_path = os.path.join(
    OUTPUT_DIR, "authentication_metadata.json"
)

with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=4)

print("\nAll four authentication models have been saved.")
print("Output folder:", OUTPUT_DIR)
print("Metadata:", metadata_path)
