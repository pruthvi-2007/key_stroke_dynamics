
import os
import csv
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ------------------------------------------
# 1. File paths and selected users
# ------------------------------------------

DATA_FILE = "keystroke_features.csv"
MODEL_DIR = "authentication_models"

METADATA_FILE = os.path.join(
    MODEL_DIR, "authentication_metadata.json"
)

OUTPUT_FILE = "evaluation_results.csv"

# Evaluate all four registered users
TARGET_USERS = ["User1", "User2", "User3", "User4"]


def main():

    # ------------------------------------------
    # 2. Load dataset and saved metadata
    # ------------------------------------------

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            "Authentication metadata not found. "
            "Run train_authentication.py first."
        )

    df = pd.read_csv(DATA_FILE)

    with open(METADATA_FILE, "r") as file:
        metadata = json.load(file)

    df = df.dropna(subset=["user_id"]).copy()
    df["user_id"] = df["user_id"].astype(str)

    feature_columns = metadata["feature_columns"]
    users = metadata["users"]
    verifiers = metadata["verifiers"]

    # Check that all four users exist
    missing_users = [
        user for user in TARGET_USERS
        if user not in users
    ]

    if missing_users:
        raise ValueError(
            f"These user IDs were not found: {missing_users}. "
            f"Available IDs: {users}. "
            "Update TARGET_USERS to match your actual IDs."
        )

    # Use the same feature columns and order as training
    X = df[feature_columns].select_dtypes(include=np.number)
    y = df["user_id"]

    if X.shape[1] != len(feature_columns):
        raise ValueError(
            "Feature columns do not match the training metadata."
        )

    # ------------------------------------------
    # 3. Recreate the original test split
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

    results = []

    print("\nKEYSTROKE AUTHENTICATION EVALUATION")
    print("=" * 55)
    print("Evaluating all four registered users")

    # ------------------------------------------
    # 4. Evaluate each user's model
    # ------------------------------------------

    for user in TARGET_USERS:

        info = verifiers[user]

        model_path = os.path.join(
            MODEL_DIR,
            info["model_file"]
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Saved model not found: {model_path}"
            )

        model = joblib.load(model_path)
        threshold = float(info["threshold"])

        # Actual labels:
        # True = attempt belongs to this user
        # False = attempt belongs to another user
        actual_genuine = (y_test == user).to_numpy()

        # Predict genuine-user probabilities
        scores = model.predict_proba(X_test)[:, 1]

        # Apply the threshold selected during training
        predicted_genuine = scores >= threshold

        # --------------------------------------
        # 5. Calculate confusion-matrix values
        # --------------------------------------

        TP = int(np.sum(
            predicted_genuine & actual_genuine
        ))

        FN = int(np.sum(
            ~predicted_genuine & actual_genuine
        ))

        FP = int(np.sum(
            predicted_genuine & ~actual_genuine
        ))

        TN = int(np.sum(
            ~predicted_genuine & ~actual_genuine
        ))

        genuine_total = TP + FN
        impostor_total = FP + TN
        total = genuine_total + impostor_total

        # --------------------------------------
        # 6. Calculate FAR, FRR and accuracy
        # --------------------------------------

        FAR = (
            FP / impostor_total * 100
            if impostor_total else float("nan")
        )

        FRR = (
            FN / genuine_total * 100
            if genuine_total else float("nan")
        )

        accuracy = (
            (TP + TN) / total * 100
            if total else float("nan")
        )

        # --------------------------------------
        # 7. Display results
        # --------------------------------------

        print("\n" + "-" * 45)
        print(f"USER: {user}")
        print("-" * 45)

        print(f"Threshold: {threshold:.4f}")
        print(f"Genuine attempts: {genuine_total}")
        print(f"Impostor attempts: {impostor_total}")
        print(f"True Accepts (TP): {TP}")
        print(f"True Rejects (TN): {TN}")
        print(f"False Accepts (FP): {FP}")
        print(f"False Rejects (FN): {FN}")
        print(f"FAR: {FAR:.2f}%")
        print(f"FRR: {FRR:.2f}%")
        print(f"Accuracy: {accuracy:.2f}%")

        # Save this user's results
        results.append({
            "User": user,
            "Threshold": threshold,
            "Genuine Attempts": genuine_total,
            "Impostor Attempts": impostor_total,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "FAR (%)": round(FAR, 2),
            "FRR (%)": round(FRR, 2),
            "Accuracy (%)": round(accuracy, 2)
        })

    # ------------------------------------------
    # 8. Save evaluation report
    # ------------------------------------------

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 55)
    print("EVALUATION COMPLETED")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

