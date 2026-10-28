
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("results/best_keystroke_model.joblib")
DATA_PATH = Path("keystroke_features.csv")

def main():
    if not MODEL_PATH.exists():
        print("Model file not found:", MODEL_PATH)
        return

    if not DATA_PATH.exists():
        print("Feature file not found:", DATA_PATH)
        return

    # Load dataset and trained model
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    # Separate labels and features
    y = df["user_id"].astype(str)
    X = df.drop(columns=["user_id", "attempt_id"], errors="ignore")

    # Use the same feature order as training, if available
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in X.columns]
        if missing:
            print("Missing features:", missing)
            return
        X = X[expected]

    # Recreate the original held-out split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Predict the test set
    predictions = model.predict(X_test)

    print("\n===== MODEL TEST RESULTS =====")
    print("Test attempts:", len(y_test))
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\n===== CONFUSION MATRIX =====")
    print("User order:", sorted(y.unique()))
    print(confusion_matrix(y_test, predictions, labels=sorted(y.unique())))

if __name__ == "__main__":
    main()
