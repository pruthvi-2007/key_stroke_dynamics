
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pynput import keyboard


PHRASE = "the quick brown fox jumps over the lazy dog"
MODEL_DIR = Path("authentication_models")
METADATA_FILE = MODEL_DIR / "authentication_metadata.json"


def calculate_statistics(values, prefix):
    values = np.asarray(values, dtype=float)

    return {
        f"{prefix}_mean": np.mean(values),
        f"{prefix}_std": np.std(values, ddof=1)
        if len(values) > 1 else 0.0,
        f"{prefix}_median": np.median(values),
        f"{prefix}_min": np.min(values),
        f"{prefix}_max": np.max(values),
        f"{prefix}_range": np.max(values) - np.min(values),
    }


def find_user_config(data, user):
    """Find the saved model filename and threshold for a user."""
    if isinstance(data, dict):
        if (
            "model_file" in data
            and "threshold" in data
            and user in str(data.get("model_file", ""))
        ):
            return data

        if user in data and isinstance(data[user], dict):
            candidate = data[user]
            if "model_file" in candidate and "threshold" in candidate:
                return candidate

        for value in data.values():
            result = find_user_config(value, user)
            if result is not None:
                return result

    elif isinstance(data, list):
        for value in data:
            result = find_user_config(value, user)
            if result is not None:
                return result

    return None


def capture_attempt():
    events = []
    press_times = {}
    last_release_time = None

    print("\nType this exact phrase, then press Enter:")
    print(PHRASE)
    print("\nStart typing below:")

    def on_press(key):
        now = __import__("time").perf_counter()

        if key == keyboard.Key.enter:
            return False

        try:
            char = key.char
        except AttributeError:
            char = None

        key_name = char if char is not None else str(key)

        flight = (
            (now - last_release_time) * 1000
            if last_release_time is not None else np.nan
        )

        events.append({
            "key": key_name,
            "press_time": now,
            "release_time": np.nan,
            "dwell_time_ms": np.nan,
            "flight_time_ms": flight,
        })

        press_times[key] = (now, len(events) - 1)

    def on_release(key):
        nonlocal last_release_time

        now = __import__("time").perf_counter()

        if key == keyboard.Key.enter:
            return False

        if key in press_times:
            start, index = press_times.pop(key)
            events[index]["release_time"] = now
            events[index]["dwell_time_ms"] = (now - start) * 1000
            last_release_time = now

    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    ) as listener:
        input()
        listener.stop()

    return events


def extract_features(events):
    if len(events) != len(PHRASE):
        raise ValueError(
            f"Expected {len(PHRASE)} key events, "
            f"but captured {len(events)}. Try again."
        )

    typed = "".join(
        " " if e["key"] == "Key.space" else e["key"].lower()
        for e in events
    )

    if typed != PHRASE:
        raise ValueError(
            "The phrase did not match exactly. Please try again."
        )

    if any(
        not np.isfinite(e["dwell_time_ms"])
        or e["dwell_time_ms"] <= 0
        for e in events
    ):
        raise ValueError("Some key timings are invalid. Please try again.")

    if any(
        not np.isfinite(e["flight_time_ms"])
        or e["flight_time_ms"] < 0
        for e in events[1:]
    ):
        raise ValueError("Some flight times are invalid. Please try again.")

    dwell = np.array(
        [e["dwell_time_ms"] for e in events], dtype=float
    )
    flight = np.array(
        [e["flight_time_ms"] for e in events[1:]], dtype=float
    )

    features = {}
    features.update(calculate_statistics(dwell, "dwell"))
    features.update(calculate_statistics(flight, "flight"))

    for position, value in enumerate(dwell, start=1):
        features[f"dwell_pos_{position:02d}"] = value

    for position, value in enumerate(flight, start=2):
        features[f"flight_pos_{position:02d}"] = value

    features["attempt_duration_ms"] = (
        events[-1]["release_time"] - events[0]["press_time"]
    ) * 1000
    features["total_dwell_ms"] = dwell.sum()
    features["total_flight_ms"] = flight.sum()

    return features


def main():
    if not METADATA_FILE.exists():
        print("Metadata file not found:", METADATA_FILE)
        return

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    users = metadata["users"]

    print("KEYSTROKE AUTHENTICATION")
    print("Registered users:", ", ".join(users))

    user = input("Which identity are you claiming? ").strip()

    if user not in users:
        print("Unknown user. Choose one of the registered IDs.")
        return

    config = find_user_config(metadata, user)

    if config is None:
        print("Could not find this user's model configuration in metadata.")
        print("Please check authentication_metadata.json.")
        return

    model_path = MODEL_DIR / config["model_file"]

    if not model_path.exists():
        print("Model file not found:", model_path)
        return

    model = joblib.load(model_path)

    try:
        events = capture_attempt()
        features = extract_features(events)
    except ValueError as error:
        print("\nAttempt could not be evaluated:", error)
        return

    feature_columns = metadata["feature_columns"]
    X = pd.DataFrame([features]).reindex(columns=feature_columns)

    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        print("Missing required features:", missing)
        print("Check that training and authentication use the same features.")
        return

    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)

    if 1 not in classes:
        print("The saved model does not contain the genuine-user class.")
        return

    genuine_probability = float(probabilities[classes.index(1)])
    threshold = float(config["threshold"])

    print("\n--- Authentication Result ---")
    print("Claimed identity:", user)
    print(f"Genuine-user probability: {genuine_probability:.4f}")
    print(f"Required threshold:        {threshold:.4f}")

    if genuine_probability >= threshold:
        print("RESULT: ACCEPTED")
    else:
        print("RESULT: REJECTED")


if __name__ == "__main__":
    main()
