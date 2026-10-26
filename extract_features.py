
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "keystroke_attempts_valid.csv"
OUTPUT_FILE = "keystroke_features.csv"
REJECTED_FILE = "feature_extraction_rejected.csv"

EXPECTED_PASSWORD = "the quick brown fox jumps over the lazy dog"
EXPECTED_USERS = {"User1", "User2", "User3", "User4"}

REQUIRED_COLUMNS = [
    "user_id",
    "attempt_id",
    "key",
    "press_time",
    "release_time",
    "dwell_time_ms",
    "flight_time_ms",
]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def key_to_character(key):
    """Convert a recorded key name into a password character."""
    if pd.isna(key):
        return None

    key = str(key)

    if key == "Key.space":
        return " "

    if len(key) == 1 and key.isprintable():
        return key.lower()

    return None


def calculate_statistics(values, prefix):
    """Calculate statistical features from a timing sequence."""
    values = np.asarray(values, dtype=float)

    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"Invalid values in {prefix}")

    return {
        f"{prefix}_mean": np.mean(values),
        f"{prefix}_std": np.std(values, ddof=1)
        if len(values) > 1 else 0.0,
        f"{prefix}_median": np.median(values),
        f"{prefix}_min": np.min(values),
        f"{prefix}_max": np.max(values),
        f"{prefix}_range": np.max(values) - np.min(values),
    }


# ============================================================
# 3. LOAD AND CHECK THE DATASET
# ============================================================

if not Path(INPUT_FILE).exists():
    raise FileNotFoundError(
        f"Cannot find {INPUT_FILE}. "
        "Place this script in the same folder as your CSV."
    )

df = pd.read_csv(INPUT_FILE)

missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

df = df[REQUIRED_COLUMNS].copy()

# Convert timing columns to numeric values.
timing_columns = [
    "press_time",
    "release_time",
    "dwell_time_ms",
    "flight_time_ms",
]

for column in timing_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("=" * 60)
print("KEYSTROKE FEATURE EXTRACTION")
print("=" * 60)
print(f"Input event rows: {len(df)}")


# ============================================================
# 4. PROCESS EACH COMPLETE TYPING ATTEMPT
# ============================================================

feature_rows = []
rejected_rows = []

groups = df.groupby(
    ["user_id", "attempt_id"],
    sort=False,
    dropna=False,
)

expected_characters = len(EXPECTED_PASSWORD)

for (user_id, attempt_id), original_group in groups:

    reasons = []

    # Sort key events chronologically.
    group = original_group.sort_values(
        "press_time",
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # A. Validate identity and event count
    # --------------------------------------------------------

    if pd.isna(user_id) or user_id not in EXPECTED_USERS:
        reasons.append("Missing or unknown user_id")

    if pd.isna(attempt_id):
        reasons.append("Missing attempt_id")

    if len(group) != expected_characters:
        reasons.append(
            f"Expected {expected_characters} events, "
            f"found {len(group)}"
        )

    # --------------------------------------------------------
    # B. Validate the exact password
    # --------------------------------------------------------

    keys = group["key"].tolist()
    converted_keys = [key_to_character(key) for key in keys]

    if any(character is None for character in converted_keys):
        reasons.append("Missing or unsupported key event")
    else:
        typed_password = "".join(converted_keys)

        if typed_password != EXPECTED_PASSWORD:
            reasons.append("Password does not match expected phrase")

    # --------------------------------------------------------
    # C. Validate timing data
    # --------------------------------------------------------

    if group[timing_columns[:3]].isna().any().any():
        reasons.append("Missing or invalid press/release/dwell value")

    if group["dwell_time_ms"].le(0).any():
        reasons.append("Non-positive dwell time")

    if (
        group["release_time"] < group["press_time"]
    ).any():
        reasons.append("Release occurs before key press")

    if not group["press_time"].is_monotonic_increasing:
        reasons.append("Press timestamps are not chronological")

    if not group["release_time"].is_monotonic_increasing:
        reasons.append("Release timestamps are not chronological")

    # Dwell time must agree with the timestamps.
    expected_dwell = (
        group["release_time"] - group["press_time"]
    ) * 1000

    dwell_difference = (
        group["dwell_time_ms"] - expected_dwell
    ).abs()

    if dwell_difference.dropna().gt(0.1).any():
        reasons.append("Dwell time inconsistent with timestamps")

    # The first key has no previous key in this attempt.
    # All subsequent flight times must be present and non-negative.
    if len(group) > 1:
        later_flights = group["flight_time_ms"].iloc[1:]

        if later_flights.isna().any():
            reasons.append("Missing flight time after first key")

        elif later_flights.lt(0).any():
            reasons.append("Negative flight time")

    # Reject invalid attempts instead of silently generating features.
    if reasons:
        rejected_rows.append({
            "user_id": user_id,
            "attempt_id": attempt_id,
            "event_count": len(group),
            "rejection_reason": " | ".join(dict.fromkeys(reasons)),
        })
        continue

    # --------------------------------------------------------
    # D. Extract features for valid attempts
    # --------------------------------------------------------

    dwell = group["dwell_time_ms"].to_numpy(dtype=float)

    # Flight time for the first key is undefined.
    # Use the remaining 42 flight times.
    flight = group["flight_time_ms"].iloc[1:].to_numpy(dtype=float)

    features = {
        "user_id": user_id,
        "attempt_id": attempt_id,
    }

    # Overall statistical features.
    features.update(calculate_statistics(dwell, "dwell"))
    features.update(calculate_statistics(flight, "flight"))

    # Position-specific dwell features.
    # Example: dwell_pos_01 is the first key's dwell time.
    for position, value in enumerate(dwell, start=1):
        features[f"dwell_pos_{position:02d}"] = value

    # Position-specific flight features.
    # flight_pos_02 is the flight time associated with key 2.
    for position, value in enumerate(flight, start=2):
        features[f"flight_pos_{position:02d}"] = value

    # Additional per-attempt timing features.
    features["attempt_duration_ms"] = (
        group["release_time"].iloc[-1]
        - group["press_time"].iloc[0]
    ) * 1000

    features["total_dwell_ms"] = dwell.sum()
    features["total_flight_ms"] = flight.sum()

    feature_rows.append(features)


# ============================================================
# 5. SAVE THE FEATURE DATASET AND REJECTION REPORT
# ============================================================

features_df = pd.DataFrame(feature_rows)

rejected_df = pd.DataFrame(
    rejected_rows,
    columns=[
        "user_id",
        "attempt_id",
        "event_count",
        "rejection_reason",
    ],
)

features_df.to_csv(OUTPUT_FILE, index=False)
rejected_df.to_csv(REJECTED_FILE, index=False)


# ============================================================
# 6. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 60)

print(f"Attempts processed: {len(groups)}")
print(f"Valid feature rows: {len(features_df)}")
print(f"Rejected attempts: {len(rejected_df)}")

if not features_df.empty:
    print(f"Features per attempt: {len(features_df.columns) - 2}")

    print("\nValid attempts per user:")
    print(features_df["user_id"].value_counts().sort_index())

print("\nFiles created:")
print(f"1. {OUTPUT_FILE}")
print(f"2. {REJECTED_FILE}")
print("\nDone!")
