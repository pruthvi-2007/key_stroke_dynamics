import pandas as pd
import math

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

INPUT_FILE = "keystroke_attempts_fixed.csv"

VALID_OUTPUT_FILE = "keystroke_attempts_valid.csv"
REJECTED_OUTPUT_FILE = "keystroke_attempts_rejected.csv"

EXPECTED_PASSWORD = "the quick brown fox jumps over the lazy dog"
EXPECTED_WORDS = EXPECTED_PASSWORD.split()

EXPECTED_USERS = {"User1", "User2", "User3", "User4"}

REQUIRED_COLUMNS = [
    "user_id",
    "attempt_id",
    "key",
    "press_time",
    "release_time",
    "dwell_time_ms",
    "flight_time_ms"
]

TIMING_TOLERANCE_MS = 0.1


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    raise SystemExit(
        f"ERROR: Cannot find '{INPUT_FILE}'. "
        "Place this script in the same folder as your CSV."
    )

print("=" * 60)
print("KEYSTROKE DATASET CLEANING")
print("=" * 60)

print(f"\nOriginal event rows: {len(df)}")

# Check required columns
missing_columns = [
    column for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise SystemExit(
        f"ERROR: Missing required columns: {missing_columns}"
    )

# Keep only the required columns
df = df[REQUIRED_COLUMNS].copy()

# Normalize identifiers and key names
df["user_id"] = df["user_id"].astype("string").str.strip()
df["attempt_id"] = df["attempt_id"].astype("string").str.strip()
df["key"] = df["key"].astype("string").str.strip()

# Convert timing fields to numbers.
# Invalid or empty values become NaN and are checked later.
TIMING_COLUMNS = [
    "press_time",
    "release_time",
    "dwell_time_ms",
    "flight_time_ms"
]

for column in TIMING_COLUMNS:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Preserve the original event order for tie-breaking.
df["_original_order"] = range(len(df))


# ==========================================================
# 3. HELPER FUNCTIONS
# ==========================================================

def is_missing(value):
    """Return True if a value is missing."""
    return pd.isna(value)


def key_to_character(key):
    """
    Convert a recorded key name to a character.

    Examples:
        't'         -> 't'
        'Key.space' -> ' '
        'Key.shift' -> None
        'Key.up'    -> None
    """

    if is_missing(key):
        return None

    key = str(key)

    if key == "Key.space":
        return " "

    # Only accept a single printable character.
    if len(key) == 1 and key.isprintable():
        return key.lower()

    return None


def reconstruct_password(keys):
    """
    Reconstruct the typed password from recorded key events.

    Returns:
        reconstructed text,
        list of unsupported or missing key events
    """

    characters = []
    invalid_keys = []

    for position, key in enumerate(keys, start=1):
        character = key_to_character(key)

        if character is None:
            invalid_keys.append(
                f"Unsupported or missing key at event {position}: {key}"
            )
        else:
            characters.append(character)

    return "".join(characters), invalid_keys


def validate_timing(group):
    """Check whether timing values are valid and consistent."""

    reasons = []

    press = group["press_time"]
    release = group["release_time"]
    dwell = group["dwell_time_ms"]
    flight = group["flight_time_ms"]

    # Required timestamps and dwell values must exist.
    for column in ["press_time", "release_time", "dwell_time_ms"]:
        if group[column].isna().any():
            reasons.append(f"Missing or invalid {column}")

    # Validate each event's timing.
    for row_index in group.index:
        p = press.loc[row_index]
        r = release.loc[row_index]
        d = dwell.loc[row_index]

        if pd.isna(p) or pd.isna(r) or pd.isna(d):
            continue

        if r < p:
            reasons.append(
                f"Release occurs before press at event row {row_index}"
            )
            continue

        if d <= 0:
            reasons.append(
                f"Non-positive dwell time at event row {row_index}"
            )

        calculated_dwell = (r - p) * 1000

        if not math.isclose(
            d,
            calculated_dwell,
            rel_tol=0,
            abs_tol=TIMING_TOLERANCE_MS
        ):
            reasons.append(
                f"Dwell time inconsistent with timestamps "
                f"at event row {row_index}"
            )

    # Ensure events are chronologically ordered.
    # Equal press timestamps are allowed, but backwards timestamps are not.
    valid_press = press.dropna()

    if not valid_press.is_monotonic_increasing:
        reasons.append("Press timestamps are not chronological")

    valid_release = release.dropna()

    if not valid_release.is_monotonic_increasing:
        reasons.append("Release timestamps are not chronological")

    # Flight time is normally undefined for the first key of an attempt.
    if len(group) > 0:
        flight_values = flight.tolist()

        # First key's flight time can be blank.
        for position, value in enumerate(flight_values):
            if position == 0:
                continue

            if pd.isna(value):
                reasons.append(
                    f"Missing or invalid flight time at event {position + 1}"
                )
                continue

            if value < 0:
                reasons.append(
                    f"Negative flight time at event {position + 1}"
                )

    return reasons


# ==========================================================
# 4. VALIDATE EACH ATTEMPT
# ==========================================================

valid_groups = []
rejected_attempts = []

# Rows are grouped into individual attempts.
grouped = df.groupby(
    ["user_id", "attempt_id"],
    sort=False,
    dropna=False
)

total_attempts = len(grouped)

print(f"Unique attempts found: {total_attempts}")
print(f"Expected password: {EXPECTED_PASSWORD}\n")

for (user_id, attempt_id), original_group in grouped:

    reasons = []

    # Sort events by press time, preserving original order for ties.
    group = original_group.sort_values(
        ["press_time", "_original_order"],
        na_position="last",
        kind="stable"
    ).copy()

    # ------------------------------------------------------
    # A. Check user and attempt identifiers
    # ------------------------------------------------------

    if is_missing(user_id) or str(user_id).strip() == "":
        reasons.append("Missing user_id")
    elif user_id not in EXPECTED_USERS:
        reasons.append(f"Unknown user_id: {user_id}")

    if is_missing(attempt_id) or str(attempt_id).strip() == "":
        reasons.append("Missing attempt_id")

    # ------------------------------------------------------
    # B. Check missing keys
    # ------------------------------------------------------

    if group["key"].isna().any():
        reasons.append("One or more key events are missing")

    keys = group["key"].tolist()

    # ------------------------------------------------------
    # C. Check the number of key events
    # ------------------------------------------------------

    expected_key_count = len(EXPECTED_PASSWORD)

    if len(group) != expected_key_count:
        reasons.append(
            f"Incorrect key-event count: expected "
            f"{expected_key_count}, found {len(group)}"
        )

    # ------------------------------------------------------
    # D. Reconstruct the password and validate spelling
    # ------------------------------------------------------

    reconstructed, invalid_keys = reconstruct_password(keys)

    reasons.extend(invalid_keys)

    actual_words = reconstructed.split()

    # Explicitly validate each word.
    if len(actual_words) != len(EXPECTED_WORDS):
        reasons.append(
            f"Incorrect word count: expected {len(EXPECTED_WORDS)}, "
            f"found {len(actual_words)}"
        )

    for word_index, expected_word in enumerate(EXPECTED_WORDS):

        if word_index >= len(actual_words):
            reasons.append(
                f"Word {word_index + 1} missing: "
                f"expected '{expected_word}'"
            )
            continue

        actual_word = actual_words[word_index]

        if actual_word != expected_word:
            reasons.append(
                f"Word {word_index + 1} incorrect: "
                f"expected '{expected_word}', found '{actual_word}'"
            )

    # Check the entire phrase, including spaces and word order.
    if reconstructed != EXPECTED_PASSWORD:
        reasons.append(
            f"Full password mismatch: typed '{reconstructed}'"
        )

    # ------------------------------------------------------
    # E. Check timing values
    # ------------------------------------------------------

    timing_reasons = validate_timing(group)
    reasons.extend(timing_reasons)

    # Remove duplicate reasons while preserving their order.
    reasons = list(dict.fromkeys(reasons))

    # ------------------------------------------------------
    # F. Keep valid attempts or record rejection reasons
    # ------------------------------------------------------

    if reasons:
        rejected_attempts.append({
            "user_id": user_id,
            "attempt_id": attempt_id,
            "event_count": len(group),
            "typed_password": reconstructed,
            "rejection_reason": " | ".join(reasons)
        })

    else:
        # First event has no previous key, so its flight time
        # is not applicable. Keep it blank.
        group.iloc[0, group.columns.get_loc("flight_time_ms")] = float("nan")

        valid_groups.append(group)


# ==========================================================
# 5. SAVE VALID DATASET
# ==========================================================

if valid_groups:
    valid_df = pd.concat(valid_groups, ignore_index=True)

    # Remove the internal sorting column.
    valid_df = valid_df.drop(columns=["_original_order"], errors="ignore")

    valid_df.to_csv(VALID_OUTPUT_FILE, index=False)

else:
    valid_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    valid_df.to_csv(VALID_OUTPUT_FILE, index=False)


# ==========================================================
# 6. SAVE REJECTED ATTEMPTS REPORT
# ==========================================================

rejected_df = pd.DataFrame(
    rejected_attempts,
    columns=[
        "user_id",
        "attempt_id",
        "event_count",
        "typed_password",
        "rejection_reason"
    ]
)

rejected_df.to_csv(REJECTED_OUTPUT_FILE, index=False)


# ==========================================================
# 7. FINAL SUMMARY
# ==========================================================

valid_attempt_count = len(valid_groups)
rejected_attempt_count = len(rejected_attempts)

print("=" * 60)
print("CLEANING COMPLETED")
print("=" * 60)

print(f"Total attempts checked : {total_attempts}")
print(f"Valid attempts         : {valid_attempt_count}")
print(f"Rejected attempts      : {rejected_attempt_count}")

print(
    f"Valid event rows       : {len(valid_df)}"
)

print("\nFiles created:")
print(f"1. {VALID_OUTPUT_FILE}")
print(f"2. {REJECTED_OUTPUT_FILE}")

if rejected_attempt_count > 0:
    print("\nRejection reasons summary:")

    for reason in rejected_df["rejection_reason"]:
        print(f"- {reason}")

print("\nDone!")