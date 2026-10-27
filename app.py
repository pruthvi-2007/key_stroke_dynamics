

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PHRASE = "the quick brown fox jumps over the lazy dog"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "authentication_models"
METADATA_FILE = MODEL_DIR / "authentication_metadata.json"
COMPONENT_DIR = BASE_DIR / "typing_component"

keystroke_capture = components.declare_component(
    "keystroke_capture",
    path=str(COMPONENT_DIR),
)


# --------------------------------------------------
# Metadata helpers
# --------------------------------------------------

def find_user_config(data, user):
    """Find a user's model filename and threshold in metadata."""
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


def resolve_model_path(model_file):
    """Resolve a model filename without duplicating its folder path."""
    configured_path = Path(str(model_file))

    if configured_path.is_absolute():
        return configured_path

    candidates = [
        MODEL_DIR / configured_path,
        BASE_DIR / configured_path,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return candidates[0]


@st.cache_resource
def load_authentication_resources():
    if not METADATA_FILE.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    users = metadata.get("users")
    feature_columns = metadata.get("feature_columns")

    if not isinstance(users, list) or not users:
        raise ValueError(
            "Metadata must contain a non-empty 'users' list."
        )

    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(
            "Metadata must contain 'feature_columns'."
        )

    models = {}
    configs = {}

    for user in users:
        config = find_user_config(metadata, user)

        if config is None:
            raise ValueError(
                f"Could not find model configuration for {user}."
            )

        model_path = resolve_model_path(config["model_file"])

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model for {user} not found: {model_path}"
            )

        models[user] = joblib.load(model_path)
        configs[user] = config

    return metadata, users, feature_columns, models, configs


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def calculate_statistics(values, prefix):
    values = np.asarray(values, dtype=float)

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": (
            float(np.std(values, ddof=1))
            if len(values) > 1 else 0.0
        ),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
    }


def extract_features(events, feature_columns):
    """
    Use browser-captured timings directly.

    Events remain in keydown/phrase order. Flight times are not
    reconstructed from adjacent press/release timestamps.
    """

    if not isinstance(events, list) or len(events) != len(PHRASE):
        raise ValueError(
            f"Expected {len(PHRASE)} keyboard events; "
            f"received {len(events) if isinstance(events, list) else 0}."
        )

    def normalize_key(event):
        key = str(event.get("key", ""))

        if key in (" ", "Space", "Spacebar", "Key.space"):
            return " "

        return key.lower()

    typed = "".join(normalize_key(event) for event in events)

    if typed != PHRASE:
        raise ValueError(
            "Captured characters do not match the required phrase. "
            "Please retry."
        )

    try:
        presses = np.asarray(
            [float(event["press_ms"]) for event in events],
            dtype=float,
        )
        releases = np.asarray(
            [float(event["release_ms"]) for event in events],
            dtype=float,
        )
        dwell = releases - presses

        # Read the exact flight times recorded at keydown.
        raw_flights = [
            event.get("flight_ms", event.get("flight_time_ms"))
            for event in events[1:]
        ]
        flight = np.asarray(
            [
                np.nan if value is None else float(value)
                for value in raw_flights
            ],
            dtype=float,
        )

    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "A keyboard timing value is missing or invalid. Retry."
        ) from exc

    if not np.isfinite(presses).all():
        raise ValueError("Press timestamps are invalid. Please retry.")

    if not np.isfinite(releases).all():
        raise ValueError("Release timestamps are invalid. Please retry.")

    if np.any(releases < presses):
        raise ValueError(
            "A key was released before its recorded press. Retry."
        )

    if np.any(dwell <= 0) or not np.isfinite(dwell).all():
        raise ValueError("Dwell-time measurements are invalid. Retry.")

    # Browser events are stored in keydown order, so this should hold.
    if np.any(np.diff(presses) < 0):
        raise ValueError(
            "Keyboard events are not in phrase order. Please retry."
        )

    # The first character has no previous key release. Its missing
    # flight time is expected, so only the remaining 42 are checked.
    if len(flight) != len(PHRASE) - 1:
        raise ValueError("Unexpected number of flight-time measurements.")

    if not np.isfinite(flight).all() or np.any(flight < 0):
        invalid_positions = [
            i + 2
            for i, value in enumerate(flight)
            if not np.isfinite(value) or value < 0
        ]
        raise ValueError(
            "Flight times are missing or invalid at character "
            f"positions {invalid_positions}. Please retry."
        )

    features = {}
    features.update(calculate_statistics(dwell, "dwell"))
    features.update(calculate_statistics(flight, "flight"))

    for position, value in enumerate(dwell, start=1):
        features[f"dwell_pos_{position:02d}"] = float(value)

    for position, value in enumerate(flight, start=2):
        features[f"flight_pos_{position:02d}"] = float(value)

    # Keep the same duration definition as the working Python script.
    features["attempt_duration_ms"] = float(
        releases[-1] - presses[0]
    )
    features["total_dwell_ms"] = float(np.sum(dwell))
    features["total_flight_ms"] = float(np.sum(flight))

    # Match training column order exactly.
    X = pd.DataFrame([features]).reindex(columns=feature_columns)

    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"Extracted features do not match training columns: {missing}"
        )

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Extracted features contain invalid numbers.")

    return X


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------

st.set_page_config(
    page_title="Keystroke Authentication",
    page_icon="🔐",
    layout="centered",
)

st.title("Keystroke Authentication")
st.write(
    "Verify your identity using the timing pattern of your typing."
)

try:
    metadata, users, feature_columns, models, configs = (
        load_authentication_resources()
    )
except Exception as exc:
    st.error(f"Could not load authentication resources: {exc}")
    st.stop()

if "reset_token" not in st.session_state:
    st.session_state.reset_token = 0

if "capture_result" not in st.session_state:
    st.session_state.capture_result = None

user = st.selectbox("Select the identity you are claiming", users)

st.markdown("**Type this exact phrase:**")
st.code(PHRASE)

st.caption(
    "Type naturally, using your physical keyboard. "
    "If you make a mistake, restart the attempt."
)

result = keystroke_capture(
    phrase=PHRASE,
    reset_token=st.session_state.reset_token,
    key=f"capture_{st.session_state.reset_token}",
    default=None,
)

if isinstance(result, dict) and result.get("completed"):
    st.session_state.capture_result = result

capture = st.session_state.capture_result

if isinstance(capture, dict) and capture.get("completed"):
    st.success("Phrase captured. You can now authenticate.")

    if st.button("Authenticate", type="primary", use_container_width=True):
        try:
            events = capture.get("keystrokes", [])
            X = extract_features(events, feature_columns)

            model = models[user]
            config = configs[user]

            probabilities = model.predict_proba(X)[0]
            classes = list(model.classes_)

            if 1 not in classes:
                raise ValueError(
                    "The model does not contain class 1 "
                    "for genuine-user attempts."
                )

            genuine_probability = float(
                probabilities[classes.index(1)]
            )
            threshold = float(config["threshold"])

            st.session_state.auth_result = {
                "user": user,
                "probability": genuine_probability,
                "threshold": threshold,
                "accepted": genuine_probability >= threshold,
            }

        except Exception as exc:
            st.session_state.auth_result = None
            st.error(f"Could not authenticate this attempt: {exc}")

    auth_result = st.session_state.get("auth_result")

    if auth_result:
        st.write(f"**Claimed identity:** {auth_result['user']}")
        st.write(
            "**Genuine-user probability:** "
            f"{auth_result['probability']:.4f}"
        )
        st.write(
            f"**Required threshold:** {auth_result['threshold']:.4f}"
        )

        if auth_result["accepted"]:
            st.success("Authentication accepted.")
        else:
            st.error("Authentication rejected.")

    if st.button("Retry", use_container_width=True):
        st.session_state.reset_token += 1
        st.session_state.capture_result = None
        st.session_state.auth_result = None
        st.rerun()

