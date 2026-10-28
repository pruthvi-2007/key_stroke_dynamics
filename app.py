
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# 1. CONFIGURATION
# =========================================================

PHRASE = "the quick brown fox jumps over the lazy dog"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "authentication_models"
METADATA_FILE = MODEL_DIR / "authentication_metadata.json"
COMPONENT_DIR = BASE_DIR / "typing_component"

keystroke_capture = components.declare_component(
    "keystroke_capture",
    path=str(COMPONENT_DIR),
)

st.set_page_config(
    page_title="KeyShield | Behavioral Authentication",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. PREMIUM CYBERSECURITY THEME
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #080d18;
        --panel: #101a2b;
        --border: #25344b;
        --text: #edf4ff;
        --muted: #94a8c4;
        --cyan: #42e8d5;
        --blue: #669cff;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, #122c45 0%, transparent 38%),
            radial-gradient(ellipse at 100% 20%, #102c32 0%, transparent 32%),
            var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 24, 0.8);
    }

    [data-testid="stSidebar"] {
        background: #0b1321;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--text) !important;
        letter-spacing: -0.03em;
    }

    p, label, .stCaption {
        color: var(--muted);
    }

    .hero {
        padding: 30px 34px;
        border: 1px solid #29415a;
        border-radius: 22px;
        background: linear-gradient(
            115deg, #13263d 0%, #101a2b 58%, #102b32 100%
        );
        margin-bottom: 25px;
    }

    .eyebrow {
        color: var(--cyan);
        font-family: 'DM Mono', monospace;
        font-size: 0.76rem;
        letter-spacing: 0.15em;
        margin-bottom: 12px;
    }

    .hero-title {
        color: #f4f8ff;
        font-size: clamp(2.2rem, 4vw, 3.2rem);
        line-height: 1.1;
        font-weight: 800;
        margin: 0 0 14px 0;
    }

    .hero-copy {
        color: #a9bdd6;
        font-size: 1rem;
        max-width: 800px;
        line-height: 1.75;
    }

    .section-card {
        background: rgba(16, 26, 43, 0.92);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px;
        margin: 12px 0 20px 0;
    }

    .section-label {
        color: var(--cyan);
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .phrase-box {
        background: #080f1c;
        border: 1px solid #2a4960;
        border-left: 4px solid var(--cyan);
        border-radius: 12px;
        padding: 20px;
        color: #e9f8ff;
        font-family: 'DM Mono', monospace;
        font-size: 1rem;
        line-height: 1.8;
        overflow-wrap: anywhere;
    }

    .metric-card {
        background: linear-gradient(145deg, #142137, #0e1727);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px;
        min-height: 110px;
    }

    .metric-label {
        color: #9bb0ca;
        font-size: 0.8rem;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #f0f6ff;
        font-size: 1.7rem;
        font-weight: 800;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
        border: 1px solid #3b6982;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #32d9c8, #65a6ff);
        color: #07111e;
        border: none;
    }

    div[data-baseweb="select"] > div {
        background: #101a2b;
        border-color: #344760;
        border-radius: 10px;
    }

    hr {
        border-color: var(--border);
    }

    footer, #MainMenu {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. MODEL AND METADATA HELPERS
# =========================================================

def find_user_config(data, user):
    """Find a user's model configuration in the metadata."""

    if isinstance(data, dict):
        if (
            "model_file" in data
            and "threshold" in data
            and user in str(data.get("model_file", ""))
        ):
            return data

        if user in data and isinstance(data[user], dict):
            candidate = data[user]

            if (
                "model_file" in candidate
                and "threshold" in candidate
            ):
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
    """Resolve model paths from the metadata."""

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
    """Load trained models without retraining them."""

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

    if (
        not isinstance(feature_columns, list)
        or not feature_columns
    ):
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


# =========================================================
# 4. FEATURE EXTRACTION
# =========================================================

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
        f"{prefix}_range": float(
            np.max(values) - np.min(values)
        ),
    }


def extract_features(events, feature_columns):
    """
    Convert browser-captured keystroke timings into the
    same feature layout used to train the existing models.
    """

    if (
        not isinstance(events, list)
        or len(events) != len(PHRASE)
    ):
        raise ValueError(
            f"Expected {len(PHRASE)} keyboard events; received "
            f"{len(events) if isinstance(events, list) else 0}."
        )

    def normalize_key(event):
        key = str(event.get("key", ""))

        if key in (" ", "Space", "Spacebar", "Key.space"):
            return " "

        return key.lower()

    typed = "".join(normalize_key(event) for event in events)

    if typed != PHRASE:
        raise ValueError(
            "The captured characters do not match the required "
            "phrase. Please clear the input and try again."
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
        raise ValueError("Press timestamps are invalid.")

    if not np.isfinite(releases).all():
        raise ValueError("Release timestamps are invalid.")

    if np.any(releases < presses):
        raise ValueError(
            "A key was released before its recorded press."
        )

    if np.any(dwell <= 0) or not np.isfinite(dwell).all():
        raise ValueError("Dwell-time measurements are invalid.")

    if np.any(np.diff(presses) < 0):
        raise ValueError(
            "Keyboard events are not in phrase order."
        )

    if (
        len(flight) != len(PHRASE) - 1
        or not np.isfinite(flight).all()
        or np.any(flight < 0)
    ):
        raise ValueError(
            "Flight-time measurements are missing or invalid."
        )

    features = {}

    features.update(calculate_statistics(dwell, "dwell"))
    features.update(calculate_statistics(flight, "flight"))

    for position, value in enumerate(dwell, start=1):
        features[f"dwell_pos_{position:02d}"] = float(value)

    for position, value in enumerate(flight, start=2):
        features[f"flight_pos_{position:02d}"] = float(value)

    features["attempt_duration_ms"] = float(
        releases[-1] - presses[0]
    )

    features["total_dwell_ms"] = float(np.sum(dwell))
    features["total_flight_ms"] = float(np.sum(flight))

    # Keep precisely the same feature order as training.
    X = pd.DataFrame([features]).reindex(
        columns=feature_columns
    )

    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()

        raise ValueError(
            f"Extracted features do not match training columns: "
            f"{missing}"
        )

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Extracted features contain invalid numerical values."
        )

    return X


# =========================================================
# 5. AUTHENTICATION
# =========================================================

def get_positive_probability(model, X):
    """
    Return the probability associated with the positive class.
    The trained binary verifier should identify genuine attempts
    using its positive class.
    """

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "This model does not support predict_proba()."
        )

    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)

    if 1 in classes:
        positive_index = classes.index(1)
    elif True in classes:
        positive_index = classes.index(True)
    else:
        raise ValueError(
            f"Cannot identify the positive class in {classes}. "
            "Check the labels used when training the verifier."
        )

    return float(probabilities[positive_index])


def authenticate_user(user, X, models, configs):
    model = models[user]
    threshold = float(configs[user]["threshold"])

    probability = get_positive_probability(model, X)

    return {
        "user": user,
        "probability": probability,
        "threshold": threshold,
        "accepted": probability >= threshold,
    }


# =========================================================
# 6. SHARED PAGE COMPONENTS
# =========================================================

def render_section_heading(label, heading, description=None):
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-label">{label}</div>
            <h2 style="margin:0 0 8px 0;">{heading}</h2>
            <div style="color:#94a8c4;line-height:1.7;">
                {description or ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">
                BEHAVIORAL BIOMETRICS · CYBERSECURITY
            </div>
            <div class="hero-title">
                KeyShield<span style="color:#42e8d5">.</span>
            </div>
            <div class="hero-copy">
                Identity verification through the unique rhythm
                of human typing. Analyze keystroke timing, extract
                behavioral features, and verify a claimed identity
                using trained machine-learning models.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 7. LOAD RESOURCES AND NAVIGATION
# =========================================================

render_hero()

try:
    (
        metadata,
        users,
        feature_columns,
        models,
        configs,
    ) = load_authentication_resources()

except Exception as exc:
    st.error("KeyShield could not load its authentication resources.")
    st.code(str(exc))

    st.info(
        "Check that authentication_models contains "
        "authentication_metadata.json and all four trained models."
    )

    st.stop()


with st.sidebar:
    st.markdown("## 🔐 KeyShield")
    st.caption("BEHAVIORAL AUTHENTICATION")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Live Authentication",
            "How It Works",
            "Model Evaluation",
        ],
        key="navigation",
    )

    st.divider()

    st.markdown("### System status")
    st.success("Metadata loaded")
    st.success(f"{len(models)} user models loaded")
    st.caption("Model files are loaded, not retrained.")


# =========================================================
# 8. LIVE AUTHENTICATION PAGE
# =========================================================

if page == "Live Authentication":

    render_section_heading(
        "LIVE VERIFICATION",
        "Verify your typing identity",
        "Select your enrolled identity and type the displayed "
        "phrase using your physical keyboard.",
    )

    left, right = st.columns([1.45, 1])

    with left:
        st.markdown(
            '<div class="section-label">01 / YOUR CHALLENGE</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="phrase-box">{PHRASE}</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Type the phrase exactly as shown. Spaces and character "
            "order are part of the verification challenge."
        )

        selected_user = st.selectbox(
            "Claimed user identity",
            users,
            key="selected_user",
        )

        st.markdown(
            '<div class="section-label">02 / LIVE KEYSTROKE CAPTURE</div>',
            unsafe_allow_html=True,
        )

        st.write(
            "Click inside the typing area below and type the phrase."
        )

        capture_data = keystroke_capture(
            phrase=PHRASE,
            key="live_keystroke_capture",
            default=None,
        )

        if not isinstance(capture_data, dict):
            capture_data = {}

        completed = bool(capture_data.get("completed", False))
        events = capture_data.get("keystrokes", [])

        if completed:
            st.success("Typing capture completed.")
        else:
            st.info(
                "Waiting for a completed typing attempt from the "
                "keyboard-capture component."
            )

        if st.button(
            "🔐 Authenticate identity",
            type="primary",
            use_container_width=True,
        ):
            try:
                if not completed:
                    raise ValueError(
                        "Finish typing the entire phrase before "
                        "authenticating."
                    )

                X = extract_features(events, feature_columns)

                result = authenticate_user(
                    selected_user,
                    X,
                    models,
                    configs,
                )

                st.session_state["last_auth_result"] = result
                st.session_state["last_auth_user"] = selected_user

            except Exception as exc:
                st.session_state.pop("last_auth_result", None)
                st.error(str(exc))

    with right:
        st.markdown(
            '<div class="section-label">VERIFICATION SUMMARY</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <h3>Behavioral identity check</h3>
                <p>
                    KeyShield compares the current typing pattern
                    against the selected user's trained verifier.
                </p>
                <hr>
                <p><b>Input</b>: keyboard timing events</p>
                <p><b>Features</b>: dwell and flight timings</p>
                <p><b>Decision</b>: probability versus user threshold</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "A high probability is the model's confidence for its "
            "positive class, not a guarantee of identity."
        )

    result = st.session_state.get("last_auth_result")
    last_user = st.session_state.get("last_auth_user")

    if result and last_user == selected_user:
        st.divider()
        st.markdown("## Authentication result")

        probability_percent = result["probability"] * 100
        threshold_percent = result["threshold"] * 100

        if result["accepted"]:
            st.success(
                f"ACCESS ACCEPTED — {selected_user}"
            )
        else:
            st.error(
                f"ACCESS DENIED — {selected_user}"
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            render_metric(
                "Genuine-class probability",
                f"{probability_percent:.2f}%",
            )

        with c2:
            render_metric(
                "Decision threshold",
                f"{threshold_percent:.2f}%",
            )

        with c3:
            render_metric(
                "Decision",
                "ACCEPTED" if result["accepted"] else "DENIED",
            )

        st.caption(
            "This is a prototype. Results depend on the trained "
            "model, capture quality, and evaluation data."
        )


# =========================================================
# 9. HOW IT WORKS PAGE
# =========================================================

elif page == "How It Works":

    render_section_heading(
        "SYSTEM OVERVIEW",
        "How KeyShield works",
        "A behavioral biometric authentication pipeline built "
        "around keyboard timing and supervised machine learning.",
    )

    st.markdown("### Authentication pipeline")

    stages = [
        (
            "01",
            "Capture",
            "Record key press and release timestamps while the "
            "user types the fixed challenge phrase.",
        ),
        (
            "02",
            "Validate",
            "Check the typed phrase and validate that timing values "
            "are complete and numerically valid.",
        ),
        (
            "03",
            "Extract features",
            "Calculate dwell time, flight time, statistical summaries, "
            "position-wise timings, and attempt duration.",
        ),
        (
            "04",
            "Verify",
            "Pass the ordered feature vector into the selected user's "
            "pre-trained binary classification model.",
        ),
        (
            "05",
            "Decide",
            "Compare the positive-class probability with that user's "
            "configured threshold to accept or reject the attempt.",
        ),
    ]

    for number, title, description in stages:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-label">STAGE {number}</div>
                <h3 style="margin:0 0 8px 0;">{title}</h3>
                <div style="color:#94a8c4;line-height:1.7;">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Signals and Systems connection")

    st.markdown(
        """
        This project treats typing as a sequence of time-dependent
        events. The press and release timestamps provide temporal
        measurements from which behavioral signals are derived.

        - **Time-domain analysis:** dwell time measures how long a
          key is held; flight time represents the interval between
          successive key presses according to the dataset's
          measurement convention.
        - **Statistical feature extraction:** mean, standard
          deviation, median, minimum, maximum, and range summarize
          timing variability.
        - **Sequence analysis:** position-wise features preserve
          timing information for individual characters and transitions.
        - **Machine learning:** the extracted timing features are
          supplied to trained user-specific verification models.
        """
    )

    st.warning(
        "The current implementation uses time-domain timing features. "
        "It does not perform FFT or DFT analysis."
    )

    st.markdown("### Important limitations")

    st.markdown(
        """
        - A fixed phrase makes the typing challenge repeatable, but
          it can also be observed or copied.
        - Typing speed can change with fatigue, keyboard hardware,
          posture, or other conditions.
        - A small dataset may not represent the full range of
          real-world typing behavior.
        - Authentication thresholds should be validated on
          independent data before any security-critical use.
        """
    )


# =========================================================
# 10. MODEL EVALUATION PAGE
# =========================================================

elif page == "Model Evaluation":

    render_section_heading(
        "MODEL PERFORMANCE",
        "Verification metrics",
        "Review the previously measured test-set performance of "
        "the four user-specific verification models.",
    )

    # These are the existing evaluation results supplied for
    # this project. They are not recalculated by this dashboard.
    evaluation = pd.DataFrame(
        [
            {
                "User": "User1",
                "Accuracy (%)": 94.34,
                "FAR (%)": 2.52,
                "FRR (%)": 15.00,
            },
            {
                "User": "User2",
                "Accuracy (%)": 93.08,
                "FAR (%)": 7.56,
                "FRR (%)": 5.00,
            },
            {
                "User": "User3",
                "Accuracy (%)": 100.00,
                "FAR (%)": 0.00,
                "FRR (%)": 0.00,
            },
            {
                "User": "User4",
                "Accuracy (%)": 100.00,
                "FAR (%)": 0.00,
                "FRR (%)": 0.00,
            },
        ]
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        render_metric("Users evaluated", "4")

    with c2:
        render_metric("Input features", str(len(feature_columns)))

    with c3:
        render_metric("Evaluation split", "80 / 20")

    st.markdown("### Previously recorded test results")

    st.dataframe(
        evaluation,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These values are the project's recorded results. "
        "They are displayed as supplied, not recomputed from the "
        "currently loaded model files."
    )

    st.markdown("### What do these metrics mean?")

    st.markdown(
        """
        **Accuracy** measures the fraction of evaluated predictions
        that were correct.

        **False Acceptance Rate (FAR)** measures how often an
        unauthorized attempt is incorrectly accepted.

        **False Rejection Rate (FRR)** measures how often a genuine
        user's attempt is incorrectly rejected.

        Lower FAR is important for preventing impostor acceptance;
        lower FRR is important for reducing inconvenience to genuine
        users. Threshold changes can affect both.
        """
    )

    st.warning(
        "A reported 100% score on a small test set does not establish "
        "perfect security. Evaluate with more independent typing "
        "sessions and realistic impostor attempts before drawing "
        "strong conclusions."
    )


# =========================================================
# 11. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#7186a4;
        font-size:0.82rem;
        padding:12px 0;
    ">
        KEYSHIELD · BEHAVIORAL BIOMETRICS · SIGNALS AND SYSTEMS
        <br>
        Academic prototype — not intended as a production
        authentication system.
    </div>
    """,
    unsafe_allow_html=True,
)