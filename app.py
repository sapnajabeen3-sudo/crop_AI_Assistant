"""
=====================================================================
 SmartCrop AI — AI-Powered Crop Recommendation System
 Developed by Sapna Jabeen
=====================================================================

WHAT THIS FILE DOES
--------------------
This is a single-file Streamlit application, but it is organized into
clearly separated sections so it stays easy to read AND easy to split
into multiple files later (utils/model_utils.py, utils/validation.py,
utils/chatbot.py, api/prediction_service.py, etc.) — see the "IoT /
API-ready design" section below for why this matters.

SECTIONS IN THIS FILE
----------------------
1. Imports & page config
2. Data / model loading functions   (cached, IoT/API-ready)
3. Input validation function        (IoT/API-ready)
4. Prediction function               (IoT/API-ready - the "core engine")
5. Chatbot logic
6. Streamlit UI (sidebar + pages)
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS & PAGE CONFIG
# ---------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from crop_info import get_crop_info

st.set_page_config(
    page_title="SmartCrop AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths to the model files (relative to this app.py file)
MODEL_PATH = os.path.join( "crop_recommendation_model.pkl")
FEATURES_PATH = os.path.join( "feature_names.pkl")
DATASET_PATH = os.path.join( "Crop_recommendation.csv")


# ---------------------------------------------------------------------------
# 2. DATA / MODEL LOADING  (cached so they only load once per session)
# ---------------------------------------------------------------------------
# NOTE: These files were saved with `joblib`, not plain `pickle`, so we must
# load them with `joblib.load()`. Using plain `pickle.load()` on a
# joblib-saved scikit-learn model can raise cryptic errors like
# "UnpicklingError: STACK_GLOBAL requires str" — we discovered this while
# testing, which is why joblib is used consistently below.

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the pre-trained crop recommendation model.
    Returns (model, error_message). If loading fails, model is None and
    error_message explains what went wrong (no raw stack trace shown to users).
    """
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except FileNotFoundError:
        return None, f"Model file not found at '{MODEL_PATH}'. Please check the file path."
    except Exception as e:
        return None, f"Could not load the model. Details: {e}"


@st.cache_resource(show_spinner=False)
def load_features():
    """
    Load the list of feature names (and their order) the model expects.
    Returns (feature_names, error_message).
    """
    try:
        feature_names = joblib.load(FEATURES_PATH)
        return list(feature_names), None
    except FileNotFoundError:
        return None, f"Feature file not found at '{FEATURES_PATH}'. Please check the file path."
    except Exception as e:
        return None, f"Could not load feature names. Details: {e}"


@st.cache_data(show_spinner=False)
def load_dataset():
    """
    Load the crop recommendation dataset (used only for the Insights page).
    Returns (dataframe, error_message).
    """
    try:
        df = pd.read_csv(DATASET_PATH)
        return df, None
    except FileNotFoundError:
        return None, f"Dataset file not found at '{DATASET_PATH}'."
    except Exception as e:
        return None, f"Could not load the dataset. Details: {e}"


# ---------------------------------------------------------------------------
# 3. INPUT VALIDATION
# ---------------------------------------------------------------------------
# Reasonable real-world bounds (a bit wider than the training data range,
# since real farms/sensors can report values outside the exact dataset range).
VALID_RANGES = {
    "N": (0, 300, "Nitrogen (N)"),
    "P": (0, 300, "Phosphorus (P)"),
    "K": (0, 300, "Potassium (K)"),
    "temperature": (-10, 60, "Temperature"),
    "humidity": (0, 100, "Humidity"),
    "ph": (0, 14, "Soil pH"),
    "rainfall": (0, 1000, "Rainfall"),
}


def validate_input(input_data: dict):
    """
    Validate a dict of {feature_name: value}.

    This function is deliberately independent from Streamlit widgets so that
    it can later validate data coming from an IoT sensor or a REST API call,
    not just from the form in this app.

    Returns:
        (is_valid: bool, error_message: str or None)
    """
    for feature, (min_val, max_val, label) in VALID_RANGES.items():
        if feature not in input_data:
            return False, f"Missing value for {label}."

        value = input_data[feature]

        # Must be numeric
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False, f"{label} must be a valid number."

        # Must be within a sensible range
        if value < min_val or value > max_val:
            return False, f"Please enter a valid {label} between {min_val} and {max_val}."

    return True, None


# ---------------------------------------------------------------------------
# 4. PREDICTION FUNCTION  (the core, IoT/API-ready engine)
# ---------------------------------------------------------------------------
def predict_crop(input_data: dict, model, feature_names: list):
    """
    Run a crop prediction given structured input data.

    This function does NOT know anything about Streamlit widgets — it just
    takes a dictionary like:
        {"N": 90, "P": 42, "K": 43, "temperature": 20.8,
         "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
    and returns a prediction. That means the exact same function can later
    be reused by:
      - A REST API endpoint (e.g. POST /predict)
      - An IoT gateway that pushes live sensor readings
      - A batch script processing many farms at once

    Returns:
        {
            "crop": "rice",
            "confidence": 94.2,              # percent, or None if unavailable
            "top_predictions": [("rice", 94.2), ("jute", 3.1), ...] or None
        }
        or raises an Exception if prediction fails.
    """
    # Arrange the input in the exact order the model was trained on
    ordered_values = [[float(input_data[feature]) for feature in feature_names]]
    input_df = pd.DataFrame(ordered_values, columns=feature_names)

    prediction = model.predict(input_df)[0]

    confidence = None
    top_predictions = None

    # Only compute confidence/top predictions if the model genuinely supports it.
    # We never invent a confidence score.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        class_labels = model.classes_

        # Pair up each class with its probability, sort by probability desc
        paired = sorted(zip(class_labels, probabilities), key=lambda x: x[1], reverse=True)

        confidence = round(paired[0][1] * 100, 1)
        top_predictions = [(label, round(prob * 100, 1)) for label, prob in paired[:3]]

    return {
        "crop": str(prediction),
        "confidence": confidence,
        "top_predictions": top_predictions,
    }


# ---------------------------------------------------------------------------
# 5. CHATBOT LOGIC
# ---------------------------------------------------------------------------
# A lightweight, rule-based chatbot. No external API is required, but the
# function signature is kept simple so an external LLM API could be dropped
# in later (e.g. call OpenAI/Anthropic here instead of the keyword rules).

CHATBOT_KNOWLEDGE = {
    "n": "**N** stands for **Nitrogen** — a key nutrient that helps plants grow leaves and stems.",
    "nitrogen": "**Nitrogen (N)** is essential for leafy, green plant growth. Too little can cause yellowing leaves.",
    "p": "**P** stands for **Phosphorus** — it supports strong root development and flowering.",
    "phosphorus": "**Phosphorus (P)** helps plants develop strong roots, flowers, and fruit.",
    "k": "**K** stands for **Potassium** — it helps plants resist disease and regulate water use.",
    "potassium": "**Potassium (K)** improves a plant's overall health, disease resistance, and water regulation.",
    "ph": "**Soil pH** measures how acidic or alkaline the soil is, on a scale of 0–14. Most crops prefer pH 6–7.5.",
    "humidity": "**Humidity** is the amount of moisture in the air, shown as a percentage. It affects how fast plants lose water.",
    "rainfall": "**Rainfall** is the amount of precipitation, measured in millimeters (mm), that the crop's location receives.",
    "temperature": "**Temperature** is the air/soil temperature in °C. Each crop has a preferred temperature range.",
    "how does this work": "You enter soil (N, P, K, pH) and environmental (temperature, humidity, rainfall) values. The trained ML model analyzes these values and predicts the crop that's most likely to grow well under those conditions.",
    "how it works": "You enter soil (N, P, K, pH) and environmental (temperature, humidity, rainfall) values. The trained ML model analyzes these values and predicts the crop that's most likely to grow well under those conditions.",
    "how to enter": "Go to the ' Crop Recommendation' page, fill in all 7 fields with your soil test and weather data, then click 'Recommend Crop'.",
    "what does the recommendation mean": "It means that, based on the values you entered, the trained ML model predicts this crop is well-suited to those soil and environmental conditions.",
}


def get_chatbot_response(user_message: str, context: dict) -> str:
    """
    Return a chatbot reply for the user's message.

    `context` may contain the last prediction, e.g.:
        {"crop": "rice", "confidence": 94.2, "inputs": {...}}
    so the bot can answer questions like "why was this crop recommended?"
    without fabricating details about the model's internal reasoning.
    """
    message = user_message.lower().strip()

    # 1) Questions about the *current* prediction (needs context)
    if any(phrase in message for phrase in ["why was this crop", "why this crop", "why did you recommend", "why was it recommended"]):
        if context.get("crop"):
            reply = (
                f"The model recommended **{context['crop'].title()}** based on the soil and "
                f"environmental values currently entered (N, P, K, temperature, humidity, pH, and rainfall). "
            )
            if context.get("confidence") is not None:
                reply += f"It did so with a model confidence of **{context['confidence']}%**. "
            reply += (
                "Note: this reflects the model's learned patterns from historical data, "
                "not a guaranteed agronomic explanation — see the Crop Information section "
                "for general growing conditions of this crop."
            )
            return reply
        return "I don't have a current recommendation yet — please fill in the form on the '🌱 Crop Recommendation' page and click 'Recommend Crop' first."

    if "current recommendation" in message or "what was recommended" in message:
        if context.get("crop"):
            return f"The current recommended crop is **{context['crop'].title()}**."
        return "No recommendation has been generated yet in this session."

    # 2) General knowledge lookups — check longer phrases first, then keywords
    for phrase in sorted(CHATBOT_KNOWLEDGE.keys(), key=len, reverse=True):
        if phrase in message:
            return CHATBOT_KNOWLEDGE[phrase]

    # 3) Fallback
    return (
        "I can help explain N, P, K, pH, humidity, rainfall, temperature, "
        "how the recommendation works, or the current prediction. "
        "Try asking something like *'What is N?'* or *'Why was this crop recommended?'*"
    )


# ---------------------------------------------------------------------------
# 6. STREAMLIT UI
# ---------------------------------------------------------------------------

# ---- Minimal custom styling (light background, green accents, soft cards) --
st.markdown(
    """
    <style>
    .stApp { background-color: #F7FAF7; }
    .info-card {
        background-color: white;
        padding: 1.2rem;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
        border: 1px solid #E5EFE5;
    }
    .result-card {
        background: linear-gradient(135deg, #E8F5E9 0%, #F1F8F2 100%);
        padding: 2rem;
        border-radius: 18px;
        text-align: center;
        border: 1px solid #C8E6C9;
        margin: 1rem 0;
    }
    .footer-text {
        text-align: center;
        color: #6B7A6B;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Session state initialization ------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, message)
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = {}  # {"crop":..., "confidence":..., "inputs": {...}}
page_options = ["Home", "Crop Recommendation", "Insights", "AI Assistant", "About"]
if st.session_state.get("page") not in page_options:
    st.session_state.page = "Home"

# ---- Load model, features, dataset -----------------------------------------
model, model_error = load_model()
feature_names, features_error = load_features()
dataset, dataset_error = load_dataset()

# ---- SIDEBAR -----------------------------------------------------------
with st.sidebar:
    st.markdown("##  SmartCrop AI")
    st.caption("AI-Powered Crop Recommendation System")
    st.divider()

    st.session_state.page = st.radio(
        "Navigate",
        page_options,
        index=page_options.index(st.session_state.page),
        label_visibility="collapsed",
    )

    st.divider()

    # ---- Compact chatbot, always available in the sidebar ----
    with st.expander(" SmartCrop Assistant", expanded=False):
        st.caption("Ask about N, P, K, pH, humidity, rainfall, or your recommendation.")

        for role, msg in st.session_state.chat_history[-6:]:
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Assistant:** {msg}")

        quick_question = st.text_input("Ask a question", key="sidebar_chat_input")
        col_a, col_b = st.columns(2)
        with col_a:
            ask_clicked = st.button("Send", key="sidebar_send", use_container_width=True)
        with col_b:
            clear_clicked = st.button("Clear chat", key="sidebar_clear", use_container_width=True)

        if ask_clicked and quick_question.strip():
            reply = get_chatbot_response(quick_question, st.session_state.last_prediction)
            st.session_state.chat_history.append(("user", quick_question))
            st.session_state.chat_history.append(("assistant", reply))
            st.rerun()

        if clear_clicked:
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown(
        '<p class="footer-text">SmartCrop AI • Developed by Sapna Jabeen</p>',
        unsafe_allow_html=True,
    )

# ---- Show a top-level warning once if model/feature files failed to load ---
if model_error or features_error:
    st.error(
        " The application could not fully load its ML model files.\n\n"
        + (f"- {model_error}\n" if model_error else "")
        + (f"- {features_error}\n" if features_error else "")
        + "\nCrop recommendations will not be available until this is fixed."
    )

# =============================================================================
# PAGE: HOME
# =============================================================================
if st.session_state.page == "Home":
    st.title(" Smart Crop Recommendation")
    st.markdown(
        "Make smarter farming decisions using soil and environmental data, "
        "powered by machine learning."
    )

    if st.button(" Start Crop Recommendation", type="primary"):
        st.session_state.page = "Crop Recommendation"
        st.rerun()

    st.write("")
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("", "Smart Recommendations", "Get crop recommendations based on agricultural conditions."),
        ("", "Soil Analysis", "Use N, P, K and pH values to understand soil conditions."),
        ("", "Environmental Factors", "Consider temperature, humidity and rainfall."),
        ("", "AI Powered", "Machine learning analyzes the provided conditions."),
    ]
    for col, (emoji, title, desc) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(
                f'<div class="info-card"><h3>{emoji} {title}</h3><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

# =============================================================================
# PAGE: CROP RECOMMENDATION
# =============================================================================
elif st.session_state.page == "Crop Recommendation":
    st.title("🌱 Crop Recommendation")
    st.caption("Enter your soil and environmental conditions below.")

    st.subheader("Soil Parameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        N = st.number_input("Nitrogen (N) — kg/ha", min_value=0.0, max_value=300.0, value=50.0, step=1.0)
    with c2:
        P = st.number_input("Phosphorus (P) — kg/ha", min_value=0.0, max_value=300.0, value=50.0, step=1.0)
    with c3:
        K = st.number_input("Potassium (K) — kg/ha", min_value=0.0, max_value=300.0, value=50.0, step=1.0)
    with c4:
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

    st.subheader("Environmental Parameters")
    e1, e2, e3 = st.columns(3)
    with e1:
        temperature = st.number_input("Temperature — °C", min_value=-10.0, max_value=60.0, value=25.0, step=0.5)
    with e2:
        humidity = st.number_input("Humidity — %", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    with e3:
        rainfall = st.number_input("Rainfall — mm", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)

    st.write("")
    recommend_clicked = st.button(" Recommend Crop", type="primary", use_container_width=True)

    if recommend_clicked:
        input_data = {
            "N": N, "P": P, "K": K,
            "temperature": temperature, "humidity": humidity,
            "ph": ph, "rainfall": rainfall,
        }

        is_valid, error_message = validate_input(input_data)

        if not is_valid:
            st.warning(f" {error_message}")
        elif model is None or feature_names is None:
            st.error("The model isn't available right now, so a recommendation can't be generated. Please check the model files.")
        else:
            try:
                result = predict_crop(input_data, model, feature_names)

                # Save to session state so the chatbot & this page can reference it
                st.session_state.last_prediction = {
                    "crop": result["crop"],
                    "confidence": result["confidence"],
                    "inputs": input_data,
                }

                info = get_crop_info(result["crop"])

                # ---- Result card ----
                st.markdown(
                    f"""
                    <div class="result-card">
                        <h2> Recommended Crop</h2>
                        <h1>{info['emoji']} {result['crop'].title()}</h1>
                        <p>Based on the soil and environmental conditions you provided,
                        the model recommends <b>{result['crop'].title()}</b>.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ---- Confidence ----
                if result["confidence"] is not None:
                    st.markdown(f"**Model confidence:** {result['confidence']}%")
                    st.progress(min(result["confidence"] / 100, 1.0))
                else:
                    st.info("Prediction generated by the trained ML model.")

                # ---- Top 3 predictions ----
                if result["top_predictions"]:
                    st.markdown("### Other possible recommendations")
                    for i, (crop_name, prob) in enumerate(result["top_predictions"], start=1):
                        st.markdown(f"{i}. **{crop_name.title()}** — {prob}%")

                # ---- Input summary table ----
                st.markdown("### Your entered values")
                summary_df = pd.DataFrame(
                    {
                        "Parameter": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)",
                                      "Temperature (°C)", "Humidity (%)", "Soil pH", "Rainfall (mm)"],
                        "Value": [N, P, K, temperature, humidity, ph, rainfall],
                    }
                )
                st.dataframe(summary_df, hide_index=True, use_container_width=True)

                # ---- Crop information (educational, clearly separated from the model) ----
                st.markdown("### About the recommended crop")
                st.markdown(
                    f"**{info['description']}**\n\n"
                    f"**Typical growing conditions:** {info['growing_conditions']}\n\n"
                    f"*This general information is provided for education and is separate "
                    f"from the ML model's prediction.*"
                )

            except Exception as e:
                st.error(f"Something went wrong while generating the recommendation. Details: {e}")

    st.divider()
    st.caption(
        "**Disclaimer:** This system provides a machine-learning-based crop recommendation "
        "for educational and decision-support purposes. Actual crop selection should also "
        "consider local climate, soil testing, water availability, disease risks, market "
        "conditions, and advice from qualified agricultural professionals."
    )

# =============================================================================
# PAGE: INSIGHTS
# =============================================================================
elif st.session_state.page == "Insights":
    st.title(" Dataset Insights")

    if dataset_error:
        st.error(dataset_error)
    else:
        st.subheader("Dataset statistics")
        s1, s2, s3 = st.columns(3)
        s1.metric("Records", len(dataset))
        s2.metric("Crop classes", dataset["label"].nunique())
        s3.metric("Features used", len(feature_names) if feature_names else 7)

        st.subheader("Crop distribution")
        crop_counts = dataset["label"].value_counts()
        st.bar_chart(crop_counts)

        st.subheader("Feature statistics")
        numeric_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        st.dataframe(dataset[numeric_cols].describe().round(2), use_container_width=True)

# =============================================================================
# PAGE: AI ASSISTANT (full-size chat, same conversation as the sidebar widget)
# =============================================================================
elif st.session_state.page == "AI Assistant":
    st.title(" SmartCrop Assistant")
    st.caption(
        "Ask about soil/environmental terms, how the recommendation works, "
        "or your current prediction."
    )

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").markdown(msg)
        else:
            st.chat_message("assistant").markdown(msg)

    user_input = st.chat_input("Ask me something, e.g. 'What is potassium?'")
    if user_input:
        reply = get_chatbot_response(user_input, st.session_state.last_prediction)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", reply))
        st.rerun()

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

# =============================================================================
# PAGE: ABOUT
# =============================================================================
elif st.session_state.page == "About":
    st.title(" About SmartCrop AI")

    st.markdown(
        """
        SmartCrop AI uses a trained machine learning model to recommend suitable
        crops based on soil and environmental parameters (Nitrogen, Phosphorus,
        Potassium, temperature, humidity, soil pH, and rainfall).

        ### Technology
        - Python
        - Pandas & NumPy
        - Scikit-learn (Random Forest Classifier)
        - Streamlit

        ### IoT & API Roadmap
        The prediction logic (`predict_crop()`) is fully separated from the
        Streamlit UI, so it can later be reused by:
        - A REST API (`POST /predict`)
        - IoT agricultural sensors sending live readings
        - Batch prediction scripts

        See the README for the full roadmap.
        """
    )

    st.divider()
    st.markdown("## Developed by Sapna Jabeen")

# ---- Global footer -----------------------------------------------------
st.markdown(
    '<p class="footer-text">SmartCrop AI • Developed by Sapna Jabeen</p>',
    unsafe_allow_html=True,
)
