# 🌾 SmartCrop AI

**AI-Powered Crop Recommendation System**

## Overview

SmartCrop AI is a Streamlit web application that recommends a suitable crop
based on soil and environmental conditions (Nitrogen, Phosphorus, Potassium,
temperature, humidity, soil pH, and rainfall), using a pre-trained
Random Forest machine learning model.

## Features

- Clean, beginner-friendly form for entering soil and environmental data
- Input validation with clear warnings (no invalid data reaches the model)
- Attractive recommendation card with the predicted crop
- Model confidence score and top-3 alternative predictions (when supported)
- Educational "About the crop" section, clearly separated from the model's prediction
- Dataset insights page (crop distribution, feature statistics)
- Built-in rule-based chatbot (sidebar + dedicated page) that explains
  N/P/K/pH/humidity/rainfall and can reference your current prediction
- Modular, IoT-ready and API-ready prediction pipeline

## Machine Learning Model

- **Type:** Random Forest Classifier (scikit-learn)
- **Input features (in this exact order):** `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`
- **Output:** one of 22 crop labels (e.g. rice, maize, mango, coffee, ...)
- The model and feature-order file were saved with **joblib**, so they are
  loaded with `joblib.load()` — using plain `pickle.load()` can raise an
  unpickling error.
- The model is used as-is (`model.predict(...)`) — it is **not** retrained,
  and no manual rule-based logic replaces its predictions.

## Input Parameters

| Parameter | Unit | Reasonable range |
|---|---|---|
| Nitrogen (N) | kg/ha | 0–300 |
| Phosphorus (P) | kg/ha | 0–300 |
| Potassium (K) | kg/ha | 0–300 |
| Temperature | °C | -10–60 |
| Humidity | % | 0–100 |
| Soil pH | — | 0–14 |
| Rainfall | mm | 0–1000 |

## How It Works

1. User enters the 7 soil/environmental values in the **Crop Recommendation** page.
2. `validate_input()` checks that all values are present, numeric, and within sensible ranges.
3. `predict_crop()` arranges the values in the order stored in `feature_names.pkl`,
   builds a single-row DataFrame, and calls `model.predict()` (and `predict_proba()`
   if supported).
4. The result is displayed as a recommendation card, with confidence and
   top-3 alternatives when available, plus a short educational blurb about the crop.

## Installation

```bash
git clone <your-repo-url>
cd smart_crop_recommendation
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Google Colab Setup

```python
!pip install streamlit pandas numpy scikit-learn joblib -q
!pip install localtunnel -q  # or use `pyngrok`

# Upload app.py, crop_info.py, and the model/ and data/ folders to Colab first
!streamlit run app.py &>/content/logs.txt &
!npx localtunnel --port 8501
```

Click the generated URL and enter the IP address shown by
`!wget -q -O - ipv4.icanhazip.com` when prompted by localtunnel.

## Project Structure

```
smart_crop_recommendation/
│
├── app.py                # Main Streamlit application
├── crop_info.py           # Educational crop descriptions
│
├── model/
│   ├── crop_recommendation_model.pkl
│   └── feature_names.pkl
│
├── data/
│   └── Crop_recommendation.csv
│
├── requirements.txt
└── README.md
```

Core functions in `app.py` (`load_model`, `load_features`, `validate_input`,
`predict_crop`, `get_chatbot_response`) are written to be UI-independent, so
splitting them into `utils/model_utils.py`, `utils/validation.py`, and
`utils/chatbot.py` later is a straightforward refactor.

## IoT Integration Roadmap

The `predict_crop(input_data, model, feature_names)` function takes a plain
dictionary and returns a plain dictionary — it has no dependency on Streamlit.
This means the same function can power a future IoT/API pipeline:

```
Sensors (soil NPK, pH, humidity, rainfall, temperature)
   ↓
ESP32 / Arduino / IoT Gateway
   ↓
MQTT / REST API  ( e.g. POST /predict )
   ↓
Data validation  (validate_input)
   ↓
predict_crop()   ← same function used by this Streamlit app
   ↓
Database + SmartCrop Dashboard
```

**Phases:**
1. Manual input + ML prediction *(current)*
2. REST API (e.g. FastAPI wrapping `predict_crop`)
3. IoT sensor integration
4. Real-time sensor dashboard
5. Automated agricultural alerts

Example future API contract:

```
POST /predict
Body: {"N": 90, "P": 42, "K": 43, "temperature": 20.8,
       "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
Response: {"recommended_crop": "rice"}
```

## Future Improvements

- Wrap `predict_crop()` in a FastAPI service for external integrations
- Add multi-language support for farmers
- Connect the chatbot to an external LLM API (kept swappable in `get_chatbot_response`)
- Add historical prediction logging per user/farm
- Fertilizer recommendation and plant disease detection modules

## Disclaimer

This system provides a machine-learning-based crop recommendation for
educational and decision-support purposes. Actual crop selection should
also consider local climate, soil testing, water availability, disease
risks, market conditions, and advice from qualified agricultural
professionals.

## Developer

**Developed by Sapna Jabeen**
