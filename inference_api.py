# inference_api.py

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import joblib

from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model

# =====================
# KONFIGURASI
# =====================

TICKERS = {
    "BBCA": "BBCA.JK",
    "BBRI": "BBRI.JK",
    "BBNI": "BBNI.JK",
    "BMRI": "BMRI.JK"
}

LOOKBACK_DAYS = 60
DEFAULT_DAYS = 7

MODELS_DIR = "models"
SCALERS_DIR = "scalers"

# =====================
# LOAD MODEL SEKALI SAAT STARTUP
# =====================

models = {}
feature_scalers = {}
close_scalers = {}
feature_cols_map = {}

for name in TICKERS:
    models[name] = load_model(f"{MODELS_DIR}/{name}_best.h5")
    feature_scalers[name] = joblib.load(f"{SCALERS_DIR}/{name}_feature_scaler.pkl")
    close_scalers[name] = joblib.load(f"{SCALERS_DIR}/{name}_close_scaler.pkl")
    feature_cols_map[name] = joblib.load(f"{SCALERS_DIR}/{name}_feature_cols.pkl")

# =====================
# FUNGSI PRESDIKSI
# =====================

def predict_future(symbol, days):
    ticker = TICKERS[symbol]
    df = yf.download(ticker, period="5y").dropna()

    feature_cols = feature_cols_map[symbol]
    features = df[feature_cols].values

    feature_scaler = feature_scalers[symbol]
    close_scaler = close_scalers[symbol]

    features_scaled = feature_scaler.transform(features)

    last_seq = features_scaled[-LOOKBACK_DAYS:, :].reshape(
        1, LOOKBACK_DAYS, len(feature_cols)
    )

    model = models[symbol]
    close_idx = feature_cols.index("Close")

    preds_scaled = []
    for _ in range(days):
        pred = model.predict(last_seq, verbose=0)[0][0]
        preds_scaled.append(pred)

        last_day = last_seq[0, -1, :].copy()
        last_day[close_idx] = pred

        new_seq = np.vstack([last_seq[0, 1:, :], last_day])
        last_seq = new_seq.reshape(1, LOOKBACK_DAYS, len(feature_cols))

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = close_scaler.inverse_transform(preds_scaled).flatten()

    last_date = df.index[-1].date()
    dates = [last_date + dt.timedelta(days=i+1) for i in range(days)]

    return [
        {"date": str(dates[i]), "predicted_close": float(preds[i])}
        for i in range(days)
    ]

# =====================
# FLASK API
# =====================

app = Flask(__name__)

@app.route("/")
def home():
    return "Stock Predictor API (BiLSTM High Precision)"

@app.route("/predict/<symbol>")
def predict(symbol):
    symbol = symbol.upper()
    if symbol not in TICKERS:
        return jsonify({"error": "Ticker harus BBCA, BBRI, BBNI atau BMRI"}), 400
    
    days = request.args.get("days", DEFAULT_DAYS, type=int)
    result = predict_future(symbol, days)

    return jsonify({
        "symbol": symbol,
        "days": days,
        "predictions": result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
