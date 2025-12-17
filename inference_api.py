from flask import Flask, jsonify
from flask_cors import CORS

import os
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from tensorflow.keras.models import load_model

# Konfigurasi dasar

MODELS_DIR = "models"
SCALERS_DIR = "scalers"

SUPPORTED_SYMBOLS = ["BBCA", "BBRI", "BBNI", "BMRI"]
LOOKBACK = 60  
DEFAULT_FORECAST_DAYS = int(os.environ.get("FORECAST_DAYS", 7))

# Kalender hari bursa
ID_HOLIDAYS = {
    # dt.date(2025, 1, 1),
}

def generate_trading_days(start_date: dt.date, n_days: int, holidays=None):
    """
    Menghasilkan n_days tanggal hari bursa setelah start_date:
    - Senin–Jumat
    - Holidays
    """
    if holidays is None:
        holidays = set()

    dates = []
    current = start_date
    while len(dates) < n_days:
        current += dt.timedelta(days=1)
        if current.weekday() < 5 and current not in holidays:
            dates.append(current)
    return dates


# APP

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# cache (untuk lazy load)
_models = {}
_feature_scalers = {}
_target_scalers = {}
_feature_cols_map = {}

def load_assets_for_symbol(symbol: str):
    """Lazy load model dan scaler untuk 1 saham"""
    if symbol in _models:
        return

    model_path = os.path.join(MODELS_DIR, f"{symbol}_best.h5")
    feat_scaler_path = os.path.join(SCALERS_DIR, f"{symbol}_feature_scaler.pkl")
    target_scaler_path = os.path.join(SCALERS_DIR, f"{symbol}_close_scaler.pkl")
    feature_cols_path = os.path.join(SCALERS_DIR, f"{symbol}_feature_cols.pkl")

    for p, label in [
        (model_path, "Model"),
        (feat_scaler_path, "Feature scaler"),
        (target_scaler_path, "Target scaler"),
        (feature_cols_path, "Feature cols"),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{label} file not found: {p}")

    _models[symbol] = load_model(model_path)
    _feature_scalers[symbol] = joblib.load(feat_scaler_path)
    _target_scalers[symbol] = joblib.load(target_scaler_path)
    _feature_cols_map[symbol] = joblib.load(feature_cols_path)

def forecast_future(
    df: pd.DataFrame,
    feature_scaler,
    target_scaler,
    model,
    feature_cols,
    days: int = DEFAULT_FORECAST_DAYS,
    lookback: int = LOOKBACK,
):
    
    """
    Forecast hari bursa ke depan
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom fitur tidak ditemukan di data Yahoo: {missing}")

    df_feat = df[feature_cols].copy().dropna()

    if len(df_feat) < lookback:
        raise ValueError(f"Data tidak cukup untuk lookback={lookback}. Tersedia: {len(df_feat)} baris.")

    scaled_features = feature_scaler.transform(df_feat.values)

    # window awal
    last_window = scaled_features[-lookback:, :]
    preds = []

    # indeks kolom Close dalam feature_cols
    if "Close" not in feature_cols:
        raise ValueError("feature_cols tidak mengandung 'Close'. Pastikan sama dengan training.")
    close_idx = feature_cols.index("Close")

    for _ in range(days):
        x_in = last_window[np.newaxis, ...]             
        y_scaled = model.predict(x_in, verbose=0)       

        # ke skala harga asli
        y = target_scaler.inverse_transform(y_scaled)[0, 0]
        preds.append(float(y))

        # geser window 1 langkah
        new_row = last_window[-1].copy()
        # masukkan Close yang sudah discale
        new_row[close_idx] = y_scaled[0, 0]
        last_window = np.vstack([last_window[1:], new_row])

    # tanggal prediksi: hari bursa
    last_date = df_feat.index[-1].date()
    trading_dates = generate_trading_days(last_date, days, holidays=ID_HOLIDAYS)

    return [{"date": d.isoformat(), "predicted_close": p} for d, p in zip(trading_dates, preds)]


# Routes
@app.route("/")
def home():
    return "Stock Predictor API (BiLSTM)"

@app.route("/predict/<symbol>", methods=["GET"])
def predict_symbol(symbol):
    symbol = symbol.upper()

    if symbol not in SUPPORTED_SYMBOLS:
        return jsonify({"error": f"Symbol {symbol} tidak didukung. Gunakan: {SUPPORTED_SYMBOLS}"}), 400

    try:
        load_assets_for_symbol(symbol)

        model = _models[symbol]
        feat_scaler = _feature_scalers[symbol]
        tgt_scaler = _target_scalers[symbol]
        feature_cols = _feature_cols_map[symbol]

        ticker_yf = symbol + ".JK"

        # Ambil periode 5 tahun
        df = yf.download(ticker_yf, period="5y", auto_adjust=False, progress=False).dropna()

        if df is None or df.empty:
            return jsonify({"error": f"Tidak ada data dari Yahoo Finance untuk {ticker_yf}"}), 500

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        days = DEFAULT_FORECAST_DAYS

        predictions = forecast_future(
            df=df,
            feature_scaler=feat_scaler,
            target_scaler=tgt_scaler,
            model=model,
            feature_cols=feature_cols,
            days=days,
            lookback=LOOKBACK,
        )

        return jsonify({"symbol": symbol, "days": days, "predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history/<symbol>", methods=["GET"])
def history(symbol):
    symbol = symbol.upper()

    if symbol not in SUPPORTED_SYMBOLS:
        return jsonify({"error": f"Symbol {symbol} tidak didukung. Gunakan: {SUPPORTED_SYMBOLS}"}), 400

    try:
        ticker_yf = symbol + ".JK"
        df = yf.download(ticker_yf, period="2y", auto_adjust=False, progress=False)

        if df is None or df.empty:
            return jsonify({"error": f"Tidak ada data untuk {ticker_yf}"}), 500

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()

        dates = df.index.strftime("%Y-%m-%d").tolist()
        close = df["Close"].astype(float).tolist()

        return jsonify({"symbol": symbol, "dates": dates, "close": close})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Main
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
