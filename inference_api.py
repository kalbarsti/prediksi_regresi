from flask import Flask, jsonify
import pandas as pd
import yfinance as yf

import os
import json
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from flask import Flask, jsonify
from tensorflow.keras.models import load_model

# =========================
# KONFIGURASI DASAR
# =========================

# Folder model & scaler (relatif terhadap root project)
MODELS_DIR = "models"
SCALERS_DIR = "scalers"

# Ticker yang di-support
SUPPORTED_SYMBOLS = ["BBCA", "BBRI", "BBNI", "BMRI"]

# Lookback window (HARUS sama dengan waktu training)
LOOKBACK = 60

# Banyak hari prediksi default
DEFAULT_FORECAST_DAYS = int(os.environ.get("FORECAST_DAYS", 7))

# =========================
# KALENDER HARI BURSA IDX
# =========================

# hari libur BEI (opsional)
# Format: dt.date(tahun, bulan, hari)
ID_HOLIDAYS = {
    # Contoh:
    # dt.date(2025, 1, 1),   # Tahun Baru
    # dt.date(2025, 3, 31),  # Libur keagamaan (contoh)
}

def generate_trading_days(start_date: dt.date, n_days: int, holidays=None):
    """
    Menghasilkan n_days tanggal HARI BURSA setelah start_date:
    - Hanya Senin–Jumat (weekday 0–4)
    - Tidak termasuk tanggal di 'holidays'
    """
    if holidays is None:
        holidays = set()

    dates = []
    current = start_date
    while len(dates) < n_days:
        current += dt.timedelta(days=1)
        # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
        if current.weekday() < 5 and current not in holidays:
            dates.append(current)
    return dates


# =========================
# LOAD MODEL & SCALER
# =========================

app = Flask(__name__)

models = {}
feature_scalers = {}
target_scalers = {}
feature_cols_map = {}


def load_assets_for_symbol(symbol: str):
    """
    Lazy load: model, scaler, dan daftar kolom fitur untuk 1 saham.
    Dipanggil pertama kali saat /predict/<symbol>.
    """
    if symbol in models:
        return  # sudah pernah di-load

    # --- path file ---
    model_path = os.path.join(MODELS_DIR, f"{symbol}_best.h5")
    feat_scaler_path = os.path.join(SCALERS_DIR, f"{symbol}_feature_scaler.pkl")
    target_scaler_path = os.path.join(SCALERS_DIR, f"{symbol}_close_scaler.pkl")
    feature_cols_path = os.path.join(SCALERS_DIR, f"{symbol}_feature_cols.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(feat_scaler_path):
        raise FileNotFoundError(f"Feature scaler not found: {feat_scaler_path}")
    if not os.path.exists(target_scaler_path):
        raise FileNotFoundError(f"Target scaler not found: {target_scaler_path}")
    if not os.path.exists(feature_cols_path):
        raise FileNotFoundError(f"Feature cols file not found: {feature_cols_path}")

    # --- load aset ---
    models[symbol] = load_model(model_path)
    feature_scalers[symbol] = joblib.load(feat_scaler_path)
    target_scalers[symbol] = joblib.load(target_scaler_path)

    #  pakai joblib.load
    feature_cols_map[symbol] = joblib.load(feature_cols_path)


# =========================
# FUNGSI FORECAST
# =========================

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
    Menghasilkan prediksi 'days' ke depan + tanggal HARI BURSA (bukan weekend/libur).
    """

    # --- Ambil fitur & scaling sama seperti saat training ---
    df_feat = df[feature_cols].copy()
    scaled_features = feature_scaler.transform(df_feat.values)

    # Window awal: lookback hari terakhir
    last_window = scaled_features[-lookback:, :]
    preds = []

    # Index kolom 'Close' di dalam feature_cols
    close_idx = feature_cols.index("Close")

    # --- Autoregressive forecasting ---
    for _ in range(days):
        x_in = last_window[np.newaxis, ...]  # shape: (1, lookback, n_features)
        y_scaled = model.predict(x_in, verbose=0)  # shape: (1, 1)
        # balik ke skala harga asli
        y = target_scaler.inverse_transform(y_scaled)[0, 0]
        preds.append(float(y))

        # update window: geser 1 langkah dan masukkan prediksi terbaru
        new_row = last_window[-1].copy()
        new_row[close_idx] = y_scaled[0, 0]
        last_window = np.vstack([last_window[1:], new_row])

    # --- Tanggal prediksi: hanya hari bursa (Mon–Fri, skip libur) ---
    last_date = df.index[-1].date()  # tanggal data terakhir dari Yahoo Finance
    trading_dates = generate_trading_days(
        start_date=last_date,
        n_days=days,
        holidays=ID_HOLIDAYS,
    )

    results = [
        {
            "date": d.isoformat(),
            "predicted_close": p,
        }
        for d, p in zip(trading_dates, preds)
    ]
    return results


# =========================
# ROUTES FLASK
# =========================

@app.route("/")
def home():
    return "Stock Predictor API (BiLSTM High Precision)"


@app.route("/predict/<symbol>", methods=["GET"])
def predict_symbol(symbol):
    symbol = symbol.upper()

    if symbol not in SUPPORTED_SYMBOLS:
        return (
            jsonify(
                {
                    "error": f"Symbol {symbol} tidak didukung. Gunakan salah satu dari {SUPPORTED_SYMBOLS}"
                }
            ),
            400,
        )

    try:
        # ---- Load model & scaler untuk symbol ini (lazy load) ----
        load_assets_for_symbol(symbol)

        model = models[symbol]
        feat_scaler = feature_scalers[symbol]
        tgt_scaler = target_scalers[symbol]
        feature_cols = feature_cols_map[symbol]

        # ---- Ambil data OHLCV terbaru dari Yahoo Finance ----
        ticker_yf = symbol + ".JK"  # asumsi di BEI
        # 5y cukup panjang untuk training & updating
        df = yf.download(ticker_yf, period="5y", auto_adjust=False).dropna()

        if df.empty:
            return (
                jsonify({"error": f"Tidak ada data dari Yahoo Finance untuk {ticker_yf}"}),
                500,
            )

        # Pastikan index datetime (kalau belum)
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

        response = {
            "symbol": symbol,
            "days": days,
            "predictions": predictions,
        }
        return jsonify(response)

    except Exception as e:
        # Untuk debugging; di produksi sebaiknya lebih dibatasi
        return jsonify({"error": str(e)}), 500


# =========================
@app.route("/history/<symbol>", methods=["GET"])
def history(symbol):
    symbol = symbol.upper()
    ticker_yf = symbol + ".JK"

    df = yf.download(ticker_yf, period="2y", auto_adjust=False).dropna()
    if df.empty:
        return jsonify({"error": f"Tidak ada data untuk {ticker_yf}"}), 500

    df = df.reset_index()
    dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()
    close = df["Close"].tolist()

    return jsonify({"symbol": symbol, "dates": dates, "close": close})
# =========================

# =========================
# ENTRY POINT (LOCAL)
# =========================

if __name__ == "__main__":
    # Untuk running lokal (python inference_api.py)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
