"""
Gold & Silver 7-Day Price Forecast
-----------------------------------
Fetches REAL historical data (Yahoo Finance futures: GC=F gold, SI=F silver),
trains an LSTM per metal, and forecasts the next 7 trading days.
Outputs a JSON file your Next.js frontend can fetch and render.

Run:
    python gold_silver_forecast.py
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
TICKERS = {"gold": "GC=F", "silver": "SI=F"}
LOOKBACK = 60          # days of history the model looks at per prediction
FORECAST_DAYS = 7
EPOCHS = 50
BATCH_SIZE = 16

# ---------------------------------------------------------------
# 1. FETCH REAL DATA
# ---------------------------------------------------------------
def fetch_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="5y", interval="1d", progress=False)
    df = df[["Close"]].dropna()
    df.columns = ["close"]
    return df

def fetch_usd_inr_rate() -> float:
    """Live USD->INR rate. Falls back to a fixed approximate rate if the fetch fails."""
    try:
        rate_df = yf.download("USDINR=X", period="5d", interval="1d", progress=False)
        return float(rate_df["Close"].dropna().iloc[-1])
    except Exception:
        return 83.0

# ---------------------------------------------------------------
# 2. PREPARE SEQUENCES
# ---------------------------------------------------------------
def make_sequences(values: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i, 0])
        y.append(values[i, 0])
    return np.array(X), np.array(y)

# ---------------------------------------------------------------
# 3. BUILD MODEL
# ---------------------------------------------------------------
def build_model(lookback: int) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# ---------------------------------------------------------------
# 4. TRAIN + EVALUATE
# ---------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values)

    X, y = make_sequences(scaled, LOOKBACK)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.9)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model(LOOKBACK)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0,
    )

    # Evaluation metrics (RMSE, MAPE) on test split
    preds = model.predict(X_test, verbose=0)
    preds_actual = scaler.inverse_transform(preds)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = float(np.sqrt(np.mean((preds_actual - y_test_actual) ** 2)))
    mape = float(np.mean(np.abs((y_test_actual - preds_actual) / y_test_actual)) * 100)

    return model, scaler, rmse, mape

# ---------------------------------------------------------------
# 5. FORECAST NEXT N DAYS (recursive)
# ---------------------------------------------------------------
def forecast_future(model, scaler, df: pd.DataFrame, days: int):
    scaled = scaler.transform(df.values)
    window = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)

    preds_scaled = []
    for _ in range(days):
        next_pred = model.predict(window, verbose=0)[0, 0]
        preds_scaled.append(next_pred)
        window = np.append(window[:, 1:, :], [[[next_pred]]], axis=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    last_date = df.index[-1]
    future_dates = []
    d = last_date
    while len(future_dates) < days:
        d = d + timedelta(days=1)
        if d.weekday() < 5:  # skip weekends, markets closed
            future_dates.append(d)

    return [
        {"date": dt.strftime("%Y-%m-%d"), "predicted_price": round(float(p), 2)}
        for dt, p in zip(future_dates, preds)
    ]

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    output = {"generated_at": datetime.utcnow().isoformat(), "currency": "INR", "metals": {}}
    usd_to_inr = fetch_usd_inr_rate()
    print(f"USD -> INR rate: {usd_to_inr:.2f}")

    for name, ticker in TICKERS.items():
        print(f"\n=== {name.upper()} ({ticker}) ===")
        df = fetch_data(ticker)
        print(f"Fetched {len(df)} days of real historical data")

        model, scaler, rmse, mape = train_and_evaluate(df)
        print(f"Model performance -> RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")

        forecast = forecast_future(model, scaler, df, FORECAST_DAYS)

        # Convert everything from USD to INR using the live exchange rate
        current_price = round(float(df["close"].iloc[-1]) * usd_to_inr, 2)
        recent_history = [round(float(v) * usd_to_inr, 2) for v in df["close"].tail(13).tolist()]
        forecast_inr = [
            {"date": f["date"], "predicted_price": round(f["predicted_price"] * usd_to_inr, 2)}
            for f in forecast
        ]
        rmse_inr = round(rmse * usd_to_inr, 2)

        output["metals"][name] = {
            "ticker": ticker,
            "current_price": current_price,
            "model": "LSTM (60-day lookback)",
            "rmse": rmse_inr,
            "mape_percent": round(mape, 2),  # percentage error is currency-independent
            "history": recent_history,
            "forecast": forecast_inr,
        }

    with open("forecast_output.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved forecast_output.json — feed this to your frontend.")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
