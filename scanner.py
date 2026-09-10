"""
MultiScan_Engulfing_HH_HL_LH_LL - version Python / GitHub Actions
------------------------------------------------------------------
Reproduit la logique de l'indicateur MT4 du meme nom, sans MT4 :
  - Recupere du H1 sur Yahoo Finance et le resample en H4 (UTC)
  - Detecte les swings (fractales HH/HL/LH/LL)
  - Detecte les bougies engulfing sur la derniere bougie H4 cloturee
  - Verifie si l'engulfing tombe dans une zone de structure (tolerance ATR)
  - Envoie une alerte Telegram si un signal valide apparait
  - Deduplique les alertes via un fichier d'etat (state.json)
"""

import os
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
    "AUDUSD", "NZDUSD", "EURGBP", "EURJPY", "EURCHF",
    "GBPJPY", "AUDJPY", "CHFJPY", "EURAUD", "EURCAD",
]

FRACTAL_WING = 2
SWING_LOOKBACK_BARS = 200
ATR_PERIOD = 14
ZONE_ATR_MULTIPLIER = 0.5
H1_HISTORY_PERIOD = "60d"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")


def to_yahoo_symbol(pair: str) -> str:
    return f"{pair}=X"


def fetch_h4(pair: str) -> pd.DataFrame:
    ticker = to_yahoo_symbol(pair)
    df = yf.download(
        ticker,
        period=H1_HISTORY_PERIOD,
        interval="60m",
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]]

    h4 = (
        df.resample("4h", origin="epoch")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )

    now = datetime.now(timezone.utc)
    if len(h4) and (h4.index[-1] + pd.Timedelta(hours=4)) > now:
        h4 = h4.iloc[:-1]

    return h4.tail(SWING_LOOKBACK_BARS + FRACTAL_WING + ATR_PERIOD + 5)


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def find_swings(df: pd.DataFrame, wing: int):
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_highs, swing_lows = [], []
    for i in range(wing, n - wing):
        left_h, right_h = highs[i - wing:i], highs[i + 1:i + 1 + wing]
        if highs[i] > left_h.max() and highs[i] > right_h.max():
            swing_highs.append((i, highs[i], df.index[i]))

        left_l, right_l = lows[i - wing:i], lows[i + 1:i + 1 + wing]
        if lows[i] < left_l.min() and lows[i] < right_l.min():
            swing_lows.append((i, lows[i], df.index[i]))

    return swing_highs, swing_lows


def classify_structure(df: pd.DataFrame, wing: int):
    swing_highs, swing_lows = find_swings(df, wing)

    last_high_label, last_low_label = "-", "-"
    last_swing_high, last_swing_low = None, None
    last_high_idx, last_low_idx = -1, -1

    if len(swing_highs) >= 2:
        last_high_label = "HH" if swing_highs[-1][1] > swing_highs[-2][1] else "LH"
    if len(swing_lows) >= 2:
        last_low_label = "HL" if swing_lows[-1][1] > swing_lows[-2][1] else "LL"

    if swing_highs:
        last_swing_high = swing_highs[-1][1]
        last_high_idx = swing_highs[-1][0]
    if swing_lows:
        last_swing_low = swing_lows[-1][1]
        last_low_idx = swing_lows[-1][0]

    if not swing_highs and not swing_lows:
        structure = "-"
    elif last_high_idx >= last_low_idx:
        structure = last_high_label
    else:
        structure = last_low_label

    return structure, last_swing_high, last_swing_low, last_high_label, last_low_label


def detect_engulfing(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return 0

    o1, c1 = df["open"].iloc[-2], df["close"].iloc[-2]
    o0, c0 = df["open"].iloc[-1], df["close"].iloc[-1]

    prev_bear, prev_bull = c1 < o1, c1 > o1
    curr_bull, curr_bear = c0 > o0, c0 < o0

    if prev_bear and curr_bull and o0 <= c1 and c0 >= o1:
        return 1
    if prev_bull and curr_bear and o0 >= c1 and c0 <= o1:
        return -1
    return 0


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True, default=str)


def send_telegram(message: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID manquants - alerte non envoyee.")
        print(message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=15)
        if resp.status_code != 200:
            print(f"[ERREUR] Telegram a repondu {resp.status_code}: {resp.text}")
    except requests.RequestException as exc:
        print(f"[ERREUR] Envoi Telegram echoue: {exc}")


def scan_pair(pair: str, state: dict) -> None:
    df = fetch_h4(pair)
    min_bars = FRACTAL_WING * 2 + ATR_PERIOD + 5
    if len(df) < min_bars:
        print(f"{pair}: donnees insuffisantes ({len(df)} bougies H4)")
        return

    structure, last_high, last_low, high_label, low_label = classify_structure(df, FRACTAL_WING)
    engulf = detect_engulfing(df)

    df = df.copy()
    df["atr"] = compute_atr(df, ATR_PERIOD)
    atr_val = df["atr"].iloc[-1]
    ref_price = df["close"].iloc[-1]
    bar_time = df.index[-1].isoformat()

    print(f"{pair}: structure={structure} engulfing={engulf} close={ref_price:.5f}")

    if pd.isna(atr_val):
        return

    if engulf == 1 and last_low is not None and abs(ref_price - last_low) <= atr_val * ZONE_ATR_MULTIPLIER:
        key = f"{pair}_bull"
        if state.get(key) != bar_time:
            state[key] = bar_time
            send_telegram(f"{pair} H4 : Engulfing HAUSSIER dans zone {low_label} (bougie {bar_time})")

    elif engulf == -1 and last_high is not None and abs(ref_price - last_high) <= atr_val * ZONE_ATR_MULTIPLIER:
        key = f"{pair}_bear"
        if state.get(key) != bar_time:
            state[key] = bar_time
            send_telegram(f"{pair} H4 : Engulfing BAISSIER dans zone {high_label} (bougie {bar_time})")


def main() -> None:
    state = load_state()
    for pair in PAIRS:
        try:
            scan_pair(pair, state)
        except Exception as exc:
            print(f"[ERREUR] {pair}: {exc}")
    save_state(state)


if __name__ == "__main__":
    main()
