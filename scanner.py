"""
FalseMove_3Vecteurs - version Python / GitHub Actions
------------------------------------------------------------------
Reproduit la logique de l'indicateur MT4 du meme nom, sans MT4 :
  - Recupere du M15 sur Yahoo Finance (limite yfinance: 60 jours d'historique)
  - Detecte les fractales (sommets/creux) avec une profondeur configurable
  - Reset quotidien : la sequence ne remonte jamais avant le jour en cours
    (identique au script MT4 qui casse la boucle des qu'on change de jour)
  - Reconstruit la sequence V1 -> N1 -> V2 (depasse V1) -> N2 (plus
    profonde que N1) -> V3 (depasse V2) = SIGNAL, separement en
    haussier et en baissier
  - Envoie une alerte Telegram intermediaire des que V2 casse la neckline
    de V1 (entree en stage 3), et une alerte finale au signal complet (stage 5)
  - Deduplique les alertes via un fichier d'etat (state.json), une entree
    par paire+sens+jour pour ne pas re-notifier a chaque execution
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

FRACTAL_DEPTH = 2          # equivalent de FractalDepth dans le .mq4
M15_HISTORY_PERIOD = "60d" # limite yfinance pour l'intervalle 15m

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")


def to_yahoo_symbol(pair: str) -> str:
    return f"{pair}=X"


def fetch_m15(pair: str) -> pd.DataFrame:
    ticker = to_yahoo_symbol(pair)
    df = yf.download(
        ticker,
        period=M15_HISTORY_PERIOD,
        interval="15m",
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
    return df


def today_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Ne garde que les bougies du jour calendaire (UTC) de la derniere bougie,
    equivalent au reset quotidien du .mq4 (IsSameDay avec la bougie en formation)."""
    if df.empty:
        return df
    last_day = df.index[-1].date()
    return df[df.index.date == last_day]


def find_fractals(highs: np.ndarray, lows: np.ndarray, depth: int):
    """Retourne deux tableaux booleens (fractal haut / fractal bas) sur des
    bougies en ordre chronologique ascendant (index 0 = plus ancien)."""
    n = len(highs)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    for i in range(depth, n - depth):
        h = highs[i]
        if all(highs[i - k] < h for k in range(1, depth + 1)) and \
           all(highs[i + k] < h for k in range(1, depth + 1)):
            is_high[i] = True
        l = lows[i]
        if all(lows[i - k] > l for k in range(1, depth + 1)) and \
           all(lows[i + k] > l for k in range(1, depth + 1)):
            is_low[i] = True
    return is_high, is_low


def analyze_pattern(day_df: pd.DataFrame, bullish: bool, depth: int) -> dict:
    """Reproduit AnalyzePattern() du .mq4 sur les bougies du jour en cours.
    Retourne stage (0-5), les niveaux V1/N1/V2/N2 et un libelle."""
    result = {"stage": 0, "v1": None, "n1": None, "v2": None, "n2": None,
              "text": "Recherche V1..."}

    highs = day_df["high"].values
    lows = day_df["low"].values
    n = len(highs)
    if n < depth * 2 + 1:
        return result

    is_frac_high, is_frac_low = find_fractals(highs, lows, depth)
    extreme_mask = is_frac_high if bullish else is_frac_low
    neck_mask = is_frac_low if bullish else is_frac_high

    # bougie en cours = derniere du jour ; on ne considere comme "visibles"
    # que les fractales confirmees (il faut `depth` bougies apres elles)
    last_confirmable = n - 1 - depth
    extreme_ks = [k for k in range(last_confirmable, -1, -1) if extreme_mask[k]]
    neck_ks = [k for k in range(last_confirmable, -1, -1) if neck_mask[k]]

    if not extreme_ks:
        return result

    v2_k = extreme_ks[0]
    v2 = highs[v2_k] if bullish else lows[v2_k]

    v1_k = next((k for k in extreme_ks[1:] if k < v2_k), None)
    if v1_k is None:
        result.update(stage=1, v1=v2, text="V1 forme, attente neckline")
        return result
    v1 = highs[v1_k] if bullish else lows[v1_k]

    n1_k = next((k for k in neck_ks if v1_k < k < v2_k), None)
    if n1_k is None:
        result.update(stage=1, v1=v1, text="V1 forme, attente neckline")
        return result
    n1 = lows[n1_k] if bullish else highs[n1_k]
    result.update(stage=2, v1=v1, n1=n1, text="Neckline V1 cassee, recherche V2")

    beyond = (v2 > v1) if bullish else (v2 < v1)
    if not beyond:
        result["text"] = "V2 ne depasse pas V1 (invalide)"
        return result
    result.update(stage=3, v2=v2, text="V2 casse la neckline V1 - N2 ?")

    n2_k = next((k for k in neck_ks if k > v2_k), None)
    if n2_k is None:
        return result
    n2 = lows[n2_k] if bullish else highs[n2_k]

    deeper = (n2 < n1) if bullish else (n2 > n1)
    if not deeper:
        result["text"] = "N2 pas assez profonde vs N1"
        return result
    result.update(stage=4, n2=n2, text="Neckline V2 cassee - attente V3")

    price = day_df["close"].iloc[-1]
    if (price > v2) if bullish else (price < v2):
        result.update(stage=5, text="SIGNAL: V3 depasse V2 !")

    return result


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
    df = fetch_m15(pair)
    min_bars = FRACTAL_DEPTH * 2 + 10
    if len(df) < min_bars:
        print(f"{pair}: donnees insuffisantes ({len(df)} bougies M15)")
        return

    day_df = today_bars(df)
    if len(day_df) < FRACTAL_DEPTH * 2 + 1:
        print(f"{pair}: pas assez de bougies aujourd'hui ({len(day_df)})")
        return

    day_key = day_df.index[-1].date().isoformat()
    bar_time = day_df.index[-1].isoformat()

    for bullish in (True, False):
        ps = analyze_pattern(day_df, bullish, FRACTAL_DEPTH)
        sens = "haussier" if bullish else "baissier"
        print(f"{pair} [{sens}]: stage={ps['stage']} - {ps['text']}")
        price = day_df["close"].iloc[-1]

        # --- Alerte intermediaire : V2 vient de casser la neckline de V1 (entree en stage 3) ---
        if ps["stage"] >= 3:
            key_s3 = f"{pair}_{sens}_{day_key}_neckline_v1"
            if not state.get(key_s3):
                state[key_s3] = bar_time
                msg = (
                    f"{pair} M15 : Neckline V1 cassee par V2 ({sens.upper()})\n"
                    f"V1={ps['v1']:.5f}  N1={ps['n1']:.5f}\n"
                    f"V2={ps['v2']:.5f}\n"
                    f"Prix actuel={price:.5f}\n"
                    f"Bougie: {bar_time}\n"
                    f"-> Recherche de N2 en cours (etape 3/5)"
                )
                send_telegram(msg)

        # --- Alerte finale : signal complet (V3 depasse V2) ---
        if ps["stage"] == 5:
            key = f"{pair}_{sens}_{day_key}"
            if state.get(key) != bar_time:
                state[key] = bar_time
                msg = (
                    f"{pair} M15 : SIGNAL False Move 3 Vecteurs {sens.upper()}\n"
                    f"V1={ps['v1']:.5f}  N1={ps['n1']:.5f}\n"
                    f"V2={ps['v2']:.5f}  N2={ps['n2']:.5f}\n"
                    f"Prix actuel={price:.5f} (depasse V2)\n"
                    f"Bougie: {bar_time}"
                )
                send_telegram(msg)


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
