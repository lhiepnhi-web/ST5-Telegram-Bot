# ================================================================
# ST5 — V1.1 TELEGRAM SIGNAL BOT
# MSR + CII
# Signal only — NO ORDER EXECUTION
# ================================================================

import os
import json
import math
import requests
import pandas as pd
import numpy as np

from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

# ================================================================
# CONFIG — FROZEN ST5 V1.1
# ================================================================

TICKERS = ["MSR", "CII"]

VOLUME_RATIO_MIN = 1.0
ROC10_MIN = 2.0
ADX14_MIN = 30.0

INITIAL_CAPITAL = 1_000_000_000
MAX6 = 6

BUY_FEE = 0.0015
SELL_FEE = 0.0015
SELL_TAX = 0.0010

TIMEZONE = "Asia/Ho_Chi_Minh"

# Telegram secrets
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# State
STATE_FILE = Path("bot_state.json")

# ================================================================
# DATA SOURCE
# ================================================================

VNSTOCK_SOURCE = os.getenv("VNSTOCK_SOURCE", "KBS")

# ================================================================
# TIME WINDOW
# ================================================================

def vietnam_now():
    return datetime.now(ZoneInfo(TIMEZONE))


def in_signal_window(now=None):
    if now is None:
        now = vietnam_now()

    # Monday = 0 ... Sunday = 6
    if now.weekday() > 4:
        return False

    t = now.hour * 60 + now.minute

    morning_start = 9 * 60 + 15
    morning_end = 11 * 60 + 30

    afternoon_start = 13 * 60
    afternoon_end = 14 * 60 + 30

    return (
        morning_start <= t <= morning_end
        or
        afternoon_start <= t <= afternoon_end
    )


# ================================================================
# TELEGRAM
# ================================================================

def telegram_send(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("TELEGRAM: secrets chưa được cấu hình.")
        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        r = requests.post(
            url,
            json=payload,
            timeout=20
        )

        if r.status_code != 200:
            print("Telegram ERROR:", r.status_code, r.text)
            return False

        print("Telegram SENT")
        return True

    except Exception as e:
        print("Telegram exception:", e)
        return False


# ================================================================
# STATE
# ================================================================

def load_state():

    if not STATE_FILE.exists():

        state = {
            "version": "ST5_V1.1",
            "positions": {
                "MSR": False,
                "CII": False
            },
            "last_signal": {
                "MSR": "WAIT",
                "CII": "WAIT"
            },
            "last_signal_date": {
                "MSR": None,
                "CII": None
            }
        }

        save_state(state)
        return state

    try:

        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        # Repair missing fields
        state.setdefault("positions", {})
        state.setdefault("last_signal", {})
        state.setdefault("last_signal_date", {})

        for ticker in TICKERS:
            state["positions"].setdefault(ticker, False)
            state["last_signal"].setdefault(ticker, "WAIT")
            state["last_signal_date"].setdefault(ticker, None)

        return state

    except Exception as e:

        print("STATE ERROR:", e)

        return {
            "version": "ST5_V1.1",
            "positions": {x: False for x in TICKERS},
            "last_signal": {x: "WAIT" for x in TICKERS},
            "last_signal_date": {x: None for x in TICKERS}
        }


def save_state(state):

    tmp = STATE_FILE.with_suffix(".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            state,
            f,
            ensure_ascii=False,
            indent=2
        )

    tmp.replace(STATE_FILE)


# ================================================================
# WILDER RMA
# ================================================================

def rma(series, period):

    return series.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


# ================================================================
# INDICATORS
# ================================================================

def calculate_indicators(df):

    df = df.copy()

    # ------------------------------------------------------------
    # Volume Ratio
    # ------------------------------------------------------------

    df["Volume_Ratio"] = (
        df["volume"] /
        df["volume"].rolling(
            20,
            min_periods=20
        ).mean().shift(1)
    )

    # ------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------

    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_Hist"] = macd - signal

    # ------------------------------------------------------------
    # ROC10
    # ------------------------------------------------------------

    df["ROC10"] = (
        (df["close"] / df["close"].shift(10)) - 1
    ) * 100

    # ------------------------------------------------------------
    # WILDER ADX14
    # Same implementation used in CII Steps 1-4
    # ------------------------------------------------------------

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (df["high"] - prev_close).abs()

    tr3 = (df["low"] - prev_close).abs()

    df["TR"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    up_move = df["high"].diff()

    down_move = -df["low"].diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0
    )

    atr = rma(df["TR"], 14)

    plus_di = (
        100 *
        rma(pd.Series(plus_dm, index=df.index), 14)
        / atr
    )

    minus_di = (
        100 *
        rma(pd.Series(minus_dm, index=df.index), 14)
        / atr
    )

    dx = (
        100 *
        (plus_di - minus_di).abs()
        /
        (plus_di + minus_di)
    )

    df["ADX14"] = rma(dx, 14)

    return df


# ================================================================
# DATA NORMALIZATION
# ================================================================

def normalize_data(df):

    df = df.copy()

    rename_map = {}

    for c in df.columns:

        cl = str(c).lower().strip()

        if cl in ["time", "date", "tradingdate", "trading_date"]:
            rename_map[c] = "date"

        elif cl == "open":
            rename_map[c] = "open"

        elif cl == "high":
            rename_map[c] = "high"

        elif cl == "low":
            rename_map[c] = "low"

        elif cl == "close":
            rename_map[c] = "close"

        elif cl == "volume":
            rename_map[c] = "volume"

    df = df.rename(columns=rename_map)

    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        x for x in required
        if x not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df[required].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna()

    df = (
        df.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    return df


# ================================================================
# FETCH DATA
# ================================================================

def fetch_ticker(ticker):

    print("=" * 70)
    print(f"FETCH {ticker}")

    try:

        from vnstock import Quote

        quote = Quote(
            source=VNSTOCK_SOURCE,
            symbol=ticker
        )

        df = quote.history(
            length="3Y",
            interval="1D"
        )

        if df is None or df.empty:
            raise RuntimeError(
                f"{ticker}: empty dataframe"
            )

        df = normalize_data(df)

        print(
            f"{ticker}: {len(df)} rows | "
            f"{df['date'].min().date()} -> "
            f"{df['date'].max().date()}"
        )

        return df

    except Exception as e:

        raise RuntimeError(
            f"{ticker} data error: {e}"
        )


# ================================================================
# SIGNAL
# ================================================================

def evaluate_signal(df):

    df = calculate_indicators(df)

    row = df.iloc[-1]

    volume_ok = (
        pd.notna(row["Volume_Ratio"])
        and row["Volume_Ratio"] > VOLUME_RATIO_MIN
    )

    macd_ok = (
        pd.notna(row["MACD_Hist"])
        and row["MACD_Hist"] > 0
    )

    roc_ok = (
        pd.notna(row["ROC10"])
        and row["ROC10"] > ROC10_MIN
    )

    adx_ok = (
        pd.notna(row["ADX14"])
        and row["ADX14"] > ADX14_MIN
    )

    entry_valid = (
        volume_ok
        and macd_ok
        and roc_ok
        and adx_ok
    )

    return {
        "date": row["date"],
        "close": float(row["close"]),
        "volume_ratio": float(row["Volume_Ratio"])
            if pd.notna(row["Volume_Ratio"]) else None,
        "macd_hist": float(row["MACD_Hist"])
            if pd.notna(row["MACD_Hist"]) else None,
        "roc10": float(row["ROC10"])
            if pd.notna(row["ROC10"]) else None,
        "adx14": float(row["ADX14"])
            if pd.notna(row["ADX14"]) else None,
        "volume_ok": volume_ok,
        "macd_ok": macd_ok,
        "roc_ok": roc_ok,
        "adx_ok": adx_ok,
        "entry_valid": entry_valid
    }


# ================================================================
# TELEGRAM MESSAGE
# ================================================================

def format_signal(ticker, result, action):

    dt = vietnam_now().strftime("%d/%m/%Y %H:%M:%S")

    if action == "BUY":

        title = "🟢 <b>ST5 — TÍN HIỆU MUA</b>"

    elif action == "EXIT":

        title = "🔴 <b>ST5 — TÍN HIỆU THOÁT</b>"

    else:

        title = "ℹ️ <b>ST5 — SIGNAL</b>"

    return f"""
{title}

<b>Mã:</b> {ticker}
<b>Giá:</b> {result['close']:.2f}
<b>Thời gian:</b> {dt}

<b>V1.1:</b>
Volume Ratio: {result['volume_ratio']:.4f} {'✅' if result['volume_ok'] else '❌'}
MACD Hist: {result['macd_hist']:.6f} {'✅' if result['macd_ok'] else '❌'}
ROC10: {result['roc10']:.4f} {'✅' if result['roc_ok'] else '❌'}
ADX14: {result['adx14']:.4f} {'✅' if result['adx_ok'] else '❌'}

<b>Quyết định:</b> {action}

ST5 V1.1 — Signal Only
Không đặt lệnh tự động.
""".strip()


# ================================================================
# MAIN
# ================================================================

def main():

    now = vietnam_now()

    print("=" * 70)
    print("ST5 V1.1 — MSR + CII TELEGRAM SIGNAL BOT")
    print("=" * 70)

    print("Vietnam time:", now.strftime(
        "%Y-%m-%d %H:%M:%S"
    ))

    print("Weekday:", now.strftime("%A"))

    # ------------------------------------------------------------
    # TIME FILTER
    # ------------------------------------------------------------

    if not in_signal_window(now):

        print("OUTSIDE SIGNAL WINDOW")
        print("No Telegram signal.")
        return

    print("SIGNAL WINDOW: ACTIVE")

    # ------------------------------------------------------------
    # LOAD STATE
    # ------------------------------------------------------------

    state = load_state()

    # ------------------------------------------------------------
    # PROCESS BOTH TICKERS
    # ------------------------------------------------------------

    for ticker in TICKERS:

        try:

            df = fetch_ticker(ticker)

            result = evaluate_signal(df)

            print("-" * 70)
            print(ticker)

            print(
                "Date:",
                result["date"]
            )

            print(
                "Close:",
                result["close"]
            )

            print(
                "Volume Ratio:",
                result["volume_ratio"]
            )

            print(
                "MACD Hist:",
                result["macd_hist"]
            )

            print(
                "ROC10:",
                result["roc10"]
            )

            print(
                "ADX14:",
                result["adx14"]
            )

            print(
                "ENTRY:",
                result["entry_valid"]
            )

            current_position = bool(
                state["positions"].get(
                    ticker,
                    False
                )
            )

            # ----------------------------------------------------
            # ENTRY
            # ----------------------------------------------------

            if (
                result["entry_valid"]
                and not current_position
            ):

                action = "BUY"

                message = format_signal(
                    ticker,
                    result,
                    action
                )

                if telegram_send(message):

                    state["positions"][ticker] = True
                    state["last_signal"][ticker] = "BUY"
                    state["last_signal_date"][ticker] = (
                        str(result["date"].date())
                    )

                    save_state(state)

                    print(
                        ticker,
                        "BUY SIGNAL SENT"
                    )

            # ----------------------------------------------------
            # EXIT
            # ----------------------------------------------------

            elif (
                not result["entry_valid"]
                and current_position
            ):

                action = "EXIT"

                message = format_signal(
                    ticker,
                    result,
                    action
                )

                if telegram_send(message):

                    state["positions"][ticker] = False
                    state["last_signal"][ticker] = "EXIT"
                    state["last_signal_date"][ticker] = (
                        str(result["date"].date())
                    )

                    save_state(state)

                    print(
                        ticker,
                        "EXIT SIGNAL SENT"
                    )

            # ----------------------------------------------------
            # WAIT
            # ----------------------------------------------------

            else:

                action = (
                    "HOLD"
                    if current_position
                    else "WAIT"
                )

                print(
                    ticker,
                    "->",
                    action
                )

                state["last_signal"][ticker] = action
                state["last_signal_date"][ticker] = (
                    str(result["date"].date())
                )

                save_state(state)

        except Exception as e:

            print(
                f"ERROR {ticker}:",
                repr(e)
            )

    # ------------------------------------------------------------
    # FINAL STATE
    # ------------------------------------------------------------

    print("=" * 70)
    print("FINAL STATE")

    for ticker in TICKERS:

        print(
            ticker,
            "| position =",
            state["positions"].get(ticker),
            "| last =",
            state["last_signal"].get(ticker)
        )

    print("=" * 70)


if __name__ == "__main__":
    main() 
