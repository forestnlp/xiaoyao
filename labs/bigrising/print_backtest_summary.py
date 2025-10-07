import os
import pandas as pd
import numpy as np

LABS_DIR = r"d:\workspace\xiaoyao\labs\bigrising"
TRADES_PATH = os.path.join(LABS_DIR, "backtest_trades.csv")

def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    wins = trades[trades["return"] > 0]["return"]
    losses = trades[trades["return"] <= 0]["return"]

    win_rate = len(wins) / len(trades)
    avg_return = trades["return"].mean()
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = losses.mean() if not losses.empty else 0.0
    gross_profit = wins.sum()
    gross_loss = -losses.sum()  # losses are negative
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

    return {
        "num_trades": int(len(trades)),
        "win_rate": float(win_rate),
        "avg_return": float(avg_return),
        "profit_factor": float(profit_factor),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
    }


def main():
    if not os.path.exists(TRADES_PATH):
        print(f"[ERROR] Trades file not found: {TRADES_PATH}")
        return

    trades = pd.read_csv(TRADES_PATH)
    summary = summarize_trades(trades)

    print("--- Backtest Summary ---")
    print(f"Trades: {summary['num_trades']}")
    print(f"Win rate: {summary['win_rate']:.2%}")
    print(f"Avg return: {summary['avg_return']:.2%}")
    print(f"Avg win: {summary['avg_win']:.2%}")
    print(f"Avg loss: {summary['avg_loss']:.2%}")
    print(f"Profit factor: {summary['profit_factor']:.2f}")


if __name__ == "__main__":
    main()