import os
import pandas as pd
import numpy as np
import talib

LABS_DIR = r"d:\workspace\xiaoyao\labs\bigrising"
DATA_PATH = r"D:\workspace\xiaoyao\data\stock_daily_price.parquet"
TRADES_OUTPUT = os.path.join(LABS_DIR, "backtest_trades.csv")

# Backtest parameters
HOLD_DAYS = 10  # fixed holding period in trading days

REQUIRED_COLUMNS = {"stock_code", "date", "open", "high", "low", "close", "volume"}


def compute_indicators(stock_df: pd.DataFrame) -> pd.DataFrame:
    stock_df = stock_df.sort_values("date").copy()
    if len(stock_df) < 35:
        return pd.DataFrame()

    close = stock_df["close"].astype(float).values
    high = stock_df["high"].astype(float).values
    low = stock_df["low"].astype(float).values

    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    atr14 = talib.ATR(high, low, close, timeperiod=14)

    stock_df["macd"] = macd
    stock_df["macdsignal"] = macdsignal
    stock_df["macdhist"] = macdhist
    stock_df["bband_upper"] = upper
    stock_df["bband_middle"] = middle
    stock_df["bband_lower"] = lower
    stock_df["atr14"] = atr14

    stock_df.dropna(inplace=True)

    # Strategy v1 signal
    rule1 = (stock_df["macd"] > stock_df["macdsignal"]) & (stock_df["macd"] > 0)
    rule2 = stock_df["close"] > stock_df["bband_upper"] * 0.98
    rule3 = (stock_df["atr14"] / stock_df["close"]) < 0.05
    stock_df["signal_v1"] = (rule1 & rule2 & rule3)

    # Next day open for realistic entry
    stock_df["next_open"] = stock_df["open"].shift(-1)

    return stock_df


def backtest_signals(df: pd.DataFrame) -> pd.DataFrame:
    trades = []
    for code, g in df.groupby("stock_code"):
        g = compute_indicators(g)
        if g.empty:
            continue

        idxs = g.index.to_list()
        dates = g["date"].values
        for i in range(len(g)):
            if not g.iloc[i]["signal_v1"]:
                continue
            entry_date = dates[i]
            entry_open = g.iloc[i]["next_open"]
            if np.isnan(entry_open):
                continue

            exit_idx = i + HOLD_DAYS
            if exit_idx >= len(g):
                continue

            exit_date = dates[exit_idx]
            exit_close = g.iloc[exit_idx]["close"]
            ret = (exit_close / entry_open) - 1.0

            trades.append({
                "stock_code": code,
                "entry_date": pd.to_datetime(entry_date),
                "entry_price": float(entry_open),
                "exit_date": pd.to_datetime(exit_date),
                "exit_price": float(exit_close),
                "hold_days": HOLD_DAYS,
                "return": float(ret)
            })

    return pd.DataFrame(trades)


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
    print("=== Backtesting strategy_v1 (MACD + BOLL + ATR) ===")
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        return

    print("Loading parquet data...")
    df = pd.read_parquet(DATA_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        return

    # Ensure dtypes and order
    df["date"] = pd.to_datetime(df["date"])  # robust date parsing
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    print("Generating trades...")
    trades = backtest_signals(df)
    print(f"Generated {len(trades)} trades.")

    print("Summarizing results...")
    summary = summarize_trades(trades)

    # Save trades for inspection
    if len(trades):
        trades.to_csv(TRADES_OUTPUT, index=False)
        print(f"Trades saved to: {TRADES_OUTPUT}")

    print("--- Summary ---")
    print(f"Trades: {summary['num_trades']}")
    print(f"Win rate: {summary['win_rate']:.2%}")
    print(f"Avg return: {summary['avg_return']:.2%}")
    print(f"Avg win: {summary['avg_win']:.2%}")
    print(f"Avg loss: {summary['avg_loss']:.2%}")
    print(f"Profit factor: {summary['profit_factor']:.2f}")

    # Also write a summary file for quick inspection
    summary_path = os.path.join(LABS_DIR, "backtest_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("--- Backtest Summary ---\n")
        f.write(f"Trades: {summary['num_trades']}\n")
        f.write(f"Win rate: {summary['win_rate']:.2%}\n")
        f.write(f"Avg return: {summary['avg_return']:.2%}\n")
        f.write(f"Avg win: {summary['avg_win']:.2%}\n")
        f.write(f"Avg loss: {summary['avg_loss']:.2%}\n")
        f.write(f"Profit factor: {summary['profit_factor']:.2f}\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

import talib
import os

# --- Constants ---
DATA_DIR = r'D:\workspace\xiaoyao\data'
PRICE_DATA_FILENAME = 'stock_daily_price.parquet'
HOLDING_PERIOD = 10  # Days to hold the stock after a buy signal

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all necessary TA features for the backtest."""
    print("  - Calculating TA features for all stocks...")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['stock_code', 'date'])

    # Group by stock to calculate features independently
    grouped = df.groupby('stock_code')
    
    # Use a dictionary to hold new feature columns
    feature_cols = {}

    feature_cols['rsi_14'] = grouped['close'].transform(lambda x: talib.RSI(x, timeperiod=14))
    
    # MACD
    macd, macdsignal, macdhist = zip(*grouped['close'].apply(lambda x: talib.MACD(x, fastperiod=12, slowperiod=26, signalperiod=9)))
    feature_cols['macd'] = [item for sublist in macd for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
    feature_cols['macdsignal'] = [item for sublist in macdsignal for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
    feature_cols['macdhist'] = [item for sublist in macdhist for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

    # Bollinger Bands
    upper, middle, lower = zip(*grouped['close'].apply(lambda x: talib.BBANDS(x, timeperiod=20)))
    feature_cols['bband_upper'] = [item for sublist in upper for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
    feature_cols['bband_middle'] = [item for sublist in middle for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]
    feature_cols['bband_lower'] = [item for sublist in lower for item in (sublist if hasattr(sublist, '__iter__') else [sublist])]

    # ATR and ADX
    feature_cols['atr_14'] = grouped.apply(lambda x: talib.ATR(x['high'], x['low'], x['close'], timeperiod=14)).reset_index(level=0, drop=True)
    feature_cols['adx_14'] = grouped.apply(lambda x: talib.ADX(x['high'], x['low'], x['close'], timeperiod=14)).reset_index(level=0, drop=True)

    # Assign new features back to the dataframe
    for name, data in feature_cols.items():
        df[name] = data

    print("  - Feature calculation complete.")
    return df

def apply_trading_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the discovered trading rules to generate buy signals."""
    print("  - Applying trading rules to generate signals...")
    
    # Rule 1: Strong Momentum
    rule1 = df['macd'] > 0.67
    
    # Rule 2: Building Momentum
    rule2 = (df['macd'] > 0) & (df['macd'] <= 0.67) & (df['bband_lower'] <= 37.65)

    # Rule 3: Volatility Contraction with Support
    rule3 = (df['macd'] <= 0) & (df['atr_14'] > 1.36) & (df['bband_lower'] <= 56.71)

    # Combine rules: A buy signal is generated if any rule is true
    df['signal'] = rule1 | rule2 | rule3
    
    # To avoid repeated signals, we only take the first signal in a sequence
    df['signal'] = (df['signal'] & (df.groupby('stock_code')['signal'].shift(1) == False))

    print(f"  - Generated {df['signal'].sum()} buy signals.")
    return df

def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Simulates trades based on signals and calculates returns."""
    print("  - Running backtest simulation...")
    
    trades = []
    signals = df[df['signal']].copy()
    
    # Group by stock to process signals for each one
    for stock_code, stock_signals in signals.groupby('stock_code'):
        stock_data = df[df['stock_code'] == stock_code].set_index('date')
        
        for date, signal_row in stock_signals.iterrows():
            buy_date = signal_row['date']
            buy_price = signal_row['close']
            
            # Find the sell date, which is HOLDING_PERIOD days after the buy date
            try:
                buy_date_location = stock_data.index.get_loc(buy_date)
                sell_date_location = buy_date_location + HOLDING_PERIOD
                
                if sell_date_location < len(stock_data):
                    sell_date = stock_data.index[sell_date_location]
                    sell_price = stock_data.loc[sell_date, 'close']
                    
                    # Calculate the return for this trade
                    trade_return = (sell_price - buy_price) / buy_price
                    
                    trades.append({
                        'stock_code': stock_code,
                        'buy_date': buy_date,
                        'buy_price': buy_price,
                        'sell_date': sell_date,
                        'sell_price': sell_price,
                        'return': trade_return
                    })
            except KeyError:
                # This can happen if the signal date is not in the stock_data index
                continue

    if not trades:
        print("  - No trades were executed.")
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    print(f"  - Executed {len(trades_df)} trades.")
    return trades_df

def report_performance(trades_df: pd.DataFrame):
    """Reports the overall performance of the backtest."""
    print("\n--- Backtest Performance Report ---")
    
    if trades_df.empty:
        print("No trades to report.")
        return

    total_return = trades_df['return'].sum()
    average_return = trades_df['return'].mean()
    win_rate = (trades_df['return'] > 0).mean()
    
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Return per Trade: {average_return:.2%}")
    print(f"Total Return (Cumulative): {total_return:.2%}")
    
    # Display summary of winning and losing trades
    print("\n--- Trade-level Summary ---")
    print("Winning Trades:")
    print(trades_df[trades_df['return'] > 0]['return'].describe())
    print("\nLosing Trades:")
    print(trades_df[trades_df['return'] <= 0]['return'].describe())

def main():
    """Main function to run the backtesting pipeline."""
    print("--- Starting Backtest ---")
    
    # 1. Load Data
    price_path = os.path.join(DATA_DIR, PRICE_DATA_FILENAME)
    if not os.path.exists(price_path):
        print(f"ERROR: Data file not found at {price_path}")
        return
        
    all_prices = pd.read_parquet(price_path)
    print(f"- Loaded {len(all_prices)} price records for {all_prices['stock_code'].nunique()} stocks.")
    
    # 2. Calculate Features
    featured_data = calculate_features(all_prices)
    
    # 3. Apply Rules
    signaled_data = apply_trading_rules(featured_data)
    
    # 4. Run Backtest
    trades_result = run_backtest(signaled_data)
    
    # 5. Report Performance
    report_performance(trades_result)

if __name__ == "__main__":
    main()