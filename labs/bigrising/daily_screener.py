import pandas as pd
import talib
import os

# --- Constants ---
DATA_DIR = r'D:\workspace\xiaoyao\data'
PRICE_DATA_FILENAME = 'stock_daily_price.parquet'
OUTPUT_FILENAME = 'daily_selections.csv'

# --- Feature Calculation ---
def calculate_features(df):
    """Calculate technical indicators for the given dataframe."""
    # MACD
    macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
    df['bband_upper'] = upper
    df['bband_middle'] = middle
    df['bband_lower'] = lower

    # ATR
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    return df

# --- Screening Logic ---
def screen_stocks_daily():
    """Load data, calculate features, and screen stocks for each day."""
    print("--- Starting Daily Stock Screening ---")

    # Load all data
    price_path = os.path.join(DATA_DIR, PRICE_DATA_FILENAME)
    all_prices = pd.read_parquet(price_path)
    all_prices['date'] = pd.to_datetime(all_prices['date'])
    
    stock_codes = all_prices['stock_code'].unique()

    print(f"- Found {len(stock_codes)} stocks. Calculating features and screening stock by stock...")
    
    all_selected_stocks = []

    # Process each stock individually to save memory
    for i, stock_code in enumerate(stock_codes):
        if (i + 1) % 200 == 0:
            print(f"  - Processing stock {i+1}/{len(stock_codes)}: {stock_code}")

        stock_df = all_prices[all_prices['stock_code'] == stock_code].copy()
        
        # Calculate features for the single stock
        stock_df = calculate_features(stock_df)
        stock_df.dropna(inplace=True)

        if stock_df.empty:
            continue

        # Apply rules using vectorized operations
        rule1 = stock_df['macd'] > 0.67
        rule2 = (stock_df['macd'] > 0) & (stock_df['macd'] <= 0.67) & (stock_df['bband_lower'] <= 37.65)
        rule3 = (stock_df['macd'] <= 0) & (stock_df['atr'] > 1.36) & (stock_df['bband_lower'] <= 56.71)

        # Get selected rows for this stock
        selected_rows = stock_df[rule1 | rule2 | rule3]
        
        if not selected_rows.empty:
            all_selected_stocks.append(selected_rows)

    # --- Save Results ---
    if all_selected_stocks:
        # Combine all selected stocks from the list into a single DataFrame
        selected_df = pd.concat(all_selected_stocks)

        # Group by date and aggregate the stock codes
        daily_selections = selected_df.groupby('date')['stock_code'].apply(lambda x: ",".join(x)).reset_index()
        daily_selections.rename(columns={'stock_code': 'selected_stocks'}, inplace=True)

        output_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILENAME)
        daily_selections.to_csv(output_path, index=False)
        print(f"- Daily selections saved to {output_path}")
    else:
        print("- No stocks met the criteria on any day.")

    print("--- Screening Complete ---")

if __name__ == '__main__':
    screen_stocks_daily()