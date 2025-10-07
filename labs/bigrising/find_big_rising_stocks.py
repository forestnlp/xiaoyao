import pandas as pd
import os

# --- Configuration ---
DATA_DIR = os.path.abspath(r'D:\workspace\xiaoyao\data')
OUTPUT_DIR = os.path.abspath(r'D:\workspace\xiaoyao\labs\bigrising')
INPUT_FILENAME = 'stock_daily_price.parquet' # Assuming this file exists in DATA_DIR
OUTPUT_FILENAME = 'big_rising_stocks.csv'

# --- Parameters ---
LOOKBACK_DAYS = 10
RISE_THRESHOLD = 0.20 # 20% rise

def find_big_rising_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies stocks that have risen by a certain threshold within a lookback period.

    Args:
        df: DataFrame with stock data, must contain 'stock_code', 'date', and 'close'.

    Returns:
        A DataFrame containing the list of stocks that met the criteria.
    """
    print(f"--- 2. Searching for stocks rising >{RISE_THRESHOLD:.0%} in {LOOKBACK_DAYS} days ---")
    
    # Ensure data is sorted for rolling calculations
    df = df.sort_values(by=['stock_code', 'date']).reset_index(drop=True)
    
    # Calculate the close price from LOOKBACK_DAYS ago
    # Using shift() within each group is efficient
    df['close_n_days_ago'] = df.groupby('stock_code')['close'].shift(LOOKBACK_DAYS - 1)
    
    # Calculate the rolling return
    df['rolling_return'] = (df['close'] / df['close_n_days_ago']) - 1
    
    # Filter for stocks that meet the threshold
    rising_stocks = df[df['rolling_return'] >= RISE_THRESHOLD].copy()
    
    if rising_stocks.empty:
        print("  - No stocks found matching the criteria.")
        return pd.DataFrame()
        
    print(f"  - Found {len(rising_stocks)} instances of significant rises.")
    
    # Get the start date of the rise period
    # Note: This retrieves the date from the original dataframe based on the shifted index
    rising_stocks['start_date'] = rising_stocks.apply(
        lambda row: df.loc[row.name - (LOOKBACK_DAYS - 1), 'date'], axis=1
    )
    
    # Select and rename columns for the final output
    result = rising_stocks[['stock_code', 'start_date', 'date', 'rolling_return']].rename(
        columns={'date': 'end_date', 'rolling_return': 'rise_percentage'}
    )
    
    # Format the percentage for better readability
    result['rise_percentage'] = result['rise_percentage'].map('{:.2%}'.format)
    
    return result

def main():
    """Main function to run the analysis."""
    print("====== Starting Big Rising Stocks Analysis ======")
    
    input_path = os.path.join(DATA_DIR, INPUT_FILENAME)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load data
    print(f"--- 1. Loading data from {input_path} ---")
    try:
        stock_df = pd.read_parquet(input_path)
        # Ensure essential columns exist
        required_cols = ['stock_code', 'date', 'close']
        if not all(col in stock_df.columns for col in required_cols):
            raise ValueError(f"Input file must contain the columns: {required_cols}")
        print(f"  - Successfully loaded data. Shape: {stock_df.shape}")
    except FileNotFoundError:
        print(f"\n[ERROR] Input file not found: {input_path}")
        print("Please ensure the file exists in the specified data directory.")
        return
    except Exception as e:
        print(f"\n[ERROR] An error occurred while loading the data: {e}")
        return

    # 2. Find the stocks
    big_risers_df = find_big_rising_stocks(stock_df)
    
    # 3. Save the results
    if not big_risers_df.empty:
        print(f"\n--- 3. Saving results to {output_path} ---")
        big_risers_df.to_csv(output_path, index=False)
        print(f"  - Save complete. Found {len(big_risers_df)} rising events.")

    print("\n====== Analysis Finished ======")

if __name__ == "__main__":
    main()