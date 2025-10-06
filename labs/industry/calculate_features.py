import pandas as pd
import os

# --- Configuration ---
DATA_DIR = os.path.abspath(r'D:\workspace\xiaoyao\data')

INPUT_FILENAME = 'wide_table.parquet'
OUTPUT_FILENAME = 'wide_table.parquet'
KEY_COLUMNS = ['date', 'stock_code']

# --- Functions ---

def get_data_path(filename: str) -> str:
    """Constructs the full, absolute path to a data file."""
    return os.path.join(DATA_DIR, filename)

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the wide table, handling potential errors."""
    print(f"--- 1. Loading Data from {file_path} ---")
    try:
        df = pd.read_parquet(file_path)
        # Ensure data is sorted for time-series operations
        df.sort_values(by=KEY_COLUMNS, inplace=True)
        print(f"  - Successfully loaded data. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {file_path}")
        print(f"Please run the 'merge_data.py' script first to generate the wide table.")
        raise

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates interday and intraday returns."""
    print("\n--- 2. Calculating Returns ---")
    
    # Group by stock to prevent data leakage across different stocks
    grouped = df.groupby('stock_code')
    
    # Interday return: (T+1 close / T close) - 1
    df['interday_return'] = (grouped['close'].shift(-1) / df['close']) - 1
    
    # Intraday return: (T close / T open) - 1
    df['intraday_return'] = (df['close'] / df['open']) - 1
    
    print("  - Calculated 'interday_return' and 'intraday_return'.")
    return df

def calculate_auction_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the ratio of auction money between T and T-1."""
    print("\n--- 3. Calculating Auction Money Ratio ---")
    
    # Group by stock to ensure correct day-over-day calculation
    grouped = df.groupby('stock_code')
    
    # Auction money ratio: (T auc_money / T-1 auc_money)
    df['auction_money_ratio'] = df['auc_money'] / grouped['auc_money'].shift(1)
    
    print("  - Calculated 'auction_money_ratio'.")
    return df

def main():
    """Main function to orchestrate the feature calculation pipeline."""
    print("====== Starting Feature Calculation Pipeline ======")
    
    input_path = get_data_path(INPUT_FILENAME)
    
    # Execute pipeline
    wide_df = load_data(input_path)
    features_df = calculate_returns(wide_df)
    features_df = calculate_auction_ratio(features_df)
    
    # Save the result
    output_path = get_data_path(OUTPUT_FILENAME)
    print(f"\n--- 4. Saving Result to {output_path} (Overwriting) ---")
    features_df.to_parquet(output_path, index=False)
    print(f"  - Successfully updated file. Final shape: {features_df.shape}")
    
    # Display a sample of the results
    print("\n====== Pipeline Finished ======")
    print("\nSample of the final DataFrame with new features:")
    
    # Define columns to display for clarity
    display_cols = KEY_COLUMNS + ['open', 'close', 'auc_money', 'interday_return', 'intraday_return', 'auction_money_ratio']
    print(features_df[display_cols].tail(10)) # Use tail to see more recent, non-NaN values

if __name__ == "__main__":
    main()