import pandas as pd
import talib
import os

# --- Constants ---
LABS_DIR = r'd:\workspace\xiaoyao\labs\bigrising'
DATA_DIR = r'D:\workspace\xiaoyao\data'
BIG_RISE_LIST_FILENAME = 'big_rising_stocks.csv'
WIDE_TABLE_FILENAME = 'wide_table.parquet'
OUTPUT_FILENAME = 'pre_rise_features.csv'
DAYS_BEFORE_RISE = 3

def load_data(file_path: str, is_parquet: bool = False) -> pd.DataFrame:
    """Loads data from a CSV or Parquet file."""
    print(f"--- Loading {os.path.basename(file_path)} ---")
    try:
        if is_parquet:
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
        print(f"  - Successfully loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"  - [ERROR] File not found: {file_path}")
        return pd.DataFrame()

def enrich_with_talib_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates a rich set of TA-Lib features for the entire dataset using a robust loop."""
    print("--- Enriching data with TA-Lib features ---")
    
    df = df.sort_values(by=['stock_code', 'date']).reset_index(drop=True)
    
    all_stocks_features = []
    unique_codes = df['stock_code'].unique()
    
    print(f"  - Found {len(unique_codes)} unique stocks to process.")

    for i, code in enumerate(unique_codes):
        if (i + 1) % 100 == 0:
            print(f"    - Processing stock {i+1}/{len(unique_codes)}...")
            
        stock_df = df[df['stock_code'] == code].copy()
        
        if len(stock_df) < 34: 
            continue

        stock_df['rsi_14'] = talib.RSI(stock_df['close'], timeperiod=14)
        
        macd, macdsignal, macdhist = talib.MACD(stock_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        stock_df['macd'] = macd
        stock_df['macdsignal'] = macdsignal
        stock_df['macdhist'] = macdhist

        upper, middle, lower = talib.BBANDS(stock_df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        stock_df['bband_upper'] = upper
        stock_df['bband_middle'] = middle
        stock_df['bband_lower'] = lower
        
        stock_df['atr_14'] = talib.ATR(stock_df['high'], stock_df['low'], stock_df['close'], timeperiod=14)
        stock_df['adx_14'] = talib.ADX(stock_df['high'], stock_df['low'], stock_df['close'], timeperiod=14)
        
        # Add Volume Features
        stock_df['vma_5'] = talib.MA(stock_df['volume'], timeperiod=5)
        stock_df['vma_20'] = talib.MA(stock_df['volume'], timeperiod=20)
        stock_df['obv'] = talib.OBV(stock_df['close'], stock_df['volume'])
        
        # Calculate Volume Ratio (avoid division by zero)
        stock_df['volume_ratio'] = stock_df['volume'] / stock_df['vma_20']
        stock_df['volume_ratio'].fillna(0, inplace=True) # Handle cases where vma_20 is NaN

        all_stocks_features.append(stock_df)

    if not all_stocks_features:
        print("  - [ERROR] No features were generated.")
        return pd.DataFrame()

    final_df = pd.concat(all_stocks_features, ignore_index=True)
    print(f"  - Feature enrichment complete. New shape: {final_df.shape}")
    
    final_df.dropna(inplace=True)
    print(f"  - Shape after dropping NaNs: {final_df.shape}")
    
    return final_df

def extract_pre_rise_features(big_rise_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Extracts feature data for the N days leading up to each big rise event."""
    print(f"\n--- Extracting features for {DAYS_BEFORE_RISE} days before each rise ---")
    
    # Ensure date columns are in datetime format for comparison
    big_rise_df['start_date'] = pd.to_datetime(big_rise_df['start_date'])
    features_df['date'] = pd.to_datetime(features_df['date'])
    
    # Use a list to collect results
    all_pre_rise_periods = []

    # Set stock_code as index for faster lookup
    features_indexed = features_df.set_index('stock_code')
    
    total_events = len(big_rise_df)
    print(f"  - Processing {total_events} rise events.")

    for i, event in big_rise_df.iterrows():
        if (i + 1) % 1000 == 0:
            print(f"    - Processing event {i+1}/{total_events}...")

        stock_code = event['stock_code']
        rise_start_date = event['start_date']
        
        try:
            # Get all data for the specific stock
            stock_features = features_indexed.loc[[stock_code]].copy()
            
            # Define the start and end of the feature extraction window
            end_date = rise_start_date - pd.Timedelta(days=1)
            start_date = end_date - pd.Timedelta(days=DAYS_BEFORE_RISE + 5) # Fetch a slightly larger window to be safe
            
            # Filter the period
            pre_rise_period = stock_features[(stock_features['date'] >= start_date) & (stock_features['date'] <= end_date)]
            
            # Get the most recent N days
            pre_rise_period = pre_rise_period.nlargest(DAYS_BEFORE_RISE, 'date')

            if not pre_rise_period.empty:
                pre_rise_period['days_before_rise'] = (rise_start_date - pre_rise_period['date']).dt.days
                pre_rise_period['rise_start_date'] = rise_start_date
                pre_rise_period['stock_code'] = stock_code # Add stock_code back
                all_pre_rise_periods.append(pre_rise_period)

        except KeyError:
            # This stock from big_rise_df might not be in features_df (e.g., insufficient history)
            continue

    if not all_pre_rise_periods:
        print("  - No pre-rise feature periods could be extracted.")
        return pd.DataFrame()

    print("  - Concatenating results...")
    final_df = pd.concat(all_pre_rise_periods, ignore_index=True)
    print(f"  - Extraction complete. Found {len(final_df)} feature rows.")
    return final_df

def main():
    """Main function to orchestrate the feature analysis."""
    print("====== Starting Pre-Rise Feature Analysis ======")
    
    big_rise_list_path = os.path.join(LABS_DIR, BIG_RISE_LIST_FILENAME)
    wide_table_path = os.path.join(DATA_DIR, WIDE_TABLE_FILENAME)
    output_path = os.path.join(LABS_DIR, OUTPUT_FILENAME)
    
    try:
        # 1. Load data
        big_rise_df = load_data(big_rise_list_path, is_parquet=False)
        wide_df = load_data(wide_table_path, is_parquet=True)
        
        # 2. Enrich features
        features_df = enrich_with_talib_features(wide_df)
        
        # 3. Extract pre-rise features
        pre_rise_features_df = extract_pre_rise_features(big_rise_df, features_df)
        
        # 4. Save results
        if not pre_rise_features_df.empty:
            print(f"\n--- Saving pre-rise features to {output_path} ---")
            # Reorder columns for clarity
            cols_to_front = ['stock_code', 'date', 'days_before_rise', 'rise_start_date']
            other_cols = [col for col in pre_rise_features_df.columns if col not in cols_to_front]
            final_df = pre_rise_features_df[cols_to_front + other_cols]
            
            final_df.to_csv(output_path, index=False)
            print(f"  - Save complete. Shape: {final_df.shape}")
            
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")

    print("\n====== Analysis Finished ======")

if __name__ == "__main__":
    main()