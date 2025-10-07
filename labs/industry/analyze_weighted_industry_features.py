import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import List

# --- Configuration ---
DATA_DIR = os.path.abspath(r'D:\workspace\xiaoyao\data')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILENAME = 'industry_features_table.parquet'
OUTPUT_FILENAME = 'weighted_industry_features_table.parquet'
OUTPUT_PLOT_DIR = os.path.join(CURRENT_DIR, 'plots')

# --- Functions ---

def get_data_path(filename: str, directory: str) -> str:
    """Constructs the full, absolute path to a data file."""
    return os.path.join(directory, filename)

def load_industry_data(file_path: str) -> pd.DataFrame:
    """Loads the industry features table."""
    print(f"--- 1. Loading Industry Data from {file_path} ---")
    try:
        df = pd.read_parquet(file_path)
        print(f"  - Successfully loaded data. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {file_path}")
        print(f"Please run the 'analyze_industry_features.py' script first to generate the table.")
        raise

def calculate_weighted_feature(df: pd.DataFrame, industry_column: str = 'ind_sw_l3_industry_name') -> pd.DataFrame:
    """Calculates the 3-day weighted moving average for the high auction ratio percentage."""
    print("\n--- 2. Calculating 3-Day Weighted High Auction Ratio ---")
    
    # Ensure data is sorted by industry and date for correct rolling calculation
    df.sort_values([industry_column, 'date'], inplace=True)
    
    # Define weights
    weights = np.array([0.33, 0.33, 0.35])
    
    # Define a function for the rolling weighted average
    def weighted_average(window):
        if len(window) < 3:
            return np.nan
        return np.dot(window, weights)

    # Group by industry and apply the rolling weighted average
    df['weighted_high_auction_ratio_pct'] = df.groupby(industry_column)['high_auction_ratio_pct'].rolling(window=3).apply(weighted_average, raw=True).reset_index(level=0, drop=True)
    
    # Drop rows with NaN values created by the rolling window
    original_rows = len(df)
    df.dropna(subset=['weighted_high_auction_ratio_pct'], inplace=True)
    print(f"  - Dropped {original_rows - len(df)} rows with NaN values after rolling calculation.")
    
    print(f"  - Successfully created weighted feature. New shape: {df.shape}")
    print("  - Displaying last 5 rows of the updated table:")
    print(df.tail())
    return df

def analyze_correlation(df: pd.DataFrame, columns: List[str]):
    """Calculates and prints the correlation matrix for the given columns."""
    print("\n--- 4. Correlation Analysis ---")
    correlation_matrix = df[columns].corr()
    print("Correlation Matrix:")
    print(correlation_matrix)

def visualize_relationships(df: pd.DataFrame, x_col: str, y_cols: List[str]):
    """Creates and saves scatter plots to visualize relationships."""
    print(f"\n--- 5. Visualizing Relationships ---")
    
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    print(f"  - Plots will be saved to: {OUTPUT_PLOT_DIR}")

    for y_col in y_cols:
        plt.figure(figsize=(12, 7))
        sns.regplot(data=df, x=x_col, y=y_col, 
                    scatter_kws={'alpha':0.5, 's': 10}, 
                    line_kws={'color':'red'})
        
        plt.title(f'Industry-Level: {x_col} vs. {y_col}', fontsize=16)
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        plt.grid(True)
        
        plot_filename = f"industry_weighted_feature_vs_{y_col}.png"
        output_path = os.path.join(OUTPUT_PLOT_DIR, plot_filename)
        plt.savefig(output_path)
        plt.close()
        print(f"  - Saved scatter plot to: {output_path}")

def main():
    """Main function to orchestrate the weighted industry feature analysis."""
    print("====== Starting Weighted Industry Feature Analysis Pipeline ======")
    
    input_path = get_data_path(INPUT_FILENAME, DATA_DIR)
    output_path = get_data_path(OUTPUT_FILENAME, DATA_DIR)
    
    # 1. Load data
    industry_df = load_industry_data(input_path)
    
    # 2. Calculate weighted feature
    weighted_df = calculate_weighted_feature(industry_df, industry_column='ind_sw_l3_industry_name')
    
    # 3. Save the new table
    print(f"\n--- 3. Saving Weighted Industry Features Table to {output_path} ---")
    weighted_df.to_parquet(output_path, index=False)
    print("  - Save complete.")

    # 4. Define columns for analysis
    analysis_columns = ['weighted_high_auction_ratio_pct', 'avg_interday_return', 'avg_intraday_return']
    
    # 5. Correlation Analysis
    analyze_correlation(weighted_df, analysis_columns)
    
    # 6. Visualization
    visualize_relationships(weighted_df, x_col='weighted_high_auction_ratio_pct', y_cols=['avg_interday_return', 'avg_intraday_return'])
    
    print("\n====== Analysis Finished ======")

if __name__ == "__main__":
    main()