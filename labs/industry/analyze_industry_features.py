import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import List

# --- Configuration ---
DATA_DIR = os.path.abspath(r'D:\workspace\xiaoyao\data')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILENAME = 'wide_table.parquet'
OUTPUT_FILENAME = 'industry_features_table.parquet'
OUTPUT_PLOT_DIR = os.path.join(CURRENT_DIR, 'plots')

# --- Functions ---

def get_data_path(filename: str, directory: str) -> str:
    """Constructs the full, absolute path to a data file."""
    return os.path.join(directory, filename)

def load_cleaned_data(file_path: str) -> pd.DataFrame:
    """Loads the cleaned data table."""
    print(f"--- 1. Loading Cleaned Data from {file_path} ---")
    try:
        df = pd.read_parquet(file_path)
        print(f"  - Successfully loaded data. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {file_path}")
        print(f"Please run the 'analyze_data.py' script first to generate the cleaned table.")
        raise

def calculate_industry_features(df: pd.DataFrame, industry_column: str = 'ind_sw_l3_industry_name') -> pd.DataFrame:
    """Calculates aggregated features at the industry level."""
    print(f"\n--- 2. Calculating Industry-Level Features (Grouping by '{industry_column}') ---")
    
    # Define aggregation logic
    def high_auction_ratio_pct(series):
        return (series > 1).sum() / series.count() if series.count() > 0 else 0

    aggregations = {
        'auction_money_ratio': high_auction_ratio_pct,
        'interday_return': 'mean',
        'intraday_return': 'mean',
        'open': 'count' # To see how many stocks are in the industry each day
    }
    
    # Group by date and industry, then aggregate
    industry_df = df.groupby(['date', industry_column]).agg(aggregations)
    
    # Rename columns for clarity
    industry_df.rename(columns={
        'auction_money_ratio': 'high_auction_ratio_pct',
        'interday_return': 'avg_interday_return',
        'intraday_return': 'avg_intraday_return',
        'open': 'stock_count'
    }, inplace=True)
    
    # Reset index to make 'date' and industry column regular columns
    industry_df.reset_index(inplace=True)
    
    print(f"  - Successfully created industry features table. Shape: {industry_df.shape}")
    print("  - Displaying last 5 rows of the new table:")
    print(industry_df.tail())
    return industry_df

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
        
        plot_filename = f"industry_{x_col}_vs_{y_col}.png"
        output_path = os.path.join(OUTPUT_PLOT_DIR, plot_filename)
        plt.savefig(output_path)
        plt.close()
        print(f"  - Saved scatter plot to: {output_path}")

def main():
    """Main function to orchestrate the industry feature analysis."""
    print("====== Starting Industry Feature Analysis Pipeline ======")
    
    input_path = get_data_path(INPUT_FILENAME, DATA_DIR)
    output_path = get_data_path(OUTPUT_FILENAME, DATA_DIR)
    
    # 1. Load data
    cleaned_df = load_cleaned_data(input_path)
    
    # 2. Calculate industry features
    industry_features_df = calculate_industry_features(cleaned_df, industry_column='ind_sw_l3_industry_name')
    
    # 3. Save the new table
    print(f"\n--- 3. Saving Industry Features Table to {output_path} ---")
    industry_features_df.to_parquet(output_path, index=False)
    print("  - Save complete.")

    # 4. Define columns for analysis
    analysis_columns = ['high_auction_ratio_pct', 'avg_interday_return', 'avg_intraday_return']
    
    # 5. Correlation Analysis
    analyze_correlation(industry_features_df, analysis_columns)
    
    # 6. Visualization
    visualize_relationships(industry_features_df, x_col='high_auction_ratio_pct', y_cols=['avg_interday_return', 'avg_intraday_return'])
    
    print("\n====== Analysis Finished ======")

if __name__ == "__main__":
    main()