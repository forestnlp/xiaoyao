import pandas as pd
import os

# 定义输入和输出路径
input_parquet_file = r'D:\workspace\millionare\stock_daily_price.parquet'
output_data_dir = r'D:\workspace\xiaoyao\qlibusing\data'
output_csv_file = os.path.join(output_data_dir, 'stock_daily_price.csv')

# 确保输出目录存在
os.makedirs(output_data_dir, exist_ok=True)

try:
    # 读取 Parquet 文件
    df = pd.read_parquet(input_parquet_file)
    
    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(output_csv_file, index=False)
    
    print(f"成功将 {input_parquet_file} 转换为 {output_csv_file}")
except FileNotFoundError:
    print(f"错误：未找到文件 {input_parquet_file}")
except Exception as e:
    print(f"发生错误：{e}")