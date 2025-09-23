#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精简版：仅导出涨停数据为parquet文件
条件：close=high_limit且日期>=2025-01-01
"""

import pandas as pd
from pathlib import Path

def extract_high_limit_stocks():
    """提取涨停数据并保存为parquet文件"""
    # 配置路径
    input_path = "d:/workspace/xiaoyao/data/stock_daily_price.parquet"
    output_path = "d:/workspace/xiaoyao/dataprocessor/high_limit_subset/high_limit_stocks_2025.parquet"
    
    print("开始提取涨停数据...")
    
    try:
        # 加载数据
        df = pd.read_parquet(input_path)
        print(f"成功加载数据: {len(df)} 条记录")
        
        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'])
        
        # 筛选日期（2025年1月1日及以后）
        df_filtered = df[df['date'] >= '2025-01-01'].copy()
        print(f"日期筛选后: {len(df_filtered)} 条记录")
        
        # 筛选涨停数据 (close == high_limit)
        high_limit_df = df_filtered[df_filtered['close'] == df_filtered['high_limit']].copy()
        print(f"涨停数据: {len(high_limit_df)} 条记录")
        
        if len(high_limit_df) > 0:
            # 确保输出目录存在
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 保存为parquet格式
            high_limit_df.to_parquet(output_path, index=False)
            print(f"涨停数据已保存到: {output_path}")
            print(f"文件大小: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
            
            # 基础统计信息
            stock_code_col = 'stock_code' if 'stock_code' in high_limit_df.columns else 'code'
            print(f"涉及股票代码数: {high_limit_df[stock_code_col].nunique()}")
            print(f"日期范围: {high_limit_df['date'].min()} 到 {high_limit_df['date'].max()}")
            
        else:
            print("没有找到涨停数据")
            
        return high_limit_df
        
    except Exception as e:
        print(f"处理失败: {e}")
        return None

if __name__ == "__main__":
    extract_high_limit_stocks()