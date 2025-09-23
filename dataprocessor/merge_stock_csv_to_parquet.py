#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 /d:/workspace/xiaoyao/redis/ 目录下的所有 stock_***.csv 文件合并为一个 parquet 文件
确保与现有 /d:/workspace/xiaoyao/data/stock_daily_price.parquet 保持字段、压缩方式一致
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import glob


def merge_stock_csv_to_parquet(csv_dir, output_parquet_file):
    """
    合并指定目录下的所有 stock_***.csv 文件到单个 parquet 文件
    
    Args:
        csv_dir: CSV 文件所在目录
        output_parquet_file: 输出的 parquet 文件路径
    """
    print(f"📁 开始处理目录: {csv_dir}")
    
    # 获取所有 stock_***.csv 文件
    csv_pattern = os.path.join(csv_dir, "stock_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print("❌ 未找到 stock_***.csv 文件")
        return False
    
    print(f"📊 找到 {len(csv_files)} 个 CSV 文件")
    
    # 按文件名排序（确保按日期顺序处理）
    csv_files.sort()
    
    # 读取并合并所有 CSV 文件
    all_dataframes = []
    total_records = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        filename = os.path.basename(csv_file)
        print(f"正在处理 ({i}/{len(csv_files)}): {filename}")
        
        try:
            # 读取 CSV 文件
            df = pd.read_csv(csv_file)
            
            # 数据验证和清洗
            # 确保 date 列是 datetime 类型
            df['date'] = pd.to_datetime(df['date'])
            
            # 确保数值列的数据类型正确
            numeric_columns = ['open', 'close', 'low', 'high', 'volume', 'money', 
                             'factor', 'high_limit', 'low_limit', 'avg', 'pre_close', 'paused']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 删除无效数据
            df = df.dropna(subset=['date', 'stock_code'])
            
            all_dataframes.append(df)
            total_records += len(df)
            print(f"  ✅ 成功读取 {len(df)} 条记录")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            continue
    
    if not all_dataframes:
        print("❌ 没有成功读取任何数据")
        return False
    
    print(f"\n📊 合并所有数据...")
    # 合并所有数据框
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 去重（按 date + stock_code）
    combined_df = combined_df.drop_duplicates(subset=['date', 'stock_code'])
    
    # 按日期和股票代码排序
    combined_df = combined_df.sort_values(['date', 'stock_code']).reset_index(drop=True)
    
    print(f"📈 总计 {len(combined_df)} 条记录（去重后）")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_parquet_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")
    
    # 转换为 pyarrow Table
    table = pa.Table.from_pandas(combined_df)
    
    # 使用与目标文件相同的压缩方式 (snappy) 和格式写入 parquet
    try:
        pq.write_table(
            table, 
            output_parquet_file,
            compression='snappy',
            version='2.6',  # 使用较新的 parquet 版本
            use_dictionary=True,
            write_batch_size=64 * 1024 * 1024  # 64MB batch size for better performance
        )
        
        print(f"✅ 成功保存到: {output_parquet_file}")
        print(f"📊 文件大小: {os.path.getsize(output_parquet_file) / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存 parquet 文件失败: {e}")
        return False


def main():
    """主函数"""
    # 设置路径
    csv_directory = "d:/workspace/xiaoyao/redis/"
    output_file = "d:/workspace/xiaoyao/dataprocessor/merged_stock_data.parquet"
    
    print("=" * 60)
    print("🚀 开始合并 stock_***.csv 文件到 parquet")
    print("=" * 60)
    
    # 执行合并
    success = merge_stock_csv_to_parquet(csv_directory, output_file)
    
    if success:
        print("\n🎉 合并完成！")
        
        # 验证结果
        try:
            print("\n📋 验证结果:")
            result_df = pd.read_parquet(output_file)
            print(f"   总行数: {len(result_df)}")
            print(f"   日期范围: {result_df['date'].min()} 到 {result_df['date'].max()}")
            print(f"   股票数量: {result_df['stock_code'].nunique()}")
            print(f"   列名: {list(result_df.columns)}")
            
        except Exception as e:
            print(f"⚠️  验证失败: {e}")
    
    else:
        print("\n❌ 合并失败！")
    
    print("=" * 60)


if __name__ == "__main__":
    main()