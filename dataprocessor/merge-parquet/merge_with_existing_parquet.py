#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将新生成的 merged_stock_data.parquet 与现有的 stock_daily_price.parquet 合并
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os


def merge_parquet_files(existing_file, new_file, output_file):
    """
    合并两个 parquet 文件
    
    Args:
        existing_file: 现有的 parquet 文件路径
        new_file: 新的 parquet 文件路径  
        output_file: 输出的合并文件路径
    """
    print("📊 开始合并 parquet 文件...")
    
    try:
        # 读取现有数据
        print(f"📖 读取现有文件: {existing_file}")
        existing_df = pd.read_parquet(existing_file)
        print(f"   现有数据行数: {len(existing_df)}")
        
        # 读取新数据
        print(f"📖 读取新文件: {new_file}")
        new_df = pd.read_parquet(new_file)
        print(f"   新数据行数: {len(new_df)}")
        
        # 合并数据
        print("🔄 合并数据中...")
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # 去重（按 date + stock_code）
        print("🧹 去重处理...")
        combined_df = combined_df.drop_duplicates(subset=['date', 'stock_code'])
        
        # 排序
        print("📅 按日期排序...")
        combined_df = combined_df.sort_values(['date', 'stock_code']).reset_index(drop=True)
        
        print(f"📈 合并后总行数: {len(combined_df)}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 转换为 pyarrow Table
        table = pa.Table.from_pandas(combined_df)
        
        # 写入 parquet（使用与源文件相同的格式）
        print(f"💾 保存合并结果: {output_file}")
        pq.write_table(
            table,
            output_file,
            compression='snappy',
            version='2.6',
            use_dictionary=True,
            write_batch_size=64 * 1024 * 1024
        )
        
        print(f"✅ 合并完成！文件大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        # 验证结果
        print("\n📋 验证结果:")
        result_df = pd.read_parquet(output_file)
        print(f"   最终行数: {len(result_df)}")
        print(f"   日期范围: {result_df['date'].min()} 到 {result_df['date'].max()}")
        print(f"   股票数量: {result_df['stock_code'].nunique()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 合并失败: {e}")
        return False


def main():
    """主函数"""
    # 设置文件路径
    # existing_file = "d:/workspace/xiaoyao/data/stock_daily_price.parquet"
    # new_file = "d:/workspace/xiaoyao/dataprocessor/merged_stock_data.parquet"
    # output_file = "d:/workspace/xiaoyao/dataprocessor/merged_combined_stock_data.parquet"

    existing_file = "d:/workspace/xiaoyao/data/stock_daily_auction.parquet"
    new_file = "d:/workspace/xiaoyao/dataprocessor/merged_stock_data.parquet"
    output_file = "d:/workspace/xiaoyao/dataprocessor/merged_combined_stock_data.parquet"

    print("=" * 60)
    print("🚀 开始合并 parquet 文件")
    print("=" * 60)
    
    # 检查文件是否存在
    if not os.path.exists(existing_file):
        print(f"❌ 现有文件不存在: {existing_file}")
        return
    
    if not os.path.exists(new_file):
        print(f"❌ 新文件不存在: {new_file}")
        return
    
    # 执行合并
    success = merge_parquet_files(existing_file, new_file, output_file)
    
    if success:
        print("\n🎉 合并成功！")
    else:
        print("\n❌ 合并失败！")
    
    print("=" * 60)


if __name__ == "__main__":
    main()