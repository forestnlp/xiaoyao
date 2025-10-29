from utils import (
    read_daily_parquet, read_processed_minutely,
    calc_daily_derivatives, calc_market_mood,
    join_daily_minutely, save_processed_data
)
import pandas as pd
import os

# -------------------------- 配置参数 --------------------------
DAILY_PARQUET_PATH = "D:/workspace/xiaoyao/data/widetable.parquet"  # 日度数据路径
PROCESSED_MINUTE_PATH = "D:/workspace/xiaoyao/data/minutely_processed"  # 预处理分钟数据路径
OUTPUT_DIR = "./processed"  # 输出目录

# -------------------------- 主执行逻辑 --------------------------
if __name__ == "__main__":
    print("="*50)
    print("开始数据预处理主流程（带进度条）")
    print("="*50)
    
    # 1. 读取数据
    print("\n1. 读取数据...")
    try:
        daily_raw = read_daily_parquet(DAILY_PARQUET_PATH)
        min_processed = read_processed_minutely(PROCESSED_MINUTE_PATH)
        # 时间范围对齐
        min_processed = min_processed[
            (min_processed["trade_date"] >= daily_raw["date"].min()) & 
            (min_processed["trade_date"] <= daily_raw["date"].max())
        ]
        print(f"日度数据：{len(daily_raw):,} 条记录，覆盖 {daily_raw['stock_code'].nunique()} 只股票")
        print(f"分钟数据：{len(min_processed):,} 条记录，覆盖 {min_processed['stock_code'].nunique()} 只股票")
    except Exception as e:
        print(f"读取数据失败：{str(e)}")
        exit()
    
    # 2. 计算日度指标与市场情绪
    print("\n2. 计算日度衍生指标与市场情绪...")
    try:
        daily_derived = calc_daily_derivatives(daily_raw)
        market_mood = calc_market_mood(daily_derived)
        print(f"日度衍生指标计算完成：{len(daily_derived):,} 条有效记录")
        print(f"市场情绪数据计算完成：{len(market_mood)} 个交易日")
    except Exception as e:
        print(f"计算日度指标失败：{str(e)}")
        exit()
    
    # 3. 关联数据
    print("\n3. 关联日度与分钟级数据...")
    try:
        strategy_core_data = join_daily_minutely(daily_derived, min_processed, market_mood)
        print(f"数据关联完成：{len(strategy_core_data):,} 条策略核心记录")
    except Exception as e:
        print(f"数据关联失败：{str(e)}")
        exit()
    
    # 4. 存储结果
    print("\n4. 存储预处理后的数据...")
    try:
        save_processed_data(strategy_core_data, market_mood, OUTPUT_DIR)
    except Exception as e:
        print(f"存储失败：{str(e)}")
        exit()
    
    print("\n" + "="*50)
    print("数据预处理全流程完成！")
    print("="*50)