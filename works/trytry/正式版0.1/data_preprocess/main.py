from utils import (
    read_daily_parquet, read_minutely_parquet,
    calc_daily_derivatives, calc_market_mood,
    calc_minutely_derivatives, join_daily_minutely,
    save_processed_data
)

# -------------------------- 配置参数（根据实际路径修改） --------------------------
DAILY_PARQUET_PATH = "D:/workspace/xiaoyao/data/widetable.parquet"  # 完整日度数据路径
MINUTE_ROOT_PATH = "D:/workspace/xiaoyao/data/stock_minutely_price/stock_minutely_price"  # 分钟级根路径
TARGET_STOCK_CODES = None# ["600460.XSHG", "688383.XSHG"]  # 测试用2只股票
OUTPUT_DIR = "./processed"  # 输出目录（自动创建）

# -------------------------- 全流程执行 --------------------------
if __name__ == "__main__":
    print("="*50)
    print("开始数据预处理（日度+分钟级）")
    print("="*50)
    
    # 1. 读取原始数据
    print("\n1. 读取原始数据...")
    try:
        daily_raw = read_daily_parquet(DAILY_PARQUET_PATH)
        min_raw = read_minutely_parquet(MINUTE_ROOT_PATH, TARGET_STOCK_CODES)
        print(f"日度数据：{len(daily_raw)} 条记录，覆盖 {daily_raw['stock_code'].nunique()} 只股票")
        print(f"分钟级数据：{len(min_raw)} 条记录，覆盖 {min_raw['stock_code'].nunique()} 只股票")
    except Exception as e:
        print(f"读取原始数据失败：{str(e)}")
        exit()
    
    # 2. 计算日度衍生指标与市场情绪
    print("\n2. 计算日度衍生指标与市场情绪...")
    try:
        daily_derived = calc_daily_derivatives(daily_raw)
        market_mood = calc_market_mood(daily_derived)
        print(f"日度衍生指标计算完成：{len(daily_derived)} 条有效记录")
        print(f"市场情绪数据计算完成：{len(market_mood)} 个交易日")
    except Exception as e:
        print(f"计算日度指标失败：{str(e)}")
        exit()
    
    # 3. 计算分钟级衍生指标
    print("\n3. 计算分钟级衍生指标...")
    try:
        min_derived = calc_minutely_derivatives(min_raw, daily_derived)
        print(f"分钟级衍生指标计算完成：{len(min_derived)} 条有效记录")
    except Exception as e:
        print(f"计算分钟级指标失败：{str(e)}")
        exit()
    
    # 4. 关联日度与分钟级数据（传入market_mood用于筛选）
    print("\n4. 关联日度与分钟级数据...")
    try:
        strategy_core_data = join_daily_minutely(daily_derived, min_derived, market_mood)
        print(f"数据关联完成：{len(strategy_core_data)} 条策略核心记录")
    except Exception as e:
        print(f"数据关联失败：{str(e)}")
        exit()
    
    # 5. 存储预处理数据
    print("\n5. 存储预处理后的数据...")
    try:
        save_processed_data(strategy_core_data, market_mood, OUTPUT_DIR)
    except Exception as e:
        print(f"数据存储失败：{str(e)}")
        exit()
    
    print("\n" + "="*50)
    print("数据预处理全流程完成！")
    print("="*50)