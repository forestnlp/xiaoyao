# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\preprocessor\minute.ipynb



# ----------------------------------------------------------------------import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import gc

# -------------------------- 配置参数 --------------------------
MINUTE_ROOT_PATH = "D:/workspace/xiaoyao/data/stock_minutely_price"
OUTPUT_PATH = "D:/workspace/xiaoyao/data/minutely_processed"
START_DATE = pd.to_datetime("2025-01-01")
END_DATE = pd.to_datetime("2025-10-27")
# 关键：新增stock_code到CORE_COLS，确保读取时一并处理其类型
CORE_COLS = ["stock_code", "time", "close", "high", "low", "volume"]  
N_JOBS = 4

# -------------------------- 1. 单股票处理（彻底修复类型冲突） --------------------------
def process_stock(stock_dir):
    stock_code = stock_dir.split("=")[1]
    stock_path = os.path.join(MINUTE_ROOT_PATH, stock_dir, "data.parquet")
    
    # 1. 读取并转换所有字段类型（含stock_code）
    try:
        # 步骤1：先读取整个文件的schema，查看所有字段类型
        parquet_file = pq.ParquetFile(stock_path)
        full_schema = parquet_file.schema.to_arrow_schema()
        
        # 步骤2：重建schema，将所有dictionary类型转为string（重点处理stock_code）
        new_fields = []
        for field in full_schema:
            field_name = field.name
            # 无论哪个字段，只要是dictionary类型，都转为string
            if str(field.type).startswith("dictionary"):
                new_field = pa.field(field_name, pa.string())
                new_fields.append(new_field)
            else:
                new_fields.append(field)
        new_full_schema = pa.schema(new_fields)
        
        # 步骤3：用新schema读取数据（确保stock_code已转为string）
        table = pq.read_table(stock_path, schema=new_full_schema, columns=CORE_COLS)
    
    except Exception as e:
        print(f"❌ 股票 {stock_code} 读取失败：{str(e)}")
        return
    
    # 2. 数据筛选（无需再补充stock_code，文件中已读取并转换）
    try:
        min_df = table.to_pandas()
        # 时间格式处理
        min_df["time"] = pd.to_datetime(min_df["time"])
        min_df["trade_date"] = min_df["time"].dt.date.astype("datetime64[ns]")
        # 筛选时间范围
        mask = (min_df["trade_date"] >= START_DATE) & (min_df["trade_date"] <= END_DATE)
        min_df = min_df[mask].copy()
    except Exception as e:
        print(f"❌ 股票 {stock_code} 数据处理失败：{str(e)}")
        return
    
    # 3. 检查数据是否为空
    if len(min_df) == 0:
        print(f"⚠️  股票 {stock_code} 无指定时间范围数据，跳过")
        return
    else:
        print(f"ℹ️  股票 {stock_code} 有效数据：{len(min_df)} 条，覆盖 {min_df['trade_date'].nunique()} 天")
    
    # 4. 计算扩展指标（无需修改）
    try:
        min_df = calc_enhanced_indicators(min_df)
    except Exception as e:
        print(f"❌ 股票 {stock_code} 指标计算失败：{str(e)}")
        return
    
    # 5. 写入二级分区（无需修改）
    try:
        save_with_double_partition(min_df, OUTPUT_PATH)
        print(f"✅ 股票 {stock_code} 处理完成")
    except Exception as e:
        print(f"❌ 股票 {stock_code} 写入失败：{str(e)}")
        return
    
    del min_df, table
    gc.collect()
    return

# -------------------------- 2. 扩展指标计算（不变） --------------------------
def calc_enhanced_indicators(min_df):
    min_df = min_df.sort_values(["trade_date", "time"]).reset_index(drop=True)
    
    # 指标1：早盘成交量占比
    morning_mask = (min_df["time"].dt.hour == 9) & (min_df["time"].dt.minute >= 30) | \
                   (min_df["time"].dt.hour == 10) & (min_df["time"].dt.minute < 30)
    min_df["daily_total_vol"] = min_df.groupby("trade_date")["volume"].transform("sum")
    min_df["morning_vol"] = min_df[morning_mask].groupby("trade_date")["volume"].transform("sum").fillna(0)
    min_df["morning_vol_ratio"] = (min_df["morning_vol"] / min_df["daily_total_vol"].replace(0, np.nan) * 100).fillna(0)
    
    # 指标2：尾盘企稳信号
    afternoon_mask = (min_df["time"].dt.hour == 14) & (min_df["time"].dt.minute >= 30) | \
                     (min_df["time"].dt.hour == 15) & (min_df["time"].dt.minute == 0)
    min_df["is_afternoon"] = afternoon_mask.astype(int)
    min_df["close_diff"] = min_df.groupby("trade_date")["close"].diff().fillna(0)
    min_df["up_streak"] = 0
    afternoon_groups = min_df[min_df["is_afternoon"] == 1].groupby("trade_date")
    for name, group in afternoon_groups:
        streak = 0
        streaks = []
        for diff in group["close_diff"]:
            streak = streak + 1 if diff > 0 else 0
            streaks.append(streak)
        min_df.loc[group.index, "up_streak"] = streaks
    min_df["afternoon_stable"] = (min_df.groupby("trade_date")["up_streak"].transform("max") >= 5).astype(int)
    
    # 指标3：日内振幅
    min_df["intraday_amplitude"] = (
        (min_df.groupby("trade_date")["high"].transform("max") - 
         min_df.groupby("trade_date")["low"].transform("min")) / 
        min_df.groupby("trade_date")["low"].transform("min") * 100
    ).fillna(0)
    
    # 指标4：量价同步性
    min_df["vol_diff"] = min_df.groupby("trade_date")["volume"].diff().fillna(0)
    min_df["vol_price_sync"] = (min_df["close_diff"] * min_df["vol_diff"] > 0).astype(int)
    
    # 指标5：收盘价靠近最高价比例
    min_df["close_to_high_ratio"] = (
        min_df["close"] / min_df.groupby("trade_date")["high"].transform("max") * 100
    ).fillna(0)
    
    # 保留必要字段（注意：stock_code已从文件读取，无需额外添加）
    keep_cols = [
        "stock_code", "trade_date", "time", "close", "volume",
        "morning_vol_ratio", "afternoon_stable", "intraday_amplitude",
        "vol_price_sync", "close_to_high_ratio"
    ]
    return min_df[keep_cols]

# -------------------------- 3. 二级分区写入（不变） --------------------------
def save_with_double_partition(df, output_root):
    stock_code = df["stock_code"].iloc[0]
    # 一级目录：stock_code=XXX
    stock_dir = os.path.join(output_root, f"stock_code={stock_code}")
    os.makedirs(stock_dir, exist_ok=True)
    
    # 二级目录：trade_date=XXX
    for trade_date, day_data in df.groupby("trade_date"):
        date_str = trade_date.strftime("%Y-%m-%d")
        date_dir = os.path.join(stock_dir, f"trade_date={date_str}")
        os.makedirs(date_dir, exist_ok=True)
        
        # 写入文件
        output_file = os.path.join(date_dir, "min_data.parquet")
        day_data.to_parquet(output_file, engine="pyarrow", index=False, compression=None)
    
    return

# -------------------------- 4. 主函数（单进程测试+多进程） --------------------------
def main():
    print("="*60)
    print("第一步：单进程测试前10只股票（修复类型冲突）")
    print("="*60)
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(f"✅ 已创建输出根目录：{OUTPUT_PATH}")
    else:
        print(f"✅ 输出根目录已存在：{OUTPUT_PATH}")
    
    try:
        all_stock_dirs = [d for d in os.listdir(MINUTE_ROOT_PATH) if d.startswith("stock_code=")]
        total_stocks = len(all_stock_dirs)
        print(f"✅ 发现 {total_stocks} 只股票目录，测试前10只...")
        
        test_success = 0
        for stock_dir in all_stock_dirs[:10]:
            stock_code = stock_dir.split("=")[1]
            try:
                process_stock(stock_dir)
                # 检查是否生成文件
                stock_output_dir = os.path.join(OUTPUT_PATH, f"stock_code={stock_code}")
                if os.path.exists(stock_output_dir) and len(os.listdir(stock_output_dir)) > 0:
                    test_success += 1
            except Exception as e:
                print(f"⚠️  测试 {stock_code} 异常：{str(e)}")
        
        print(f"\n单进程测试结果：{test_success}/10 只股票处理成功")
        if test_success == 0:
            print("❌ 单进程测试失败，请检查文件路径或字段类型！")
            return
        else:
            print("✅ 单进程测试通过，准备多进程处理...")
    
    except Exception as e:
        print(f"❌ 测试阶段失败：{str(e)}")
        return
    
    # 多进程处理剩余股票
    print("\n" + "="*60)
    print("第二步：多进程处理剩余股票")
    print("="*60)
    
    Parallel(
        n_jobs=N_JOBS,
        verbose=10,
        batch_size=2,
        backend="threading"
    )(
        delayed(process_stock)(stock_dir) for stock_dir in all_stock_dirs[10:]
    )
    
    # 统计结果
    processed_stocks = 0
    for stock_dir in all_stock_dirs:
        stock_code = stock_dir.split("=")[1]
        stock_output_dir = os.path.join(OUTPUT_PATH, f"stock_code={stock_code}")
        if os.path.exists(stock_output_dir) and len(os.listdir(stock_output_dir)) > 0:
            processed_stocks += 1
    
    print("\n" + "="*60)
    print("所有处理完成！")
    print(f"📊 统计：总股票数={total_stocks}，成功处理={processed_stocks}，成功率={processed_stocks/total_stocks:.2%}")
    print(f"结果路径：{OUTPUT_PATH}")
    print("="*60)

if __name__ == "__main__":
    main()

