import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm

# -------------------------- 配置参数 --------------------------
MINUTE_ROOT_PATH = "D:/workspace/xiaoyao/data/stock_minutely_price"
OUTPUT_MINUTE_PATH = "D:/workspace/xiaoyao/data/minutely_processed"
START_DATE = pd.to_datetime("2025-01-01")
END_DATE = pd.to_datetime("2025-10-27")
MIN_CORE_COLS = ["time", "close", "volume"]  # 仅核心字段

# -------------------------- 1. 单股票处理（修复类型冲突） --------------------------
def process_single_stock(stock_dir):
    stock_code = stock_dir.split("=")[1]
    stock_path = os.path.join(MINUTE_ROOT_PATH, stock_dir, "data.parquet")
    
    # 核心修复：读取时强制统一字段类型（dictionary→string）
    try:
        # 1. 先读取Parquet文件的schema，检查字段类型
        parquet_file = pq.ParquetFile(stock_path)
        schema = parquet_file.schema.to_arrow_schema()
        
        # 2. 重建schema：将所有dictionary类型字段转为string
        new_fields = []
        for field in schema:
            # 检查是否为dictionary类型
            if str(field.type).startswith("dictionary"):
                # 强制转为string类型
                new_field = field.with_type(pa.string())
                new_fields.append(new_field)
            else:
                new_fields.append(field)
        new_schema = pa.schema(new_fields)
        
        # 3. 用新schema读取数据（确保所有字段类型统一）
        min_data = pq.read_table(stock_path, columns=MIN_CORE_COLS, schema=new_schema).to_pandas()
    
    except Exception as e:
        print(f"跳过 {stock_code}：读取失败（{str(e)}）")
        return
    
    # 后续数据处理逻辑不变
    min_data["time"] = pd.to_datetime(min_data["time"])
    min_data["trade_date"] = min_data["time"].dt.date.astype("datetime64[ns]")
    min_data = min_data[(min_data["trade_date"] >= START_DATE) & (min_data["trade_date"] <= END_DATE)]
    if len(min_data) == 0:
        print(f"跳过 {stock_code}：无指定时间范围数据")
        return
    
    # 补充股票代码（此时字段类型已统一为string）
    min_data["stock_code"] = stock_code
    
    # 计算指标+保存
    min_data = calc_minute_indicators_single_stock(min_data)
    save_single_stock_result(min_data, OUTPUT_MINUTE_PATH)
    return

# -------------------------- 2. 指标计算（不变） --------------------------
def calc_minute_indicators_single_stock(min_data):
    # 早盘成交量占比
    morning_mask = (min_data["time"].dt.hour == 9) & (min_data["time"].dt.minute >= 30) | \
                   (min_data["time"].dt.hour == 10) & (min_data["time"].dt.minute < 30)
    daily_volume = min_data.groupby("trade_date")["volume"].sum().reset_index()
    daily_volume.columns = ["trade_date", "daily_total_volume"]
    morning_volume = min_data[morning_mask].groupby("trade_date")["volume"].sum().reset_index()
    morning_volume.columns = ["trade_date", "morning_volume"]
    
    min_data = min_data.merge(daily_volume, on="trade_date", how="left")
    min_data = min_data.merge(morning_volume, on="trade_date", how="left")
    min_data["morning_volume_ratio"] = (min_data["morning_volume"] / min_data["daily_total_volume"].replace(0, np.nan) * 100).fillna(0)
    min_data = min_data.drop(columns=["daily_total_volume", "morning_volume"])
    
    # 尾盘企稳信号
    afternoon_mask = (min_data["time"].dt.hour == 14) & (min_data["time"].dt.minute >= 30) | \
                     (min_data["time"].dt.hour == 15) & (min_data["time"].dt.minute == 0)
    
    def check_stabilize(group):
        afternoon_data = group[afternoon_mask].sort_values("time")
        if len(afternoon_data) < 5:
            group["afternoon_stabilize"] = 0
            return group
        current_streak = 0
        stabilize_flag = 0
        for idx, row in afternoon_data.iterrows():
            if idx == afternoon_data.index[0]:
                current_streak = 1
            else:
                prev_close = afternoon_data.loc[afternoon_data.index[afternoon_data.index.get_loc(idx)-1], "close"]
                if row["close"] >= prev_close:
                    current_streak += 1
                else:
                    current_streak = 0
            if current_streak >= 5:
                stabilize_flag = 1
                break
        group["afternoon_stabilize"] = stabilize_flag
        return group
    
    min_data = min_data.groupby("trade_date").apply(check_stabilize).reset_index(drop=True)
    
    # 保留最终字段
    final_cols = ["stock_code", "trade_date", "time", "close", "volume", "morning_volume_ratio", "afternoon_stabilize"]
    return min_data[final_cols]

# -------------------------- 3. 结果保存（不变） --------------------------
def save_single_stock_result(min_data, output_path):
    pq.write_to_dataset(
        table=pa.Table.from_pandas(min_data),
        root_path=output_path,
        partition_cols=["trade_date"],
        filesystem=None,
        use_dictionary=False
    )
    return

# -------------------------- 4. 主执行（不变） --------------------------
if __name__ == "__main__":
    print("="*50)
    print("按股票代码分区处理分钟数据（修复类型冲突）")
    print("="*50)
    
    all_dirs = [d for d in os.listdir(MINUTE_ROOT_PATH) if d.startswith("stock_code=")]
    total_stocks = len(all_dirs)
    print(f"发现 {total_stocks} 只股票，开始依次处理...")
    
    for stock_dir in tqdm(all_dirs, desc="处理股票", total=total_stocks, unit="只"):
        process_single_stock(stock_dir)
    
    print("\n" + "="*50)
    print(f"所有股票处理完成！结果保存至：{OUTPUT_MINUTE_PATH}")
    print("="*50)