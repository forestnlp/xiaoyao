import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm  # 进度条库

# -------------------------- 1. 数据读取函数 --------------------------
def read_daily_parquet(daily_path):
    """读取日度数据，带进度提示"""
    print("读取日度数据...")
    daily_data = pq.read_table(daily_path).to_pandas()
    # 时间格式标准化+删除停牌数据
    daily_data["date"] = pd.to_datetime(daily_data["date"])
    daily_data = daily_data[daily_data["paused"] == 0.0].reset_index(drop=True)
    # 保留核心字段
    core_cols = [
        "date", "stock_code", "stock_name", "open", "high", "low", "close", 
        "volume", "money", "pre_close", "high_limit", "low_limit", "turnover_ratio",
        "circulating_market_cap", "concept_name_list", "sw_l2_industry_name", "auc_volume"
    ]
    daily_data = daily_data[core_cols]
    return daily_data

def read_raw_minutely_with_progress(minute_root_path, start_date, end_date):
    """带进度条读取原始分钟数据"""
    min_data_list = []
    stock_dirs = [d for d in os.listdir(minute_root_path) if d.startswith("stock_code=")]
    total_stocks = len(stock_dirs)
    
    # 使用tqdm创建进度条
    for dir_name in tqdm(stock_dirs, desc="读取分钟数据", total=total_stocks, unit="只股票"):
        stock_code = dir_name.split("=")[1]
        min_path = os.path.join(minute_root_path, dir_name, "data.parquet")
        
        if not os.path.exists(min_path):
            continue  # 进度条模式下简化警告输出
        
        try:
            # 处理stock_code字段类型
            parquet_file = pq.ParquetFile(min_path)
            schema = parquet_file.schema.to_arrow_schema()
            new_fields = []
            for field in schema:
                if field.name == "stock_code" and str(field.type).startswith("dictionary"):
                    new_fields.append(field.with_type(pa.string()))
                else:
                    new_fields.append(field)
            new_schema = pa.schema(new_fields)
            min_data = pq.read_table(min_path, schema=new_schema).to_pandas()
            
            # 补充字段+过滤时间
            min_data["stock_code"] = stock_code
            min_data["time"] = pd.to_datetime(min_data["time"])
            min_data["trade_date"] = min_data["time"].dt.date.astype("datetime64[ns]")
            min_data = min_data[(min_data["trade_date"] >= start_date) & (min_data["trade_date"] <= end_date)]
            
            if len(min_data) > 0:
                min_data_list.append(min_data)
        
        except Exception:
            continue  # 进度条模式下简化错误输出
    
    if not min_data_list:
        raise ValueError("未读取到有效分钟数据，请检查路径或时间范围")
    
    # 分批次合并（带进度条）
    batch_size = 100
    min_all_data = pd.DataFrame()
    for i in tqdm(range(0, len(min_data_list), batch_size), desc="合并分钟数据", unit="批"):
        batch = min_data_list[i:i+batch_size]
        min_all_data = pd.concat([min_all_data, pd.concat(batch, ignore_index=True)], ignore_index=True)
    
    return min_all_data

def read_processed_minutely(processed_minute_path):
    """读取预处理后的分钟数据（带进度提示）"""
    if not os.path.exists(processed_minute_path):
        raise ValueError(f"预处理分钟数据路径不存在：{processed_minute_path}")
    
    print("读取预处理后的分钟数据...")
    # 读取所有分区（大型数据集时自动分块）
    min_data = pq.read_table(processed_minute_path).to_pandas()
    # 格式转换
    min_data["trade_date"] = pd.to_datetime(min_data["trade_date"])
    min_data["time"] = pd.to_datetime(min_data["time"])
    return min_data

# -------------------------- 2. 日度数据处理函数 --------------------------
def calc_daily_derivatives(daily_data):
    """计算日度衍生指标（带进度条）"""
    daily_df = daily_data.copy()
    total_stocks = daily_df["stock_code"].nunique()
    
    # 2.1 涨停状态
    daily_df["is_close_limit_up"] = (daily_df["close"] >= daily_df["high_limit"] - 0.01).astype(int)
    daily_df["is_touch_limit_up"] = (daily_df["high"] >= daily_df["high_limit"] - 0.01).astype(int)
    
    # 2.2 连续涨停天数（带进度条）
    daily_df = daily_df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    def calculate_consec_limit_up(group):
        group["break_point"] = (group["is_close_limit_up"] != group["is_close_limit_up"].shift(1)).cumsum()
        group["consec_count"] = group.groupby("break_point").cumcount() + 1
        group["consec_limit_up_days"] = group["consec_count"] * group["is_close_limit_up"]
        return group["consec_limit_up_days"]
    
    # 用tqdm包装groupby.apply
    tqdm.pandas(desc="计算连续涨停天数")
    daily_df["consec_limit_up_days"] = daily_df.groupby("stock_code", group_keys=False).progress_apply(calculate_consec_limit_up)
    daily_df = daily_df.drop(columns=["break_point", "consec_count"], errors="ignore")
    
    # 2.3 流通市值百分位
    def calc_market_cap_percentile(group):
        group["cap_percentile"] = group["circulating_market_cap"].rank(pct=True) * 100
        return group
    
    tqdm.pandas(desc="计算市值百分位")
    daily_df = daily_df.groupby("date", group_keys=False).progress_apply(calc_market_cap_percentile).reset_index(drop=True)
    daily_df["is_cap_valid"] = ((daily_df["cap_percentile"] >= 5) & (daily_df["cap_percentile"] <= 95)).astype(int)
    
    # 2.4 量比与换手率比（带进度条）
    rolling_window = 5
    tqdm.pandas(desc="计算量比指标")
    daily_df["volume_5d_avg"] = daily_df.groupby("stock_code")["volume"].rolling(window=rolling_window, min_periods=3).mean().reset_index(drop=True)
    daily_df["turnover_5d_avg"] = daily_df.groupby("stock_code")["turnover_ratio"].rolling(window=rolling_window, min_periods=3).mean().reset_index(drop=True)
    daily_df["volume_ratio"] = daily_df["volume"] / daily_df["volume_5d_avg"].replace(0, np.nan)
    daily_df["turnover_ratio_ratio"] = daily_df["turnover_ratio"] / daily_df["turnover_5d_avg"].replace(0, np.nan)
    
    # 2.5 其他指标
    daily_df["leading_activity"] = daily_df["volume_ratio"] * daily_df["turnover_ratio_ratio"]
    daily_df["auc_volume_5d_avg"] = daily_df.groupby("stock_code")["auc_volume"].rolling(window=rolling_window, min_periods=3).mean().reset_index(drop=True)
    daily_df["auc_ratio"] = daily_df["auc_volume"] / daily_df["auc_volume_5d_avg"].replace(0, np.nan)
    
    # 2.6 ATR波动率
    daily_df["true_range"] = pd.DataFrame({
        "tr1": daily_df["high"] - daily_df["low"],
        "tr2": abs(daily_df["high"] - daily_df["pre_close"]),
        "tr3": abs(daily_df["low"] - daily_df["pre_close"])
    }).max(axis=1)
    daily_df["atr_14d"] = daily_df.groupby("stock_code")["true_range"].rolling(window=14, min_periods=10).mean().reset_index(drop=True)
    
    # 2.7 竞价溢价率
    daily_df["auction_premium"] = (daily_df["open"] - daily_df["pre_close"]) / daily_df["pre_close"] * 100
    
    # 过滤无效数据
    daily_df = daily_df[
        (daily_df["is_cap_valid"] == 1) & 
        (daily_df["atr_14d"].notna())
    ].reset_index(drop=True)
    
    return daily_df

def calc_market_mood(daily_df):
    """计算市场情绪（带进度条）"""
    # 3.1 全市场赚钱效应
    daily_grouped = daily_df.groupby("date").agg({
        "stock_code": "count",
        "is_close_limit_up": "sum",
        "close": lambda x: ((x / daily_df.loc[x.index, "pre_close"] - 1) >= 0.05).sum(),
        "pre_close": lambda x: ((daily_df.loc[x.index, "close"] / x - 1) <= -0.05).sum()
    }).rename(columns={
        "stock_code": "total_stocks",
        "is_close_limit_up": "limit_up_count",
        "close": "up5pct_count",
        "pre_close": "down5pct_count"
    }).reset_index()
    daily_grouped["market_profit_effect"] = (
        daily_grouped["up5pct_count"] / daily_grouped["total_stocks"] - 
        daily_grouped["down5pct_count"] / daily_grouped["total_stocks"]
    )
    
    # 3.2 行业热度（带进度条）
    print("计算行业热度...")
    industry_heat = daily_df.groupby(["date", "sw_l2_industry_name"]).agg({
        "stock_code": "count",
        "is_close_limit_up": "sum"
    }).rename(columns={"stock_code": "industry_total", "is_close_limit_up": "industry_limit_up"}).reset_index()
    industry_heat["industry_heat_sw"] = industry_heat["industry_limit_up"] / industry_heat["industry_total"] * 100
    industry_heat["industry_heat_rank"] = industry_heat.groupby("date")["industry_heat_sw"].rank(ascending=False)
    top3_industry = industry_heat[industry_heat["industry_heat_rank"] <= 3].groupby("date").agg({
        "sw_l2_industry_name": lambda x: list(x),
        "industry_heat_sw": lambda x: list(x)
    }).rename(columns={"sw_l2_industry_name": "top3_industry", "industry_heat_sw": "top3_industry_heat"}).reset_index()
    
    # 3.3 概念热度（带进度条）
    print("计算概念热度...")
    def normalize_concept_field(val):
        if isinstance(val, list):
            return ",".join([str(item).strip() for item in val if item])
        else:
            return str(val).strip().strip("[]").strip("''").replace("' ", ",").replace(" '", ",").replace(" ", "")
    
    tqdm.pandas(desc="标准化概念字段")
    daily_df["concept_str"] = daily_df["concept_name_list"].progress_apply(normalize_concept_field)
    
    # 拆分概念
    concept_explode = daily_df.assign(
        single_concept=lambda x: x["concept_str"].str.split(",")
    ).explode("single_concept")
    concept_explode = concept_explode[
        (concept_explode["single_concept"] != "") & 
        (concept_explode["single_concept"].notna())
    ].reset_index(drop=True)
    
    # 计算概念热度
    concept_heat = concept_explode.groupby(["date", "single_concept"]).agg({
        "stock_code": "count",
        "is_close_limit_up": "sum"
    }).rename(columns={"stock_code": "concept_total", "is_close_limit_up": "concept_limit_up"}).reset_index()
    concept_heat["concept_heat_single"] = concept_heat["concept_limit_up"] / concept_heat["concept_total"] * 100
    concept_heat["concept_heat_rank"] = concept_heat.groupby("date")["concept_heat_single"].rank(ascending=False)
    top3_concept = concept_heat[concept_heat["concept_heat_rank"] <= 3].groupby("date").agg({
        "single_concept": lambda x: list(x),
        "concept_heat_single": lambda x: list(x)
    }).rename(columns={"single_concept": "top3_concept", "concept_heat_single": "top3_concept_heat"}).reset_index()
    
    # 合并情绪数据
    market_mood = daily_grouped.merge(top3_industry, on="date", how="left").merge(top3_concept, on="date", how="left")
    market_mood["top3_industry"] = market_mood["top3_industry"].fillna("[]").apply(lambda x: x if isinstance(x, list) else [])
    market_mood["top3_concept"] = market_mood["top3_concept"].fillna("[]").apply(lambda x: x if isinstance(x, list) else [])
    market_mood.set_index("date", inplace=True)
    
    return market_mood

# -------------------------- 3. 分钟级数据处理函数 --------------------------
def calc_minutely_indicators(min_all_data):
    """计算分钟级衍生指标（带进度条）"""
    min_df = min_all_data.copy()
    total_groups = min_df.groupby(["stock_code", "trade_date"]).ngroups
    
    # 3.1 早盘成交量占比
    min_df["is_morning"] = min_df["time"].dt.time.between(
        datetime.strptime("09:30:00", "%H:%M:%S").time(),
        datetime.strptime("10:30:00", "%H:%M:%S").time()
    ).astype(int)
    
    # 计算成交量（带进度条）
    print("计算早盘成交量占比...")
    daily_volume = min_df.groupby(["stock_code", "trade_date"])["volume"].sum().reset_index().rename(columns={"volume": "daily_total_volume"})
    morning_volume = min_df[min_df["is_morning"] == 1].groupby(["stock_code", "trade_date"])["volume"].sum().reset_index().rename(columns={"volume": "morning_volume"})
    min_df = min_df.merge(daily_volume, on=["stock_code", "trade_date"], how="left")
    min_df = min_df.merge(morning_volume, on=["stock_code", "trade_date"], how="left")
    min_df["morning_volume_ratio"] = (min_df["morning_volume"] / min_df["daily_total_volume"].replace(0, np.nan) * 100).fillna(0)
    
    # 3.2 尾盘企稳信号（带进度条）
    min_df["is_afternoon"] = min_df["time"].dt.time.between(
        datetime.strptime("14:30:00", "%H:%M:%S").time(),
        datetime.strptime("15:00:00", "%H:%M:%S").time()
    ).astype(int)
    
    def check_stabilize(group):
        afternoon = group[group["is_afternoon"] == 1].sort_values("time")
        if len(afternoon) < 5:
            group["afternoon_stabilize"] = 0
            return group
        afternoon["is_stable"] = (afternoon["close"] >= afternoon["close"].shift(1)).astype(int)
        afternoon["stable_streak"] = afternoon["is_stable"].groupby((afternoon["is_stable"] != afternoon["is_stable"].shift()).cumsum()).cumcount() + 1
        group["afternoon_stabilize"] = 1 if (afternoon["stable_streak"] >= 5).any() else 0
        return group
    
    # 用tqdm包装分组处理
    tqdm.pandas(desc="计算尾盘企稳信号", total=total_groups)
    min_df = min_df.groupby(["stock_code", "trade_date"], group_keys=False).progress_apply(check_stabilize).reset_index(drop=True)
    
    # 保留核心字段
    core_cols = [
        "stock_code", "trade_date", "time", "open", "high", "low", "close", "volume",
        "daily_total_volume", "morning_volume_ratio", "afternoon_stabilize"
    ]
    return min_df[core_cols]

# -------------------------- 4. 数据关联函数 --------------------------
def join_daily_minutely(daily_df, min_df, market_mood_df):
    """关联日度与分钟数据（带进度条）"""
    # 日度核心字段
    daily_join = daily_df[
        ["date", "stock_code", "is_close_limit_up", "consec_limit_up_days", "leading_activity", 
         "auc_ratio", "auction_premium", "atr_14d", "sw_l2_industry_name", "concept_str"]
    ].rename(columns={"date": "trade_date"})
    
    # 关联市场情绪
    market_mood_flat = market_mood_df.reset_index()[["date", "top3_industry", "top3_concept"]]
    market_mood_flat.rename(columns={"date": "trade_date"}, inplace=True)
    daily_join = daily_join.merge(market_mood_flat, on="trade_date", how="left")
    
    # 关联日度与分钟数据（带进度条）
    print("关联日度与分钟级数据...")
    joined_data = min_df.merge(daily_join, on=["stock_code", "trade_date"], how="inner")
    
    # 筛选逻辑
    def is_valid_leading(row):
        cond1 = row["consec_limit_up_days"] >= 1
        cond2 = row["sw_l2_industry_name"] in row["top3_industry"] if isinstance(row["top3_industry"], list) else False
        cond3 = any(concept in row["top3_concept"] for concept in row["concept_str"].split(",")) if (isinstance(row["top3_concept"], list) and row["concept_str"]) else False
        return cond1 and (cond2 or cond3)
    
    tqdm.pandas(desc="筛选策略核心记录")
    joined_data["is_valid_leading"] = joined_data.progress_apply(is_valid_leading, axis=1)
    joined_data = joined_data[joined_data["is_valid_leading"]].reset_index(drop=True)
    
    # 降低门槛
    if len(joined_data) == 0:
        print("警告：严格筛选无数据，降低门槛至仅连续涨停≥1板")
        joined_data = min_df.merge(daily_join, on=["stock_code", "trade_date"], how="inner")
        joined_data = joined_data[joined_data["consec_limit_up_days"] >= 1].reset_index(drop=True)
    
    return joined_data

# -------------------------- 5. 数据存储函数 --------------------------
def save_processed_data(joined_data, market_mood_data, output_dir):
    """存储预处理数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储策略核心数据
    core_output_path = os.path.join(output_dir, "strategy_core_data.parquet")
    arrow_table = pa.Table.from_pandas(joined_data)
    pq.write_to_dataset(
        table=arrow_table,
        root_path=core_output_path,
        partition_cols=["stock_code", "trade_date"]
    )
    print(f"策略核心数据已存储至：{core_output_path}")
    
    # 存储市场情绪数据
    mood_output_path = os.path.join(output_dir, "market_mood_data.parquet")
    market_mood_data.to_parquet(mood_output_path, engine="pyarrow", index=True)
    print(f"市场情绪数据已存储至：{mood_output_path}")
    
    # 数据概览
    if len(joined_data) > 0:
        print(f"\n数据概览：")
        print(f"1. 策略核心数据：{len(joined_data)} 条记录，覆盖 {joined_data['stock_code'].nunique()} 只股票")
    else:
        print("\n警告：策略核心数据无有效记录")
    print(f"2. 市场情绪数据：{len(market_mood_data)} 个交易日")