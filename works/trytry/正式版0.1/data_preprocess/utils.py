import pandas as pd
import pyarrow as pa  # 用于类型转换与Table创建
import pyarrow.parquet as pq
import os
from datetime import datetime
import numpy as np

# -------------------------- 1. 数据读取函数 --------------------------
def read_daily_parquet(daily_path):
    """读取日度widetable.parquet数据，返回处理后的DataFrame"""
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

def read_minutely_parquet(minute_root_path, target_stock_codes=None):
    """批量读取分钟级Parquet数据，返回含stock_code与trade_date的DataFrame"""
    min_data_list = []
    for dir_name in os.listdir(minute_root_path):
        if not dir_name.startswith("stock_code="):
            continue
        stock_code = dir_name.split("=")[1]
        if target_stock_codes and stock_code not in target_stock_codes:
            continue
        
        min_path = os.path.join(minute_root_path, dir_name, "data.parquet")
        if not os.path.exists(min_path):
            print(f"警告：{stock_code} 无分钟数据，跳过")
            continue
        
        # 读取时强制转换stock_code类型（解决字典vs字符串冲突）
        try:
            parquet_file = pq.ParquetFile(min_path)
            schema = parquet_file.schema.to_arrow_schema()
            # 处理stock_code字段类型
            new_fields = []
            for field in schema:
                if field.name == "stock_code":
                    if str(field.type).startswith("dictionary"):
                        new_field = field.with_type(pa.string())
                        new_fields.append(new_field)
                    else:
                        new_fields.append(field)
                else:
                    new_fields.append(field)
            new_schema = pa.schema(new_fields)
            min_data = pq.read_table(min_path, schema=new_schema).to_pandas()
        except Exception as e:
            print(f"警告：{stock_code} 读取失败，错误：{str(e)}，跳过该股票")
            continue
        
        # 覆盖stock_code+处理时间格式
        min_data["stock_code"] = stock_code
        min_data["time"] = pd.to_datetime(min_data["time"])
        min_data["trade_date"] = min_data["time"].dt.date.astype("datetime64[ns]")
        # 校验核心字段
        min_core_cols = ["stock_code", "trade_date", "time", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in min_core_cols if col not in min_data.columns]
        if missing_cols:
            print(f"警告：{stock_code} 缺失字段 {missing_cols}，跳过")
            continue
        
        min_data_list.append(min_data[min_core_cols])
    
    if not min_data_list:
        raise ValueError("未读取到任何分钟级数据，请检查路径或股票数据完整性")
    
    min_all_data = pd.concat(min_data_list, ignore_index=True)
    return min_all_data

# -------------------------- 2. 日度数据清洗与衍生指标函数 --------------------------
def calc_daily_derivatives(daily_data):
    """
    计算日度衍生指标：涨停状态、市值百分位、量比、换手率比、ATR等
    :param daily_data: 原始日度DataFrame（read_daily_parquet输出）
    :return: 含衍生指标的日度DataFrame
    """
    daily_df = daily_data.copy()
    
    # 2.1 涨停状态（收盘涨停+触达涨停）
    daily_df["is_close_limit_up"] = (daily_df["close"] >= daily_df["high_limit"] - 0.01).astype(int)  # 容忍0.01元误差
    daily_df["is_touch_limit_up"] = (daily_df["high"] >= daily_df["high_limit"] - 0.01).astype(int)
    
    # 2.2 连续涨停天数（按股票分组，向前追溯，修复索引对齐问题）
    daily_df = daily_df.sort_values(["stock_code", "date"]).reset_index(drop=True)  # 确保按股票和日期排序
    
    def calculate_consec_limit_up(group):
        """分组内计算连续涨停天数，确保索引与原始分组一致"""
        # 标识涨停中断点（当前与前一天状态不同时+1）
        group["break_point"] = (group["is_close_limit_up"] != group["is_close_limit_up"].shift(1)).cumsum()
        # 按中断点分组累计计数（仅涨停日有效）
        group["consec_count"] = group.groupby("break_point").cumcount() + 1
        # 非涨停日计数清零
        group["consec_limit_up_days"] = group["consec_count"] * group["is_close_limit_up"]
        return group["consec_limit_up_days"]  # 返回Series，保留分组内索引
    
    # 应用函数计算连续涨停天数（移除include_groups参数，兼容低版本pandas）
    daily_df["consec_limit_up_days"] = daily_df.groupby("stock_code", group_keys=False).apply(calculate_consec_limit_up)
    # 清理临时列（若有）
    daily_df = daily_df.drop(columns=["break_point", "consec_count"], errors="ignore")
    
    # 2.3 流通市值百分位（每日排除前5%超大盘、后5%超小盘）
    def calc_market_cap_percentile(group):
        group["cap_percentile"] = group["circulating_market_cap"].rank(pct=True) * 100
        return group
    # 移除include_groups参数，兼容低版本pandas
    daily_df = daily_df.groupby("date", group_keys=False).apply(calc_market_cap_percentile).reset_index(drop=True)
    daily_df["is_cap_valid"] = ((daily_df["cap_percentile"] >= 5) & (daily_df["cap_percentile"] <= 95)).astype(int)  # 中间90%
    
    # 2.4 量比与换手率比（过去5日均值）
    rolling_window = 5
    daily_df["volume_5d_avg"] = daily_df.groupby("stock_code")["volume"].rolling(window=rolling_window, min_periods=3).mean().reset_index(drop=True)
    daily_df["turnover_5d_avg"] = daily_df.groupby("stock_code")["turnover_ratio"].rolling(window=rolling_window, min_periods=3).mean().reset_index(drop=True)
    daily_df["volume_ratio"] = daily_df["volume"] / daily_df["volume_5d_avg"].replace(0, np.nan)  # 避免除0错误
    daily_df["turnover_ratio_ratio"] = daily_df["turnover_ratio"] / daily_df["turnover_5d_avg"].replace(0, np.nan)  # 避免除0错误
    
    # 2.5 龙头活跃度（量比×换手率比）+ 竞价量比（可选排序用）
    daily_df["leading_activity"] = daily_df["volume_ratio"] * daily_df["turnover_ratio_ratio"]
    daily_df["auc_volume_5d_avg"] = daily_df.groupby("stock_code")["auc_volume"].rolling(window=rolling_window, min_periods=3).mean().reset_index(drop=True)
    daily_df["auc_ratio"] = daily_df["auc_volume"] / daily_df["auc_volume_5d_avg"].replace(0, np.nan)  # 避免除0错误
    
    # 2.6 ATR波动率（14日真实波幅均值）
    daily_df["true_range"] = pd.DataFrame({
        "tr1": daily_df["high"] - daily_df["low"],
        "tr2": abs(daily_df["high"] - daily_df["pre_close"]),
        "tr3": abs(daily_df["low"] - daily_df["pre_close"])
    }).max(axis=1)
    daily_df["atr_14d"] = daily_df.groupby("stock_code")["true_range"].rolling(window=14, min_periods=10).mean().reset_index(drop=True)
    
    # 2.7 竞价溢价率（用日线open，替代9:25数据）
    daily_df["auction_premium"] = (daily_df["open"] - daily_df["pre_close"]) / daily_df["pre_close"] * 100
    
    # 过滤无效数据（市值不在中间90%、ATR为空）
    daily_df = daily_df[
        (daily_df["is_cap_valid"] == 1) & 
        (daily_df["atr_14d"].notna())
    ].reset_index(drop=True)
    
    return daily_df


def calc_market_mood(daily_df):
    """计算市场情绪数据，返回按date索引的DataFrame"""
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
    
    # 3.2 申万二级行业热度（Top3）
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
    
    # 3.3 单概念热度（Top3，修复字段格式）
    def normalize_concept_field(val):
        if isinstance(val, list):
            return ",".join([str(item).strip() for item in val if item])
        elif isinstance(val, (dict, int, float)):
            return ""
        else:
            str_val = str(val).strip().strip("[]").strip("''").replace("' ", ",").replace(" '", ",").replace(" ", "")
            return str_val
    daily_df["concept_str"] = daily_df["concept_name_list"].apply(normalize_concept_field)
    
    # 拆分概念并过滤无效值
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

# -------------------------- 3. 分钟级数据清洗与衍生指标函数 --------------------------
def calc_minutely_derivatives(min_all_data, daily_df):
    """
    计算分钟级衍生指标：封板时间、早盘成交量占比、尾盘企稳信号
    :param min_all_data: 原始分钟级DataFrame（read_minutely_parquet输出）
    :param daily_df: 含衍生指标的日度DataFrame（calc_daily_derivatives输出）
    :return: 含衍生指标的分钟级DataFrame
    """
    min_df = min_all_data.copy()
    daily_core = daily_df[["date", "stock_code", "high_limit", "is_close_limit_up"]].rename(columns={"date": "trade_date"})
    
    # 3.1 关联日度数据（判断是否涨停日）
    min_df = min_df.merge(daily_core, on=["stock_code", "trade_date"], how="left")
    min_df = min_df[min_df["is_close_limit_up"].notna()].reset_index(drop=True)  # 过滤无日度数据的分钟记录
    
    # 3.2 封板时间（仅涨停日计算：首次触达涨停且30分钟内不跌破）
    def get_limit_up_time(group):
        if group["is_close_limit_up"].iloc[0] != 1:  # 非涨停日，无封板时间
            group["limit_up_time"] = pd.NaT
            return group
        # 筛选触达涨停的分钟记录
        limit_up_mins = group[group["close"] >= group["high_limit"].iloc[0] - 0.01].copy()
        if limit_up_mins.empty:
            group["limit_up_time"] = pd.NaT
            return group
        # 找首次触达涨停的时间，并验证后续30分钟是否跌破
        first_limit_up_time = limit_up_mins["time"].min()
        # 计算30分钟后时间点
        thirty_mins_later = first_limit_up_time + pd.Timedelta(minutes=30)
        # 筛选首次封板后30分钟内的记录
        after_limit_up = group[group["time"].between(first_limit_up_time, thirty_mins_later)]
        if after_limit_up["close"].min() >= group["high_limit"].iloc[0] - 0.01:
            group["limit_up_time"] = first_limit_up_time
        else:
            group["limit_up_time"] = pd.NaT
        return group
    min_df = min_df.sort_values(["stock_code", "trade_date", "time"]).reset_index(drop=True)
    # 移除include_groups参数，兼容低版本pandas
    min_df = min_df.groupby(["stock_code", "trade_date"], group_keys=False).apply(get_limit_up_time).reset_index(drop=True)
    
    # 3.3 早盘成交量占比（09:30-10:30成交量 / 当日总成交量）
    # 标记早盘时段（9:30-10:30）
    min_df["is_morning"] = min_df["time"].dt.time.between(datetime.strptime("09:30:00", "%H:%M:%S").time(), 
                                                          datetime.strptime("10:30:00", "%H:%M:%S").time()).astype(int)
    # 计算当日总成交量与早盘成交量
    daily_volume = min_df.groupby(["stock_code", "trade_date"])["volume"].sum().reset_index().rename(columns={"volume": "daily_total_volume"})
    morning_volume = min_df[min_df["is_morning"] == 1].groupby(["stock_code", "trade_date"])["volume"].sum().reset_index().rename(columns={"volume": "morning_volume"})
    # 合并计算占比
    min_df = min_df.merge(daily_volume, on=["stock_code", "trade_date"], how="left")
    min_df = min_df.merge(morning_volume, on=["stock_code", "trade_date"], how="left")
    min_df["morning_volume_ratio"] = min_df["morning_volume"] / min_df["daily_total_volume"].replace(0, np.nan) * 100  # 避免除0
    min_df["morning_volume_ratio"] = min_df["morning_volume_ratio"].fillna(0)  # 无早盘数据时记为0
    
    # 3.4 尾盘企稳信号（14:30-15:00，连续5分钟不下跌）
    # 标记尾盘时段（14:30-15:00）
    min_df["is_afternoon"] = min_df["time"].dt.time.between(datetime.strptime("14:30:00", "%H:%M:%S").time(), 
                                                            datetime.strptime("15:00:00", "%H:%M:%S").time()).astype(int)
    # 仅尾盘时段计算连续上涨/企稳
    def check_afternoon_stabilize(group):
        afternoon_group = group[group["is_afternoon"] == 1].sort_values("time")
        if len(afternoon_group) < 5:  # 尾盘数据不足5分钟，无企稳信号
            group["afternoon_stabilize"] = 0
            return group
        # 计算连续5分钟是否无下跌（后一分钟close >= 前一分钟close）
        afternoon_group["is_stable"] = (afternoon_group["close"] >= afternoon_group["close"].shift(1)).astype(int)
        afternoon_group["stable_streak"] = afternoon_group["is_stable"].groupby((afternoon_group["is_stable"] != afternoon_group["is_stable"].shift()).cumsum()).cumcount() + 1
        # 若存在连续5分钟企稳，标记为1
        group["afternoon_stabilize"] = 1 if (afternoon_group["stable_streak"] >= 5).any() else 0
        return group
    # 移除include_groups参数，兼容低版本pandas
    min_df = min_df.groupby(["stock_code", "trade_date"], group_keys=False).apply(check_afternoon_stabilize).reset_index(drop=True)
    
    return min_df

# -------------------------- 4. 日度与分钟级数据关联函数 --------------------------
def join_daily_minutely(daily_df, min_df, market_mood_df):
    """关联日度与分钟级数据，修复龙头筛选逻辑，返回策略核心数据"""
    # 日度核心字段（含Top3行业/概念，解决关联记录为0问题）
    daily_join = daily_df[
        ["date", "stock_code", "is_close_limit_up", "consec_limit_up_days", "leading_activity", 
         "auc_ratio", "auction_premium", "atr_14d", "sw_l2_industry_name", "concept_str"]
    ].rename(columns={"date": "trade_date"})
    
    # 从market_mood获取每日Top3行业/概念，合并到日度数据
    market_mood_flat = market_mood_df.reset_index()[["date", "top3_industry", "top3_concept"]]
    market_mood_flat.rename(columns={"date": "trade_date"}, inplace=True)
    daily_join = daily_join.merge(market_mood_flat, on="trade_date", how="left")
    
    # 关联日度与分钟级数据
    joined_data = min_df.merge(daily_join, on=["stock_code", "trade_date"], how="inner")
    
    # 修复龙头筛选逻辑（降低门槛，避免记录为0）
    def is_valid_leading(row):
        # 条件1：连续涨停≥1板（原2板门槛过高，先降低验证）
        cond1 = row["consec_limit_up_days"] >= 1
        # 条件2：属于Top3行业或Top3概念
        cond2 = row["sw_l2_industry_name"] in row["top3_industry"] if isinstance(row["top3_industry"], list) else False
        cond3 = any(concept in row["top3_concept"] for concept in row["concept_str"].split(",")) if (isinstance(row["top3_concept"], list) and row["concept_str"]) else False
        return cond1 and (cond2 or cond3)
    
    # 应用筛选条件
    joined_data["is_valid_leading"] = joined_data.apply(is_valid_leading, axis=1)
    joined_data = joined_data[joined_data["is_valid_leading"]].reset_index(drop=True)
    
    # 若仍无数据，进一步降低门槛（仅保留连续涨停≥1板）
    if len(joined_data) == 0:
        print("警告：严格筛选无数据，降低门槛至仅连续涨停≥1板")
        joined_data = min_df.merge(daily_join, on=["stock_code", "trade_date"], how="inner")
        joined_data = joined_data[joined_data["consec_limit_up_days"] >= 1].reset_index(drop=True)
    
    return joined_data

# -------------------------- 5. 数据存储函数 --------------------------
def save_processed_data(joined_data, market_mood_data, output_dir):
    """存储预处理数据，修复pyarrow.Table报错问题"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 5.1 存储策略核心数据（用pa.Table替代pq.Table，解决属性不存在问题）
    core_output_path = os.path.join(output_dir, "strategy_core_data.parquet")
    # 转换为Arrow Table
    arrow_table = pa.Table.from_pandas(joined_data)
    pq.write_to_dataset(
        table=arrow_table,
        root_path=core_output_path,
        partition_cols=["stock_code", "trade_date"],
        filesystem=None
    )
    print(f"策略核心数据已存储至：{core_output_path}")
    
    # 5.2 存储市场情绪数据
    mood_output_path = os.path.join(output_dir, "market_mood_data.parquet")
    market_mood_data.to_parquet(mood_output_path, engine="pyarrow", index=True)
    print(f"市场情绪数据已存储至：{mood_output_path}")
    
    # 数据概览（兼容无数据场景）
    if len(joined_data) > 0:
        print(f"\n数据概览：")
        print(f"1. 策略核心数据：{len(joined_data)} 条记录，覆盖 {joined_data['stock_code'].nunique()} 只股票，时间范围 {joined_data['trade_date'].min()} 至 {joined_data['trade_date'].max()}")
    else:
        print("\n警告：策略核心数据无有效记录，建议检查筛选条件或测试股票的涨停数据")
    print(f"2. 市场情绪数据：{len(market_mood_data)} 个交易日，时间范围 {market_mood_data.index.min()} 至 {market_mood_data.index.max()}")