# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\preprocessor\tradedate.ipynb



# ----------------------------------------------------------------------import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import gc

# -------------------------- 配置参数（针对widetable全量文件） --------------------------
WIDETABLE_PATH = "D:/workspace/xiaoyao/data/widetable.parquet"  # 全量日线文件路径
OUTPUT_PATH = "D:/workspace/xiaoyao/data/daily_processed"  # 日线指标输出路径
START_DATE = pd.to_datetime("2025-01-01")  # 数据时间范围
END_DATE = pd.to_datetime("2025-10-27")
# 假设widetable包含的基础字段（根据实际字段调整，缺失字段会自动兼容）
EXPECTED_COLS = ["stock_code", "trade_date", "close", "open", "high", "low", "volume", "circulating_capital", "pe_ttm", "pb"]
N_JOBS = 4  # 并行进程数（建议≤CPU核心数，避免内存过载）

# -------------------------- 1. 预加载全量数据并拆分股票（关键适配） --------------------------
def load_and_split_widetable():
    """加载widetable全量文件，按stock_code拆分数据块，返回股票代码列表+数据字典"""
    print("="*60)
    print("正在加载widetable.parquet全量数据...")
    print("="*60)
    
    try:
        # 1. 读取全量文件，仅保留需要的字段（过滤无用字段，减少内存）
        table = pq.read_table(WIDETABLE_PATH)
        # 检查并保留实际存在的字段（避免字段缺失导致报错）
        exist_cols = [col for col in EXPECTED_COLS if col in table.column_names]
        table = table.select(exist_cols)
        
        # 2. 转换dictionary类型字段（如stock_code/trade_date可能为字典编码）
        new_fields = []
        for field in table.schema:
            if str(field.type).startswith("dictionary"):
                if field.name in ["stock_code", "trade_date"]:
                    new_fields.append(field.with_type(pa.string()))  # 字符串字段转string
                else:
                    new_fields.append(field.with_type(pa.float64()))  # 数值字段转float64
            else:
                new_fields.append(field)
        table = table.cast(pa.schema(new_fields))
        
        # 3. 转为DataFrame并预处理时间格式
        wide_df = table.to_pandas()
        wide_df["trade_date"] = pd.to_datetime(wide_df["trade_date"])
        print(f"✅ 全量数据加载完成：共 {len(wide_df)} 条记录，覆盖 {wide_df['stock_code'].nunique()} 只股票")
        
        # 4. 筛选时间范围（先筛选再拆分，减少后续计算量）
        mask = (wide_df["trade_date"] >= START_DATE) & (wide_df["trade_date"] <= END_DATE)
        wide_df = wide_df[mask].copy()
        if len(wide_df) == 0:
            print(f"❌ 筛选后无数据（{START_DATE.strftime('%Y-%m-%d')} 至 {END_DATE.strftime('%Y-%m-%d')}）")
            return [], {}
        
        # 5. 按stock_code拆分数据块，存入字典（key=股票代码，value=单股票DataFrame）
        stock_data_dict = {}
        for stock_code, stock_df in wide_df.groupby("stock_code"):
            stock_data_dict[stock_code] = stock_df.sort_values("trade_date").reset_index(drop=True)
        
        # 6. 释放全量数据内存
        del wide_df, table
        gc.collect()
        
        return list(stock_data_dict.keys()), stock_data_dict
    
    except Exception as e:
        print(f"❌ 加载widetable失败：{str(e)}")
        return [], {}

# -------------------------- 2. 单股票日线指标计算（基于数据块） --------------------------
def process_stock_from_dict(stock_code, stock_df):
    """基于拆分后的单股票数据块计算指标，避免重复加载全量文件"""
    # 1. 检查单股票数据是否为空
    if len(stock_df) == 0:
        print(f"⚠️  股票 {stock_code} 无有效数据，跳过")
        return
    
    # 2. 计算核心指标（兼容缺失字段）
    try:
        stock_df = calc_daily_indicators(stock_df)
    except Exception as e:
        print(f"❌ 股票 {stock_code} 指标计算失败：{str(e)}")
        return
    
    # 3. 二级分区存储（与分钟数据结构一致）
    try:
        save_daily_with_double_partition(stock_df, OUTPUT_PATH)
        print(f"✅ 股票 {stock_code} 处理完成（{len(stock_df)} 条数据）")
    except Exception as e:
        print(f"❌ 股票 {stock_code} 写入失败：{str(e)}")
        return
    
    # 释放单股票数据内存
    del stock_df
    gc.collect()
    return

# -------------------------- 3. 日线指标计算核心函数（兼容缺失字段） --------------------------
def calc_daily_indicators(stock_df):
    """计算12个核心指标，自动兼容widetable中缺失的字段"""
    # 3.1 基础量价指标（核心字段，缺失会报错，确保widetable有这些字段）
    # 涨跌幅（%）
    stock_df["pct_change"] = stock_df["close"].pct_change() * 100
    # 振幅（%）
    stock_df["amplitude"] = (stock_df["high"] - stock_df["low"]) / stock_df["close"].shift(1) * 100
    # 换手率（%）：有流通股本用流通股本，无则用“成交量/近20日均量”替代
    if "circulating_capital" in stock_df.columns and stock_df["circulating_capital"].notna().any():
        stock_df["turnover_rate"] = (stock_df["volume"] / stock_df["circulating_capital"]) * 100
    else:
        stock_df["turnover_rate"] = (stock_df["volume"] / stock_df["volume"].rolling(20, min_periods=5).mean()) * 100
    
    # 3.2 情绪关联指标
    # 是否涨停（10%涨停→1.098，ST股5%→1.048，可根据需要调整）
    stock_df["is_limit_up"] = (stock_df["close"] / stock_df["close"].shift(1) >= 1.098).astype(int)
    # 连续涨停天数
    stock_df["consec_limit_days"] = 0
    current_streak = 0
    for i in range(len(stock_df)):
        if stock_df.iloc[i]["is_limit_up"] == 1:
            current_streak += 1
        else:
            current_streak = 0
        stock_df.iloc[i, stock_df.columns.get_loc("consec_limit_days")] = current_streak
    # 市场赚钱效应（先占位，后续全量统计后补充）
    stock_df["market_profit_effect"] = np.nan
    
    # 3.3 估值安全指标（widetable有则用，无则留空）
    if "pe_ttm" in stock_df.columns:
        stock_df["pe_ttm"] = stock_df["pe_ttm"]  # 复用原始PE数据
    else:
        stock_df["pe_ttm"] = np.nan
    if "pb" in stock_df.columns:
        stock_df["pb"] = stock_df["pb"]  # 复用原始PB数据
    else:
        stock_df["pb"] = np.nan
    
    # 3.4 趋势强度指标
    # 均线（5日、10日、20日）
    stock_df["ma5"] = stock_df["close"].rolling(5, min_periods=3).mean()
    stock_df["ma10"] = stock_df["close"].rolling(10, min_periods=5).mean()
    stock_df["ma20"] = stock_df["close"].rolling(20, min_periods=10).mean()
    # 均线多头排列（ma5>ma10>ma20，且均大于前一日均线）
    stock_df["is_ma_bull"] = (
        (stock_df["ma5"] > stock_df["ma10"]) & 
        (stock_df["ma10"] > stock_df["ma20"]) & 
        (stock_df["ma5"] > stock_df["ma5"].shift(1)) & 
        (stock_df["ma10"] > stock_df["ma10"].shift(1))
    ).astype(int)
    # 近5日涨幅（%）
    stock_df["profit_5d"] = (stock_df["close"] / stock_df["close"].shift(5) - 1) * 100
    
    # 保留最终字段（仅核心指标，剔除中间变量）
    keep_cols = [
        "stock_code", "trade_date", "close", "open", "high", "low", "volume",
        # 基础量价
        "pct_change", "turnover_rate", "amplitude",
        # 情绪关联
        "is_limit_up", "consec_limit_days", "market_profit_effect",
        # 估值安全
        "pe_ttm", "pb",
        # 趋势强度
        "ma5", "ma10", "ma20", "is_ma_bull", "profit_5d"
    ]
    # 过滤实际存在的字段（避免缺失字段报错）
    keep_cols = [col for col in keep_cols if col in stock_df.columns]
    return stock_df[keep_cols]

# -------------------------- 4. 二级分区存储（与分钟数据结构对齐） --------------------------
def save_daily_with_double_partition(stock_df, output_root):
    """按“stock_code=XXX/trade_date=YYYY-MM-DD”存储，与分钟数据路径一致"""
    stock_code = stock_df["stock_code"].iloc[0]
    # 1. 创建一级目录（股票代码）
    stock_dir = os.path.join(output_root, f"stock_code={stock_code}")
    os.makedirs(stock_dir, exist_ok=True)
    
    # 2. 按交易日分组，创建二级目录并写入
    for _, day_data in stock_df.groupby("trade_date"):
        trade_date_str = day_data["trade_date"].iloc[0].strftime("%Y-%m-%d")
        date_dir = os.path.join(stock_dir, f"trade_date={trade_date_str}")
        os.makedirs(date_dir, exist_ok=True)
        
        # 3. 写入文件（命名为daily_data.parquet，与分钟数据区分）
        output_file = os.path.join(date_dir, "daily_data.parquet")
        day_data.to_parquet(output_file, engine="pyarrow", index=False, compression=None)
    return

# -------------------------- 5. 补充全市场赚钱效应（基于已存储的日线数据） --------------------------
def calc_market_profit_effect():
    """统计全市场每日涨超5%个股比例，补充到个股的market_profit_effect字段"""
    print("\n" + "="*60)
    print("开始计算全市场赚钱效应（补充情绪指标）")
    print("="*60)
    
    # 1. 收集所有已处理的股票目录
    stock_dirs = [d for d in os.listdir(OUTPUT_PATH) if d.startswith("stock_code=")]
    if len(stock_dirs) == 0:
        print("❌ 无已处理的股票数据，无法计算赚钱效应")
        return
    
    # 2. 统计每日涨超5%的个股数量和总数量
    daily_stat = {}  # key=trade_date，value={"profit_5pct": 涨超5%数量, "total": 总数量}
    for stock_dir in tqdm(stock_dirs, desc="统计每日涨超5%个股", total=len(stock_dirs)):
        stock_code = stock_dir.split("=")[1]
        stock_path = os.path.join(OUTPUT_PATH, stock_dir)
        date_dirs = [d for d in os.listdir(stock_path) if d.startswith("trade_date=")]
        
        for date_dir in date_dirs:
            trade_date = pd.to_datetime(date_dir.split("=")[1])
            daily_file = os.path.join(stock_path, date_dir, "daily_data.parquet")
            if not os.path.exists(daily_file):
                continue
            
            # 读取单交易日数据
            day_df = pd.read_parquet(daily_file)
            if len(day_df) == 0 or "pct_change" not in day_df.columns:
                continue
            
            # 判断是否涨超5%
            pct = day_df.iloc[0]["pct_change"]
            if pd.isna(pct):
                continue
            
            # 更新统计
            if trade_date not in daily_stat:
                daily_stat[trade_date] = {"profit_5pct": 0, "total": 0}
            daily_stat[trade_date]["total"] += 1
            if pct >= 5:
                daily_stat[trade_date]["profit_5pct"] += 1
    
    # 3. 计算赚钱效应（涨超5%数量/总数量 × 100）并补充到个股数据
    for stock_dir in tqdm(stock_dirs, desc="补充赚钱效应到个股", total=len(stock_dirs)):
        stock_code = stock_dir.split("=")[1]
        stock_path = os.path.join(OUTPUT_PATH, stock_dir)
        date_dirs = [d for d in os.listdir(stock_path) if d.startswith("trade_date=")]
        
        for date_dir in date_dirs:
            trade_date = pd.to_datetime(date_dir.split("=")[1])
            daily_file = os.path.join(stock_path, date_dir, "daily_data.parquet")
            if not os.path.exists(daily_file) or trade_date not in daily_stat:
                continue
            
            # 读取并更新赚钱效应字段
            day_df = pd.read_parquet(daily_file)
            if "market_profit_effect" in day_df.columns:
                profit_effect = (daily_stat[trade_date]["profit_5pct"] / daily_stat[trade_date]["total"]) * 100
                day_df["market_profit_effect"] = round(profit_effect, 2)
                # 覆盖写入更新后的数据
                day_df.to_parquet(daily_file, engine="pyarrow", index=False, compression=None)
    
    print(f"✅ 赚钱效应计算完成：覆盖 {len(daily_stat)} 个交易日")
    return

# -------------------------- 6. 主函数（全流程控制） --------------------------
def main():
    # 1. 加载widetable并拆分股票数据
    stock_codes, stock_data_dict = load_and_split_widetable()
    if len(stock_codes) == 0:
        print("❌ 无股票数据可处理，程序退出")
        return
    total_stocks = len(stock_codes)
    print(f"✅ 共拆分出 {total_stocks} 只股票的有效数据")
    
    # 2. 单进程测试前5只股票（验证逻辑）
    print("\n" + "="*40)
    print("第一步：单进程测试前5只股票")
    print("="*40)
    test_success = 0
    for stock_code in stock_codes[:5]:
        try:
            process_stock_from_dict(stock_code, stock_data_dict[stock_code].copy())
            # 检查是否生成文件
            stock_output_dir = os.path.join(OUTPUT_PATH, f"stock_code={stock_code}")
            if os.path.exists(stock_output_dir) and len(os.listdir(stock_output_dir)) > 0:
                test_success += 1
        except Exception as e:
            print(f"⚠️  测试 {stock_code} 异常：{str(e)}")
    
    print(f"\n单进程测试结果：{test_success}/5 只股票处理成功")
if __name__ == "__main__":
    main()