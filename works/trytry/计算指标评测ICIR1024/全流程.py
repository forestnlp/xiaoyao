# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\trytry\计算指标评测ICIR1024\全流程.ipynb



# ----------------------------------------------------------------------import pandas as pd
import numpy as np
import talib as ta
import os
from scipy.stats import spearmanr

# --------------------------
# 1. 配置参数（请根据实际路径修改）
# --------------------------
CONFIG = {
    "raw_data_path": r'D:\workspace\xiaoyao\data\widetable.parquet',  # 原始行情数据路径
    "factor_output_path": r'./factor_data.parquet',  # 因子数据保存路径
    "log_path": r'./factor_calc_log.txt'  # 日志路径
}

# --------------------------
# 2. 工具函数
# --------------------------
def init_log():
    """初始化日志"""
    with open(CONFIG["log_path"], 'w', encoding='utf-8') as f:
        f.write(f"【因子计算启动】{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def log_msg(msg):
    """日志输出"""
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {msg}"
    print(log_line)
    with open(CONFIG["log_path"], 'a', encoding='utf-8') as f:
        f.write(log_line + "\n")

# --------------------------
# 3. 加载原始数据
# --------------------------
def load_raw_data():
    log_msg("开始加载原始行情数据...")
    # 读取parquet格式数据
    df = pd.read_parquet(CONFIG["raw_data_path"])
    # 基础预处理
    df["date"] = pd.to_datetime(df["date"])  # 日期格式转换
    df = df.sort_values(by=["stock_code", "date"]).reset_index(drop=True)  # 按股票、日期排序
    df = df.dropna(subset=["close", "open", "volume", "high", "low"])  # 过滤核心字段缺失值
    df = df.drop_duplicates(subset=["stock_code", "date"], keep="first")  # 去重
    
    # 验证核心字段
    must_have_cols = ['stock_code', 'date', 'close', 'open', 'volume', 'high', 'low']
    missing_cols = [col for col in must_have_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"原始数据缺少必需字段：{missing_cols}")
    
    log_msg(f"✅ 原始数据加载完成：{len(df)}条记录，{df['stock_code'].nunique()}只股票")
    return df

# --------------------------
# 4. 计算核心因子（可根据需求增删）
# --------------------------
def calculate_factors(df):
    log_msg("开始计算因子指标...")
    df = df.copy()
    
    # 4.1 趋势类因子
    # 连续上涨天数
    def calc_consecutive_up_days(close_series):
        up = close_series > close_series.shift(1)
        consecutive_up = up.groupby(up.ne(up.shift()).cumsum()).cumsum()
        return consecutive_up.astype(int)
    df['consecutive_up_days'] = df.groupby('stock_code', group_keys=False)['close'].apply(calc_consecutive_up_days)
    
    # 均线指标（MA5、MA20）
    df['ma5'] = df.groupby('stock_code')['close'].transform(lambda x: ta.SMA(x, 5))
    df['ma20'] = df.groupby('stock_code')['close'].transform(lambda x: ta.SMA(x, 20))
    df['ma5_ma20_ratio'] = df['ma5'] / df['ma20'].replace(0, 0.0001)  # 均线比值因子
    
    # 4.2 量能类因子
    # 5日量能比
    df['volume_ratio_5d'] = df.groupby('stock_code')['volume'].transform(
        lambda x: x / x.rolling(5, min_periods=1).mean().shift(1).replace(0, 0.0001)
    )
    # 连续放量天数（量能比≥1.0）
    def calc_consecutive_volume(group):
        vol_over = (group['volume_ratio_5d'] >= 1.0).astype(int)
        consecutive_vol = vol_over.groupby(vol_over.ne(vol_over.shift()).cumsum()).cumsum()
        group['consecutive_volume_days'] = consecutive_vol
        return group
    df = df.groupby('stock_code', group_keys=False).apply(calc_consecutive_volume)
    
    # 4.3 震荡类因子
    # RSI14
    df['rsi14'] = df.groupby('stock_code')['close'].transform(lambda x: ta.RSI(x, 14))
    # 布林带下限距离
    def calc_bollinger_lower_dist(group):
        upper, mid, lower = ta.BBANDS(group['close'], 20, 2, 2)
        group['bollinger_lower_dist'] = (group['close'] - lower) / lower.replace(0, 0.0001)
        return group
    df = df.groupby('stock_code', group_keys=False).apply(calc_bollinger_lower_dist)
    
    # 4.4 价格类因子
    # 30日涨幅
    df['rise_ratio_30d'] = df.groupby('stock_code')['close'].transform(
        lambda x: (x - x.shift(30)) / x.shift(30).replace(0, 0.0001)
    )
    # 单日涨幅
    df['daily_rise_ratio'] = (df['close'] - df['open']) / df['open'].replace(0, 0.0001)
    
    # 4.5 竞价类因子（若有auc_volume字段）
    if 'auc_volume' in df.columns:
        df['auction_volume_ratio'] = df['auc_volume'] / df.groupby('stock_code')['volume'].shift(1).replace(0, 0.0001)
        df['auction_rise_ratio'] = (df['open'] - df.groupby('stock_code')['close'].shift(1)) / df.groupby('stock_code')['close'].shift(1).replace(0, 0.0001)
        df['is_high_open'] = (df['open'] > df.groupby('stock_code')['close'].shift(1)).astype(int)
    else:
        log_msg("⚠️ 原始数据无auc_volume字段，竞价类因子填充为NaN")
        df['auction_volume_ratio'] = np.nan
        df['auction_rise_ratio'] = np.nan
        df['is_high_open'] = np.nan

    log_msg("✅ 所有因子计算完成")
    return df

# --------------------------
# 5. 保存因子数据
# --------------------------
def save_factor_data(df):
    # 保留核心字段（基础行情+所有因子）
    keep_cols = [
        'stock_code', 'date', 'close', 'open', 'volume', 'high', 'low',
        # 趋势类因子
        'consecutive_up_days', 'ma5', 'ma20', 'ma5_ma20_ratio',
        # 量能类因子
        'volume_ratio_5d', 'consecutive_volume_days',
        # 震荡类因子
        'rsi14', 'bollinger_lower_dist',
        # 价格类因子
        'rise_ratio_30d', 'daily_rise_ratio',
        # 竞价类因子
        'auction_volume_ratio', 'auction_rise_ratio', 'is_high_open'
    ]
    factor_df = df[keep_cols].copy()
    # 保存为parquet格式（压缩率高，读取快）
    factor_df.to_parquet(CONFIG["factor_output_path"], index=False)
    log_msg(f"✅ 因子数据保存完成：{CONFIG['factor_output_path']}")
    return factor_df

# --------------------------
# 主函数：因子计算流程
# --------------------------
def run_factor_calculation():
    try:
        init_log()
        raw_df = load_raw_data()
        factor_df = calculate_factors(raw_df)
        factor_df = save_factor_data(factor_df)
        log_msg(f"【因子计算全流程完成】累计{len(factor_df)}条因子记录")
        return factor_df
    except Exception as e:
        error_msg = f"❌ 因子计算失败：{str(e)}"
        log_msg(error_msg)
        raise

# 执行因子计算（运行该Cell时自动执行）
if __name__ == "__main__":
    factor_data = run_factor_calculation()

import pandas as pd
import numpy as np
import os

# --------------------------
# 1. 配置参数（与前一个Cell联动，可按需调整）
# --------------------------
CONFIG = {
    "factor_input_path": r'./factor_data.parquet',  # 前一个Cell输出的因子数据路径
    "selection_output_path": r'./selection_result.csv',  # 选股结果保存路径
    "daily_result_dir": r'./daily_selection_results',  # 每日选股结果目录
    "log_path": r'./selection_log.txt',  # 选股日志路径
    "top_n": 50,  # 每日选股数量（目标20-50只）
    # 选股条件参数（可按需放宽/收紧）
    "selection_params": {
        # 趋势条件
        "consecutive_up_days_min": 2,        # 连续上涨≥2天
        "consecutive_up_days_max": 8,        # 连续上涨≤8天
        "ma_trend": True,                    # 要求MA5≥MA20（均线多头）
        # 价格条件
        "rise_ratio_30d_min": 0.05,          # 30日涨幅≥5%
        "rise_ratio_30d_max": 0.5,           # 30日涨幅≤50%
        "daily_rise_min": -0.01,             # 单日涨幅≥-1%（允许小幅回调）
        "daily_rise_max": 0.05,              # 单日涨幅≤5%（避免暴涨）
        # 量能条件
        "volume_ratio_min": 0.7,             # 量能比≥0.7
        "consecutive_volume_min": 1,         # 连续放量≥1天
        # 风险条件
        "rsi_max": 70,                       # RSI≤70（避免过度超买）
        # 竞价条件（若有数据）
        "high_open_min": 0.003,              # 高开≥0.3%
        "high_open_max": 0.03,               # 高开≤3%
        "auction_volume_ratio_min": 0.02     # 竞价量比≥2%
    },
    # 得分权重（突出核心因子）
    "score_weights": {
        "auction_score": 0.3,    # 竞价得分权重
        "volume_score": 0.3,     # 量能得分权重
        "trend_score": 0.25,     # 趋势得分权重
        "risk_score": 0.15       # 风险得分权重
    }
}

# --------------------------
# 2. 工具函数
# --------------------------
def init_environment():
    """初始化环境（创建目录+日志）"""
    os.makedirs(CONFIG["daily_result_dir"], exist_ok=True)
    with open(CONFIG["log_path"], 'w', encoding='utf-8') as f:
        f.write(f"【因子选股启动】{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def log_msg(msg):
    """日志输出"""
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {msg}"
    print(log_line)
    with open(CONFIG["log_path"], 'a', encoding='utf-8') as f:
        f.write(log_line + "\n")

# --------------------------
# 3. 加载因子数据
# --------------------------
def load_factor_data():
    log_msg("开始加载因子数据...")
    # 读取前一个Cell生成的因子数据
    df = pd.read_parquet(CONFIG["factor_input_path"])
    df["date"] = pd.to_datetime(df["date"])  # 确保日期格式一致
    # 过滤无效因子记录
    df = df.dropna(subset=["ma5", "ma20", "rsi14", "rise_ratio_30d", "volume_ratio_5d"])
    log_msg(f"✅ 因子数据加载完成：{len(df)}条记录，{df['stock_code'].nunique()}只股票")
    return df

# --------------------------
# 4. 单日选股逻辑（筛选+得分排序）
# --------------------------
def select_single_day(daily_df):
    params = CONFIG["selection_params"]
    weights = CONFIG["score_weights"]
    
    # --------------------------
    # 步骤1：形态筛选（过滤不符合条件的标的）
    # --------------------------
    # 趋势条件
    cond1 = daily_df['consecutive_up_days'].between(params['consecutive_up_days_min'], params['consecutive_up_days_max'])
    cond2 = (daily_df['ma5'] >= daily_df['ma20']) if params['ma_trend'] else True
    # 价格条件
    cond3 = daily_df['rise_ratio_30d'].between(params['rise_ratio_30d_min'], params['rise_ratio_30d_max'])
    cond4 = daily_df['daily_rise_ratio'].between(params['daily_rise_min'], params['daily_rise_max'])
    # 量能条件
    cond5 = daily_df['volume_ratio_5d'] >= params['volume_ratio_min']
    cond6 = daily_df['consecutive_volume_days'] >= params['consecutive_volume_min']
    # 风险条件
    cond7 = daily_df['rsi14'] <= params['rsi_max']
    # 竞价条件（若数据非空则启用）
    cond8 = True
    cond9 = True
    if not pd.isna(daily_df['auction_rise_ratio']).all():
        cond8 = daily_df['auction_rise_ratio'].between(params['high_open_min'], params['high_open_max'])
        cond9 = daily_df['auction_volume_ratio'] >= params['auction_volume_ratio_min']
    
    # 组合筛选条件
    total_cond = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8 & cond9
    filtered_df = daily_df[total_cond].copy()
    if len(filtered_df) == 0:
        return pd.DataFrame()
    
    # --------------------------
    # 步骤2：因子得分排序（按权重计算综合得分）
    # --------------------------
    # 1. 竞价得分（0-100分）
    if 'auction_rise_ratio' in filtered_df.columns and not pd.isna(filtered_df['auction_rise_ratio']).all():
        filtered_df['auction_rise_score'] = ((filtered_df['auction_rise_ratio'] - params['high_open_min']) / 
                                           (params['high_open_max'] - params['high_open_min'] + 1e-8)) * 50
        filtered_df['auction_vol_score'] = ((filtered_df['auction_volume_ratio'] - params['auction_volume_ratio_min']) / 
                                          (0.1 - params['auction_volume_ratio_min'] + 1e-8)) * 50
        filtered_df['auction_score'] = (filtered_df['auction_rise_score'] + filtered_df['auction_vol_score']).clip(0, 100)
    else:
        filtered_df['auction_score'] = 50  # 无数据时赋默认分
    
    # 2. 量能得分（0-100分）
    filtered_df['volume_score'] = ((filtered_df['volume_ratio_5d'] - params['volume_ratio_min']) / 
                                  (2.0 - params['volume_ratio_min'] + 1e-8)) * 100
    filtered_df['volume_score'] = filtered_df['volume_score'].clip(0, 100)
    
    # 3. 趋势得分（0-100分）
    filtered_df['up_days_score'] = (filtered_df['consecutive_up_days'] / params['consecutive_up_days_max']) * 60
    filtered_df['ma_score'] = ((filtered_df['ma5'] - filtered_df['ma20']) / filtered_df['ma20'].replace(0, 0.0001) * 1000).clip(0, 40)
    filtered_df['trend_score'] = filtered_df['up_days_score'] + filtered_df['ma_score']
    
    # 4. 风险得分（0-100分）
    filtered_df['rsi_score'] = 100 - ((filtered_df['rsi14'] - 30) / (params['rsi_max'] - 30 + 1e-8)) * 100
    filtered_df['risk_score'] = filtered_df['rsi_score'].clip(0, 100)
    
    # 5. 综合得分（按权重加权）
    filtered_df['total_score'] = (
        filtered_df['auction_score'] * weights['auction_score'] +
        filtered_df['volume_score'] * weights['volume_score'] +
        filtered_df['trend_score'] * weights['trend_score'] +
        filtered_df['risk_score'] * weights['risk_score']
    )
    
    # 取前N只标的
    selected_df = filtered_df.sort_values(by='total_score', ascending=False).head(CONFIG["top_n"]).reset_index(drop=True)
    return selected_df

# --------------------------
# 5. 全周期选股（遍历所有交易日）
# --------------------------
def run_selection(factor_df):
    log_msg("开始全周期选股...")
    # 获取所有交易日
    trade_dates = sorted(factor_df['date'].dt.date.unique())
    all_results = []
    
    for trade_date in trade_dates:
        log_msg(f"\n===== 处理交易日：{trade_date.strftime('%Y-%m-%d')} =====")
        # 提取当日数据
        daily_df = factor_df[factor_df['date'].dt.date == trade_date].copy()
        if len(daily_df) < 500:  # 当日数据过少，跳过
            log_msg(f"⚠️ 当日数据异常（记录数{len(daily_df)}<500），跳过")
            continue
        
        # 执行单日选股
        selected_df = select_single_day(daily_df)
        if selected_df.empty:
            log_msg("⚠️ 无符合条件标的，跳过")
            continue
        
        # 补充交易日信息并保存每日结果
        selected_df['trade_date'] = trade_date.strftime('%Y-%m-%d')
        date_str = trade_date.strftime('%Y%m%d')
        daily_save_path = os.path.join(CONFIG["daily_result_dir"], f"selection_{date_str}.csv")
        selected_df.to_csv(daily_save_path, index=False, encoding='utf-8-sig')
        
        # 收集全周期结果
        all_results.append(selected_df)
        log_msg(f"✅ 当日选股完成：{len(selected_df)}只标的")
    
    # 保存全周期选股结果
    if all_results:
        final_result = pd.concat(all_results, ignore_index=True)
        final_result.to_csv(CONFIG["selection_output_path"], index=False, encoding='utf-8-sig')
        log_msg(f"\n✅ 全周期选股完成！累计{len(final_result)}条记录，路径：{CONFIG['selection_output_path']}")
        return final_result
    else:
        log_msg(f"\n⚠️ 无选股结果，请调整选股参数")
        return pd.DataFrame()

# --------------------------
# 主函数：选股全流程
# --------------------------
def main_selection():
    try:
        init_environment()
        factor_df = load_factor_data()
        selection_result = run_selection(factor_df)
        return selection_result
    except Exception as e:
        error_msg = f"❌ 选股失败：{str(e)}"
        log_msg(error_msg)
        raise

# 执行选股（运行该Cell时自动执行）
if __name__ == "__main__":
    selection_data = main_selection()

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os

# --------------------------
# 1. 配置参数（与前两个Cell联动）
# --------------------------
CONFIG = {
    "factor_input_path": r'./factor_data.parquet',  # 第一个Cell的因子数据
    "selection_input_path": r'./selection_result.csv',  # 第二个Cell的选股结果
    "icir_output_path": r'./icir_result.csv',  # IC/IR结果保存路径
    "log_path": r'./icir_calc_log.txt',  # 计算日志路径
    "future_return_days": 5,  # 预测未来5日收益率（可调整为1/10日）
    "buy_delay": 1,  # 选股后T+1买入（与回测规则对齐）
    "sell_delay": 5,  # T+5卖出（与future_return_days对应）
}

# --------------------------
# 2. 工具函数
# --------------------------
def init_log():
    """初始化IC/IR计算日志"""
    with open(CONFIG["log_path"], 'w', encoding='utf-8') as f:
        f.write(f"【IC/IR计算启动】{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def log_msg(msg):
    """日志输出"""
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {msg}"
    print(log_line)
    with open(CONFIG["log_path"], 'a', encoding='utf-8') as f:
        f.write(log_line + "\n")

# --------------------------
# 3. 加载基础数据（因子+选股结果+收盘价）
# --------------------------
def load_basic_data():
    log_msg("开始加载IC/IR计算所需数据...")
    
    # 3.1 加载因子数据（获取收盘价和交易序列，用于计算收益率）
    factor_df = pd.read_parquet(CONFIG["factor_input_path"])
    factor_df["date"] = pd.to_datetime(factor_df["date"])
    # 按股票代码和日期排序，添加交易序列（用于匹配延迟后的价格）
    factor_df = factor_df.sort_values(by=["stock_code", "date"]).reset_index(drop=True)
    factor_df["trade_seq"] = factor_df.groupby("stock_code").cumcount()  # 每个股票的交易日序列
    price_df = factor_df[["stock_code", "date", "trade_seq", "close"]].dropna(subset=["close"])
    
    # 3.2 加载选股结果（获取因子值和选股日期）
    selection_df = pd.read_csv(CONFIG["selection_input_path"])
    selection_df["date"] = pd.to_datetime(selection_df["date"])  # 选股日期（T日）
    # 合并交易序列（与价格数据对齐）
    selection_df = pd.merge(
        selection_df,
        factor_df[["stock_code", "date", "trade_seq"]],
        on=["stock_code", "date"],
        how="left"
    ).dropna(subset=["trade_seq", "total_score"])
    selection_df["trade_seq"] = selection_df["trade_seq"].astype(int)
    
    log_msg(f"✅ 数据加载完成：")
    log_msg(f"- 因子数据：{len(factor_df)}条记录")
    log_msg(f"- 选股结果：{len(selection_df)}条记录")
    return price_df, selection_df

# --------------------------
# 4. 计算未来N日收益率（与选股规则对齐：T+1买→T+5卖）
# --------------------------
def calculate_future_returns(price_df, selection_df):
    log_msg(f"开始计算未来{CONFIG['future_return_days']}日收益率...")
    
    # 构建价格映射表：(股票代码, 交易序列) → 收盘价
    price_map = price_df.set_index(["stock_code", "trade_seq"])["close"].to_dict()
    
    # 计算买入和卖出序列（T日选股→T+buy_delay买入→T+sell_delay卖出）
    selection_df["buy_seq"] = selection_df["trade_seq"] + CONFIG["buy_delay"]
    selection_df["sell_seq"] = selection_df["trade_seq"] + CONFIG["sell_delay"]
    
    # 匹配买入价和卖出价
    def get_price(stock_code, seq):
        return price_map.get((stock_code, seq), np.nan)
    selection_df["buy_price"] = selection_df.apply(lambda x: get_price(x["stock_code"], x["buy_seq"]), axis=1)
    selection_df["sell_price"] = selection_df.apply(lambda x: get_price(x["stock_code"], x["sell_seq"]), axis=1)
    
    # 计算收益率（百分比）
    selection_df["return_rate"] = (selection_df["sell_price"] - selection_df["buy_price"]) / \
                                selection_df["buy_price"].replace(0, 0.0001) * 100
    # 过滤有效收益率记录
    valid_df = selection_df.dropna(subset=["buy_price", "sell_price", "return_rate"]).copy()
    
    log_msg(f"✅ 收益率计算完成：")
    log_msg(f"- 有效记录数：{len(valid_df)}条（与回测有效交易数一致）")
    log_msg(f"- 平均收益率：{valid_df['return_rate'].mean():.2f}%")
    return valid_df

# --------------------------
# 5. 计算IC/IR（核心：因子值与未来收益率的相关性）
# --------------------------
def calculate_ic_ir(valid_df):
    log_msg("开始计算IC/IR指标...")
    
    # 5.1 按选股日分组，计算每日IC（Spearman秩相关系数）
    def calc_daily_ic(group):
        # 单组至少2条记录才计算相关系数，避免随机误差
        if len(group) < 2:
            return pd.Series({
                "IC": np.nan,
                "P_Value": np.nan,
                "Trade_Count": len(group),
                "Avg_Return": np.nan
            })
        # 计算因子值（total_score）与收益率（return_rate）的Spearman相关系数
        ic, p_value = spearmanr(group["total_score"], group["return_rate"])
        return pd.Series({
            "IC": ic,
            "P_Value": p_value,
            "Trade_Count": len(group),
            "Avg_Return": group["return_rate"].mean()
        })
    
    # 执行每日IC计算
    daily_ic = valid_df.groupby("date", observed=True).apply(
        calc_daily_ic, include_groups=False  # 消除pandas版本警告
    ).reset_index()
    # 过滤无效IC值（剔除NaN）
    daily_ic = daily_ic.dropna(subset=["IC"]).sort_values(by="date")
    
    # 5.2 计算核心指标
    ic_mean = daily_ic["IC"].mean()  # 平均IC
    ic_std = daily_ic["IC"].std()    # IC标准差
    ir = ic_mean / ic_std if ic_std != 0 else 0  # 信息比率（IR）
    significant_days = len(daily_ic[daily_ic["P_Value"] < 0.05])  # IC显著（P<0.05）的天数
    positive_ic_ratio = np.mean(daily_ic["IC"] > 0)  # IC为正的天数占比
    
    # 5.3 输出结果
    print("\n" + "="*60)
    print(f"因子IC/IR有效性分析结果（预测未来{CONFIG['future_return_days']}日收益）")
    print("="*60)
    print(f"平均IC值：{ic_mean:.4f}        （绝对值>0.05为强因子，>0.02为弱因子）")
    print(f"IC标准差：{ic_std:.4f}         （越小说明因子预测越稳定）")
    print(f"信息比率IR：{ir:.4f}           （>0.5为有效因子，>1.0为优秀因子）")
    print(f"有效计算日数：{len(daily_ic)}天  （每日至少2条有效交易记录）")
    print(f"IC显著天数（P<0.05）：{significant_days}/{len(daily_ic)}  （占比：{significant_days/len(daily_ic):.2%}）")
    print(f"IC为正天数占比：{positive_ic_ratio:.2%}  （>50%说明因子正向预测能力）")
    print("="*60)
    
    # 5.4 保存结果（每日IC明细+核心指标）
    # 保存每日IC明细
    daily_ic.to_csv(CONFIG["icir_output_path"], index=False, encoding="utf-8-sig")
    # 保存核心指标汇总
    summary_content = f"""
【IC/IR计算汇总】
计算规则：T日选股（因子值total_score）→ T+1买入 → T+5卖出
计算周期：{daily_ic['date'].min().strftime('%Y-%m-%d')} ~ {daily_ic['date'].max().strftime('%Y-%m-%d')}
核心指标：
- 平均IC值：{ic_mean:.4f}
- IC标准差：{ic_std:.4f}
- 信息比率IR：{ir:.4f}
- 有效计算日数：{len(daily_ic)}天
- IC显著天数占比：{significant_days/len(daily_ic):.2%}
- IC为正天数占比：{positive_ic_ratio:.2%}
"""
    with open(CONFIG["icir_output_path"].replace(".csv", "_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    log_msg(f"✅ IC/IR结果保存完成：")
    log_msg(f"- 每日明细：{CONFIG['icir_output_path']}")
    log_msg(f"- 核心汇总：{CONFIG['icir_output_path'].replace('.csv', '_summary.txt')}")
    return daily_ic, ic_mean, ir

# --------------------------
# 主函数：IC/IR计算全流程
# --------------------------
def main_icir_calc():
    try:
        init_log()
        # 步骤1：加载数据
        price_df, selection_df = load_basic_data()
        # 步骤2：计算收益率
        valid_df = calculate_future_returns(price_df, selection_df)
        if len(valid_df) == 0:
            log_msg("❌ 无有效收益率数据，无法计算IC/IR")
            return None, None, None
        # 步骤3：计算IC/IR
        daily_ic, ic_mean, ir = calculate_ic_ir(valid_df)
        return daily_ic, ic_mean, ir
    except Exception as e:
        error_msg = f"❌ IC/IR计算失败：{str(e)}"
        log_msg(error_msg)
        raise

# 执行IC/IR计算（运行该Cell时自动执行）
if __name__ == "__main__":
    daily_ic_result, ic_mean_value, ir_value = main_icir_calc()