# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\trytry\养家心法\指标计算.ipynb



# ----------------------------------------------------------------------import pandas as pd
import numpy as np
import talib as ta
import os
from datetime import datetime

# --------------------------
# 1. 配置参数（不变）
# --------------------------
CONFIG = {
    "widetable_path": r'D:\workspace\xiaoyao\data\widetable.parquet',
    "factor_output_path": r'./yangjia_factor_data.parquet',
    "log_path": r'./yangjia_factor_calc_log.txt',
    "market_strength": {
        "trend_weight": 0.4,
        "activity_weight": 0.3,
        "profit_weight": 0.3
    }
}

# --------------------------
# 2. 工具函数（不变）
# --------------------------
def init_log():
    with open(CONFIG["log_path"], 'w', encoding='utf-8') as f:
        f.write(f"【养家心法因子计算启动】{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def log_msg(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {msg}"
    print(log_line)
    with open(CONFIG["log_path"], 'a', encoding='utf-8') as f:
        f.write(log_line + "\n")

# --------------------------
# 3. 加载宽表数据（不变）
# --------------------------
def load_widetable_data():
    log_msg("开始加载股票宽表数据...")
    core_cols = [
        'date', 'stock_code', 'stock_name',
        'open', 'close', 'high', 'low', 'pre_close', 'volume', 'money',
        'auc_volume', 'auc_money',
        'sw_l1_industry_code', 'sw_l1_industry_name',
        'circulating_market_cap', 'turnover_ratio'
    ]
    df = pd.read_parquet(CONFIG["widetable_path"], columns=core_cols)
    
    # 预处理
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["stock_code", "date"]).reset_index(drop=True)
    if 'paused' in df.columns:
        df = df[df['paused'] == 0.0].copy()
    df = df[df['circulating_market_cap'] >= 50].copy()  # 流通市值≥50亿
    df = df.dropna(subset=["close", "volume", "sw_l1_industry_name"])
    
    log_msg(f"✅ 宽表数据加载完成：{len(df)}条记录，{df['stock_code'].nunique()}只股票，{df['sw_l1_industry_name'].nunique()}个申万一级行业")
    return df

# --------------------------
# 4. 计算养家核心因子（兼容低版本pandas）
# --------------------------
def calculate_yangjia_factors(df):
    log_msg("开始计算养家心法因子...")
    df = df.copy()
    
    # 4.1 基础趋势/量能因子（关键修复：移除include_groups，用传统方式计算连续上涨天数）
    def calc_consecutive_up(group):
        # 低版本pandas中，group包含分组键，这里显式取需要的列计算
        up = group['close'] > group['pre_close']  # 仅用close和pre_close列
        consecutive_up = up.groupby(up.ne(up.shift()).cumsum()).cumsum()
        return consecutive_up.astype(int)
    # 移除include_groups参数，兼容低版本
    df['consecutive_up_days'] = df.groupby('stock_code', group_keys=False).apply(calc_consecutive_up)
    
    # 5日量能比（不变）
    df['volume_ratio_5d'] = df.groupby('stock_code')['volume'].transform(
        lambda x: x / x.rolling(5, min_periods=1).mean().shift(1).replace(0, 0.0001)
    )
    
    # 均线因子（不变）
    df['ma5'] = df.groupby('stock_code')['close'].transform(lambda x: ta.SMA(x, 5))
    df['ma20'] = df.groupby('stock_code')['close'].transform(lambda x: ta.SMA(x, 20))
    df['is_ma_bull'] = (df['ma5'] >= df['ma20']).astype(int)
    
    # 4.2 市场强弱因子（不变，已修复列名问题）
    daily_market = df.groupby('date').agg(
        market_rise_ratio=('close', lambda x: (x > x.shift(1)).sum() / len(x) if len(x) > 1000 else 0),
        market_avg_turnover=('turnover_ratio', 'mean'),
        market_zt_ratio=('close', lambda x: ((x / x.shift(1) - 1) >= 0.095).sum() / len(x) if len(x) > 1000 else 0),
        market_avg_money=('money', 'mean')
    ).reset_index()
    
    daily_market['trend_score'] = daily_market['market_rise_ratio'].clip(0.3, 0.7).apply(lambda x: (x-0.3)/0.4*100)
    daily_market['activity_score'] = daily_market['market_avg_turnover'].clip(1, 5).apply(lambda x: (x-1)/4*100)
    daily_market['profit_score'] = daily_market['market_zt_ratio'].clip(0.02, 0.1).apply(lambda x: (x-0.02)/0.08*100)
    weights = CONFIG["market_strength"]
    daily_market['market_strength_score'] = (
        daily_market['trend_score'] * weights['trend_weight'] +
        daily_market['activity_score'] * weights['activity_weight'] +
        daily_market['profit_score'] * weights['profit_weight']
    )
    
    df = pd.merge(df, daily_market[['date', 'market_strength_score']], on='date', how='left')
    
    # 4.3 板块强势因子（不变）
    daily_industry = df.groupby(['date', 'sw_l1_industry_name']).agg(
        industry_avg_rise=('close', lambda x: ((x / x.shift(1) - 1).mean()) * 100),
        industry_volume_ratio=('volume_ratio_5d', 'mean'),
        industry_zt_ratio=('close', lambda x: ((x / x.shift(1) - 1) >= 0.095).sum() / len(x))
    ).reset_index()
    
    daily_industry['industry_strength_score'] = (
        daily_industry['industry_avg_rise'].clip(1, 5).apply(lambda x: (x-1)/4*40) +
        daily_industry['industry_volume_ratio'].clip(1.0, 2.0).apply(lambda x: (x-1)/1*30) +
        daily_industry['industry_zt_ratio'].clip(0.05, 0.2).apply(lambda x: (x-0.05)/0.15*30)
    )
    
    df = pd.merge(
        df,
        daily_industry[['date', 'sw_l1_industry_name', 'industry_strength_score', 'industry_avg_rise']],
        on=['date', 'sw_l1_industry_name'],
        how='left'
    )
    
    # 4.4 龙头特征因子（不变）
    df['leader_drive'] = ((df['close'] / df['pre_close'] - 1) * 100) - df['industry_avg_rise']
    df['leader_drive_score'] = df['leader_drive'].clip(-2, 8).apply(lambda x: (x+2)/10*100)
    
    df['auction_volume_ratio'] = df['auc_volume'] / df.groupby('stock_code')['volume'].shift(1).replace(0, 0.0001)
    df['fund_focus_score'] = (
        df['auction_volume_ratio'].clip(0.02, 0.1).apply(lambda x: (x-0.02)/0.08*50) +
        df['turnover_ratio'].clip(2, 8).apply(lambda x: (x-2)/6*50)
    )
    
    df['leader_trend_score'] = (
        df['consecutive_up_days'].clip(2, 7).apply(lambda x: (x-2)/5*60) +
        df['is_ma_bull'] * 40
    )
    
    df['leader_total_score'] = (
        df['leader_drive_score'] * 0.4 +
        df['fund_focus_score'] * 0.3 +
        df['leader_trend_score'] * 0.3
    )
    
    log_msg("✅ 养家心法因子计算完成")
    return df

# --------------------------
# 5. 保存因子数据（不变）
# --------------------------
def save_yangjia_factor(df):
    keep_cols = [
        'date', 'stock_code', 'stock_name', 'open', 'close', 'volume',
        'consecutive_up_days', 'volume_ratio_5d', 'ma5', 'ma20', 'is_ma_bull',
        'market_strength_score',
        'sw_l1_industry_name', 'industry_strength_score',
        'leader_drive_score', 'fund_focus_score', 'leader_trend_score', 'leader_total_score'
    ]
    factor_df = df[keep_cols].copy()
    factor_df.to_parquet(CONFIG["factor_output_path"], index=False)
    log_msg(f"✅ 养家因子保存完成：{CONFIG['factor_output_path']}")
    return factor_df

# --------------------------
# 主函数（不变）
# --------------------------
def run_yangjia_factor_calc():
    try:
        init_log()
        widetable_df = load_widetable_data()
        factor_df = calculate_yangjia_factors(widetable_df)
        factor_df = save_yangjia_factor(factor_df)
        log_msg(f"【养家因子计算完成】累计{len(factor_df)}条因子记录")
        return factor_df
    except Exception as e:
        log_msg(f"❌ 因子计算失败：{str(e)}")
        raise

# 执行计算
if __name__ == "__main__":
    yangjia_factor = run_yangjia_factor_calc()

