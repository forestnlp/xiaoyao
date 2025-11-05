# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\preprocessor\factors.ipynb



# ----------------------------------------------------------------------import pandas as pd
import numpy as np
import talib as ta
import os

# --------------------------
# 配置参数（本地存储路径）
# --------------------------
CONFIG = {
    "raw_data_path": r'D:\workspace\xiaoyao\data\widetable.parquet',  # 原始widetable路径
    "factortable_output_path": r'./factortable.parquet',  # 本地输出路径
    "sample_output_path": r'./factortable_sample.csv',  # 样本文件
    "log_path": r'./calc_factortable_log.txt'  # 日志路径
}

# --------------------------
# 工具函数
# --------------------------
def init_environment():
    """初始化本地目录和日志"""
    os.makedirs(os.path.dirname(CONFIG["factortable_output_path"]), exist_ok=True)
    with open(CONFIG["log_path"], 'w', encoding='utf-8') as f:
        f.write(f"【Factortable计算启动】{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_msg("✅ 环境初始化完成，输出路径：./factortable.parquet")

def log_msg(msg):
    """日志输出到控制台和文件"""
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {msg}"
    print(log_line)
    with open(CONFIG["log_path"], 'a', encoding='utf-8') as f:
        f.write(log_line + "\n")

# --------------------------
# 核心逻辑：计算实验所需指标
# --------------------------
def calculate_factortable():
    try:
        init_environment()
        
        # 1. 加载widetable并预处理（参考旧代码逻辑）
        log_msg("加载widetable数据...")
        df = pd.read_parquet(CONFIG["raw_data_path"])
        
        # 验证必需字段（与旧代码一致）
        must_have_cols = ['stock_code', 'date', 'close', 'open', 'volume', 'high', 'low']
        missing_cols = [col for col in must_have_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"widetable缺少必需字段：{missing_cols}")
        log_msg(f"✅ 数据加载完成：{len(df)}条记录，{df['stock_code'].nunique()}只股票")
        
        # 预处理（排序+日期转换+去重）
        df = df.sort_values(by=['stock_code', 'date']).reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')  # 字符串转datetime
        df = df.dropna(subset=['date', 'close', 'open'])  # 过滤无效日期和价格
        df = df.drop_duplicates(subset=['stock_code', 'date'], keep='first')  # 去重
        log_msg(f"✅ 预处理完成：{len(df)}条有效记录")

        # 2. 计算实验所需核心指标（参考旧代码，仅保留必要指标）
        log_msg("计算核心指标...")
        
        # （1）连续上涨天数（与旧代码一致）
        def calc_consecutive_up_days(close_series):
            up = close_series > close_series.shift(1)
            consecutive_up = up.groupby(up.ne(up.shift()).cumsum()).cumsum()
            return consecutive_up.astype(int)
        df['consecutive_up_days'] = df.groupby('stock_code', group_keys=False)['close'].apply(calc_consecutive_up_days)
        log_msg("✅ 连续上涨天数计算完成")
        
        # （2）竞价相关指标（与旧代码一致，适配widetable的auc_volume）
        if 'auc_volume' in df.columns:
            df['auction_volume_ratio'] = df['auc_volume'] / \
                                       df.groupby('stock_code')['volume'].shift(1).replace(0, 0.0001)
        else:
            df['auction_volume_ratio'] = np.nan
            log_msg("⚠️ 无auc_volume字段，auction_volume_ratio填充为NaN")
        df['auction_rise_ratio'] = (df['open'] - df.groupby('stock_code')['close'].shift(1)) / \
                                  df.groupby('stock_code')['close'].shift(1).replace(0, 0.0001)
        df['is_high_open'] = df['open'] > df.groupby('stock_code')['close'].shift(1)
        log_msg("✅ 竞价指标计算完成")
        
        # （3）均线指标（与旧代码一致）
        df['ma5'] = df.groupby('stock_code')['close'].transform(lambda x: ta.SMA(x, 5))
        df['ma20'] = df.groupby('stock_code')['close'].transform(lambda x: ta.SMA(x, 20))
        log_msg("✅ 均线指标计算完成")
        
        # （4）MACD指标（与旧代码一致）
        def calc_macd(close_series):
            macd, signal, hist = ta.MACD(close_series, 12, 26, 9)
            return pd.DataFrame({
                'macd_line': macd,
                'signal_line': signal,
                'macd_hist': hist
            }, index=close_series.index)
        df = df.join(df.groupby('stock_code', group_keys=False)['close'].apply(calc_macd))
        log_msg("✅ MACD指标计算完成")
        
        # （5）量能指标（与旧代码一致）
        df['volume_ratio_5d'] = df.groupby('stock_code')['volume'].transform(
            lambda x: x / x.rolling(5, min_periods=1).mean().shift(1).replace(0, 0.0001)
        )
        log_msg("✅ 量能指标计算完成")
        
        # （6）回调指标（核心修复：参考旧代码，避免整数转换错误）
        def calc_30d_high(group):
            # 30日最高价（用close还是high？旧代码用close，这里保持一致）
            group['high_30d'] = group['close'].rolling(30, min_periods=5).max()  # 降低min_periods避免过多NaN
            
            # 计算最高价对应的索引（不立即转int，避免NaN错误）
            def get_high_idx(window):
                return window.idxmax()  # 可能返回NaN，但先保留float
            group['high_idx_30d'] = group['close'].rolling(30, min_periods=5).apply(get_high_idx, raw=False)
            
            # 匹配最高价日期（旧代码逻辑：用索引定位）
            # 处理NaN索引：填充为当前行索引（避免后续loc报错）
            group['high_idx_30d'] = group['high_idx_30d'].fillna(group.index.to_series())
            # 转换为整数索引（此时已无NaN，可安全转换）
            group['high_idx_30d'] = group['high_idx_30d'].astype(int)
            # 用索引匹配日期
            group['high_date_30d'] = group.loc[group['high_idx_30d'], 'date'].values
            return group.drop(columns=['high_idx_30d'])
        
        # 应用计算并合并结果（参考旧代码的join方式）
        high_30d_df = df.groupby('stock_code', group_keys=False)[['close', 'date']].apply(calc_30d_high)
        df = df.join(high_30d_df[['high_30d', 'high_date_30d']], how='left')
        
        # 转换日期并过滤无效值（关键：删除日期为空的记录）
        df['high_date_30d'] = pd.to_datetime(df['high_date_30d'], errors='coerce')
        df = df.dropna(subset=['high_date_30d'])
        
        # 计算回撤比例和天数（与旧代码一致）
        df['pullback_ratio'] = (df['high_30d'] - df['close']) / df['high_30d'].replace(0, 0.0001)
        df['pullback_days'] = (df['date'] - df['high_date_30d']).dt.days
        log_msg("✅ 回调指标计算完成（修复整数转换错误）")
        
        # （7）支撑位与RSI（与旧代码一致）
        df['bollinger_lower'] = df.groupby('stock_code', group_keys=False)['close'].apply(
            lambda x: ta.BBANDS(x, 20, 2, 2)[2]
        )
        df['rsi14'] = df.groupby('stock_code')['close'].transform(lambda x: ta.RSI(x, 14))
        log_msg("✅ 支撑位与RSI计算完成")
        
        # （8）振幅（与旧代码一致，用close做分母）
        df['amplitude'] = (df['high'] - df['low']) / df['close'].replace(0, 0.0001)
        log_msg("✅ 振幅计算完成")
        
        # 3. 筛选字段并保存（仅保留实验所需）
        log_msg("筛选字段并保存...")
        keep_cols = [
            # 基础字段
            'stock_code', 'date', 'close', 'open', 'volume', 'high', 'low', 'amplitude',
            # 趋势与竞价
            'consecutive_up_days', 'auction_rise_ratio', 'auction_volume_ratio', 'is_high_open',
            # 均线与MACD
            'ma5', 'ma20', 'macd_line', 'signal_line', 'macd_hist',
            # 量能与回调
            'volume_ratio_5d', 'high_30d', 'high_date_30d', 'pullback_ratio', 'pullback_days',
            # 支撑与超买超卖
            'bollinger_lower', 'rsi14'
        ]
        df_final = df[keep_cols].copy()
        # 最终过滤关键指标（确保选股可用）
        df_final = df_final.dropna(subset=['close', 'ma5', 'ma20', 'high_30d', 'rsi14'])
        log_msg(f"✅ 最终Factortable：{len(df_final)}条记录，{df_final['stock_code'].nunique()}只股票")
        
        # 保存到本地
        df_final.to_parquet(CONFIG["factortable_output_path"], index=False)
        df_final.sample(5, random_state=42).to_csv(CONFIG["sample_output_path"], index=False, encoding='utf-8-sig')
        log_msg(f"✅ 保存完成：")
        log_msg(f"📁 主文件：{CONFIG['factortable_output_path']}")
        log_msg(f"📁 样本文件：{CONFIG['sample_output_path']}")

    except Exception as e:
        log_msg(f"❌ 计算失败：{str(e)}")
        raise

# --------------------------
# 执行入口
# --------------------------
if __name__ == "__main__":
    calculate_factortable()

