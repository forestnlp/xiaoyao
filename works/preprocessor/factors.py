# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\preprocessor\factors.ipynb



# ----------------------------------------------------------------------import pandas as pd
import numpy as np

# 读取widetable.parquet文件
file_path = r'D:\workspace\xiaoyao\data\widetable.parquet'
df = pd.read_parquet(file_path)

# --------------------------
# 基础准备：数据排序与初始化
# --------------------------
# 按股票代码+日期正序排列，确保时间逻辑正确（规避隐性未来函数）
df = df.sort_values(by=['stock_code', 'date']).reset_index(drop=True)

# --------------------------
# 1. 趋势类指标：移动平均线（MA）
# --------------------------
df['ma5'] = df.groupby('stock_code')['close'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()  # 5日移动平均
)
df['ma10'] = df.groupby('stock_code')['close'].transform(
    lambda x: x.rolling(window=10, min_periods=1).mean()  # 10日移动平均
)
df['ma20'] = df.groupby('stock_code')['close'].transform(
    lambda x: x.rolling(window=20, min_periods=1).mean()  # 20日移动平均
)
df['ma60'] = df.groupby('stock_code')['close'].transform(
    lambda x: x.rolling(window=60, min_periods=1).mean()  # 60日移动平均
)

# --------------------------
# 2. 震荡类指标：相对强弱指数（RSI）
# --------------------------
def calculate_rsi(series, window=14):
    delta = series.diff()  # 计算价格涨跌差
    gain = delta.where(delta > 0, 0)  # 上涨幅度（下跌记为0）
    loss = -delta.where(delta < 0, 0)  # 下跌幅度（上涨记为0）
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()  # 平均上涨幅度
    avg_loss = loss.rolling(window=window, min_periods=1).mean()  # 平均下跌幅度
    
    avg_loss = avg_loss.replace(0, 0.0001)  # 避免除零错误
    rs = avg_gain / avg_loss  # 相对强弱
    rsi = 100 - (100 / (1 + rs))  # RSI值（0-100）
    return rsi

df['rsi14'] = df.groupby('stock_code')['close'].transform(
    lambda x: calculate_rsi(x, window=14)  # 14日RSI
)

# --------------------------
# 3. 趋势类指标：MACD（指数平滑异同平均线）
# --------------------------
def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()  # 快速EMA（12日）
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()  # 慢速EMA（26日）
    macd_line = ema_fast - ema_slow  # MACD线（快慢EMA差值）
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()  # 信号线（9日EMA）
    macd_hist = macd_line - signal_line  # MACD柱状图（MACD线-信号线）
    return pd.DataFrame({
        'macd_line': macd_line,
        'signal_line': signal_line,
        'macd_hist': macd_hist
    })

# 按股票分组计算MACD，合并结果（加rsuffix避免列名冲突）
macd_results = df.groupby('stock_code')['close'].apply(calculate_macd)
df = df.join(macd_results.reset_index(level=0, drop=True), rsuffix='_calc')

# --------------------------
# 4. 波动类指标：布林带（Bollinger Bands）
# --------------------------
def calculate_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window, min_periods=1).mean()  # 中轨（20日MA）
    rolling_std = series.rolling(window=window, min_periods=1).std().replace(0, 0.0001)  # 标准差（避免除零）
    upper_band = rolling_mean + (rolling_std * num_std)  # 上轨（中轨+2倍标准差）
    lower_band = rolling_mean - (rolling_std * num_std)  # 下轨（中轨-2倍标准差）
    return pd.DataFrame({
        'bollinger_mid': rolling_mean,
        'bollinger_upper': upper_band,
        'bollinger_lower': lower_band
    })

# 按股票分组计算布林带，合并结果（加rsuffix避免列名冲突）
bollinger_results = df.groupby('stock_code')['close'].apply(calculate_bollinger_bands)
df = df.join(bollinger_results.reset_index(level=0, drop=True), rsuffix='_calc')

# --------------------------
# 5. 量价类指标：成交量加权平均价（VWAP）
# --------------------------
def calculate_vwap(group):
    volume = group['volume'].replace(0, 0.0001)  # 成交量（避免除零）
    # VWAP = 累计（成交额/成交量） / 累计天数（按股票分组内的交易日计数）
    vwap = (group['money'] / volume).cumsum() / np.arange(1, len(group) + 1)
    return vwap

df['vwap'] = df.groupby('stock_code', group_keys=False).apply(calculate_vwap)

# --------------------------
# 6. 趋势类指标：动量指标（Momentum）
# --------------------------
def calculate_momentum(series, period=14):
    # 动量 = 当日收盘价 - 14日前收盘价（反映价格趋势强度）
    return series - series.shift(period)

df['momentum14'] = df.groupby('stock_code')['close'].transform(
    lambda x: calculate_momentum(x, period=14)  # 14日动量
)

# --------------------------
# 7. 量能类指标：成交量对比（昨日比、5日均比）
# --------------------------
# 7.1 成交量与昨日比（当日成交量 / 昨日成交量）
df['volume_ratio_vs_yesterday'] = df.groupby('stock_code')['volume'].transform(
    lambda x: x / x.shift(1).replace(0, 0.0001)
)
# 7.2 成交量与5日均比（当日成交量 / 过去5日平均成交量，不含今日）
df['volume_ratio_vs_5d_avg'] = df.groupby('stock_code')['volume'].transform(
    lambda x: x / x.rolling(window=5, min_periods=1).mean().shift(1).replace(0, 0.0001)
)

# --------------------------
# 8. 量能类指标：竞价量对比（昨日比、5日均比）
# --------------------------
# 8.1 竞价量与昨日比（当日竞价量 / 昨日竞价量）
df['auc_volume_ratio_vs_yesterday'] = df.groupby('stock_code')['auc_volume'].transform(
    lambda x: x / x.shift(1).replace(0, 0.0001)
)
# 8.2 竞价量与5日均比（当日竞价量 / 过去5日平均竞价量，不含今日）
df['auc_volume_ratio_vs_5d_avg'] = df.groupby('stock_code')['auc_volume'].transform(
    lambda x: x / x.rolling(window=5, min_periods=1).mean().shift(1).replace(0, 0.0001)
)

# --------------------------
# 9. 波动类指标：波动率（20日价格波动幅度）
# --------------------------
def calculate_volatility(series, window=20):
    open_price = series['open'].replace(0, 0.0001)  # 开盘价（避免除零）
    daily_range = (series['high'] - series['low']) / open_price  # 当日波动幅度（高低差/开盘价）
    return daily_range.rolling(window=window, min_periods=1).mean()  # 20日平均波动幅度

df['volatility'] = df.groupby('stock_code', group_keys=False).apply(
    lambda x: calculate_volatility(x, window=20)
)

# --------------------------
# 10. 盘口类指标：五档盘口量比（买盘/卖盘）及对比
# --------------------------
# 10.1 计算买盘、卖盘总量（买1-5、卖1-5合计）
df['buy_total'] = df['b1_v'] + df['b2_v'] + df['b3_v'] + df['b4_v'] + df['b5_v']
df['sell_total'] = df['a1_v'] + df['a2_v'] + df['a3_v'] + df['a4_v'] + df['a5_v']
# 10.2 当日盘口量比（买盘总量/卖盘总量，避免除零）
df['order_book_volume_ratio'] = df.apply(
    lambda row: row['buy_total'] / row['sell_total'] if row['sell_total'] != 0 else np.nan,
    axis=1
)
# 10.3 盘口量比与昨日比（当日盘口量比 / 昨日盘口量比）
df['obv_ratio_vs_yesterday'] = df.groupby('stock_code')['order_book_volume_ratio'].transform(
    lambda x: x / x.shift(1).replace(0, np.nan)
)
# 10.4 盘口量比与5日均比（当日盘口量比 / 过去5日平均盘口量比，不含今日）
df['obv_ratio_vs_5d_avg'] = df.groupby('stock_code')['order_book_volume_ratio'].transform(
    lambda x: x / x.rolling(window=5, min_periods=1).mean().shift(1).replace(0, np.nan)
)

# --------------------------
# 11. 新增：活跃度指标（换手率、成交额、振幅对比）
# --------------------------
# 11.1 换手率对比（基于原始turnover_ratio字段）
# 换手率与昨日比
df['turnover_ratio_vs_yesterday'] = df.groupby('stock_code')['turnover_ratio'].transform(
    lambda x: x / x.shift(1).replace(0, np.nan)
)
# 换手率与5日均比
df['turnover_ratio_vs_5d_avg'] = df.groupby('stock_code')['turnover_ratio'].transform(
    lambda x: x / x.rolling(window=5, min_periods=1).mean().shift(1).replace(0, np.nan)
)

# 11.2 成交额对比（基于原始money字段）
# 成交额与昨日比
df['money_ratio_vs_yesterday'] = df.groupby('stock_code')['money'].transform(
    lambda x: x / x.shift(1).replace(0, 0.0001)
)
# 成交额与5日均比
df['money_ratio_vs_5d_avg'] = df.groupby('stock_code')['money'].transform(
    lambda x: x / x.rolling(window=5, min_periods=1).mean().shift(1).replace(0, 0.0001)
)

# 11.3 振幅及对比（振幅=（最高价-最低价）/ 昨日收盘价 * 100）
# 当日振幅（百分比）
df['amplitude'] = (df['high'] - df['low']) / df['pre_close'] * 100
# 振幅与昨日比
df['amplitude_vs_yesterday'] = df.groupby('stock_code')['amplitude'].transform(
    lambda x: x / x.shift(1).replace(0, np.nan)
)
# 振幅与5日均比
df['amplitude_vs_5d_avg'] = df.groupby('stock_code')['amplitude'].transform(
    lambda x: x / x.rolling(window=5, min_periods=1).mean().shift(1).replace(0, np.nan)
)

# --------------------------
# 异常值处理：规避无穷值/负无穷值
# --------------------------
df = df.replace([np.inf, -np.inf], np.nan)

# 将df写入到parquet文件里
df.to_parquet(r'D:\workspace\xiaoyao\data\factortable.parquet', index=False)

# 将parquet读取后，随机采样5条数据，并导出为csv存放在本地目录
import pandas as pd

df = pd.read_parquet(r'D:\workspace\xiaoyao\data\factortable.parquet')
df.sample(5).to_csv('./sample.csv', index=False)


