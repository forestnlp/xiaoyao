import baostock as bs
import pandas as pd
import talib
import time
import numpy as np

# --- 批量股票技术分析脚本 ---

# 封装数据获取函数
def fetch_stock_data(symbol):
    lg = bs.login()
    if lg.error_code != '0':
        return None # 登录失败

    try:
        fields = "date,code,open,high,low,close,volume"
        rs = bs.query_history_k_data_plus(
            symbol, fields, start_date='2024-01-01', frequency="d", adjustflag="2"
        )
        if rs.error_code != '0':
            return None

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        return pd.DataFrame(data_list, columns=rs.fields)

    finally:
        bs.logout()

# 封装因子计算函数
def calculate_all_factors(df):
    if df is None or df.empty or len(df) < 35: # 确保有足够数据计算最长的均线
        return None

    # 数据类型转换
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # TA-Lib 计算
    df['SMA_10'] = talib.SMA(df['close'].values, timeperiod=10)
    df['SMA_30'] = talib.SMA(df['close'].values, timeperiod=30)
    df['RSI_14'] = talib.RSI(df['close'].values, timeperiod=14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['close'].values)
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['close'].values)
    
    # 手动计算OBV
    daily_change = df['close'].diff()
    direction = np.sign(daily_change.fillna(0))
    directional_volume = direction * df['volume']
    df['OBV_manual'] = directional_volume.cumsum()
    
    return df

# 核心：封装完整的分析与解读函数
def analyze_stock(symbol):
    print(f"\n{'='*20} 分析报告: {symbol} {'='*20}")
    
    # 1. 获取并计算
    df_raw = fetch_stock_data(symbol)
    df_factors = calculate_all_factors(df_raw)

    if df_factors is None:
        print("数据不足或获取失败，无法生成分析报告。")
        return

    latest = df_factors.iloc[-1]

    # 2. 指标解读
    sma_10, sma_30 = latest['SMA_10'], latest['SMA_30']
    trend_signal = "看涨 (金叉)" if sma_10 > sma_30 else "看跌 (死叉)"
    rsi_14 = latest['RSI_14']
    if rsi_14 > 70: rsi_signal = "超买区域"
    elif rsi_14 < 30: rsi_signal = "超卖区域"
    else: rsi_signal = "中性区域"
    macd_hist = latest['MACD_Hist']
    if macd_hist > 0: macd_signal = "多头动能增强"
    else: macd_signal = "空头动能增强"

    # 3. 综合评估与情景分析
    print("--- 1. 最新关键指标 ---")
    print(latest)
    print("\n--- 2. 多维度量化解读 ---")
    print(f"- 趋势: {trend_signal}")
    print(f"- 动能: {rsi_signal} (RSI: {rsi_14:.2f}), {macd_signal} (MACD柱: {macd_hist:.4f})")

    print("\n--- 3. 未来走势预测与情景分析 ---")
    bullish_score = 0
    bearish_score = 0
    if sma_10 > sma_30: bullish_score += 2
    else: bearish_score += 2
    if rsi_14 > 50 and rsi_14 < 70: bullish_score += 1
    if rsi_14 < 50 and rsi_14 > 30: bearish_score += 1
    if macd_hist > 0: bullish_score += 1.5
    else: bearish_score += 1.5

    if bullish_score > bearish_score + 1:
        print("结论: 看涨概率较高。")
    elif bearish_score > bullish_score + 1:
        print("结论: 看跌或回调概率较高。")
    else:
        print("结论: 震荡概率较高。")

    print("\n免责声明：本分析仅为量化指标解读，不构成任何投资建议。")
    print(f"{'='*55}")


if __name__ == "__main__":
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 批量分析任务启动...")
    
    stocks_to_analyze = ["sh.600812", "sh.600666", "sh.601595"]
    for stock_code in stocks_to_analyze:
        analyze_stock(stock_code)
    
    t_end = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] 批量分析任务结束, 总耗时: {t_end - t_start:.2f} 秒。")
