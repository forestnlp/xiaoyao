import baostock as bs
import pandas as pd
import talib
import time
import numpy as np

# --- 最终版：BaoStock + TA-Lib + 手动OBV 综合分析脚本 ---

def fetch_stock_data(symbol="sh.600570", start_date='2024-01-01'):
    """使用BaoStock获取单只股票的历史K线数据。"""
    print(f"[{time.strftime('%H:%M:%S')}] 1. 开始执行数据获取...")
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录BaoStock失败: {lg.error_msg}")
        return None

    try:
        fields = "date,code,open,high,low,close,volume,amount,turn,pctChg"
        rs = bs.query_history_k_data_plus(
            symbol, fields, start_date=start_date, frequency="d", adjustflag="2"
        )
        if rs.error_code != '0':
            print(f"查询K线数据失败: {rs.error_msg}")
            return None

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        print(f"[{time.strftime('%H:%M:%S')}] 数据查询完成, 共 {len(df)} 条。")
        return df

    finally:
        bs.logout()


def calculate_factors_and_obv(df):
    """为DataFrame计算技术因子，并手动计算OBV。"""
    print(f"[{time.strftime('%H:%M:%S')}] 2. 开始执行因子计算...")
    if df is None or df.empty:
        print("数据为空，无法计算。")
        return None

    # --- 数据类型转换 ---
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- TA-Lib 计算部分 ---
    df['SMA_10'] = talib.SMA(df['close'].values, timeperiod=10)
    df['SMA_30'] = talib.SMA(df['close'].values, timeperiod=30)
    df['RSI_14'] = talib.RSI(df['close'].values, timeperiod=14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
    print(f"[{time.strftime('%H:%M:%S')}] TA-Lib指标计算完成。")

    # --- 手动计算OBV，绕开Bug ---
    # 价格上涨，OBV增加；价格下跌，OBV减少
    daily_change = df['close'].diff()
    # 如果价格上涨，方向为1；下跌为-1；不变为0
    direction = np.sign(daily_change.fillna(0))
    directional_volume = direction * df['volume']
    df['OBV_manual'] = directional_volume.cumsum()
    print(f"[{time.strftime('%H:%M:%S')}] 手动OBV计算完成。")

    return df

if __name__ == "__main__":
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 最终分析脚本启动...")
    
    stock_df = fetch_stock_data(symbol="sh.600570")
    
    if stock_df is not None:
        factors_df = calculate_factors_and_obv(stock_df)
        if factors_df is not None:
            print("\n--- 最终结果 (最新一个交易日) ---")
            # 打印最后一行，确保所有数据可见
            print(factors_df.iloc[-1])
    
    t_end = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] 脚本运行结束, 总耗时: {t_end - t_start:.2f} 秒。")
