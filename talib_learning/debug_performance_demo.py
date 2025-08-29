import baostock as bs
import pandas as pd
import talib
import time

# --- 性能调试专用脚本 ---

def fetch_stock_data(symbol="sh.600570", start_date='2024-01-01'):
    print(f"[{time.strftime('%H:%M:%S')}] 开始执行数据获取...")
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录BaoStock失败: {lg.error_msg}")
        return None
    print(f"[{time.strftime('%H:%M:%S')}] 登录成功, 开始查询K线...")

    try:
        fields = "date,code,open,high,low,close,volume"
        rs = bs.query_history_k_data_plus(
            symbol, 
            fields, 
            start_date=start_date, 
            frequency="d",
            adjustflag="2"
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


def calculate_simple_factor(df):
    print(f"[{time.strftime('%H:%M:%S')}] 开始执行因子计算...")
    if df is None or df.empty:
        print("数据为空，无法计算。")
        return None

    # 数据类型转换
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 计算SMA
    df['SMA_10'] = talib.SMA(df['close'].values, timeperiod=10)
    print(f"[{time.strftime('%H:%M:%S')}] SMA计算完成。")

    # 新增MACD计算
    macd, macdsignal, macdhist = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Hist'] = macdhist
    print(f"[{time.strftime('%H:%M:%S')}] MACD计算完成。")

    # 新增布林带计算
    upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower
    print(f"[{time.strftime('%H:%M:%S')}] 布林带计算完成。")
    
    print(f"[{time.strftime('%H:%M:%S')}] 因子计算完成。")
    return df

if __name__ == "__main__":
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] 调试脚本启动...")
    
    stock_df = fetch_stock_data(symbol="sh.600570")
    
    if stock_df is not None:
        factors_df = calculate_simple_factor(stock_df)
        if factors_df is not None:
            print("\n--- 调试结果 (最近5行) ---")
            print(factors_df[['date', 'close', 'SMA_10', 'MACD', 'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower']].tail())
    
    t_end = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] 脚本运行结束, 总耗时: {t_end - t_start:.2f} 秒。")

