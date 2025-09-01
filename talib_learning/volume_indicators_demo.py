import baostock as bs
import pandas as pd
import talib

# --- 交易量指标计算演示 --- 

def fetch_stock_data(symbol="sh.600570", start_date='2023-01-01', end_date='2024-01-01'):
    """使用BaoStock获取单只股票的历史K线数据，包含交易量。"""
    print(f"--- 正在从BaoStock获取 {symbol} 的历史数据 ---")
    
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录BaoStock失败: {lg.error_msg}")
        return None

    try:
        fields = "date,code,open,high,low,close,volume,amount"
        rs = bs.query_history_k_data_plus(
            symbol, 
            fields, 
            start_date=start_date, 
            end_date=end_date,
            frequency="d", # d=日线
            adjustflag="2" # 2:前复权
        )
        if rs.error_code != '0':
            print(f"查询K线数据失败: {rs.error_msg}")
            return None

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            print(f"未能获取到 {symbol} 的任何数据。")
            return None

        df = pd.DataFrame(data_list, columns=rs.fields)
        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"成功获取 {len(df)} 天的日线数据。")
        return df

    finally:
        bs.logout()

def calculate_volume_indicators(df):
    """使用TA-Lib计算交易量指标。"""
    print("\n--- 正在使用TA-Lib计算交易量指标 ---")
    if df is None or df.empty:
        print("数据为空，无法计算指标。")
        return None

    # 确保数据列存在且为数值类型
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"缺少必要的列：{required_cols}")
        return None

    # 显式转换为 float64 类型，以满足 TA-Lib 的严格要求
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    # --- OBV (On Balance Volume) --- 
    df['OBV'] = talib.OBV(df['close'].values, df['volume'].values)

    # --- AD (Accumulation/Distribution Line) --- 
    df['AD'] = talib.AD(df['high'].values, df['low'].values, df['close'].values, df['volume'].values)

    # --- MFI (Money Flow Index) --- 
    # MFI 需要 high, low, close, volume
    df['MFI'] = talib.MFI(df['high'].values, df['low'].values, df['close'].values, df['volume'].values, timeperiod=14)

    print("交易量指标计算完成。")
    return df

if __name__ == "__main__":
    # 获取数据
    stock_df = fetch_stock_data(symbol="sh.600000") # 以浦发银行为例
    
    if stock_df is not None:
        # 计算交易量指标
        indicators_df = calculate_volume_indicators(stock_df)
        
        if indicators_df is not None:
            print("\n--- 交易量指标结果 (最新5个交易日) ---")
            print(indicators_df[['date', 'close', 'volume', 'OBV', 'AD', 'MFI']].tail())
