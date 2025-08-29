import baostock as bs
import pandas as pd
import talib

# --- BaoStock + TA-Lib 综合应用演示 ---

def fetch_stock_data(symbol="sh.600570", start_date='2024-01-01'):
    """使用BaoStock获取单只股票的历史K线数据。"""
    print(f"--- 1. 正在从BaoStock获取 {symbol} 的历史数据 ---")
    
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录BaoStock失败: {lg.error_msg}")
        return None

    try:
        # 查询字段可以按需调整
        fields = "date,code,open,high,low,close,volume,amount,turn,pctChg"
        rs = bs.query_history_k_data_plus(
            symbol, 
            fields, 
            start_date=start_date, 
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
        print(f"成功获取 {len(df)} 天的日线数据。")
        return df

    finally:
        bs.logout()
        # print("BaoStock已登出。")

def convert_data_types(dataframe):
    """将DataFrame中的数据列转换为TA-Lib所需的数值类型。"""
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    
    # OBV函数要求volume必须是float类型，我们在这里做一次强制转换
    if 'volume' in dataframe.columns:
        dataframe['volume'] = dataframe['volume'].astype(float)

    return dataframe

def calculate_factors(df):
    """使用TA-Lib为DataFrame计算技术因子。"""
    print("\n--- 2. 正在使用TA-Lib计算技术因子 ---")
    if df is None or df.empty:
        print("数据为空，无法计算因子。")
        return None

    # 确保数据类型正确
    df = convert_data_types(df)

    # --- 趋势指标: 计算移动平均线 (SMA) ---
    df['SMA_10'] = talib.SMA(df['close'].values, timeperiod=10)
    df['SMA_30'] = talib.SMA(df['close'].values, timeperiod=30)

    # --- 动能指标: 计算RSI和MACD ---
    df['RSI_14'] = talib.RSI(df['close'].values, timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Hist'] = macdhist

    # --- 新增波动率指标: 布林带 (Bollinger Bands) ---
    upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower

    # --- 新增成交量指标: 能量潮 (On-Balance Volume, OBV) ---
    df['OBV'] = talib.OBV(df['close'].values, df['volume'].values)

    print("因子计算完成。")
    return df

if __name__ == "__main__":
    # 获取数据
    stock_df = fetch_stock_data(symbol="sh.600570")
    
    if stock_df is not None:
        # 计算因子
        factors_df = calculate_factors(stock_df)
        
        if factors_df is not None:
            print("\n--- 3. 结果展示 (最新一个交易日) ---")
            print(factors_df.iloc[-1].round(2))