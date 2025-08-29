import talib
import numpy as np
import pandas as pd

# --- TA-Lib 库使用演示 ---

def create_sample_data(days=100):
    """创建一个模拟的股票历史数据DataFrame，用于演示。"""
    data = {
        'date': pd.to_datetime(pd.date_range(start='2024-01-01', periods=days, freq='D')),
        'open': np.random.uniform(98, 102, size=days).cumsum() + 100,
    }
    data['high'] = data['open'] + np.random.uniform(0, 5, size=days)
    data['low'] = data['open'] - np.random.uniform(0, 5, size=days)
    data['close'] = data['open'] + np.random.uniform(-2, 2, size=days)
    data['volume'] = np.random.randint(100000, 500000, size=days)
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    print("--- 1. 创建模拟数据 ---")
    print("模拟数据的前5行:")
    print(df.head())
    print("-" * 30)
    return df

def calculate_sma(dataframe):
    """计算简单移动平均线 (SMA)"""
    print("\n--- 2. 计算10日和30日简单移动平均线 (SMA) ---")
    # 从DataFrame中提取收盘价,并转换为NumPy数组
    close_prices = dataframe['close'].values
    
    # 计算SMA
    sma_10 = talib.SMA(close_prices, timeperiod=10)
    sma_30 = talib.SMA(close_prices, timeperiod=30)
    
    # 将计算结果添加回DataFrame以便查看
    dataframe['SMA_10'] = sma_10
    dataframe['SMA_30'] = sma_30
    
    print("计算结果的后5行 (注意前面的值为NaN，因为数据不足): ")
    print(dataframe[['close', 'SMA_10', 'SMA_30']].tail())
    print("-" * 30)

def calculate_rsi(dataframe):
    """计算相对强弱指数 (RSI)"""
    print("\n--- 3. 计算14日相对强弱指数 (RSI) ---")
    close_prices = dataframe['close'].values
    
    # 计算RSI
    rsi_14 = talib.RSI(close_prices, timeperiod=14)
    dataframe['RSI_14'] = rsi_14
    
    print("计算结果的后5行:")
    print(dataframe[['close', 'RSI_14']].tail())
    print("-" * 30)

def calculate_macd(dataframe):
    """计算平滑异同移动平均线 (MACD)"""
    print("\n--- 4. 计算MACD ---")
    close_prices = dataframe['close'].values
    
    # 计算MACD
    # 注意: MACD会返回三个值
    macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    
    dataframe['MACD'] = macd
    dataframe['MACD_Signal'] = macdsignal
    dataframe['MACD_Hist'] = macdhist
    
    print("计算结果的后5行:")
    print(dataframe[['close', 'MACD', 'MACD_Signal', 'MACD_Hist']].tail())
    print("-" * 30)

if __name__ == "__main__":
    # 1. 准备数据
    # 在真实场景中，这里应该是用 akshare 或 baostock 获取数据
    stock_data = create_sample_data()
    
    # 2. 计算各种技术因子
    calculate_sma(stock_data)
    calculate_rsi(stock_data)
    calculate_macd(stock_data)
    
    print("\nTA-Lib Demo 运行结束。")
