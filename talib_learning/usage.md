# TA-Lib (Python Wrapper) 使用指南

## 1. 简介

TA-Lib (Technical Analysis Library) 是一个广泛用于金融市场数据技术分析的开源库。其 Python 包装器 `talib` 允许我们方便地调用其内置的大量技术指标函数。

## 2. 核心概念

TA-Lib 的 Python 库函数**不直接接受** Pandas DataFrame 或 Series 作为输入。它要求输入的数据是 **NumPy 数组**。

因此，标准的使用流程是：

1.  从数据源（如 AkShare, BaoStock 或 CSV 文件）加载数据到 Pandas DataFrame。
2.  从 DataFrame 中提取需要的列（如 'close', 'high', 'low', 'volume'）。
3.  将这些列转换为 NumPy 数组（通过 `.values` 属性）。
4.  将 NumPy 数组作为参数传递给 TA-Lib 的函数。
5.  函数的计算结果通常也是一个或多个 NumPy 数组。

## 3. 常用函数示例

假设我们有一个名为 `df` 的 Pandas DataFrame，其中包含了 `close`, `high`, `low` 等列。

```python
import talib
import pandas as pd
import numpy as np

# 示例数据
data = {
    'open': np.random.random(100) * 10 + 100,
    'high': np.random.random(100) * 10 + 110,
    'low': np.random.random(100) * 10 + 90,
    'close': np.random.random(100) * 10 + 100,
}
df = pd.DataFrame(data)

# 从DataFrame中获取NumPy数组
close_prices = df['close'].values
high_prices = df['high'].values
low_prices = df['low'].values
```

### 示例1：计算简单移动平均线 (SMA)

`SMA(real, timeperiod=30)`

```python
# 计算10日简单移动平均线
sma_10 = talib.SMA(close_prices, timeperiod=10)
print(sma_10)
```

### 示例2：计算相对强弱指数 (RSI)

`RSI(real, timeperiod=14)`

```python
# 计算14日RSI
rsi_14 = talib.RSI(close_prices, timeperiod=14)
print(rsi_14)
```

### 示例3：计算平滑异同移动平均线 (MACD)

MACD 函数会返回三个 NumPy 数组：`macd`, `macdsignal`, `macdhist`。

`MACD(real, fastperiod=12, slowperiod=26, signalperiod=9)`

```python
# 计算默认参数的MACD
macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
print("MACD:", macd)
print("Signal:", macdsignal)
print("Hist:", macdhist)
```

### 示例4：计算布林带 (Bollinger Bands)

布林带函数返回三个 NumPy 数组：`upperband`, `middleband`, `lowerband`。

`BBANDS(real, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)`

```python
# 计算20日、2个标准差的布林带
upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
print("Upper Band:", upper)
print("Middle Band:", middle)
print("Lower Band:", lower)
```
