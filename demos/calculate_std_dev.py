
import numpy as np
import pandas as pd

# 1. 准备数据：假设这是你投资组合过去10天的每日总价值
portfolio_values = pd.Series([
    10000, 10100, 10050, 10200, 10150, 
    10300, 10350, 10320, 10400, 10500
])

# 2. 计算每日回报率
# pct_change() 函数可以轻松计算出每日变化百分比
daily_returns = portfolio_values.pct_change()

# 丢弃第一天的空值 (因为第一天没有前一天的数据)
daily_returns = daily_returns.dropna()

print("--- 每日回报率 ---")
print(daily_returns)

# 3. 计算每日回报率的标准差
daily_std_dev = daily_returns.std()
# 也可以用 numpy: daily_std_dev = np.std(daily_returns)

print(f"\n每日标准差: {daily_std_dev:.4f}")

# 4. 年化标准差 (假设一年有252个交易日)
annualized_std_dev = daily_std_dev * np.sqrt(252)

print(f"年化标准差 (年化波动率): {annualized_std_dev:.4f}")

