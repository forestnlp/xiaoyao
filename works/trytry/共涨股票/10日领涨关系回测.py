# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\trytry\共涨股票\10日领涨关系回测.ipynb



# ----------------------------------------------------------------------import pandas as pd

# 1. 手动加载数据
def load_data():
    acc_df = pd.read_csv('./2025_test_accuracy_result.csv')
    leading_group = acc_df['leading_group'].iloc[0].split(',')
    target_stock = '600570.XSHG'
    
    # 读取价格数据
    price_df = pd.read_parquet(r'D:\workspace\xiaoyao\data\stock_daily_price.parquet')
    price_df['date'] = pd.to_datetime(price_df['date']).dt.date  # 只保留日期部分
    mask = (price_df['date'] >= pd.to_datetime('2025-01-02').date()) & \
           (price_df['date'] <= pd.to_datetime('2025-10-16').date()) & \
           (price_df['stock_code'].isin([target_stock] + leading_group))
    price_df = price_df[mask][['date', 'stock_code', 'close']]
    
    # 转为宽表并填充
    pivot_df = price_df.pivot(index='date', columns='stock_code', values='close').fillna(method='ffill')
    return pivot_df, leading_group, target_stock

# 2. 纯手动回测（不用任何框架）
def manual_backtest():
    pivot_df, leading_group, target_stock = load_data()
    dates = pivot_df.index.tolist()
    initial_cash = 1000000.0
    cash = initial_cash
    shares = 0  # 持仓股数
    trade_log = []
    cycle = 10  # 10日周期

    for i in range(len(dates)):
        current_date = dates[i]
        # 每10天操作一次
        if (i + 1) % cycle != 0:
            continue
        
        # 确保有10天历史数据
        if i < cycle - 1:
            continue
        
        # 计算领先组信号
        signal_sum = 0.0
        valid_count = 0
        for code in leading_group:
            if code not in pivot_df.columns:
                continue
            prices = pivot_df[code].iloc[i-9:i+1]
            if len(prices.dropna()) == 10:
                signal_sum += (prices.iloc[-1] / prices.iloc[0] - 1)
                valid_count += 1
        if valid_count == 0:
            print(f"{current_date} 无有效信号")
            continue
        signal = signal_sum / valid_count

        # 目标股当前价格
        current_price = pivot_df[target_stock].iloc[i]
        if pd.isna(current_price) or current_price <= 0:
            print(f"{current_date} 价格异常")
            continue

        # 交易逻辑
        if signal > 0 and shares == 0:
            # 买入：100股倍数，留10%资金
            max_shares = int((cash * 0.9) / current_price / 100) * 100
            if max_shares > 0:
                cost = max_shares * current_price * 1.001  # 含手续费
                if cost <= cash:
                    cash -= cost
                    shares = max_shares
                    trade_log.append({'日期': current_date, '操作': '买入', '股数': max_shares, '价格': current_price})
                    print(f"{current_date} 买入{max_shares}股，花费{cost:.2f}元")
        
        elif signal < 0 and shares > 0:
            # 卖出
            revenue = shares * current_price * 0.999  # 含手续费
            cash += revenue
            trade_log.append({'日期': current_date, '操作': '卖出', '股数': shares, '价格': current_price})
            print(f"{current_date} 卖出{shares}股，收入{revenue:.2f}元")
            shares = 0

    # 计算最终资产（现金+持仓市值）
    final_price = pivot_df[target_stock].iloc[-1] if shares > 0 else 0
    final_value = cash + shares * final_price
    print(f"\n===== 最终结果 =====")
    print(f"初始资金：{initial_cash:.2f}元")
    print(f"最终资金：{final_value:.2f}元")
    print(f"总收益：{final_value - initial_cash:.2f}元")
    print(f"总收益率：{(final_value / initial_cash - 1) * 100:.2f}%")
    print(f"交易次数：{len(trade_log)}次")

    # 保存日志
    pd.DataFrame(trade_log).to_csv('./manual_trade_log.csv', index=False)

if __name__ == "__main__":
    manual_backtest()

