import backtrader as bt
import akshare as ak
import pandas as pd
import datetime
import talib # Added import
import numpy as np # Added import

print("Executing volume_mfi_strategy.py")

# Custom TA-Lib MFI Indicator
class TalibMFI(bt.Indicator):
    lines = ('mfi',)
    params = (('period', 14),)

    # The indicator takes a single data feed as input
    def __init__(self):
        self.addminperiod(self.p.period)

    def next(self):
        # Ensure enough data is available
        if len(self.datas[0]) < self.p.period:
            self.lines.mfi[0] = float('nan')
            return

        # Get numpy arrays from backtrader data feeds
        high = np.array(self.datas[0].high.array[-self.p.period:])
        low = np.array(self.datas[0].low.array[-self.p.period:])
        close = np.array(self.datas[0].close.array[-self.p.period:])
        volume = np.array(self.datas[0].volume.array[-self.p.period:])

        # Check for NaNs in input data before passing to TA-Lib
        if np.isnan(high).any() or np.isnan(low).any() or np.isnan(close).any() or np.isnan(volume).any():
            self.lines.mfi[0] = float('nan')
            return

        # Calculate MFI using TA-Lib
        mfi_val = talib.MFI(high, low, close, volume, timeperiod=self.p.period)[-1]
        self.lines.mfi[0] = mfi_val

class VolumeMFIStrategy(bt.Strategy):
    params = (
        ('mfi_period', 50), # Changed from 14 to 50
        ('mfi_overbought', 80),
        ('mfi_oversold', 20),
        ('trend_period', 200),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume

        # Trend filter
        self.sma_trend = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.trend_period)

        # Volume indicator (using custom TalibMFI)
        self.mfi = TalibMFI(
            self.datas[0], # Pass the main data feed
            period=self.params.mfi_period)
            
        # Crossover signals for MFI
        self.mfi_cross_oversold_up = bt.indicators.CrossUp(
            self.mfi.lines.mfi, self.params.mfi_oversold) # Use mfi.lines.mfi

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

    def next(self):
        # Ensure all indicators have enough data
        if len(self.data) < max(self.params.trend_period, self.params.mfi_period):
            return

        # Get MFI value
        mfi_val = self.mfi.lines.mfi[0]

        # Log current values for debugging
        self.log(f'Close: {self.dataclose[0]:.2f}, SMA_Trend: {self.sma_trend[0]:.2f}, MFI: {mfi_val:.2f}, MFI_Cross_Oversold_Up: {self.mfi_cross_oversold_up[0]}')

        # Check if MFI is a valid number
        if np.isnan(mfi_val):
            self.log("MFI is NaN, skipping trade decision.")
            return

        # Check if we are in the market
        if not self.position:
            # Not in market, check for a buy signal
            # is_uptrend = self.dataclose[0] > self.sma_trend[0] # Removed for now
            # if is_uptrend: # Only check MFI if in uptrend
            if self.mfi_cross_oversold_up[0] == 1.0:
                self.log(f'BUY CREATE, Close: {self.dataclose[0]:.2f}, MFI: {mfi_val:.2f}')
                self.buy()
            else:
                self.log(f'No MFI buy signal. MFI: {mfi_val:.2f}, Cross: {self.mfi_cross_oversold_up[0]}')
            # else:
            #     self.log(f'Not in uptrend. Close: {self.dataclose[0]:.2f}, SMA_Trend: {self.sma_trend[0]:.2f}')
        else:
            # In market, check for a sell signal
            is_overbought = self.mfi.lines.mfi[0] > self.params.mfi_overbought
            trend_reversed = self.dataclose[0] < self.sma_trend[0]
            if is_overbought or trend_reversed:
                self.log(f'SELL CREATE, Close: {self.dataclose[0]:.2f}, MFI: {mfi_val:.2f}, Overbought: {is_overbought}, Trend Reversed: {trend_reversed}')

def get_stock_data(symbol, start_date):
    print(f"--- 正在获取 {symbol} 的历史数据 ---")
    stock_hist_df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, adjust="qfq")
    
    # Debugging prints for data quality
    print("--- 原始数据 NaN 统计 ---")
    print(stock_hist_df.isnull().sum())
    print("--- 原始数据头部 ---")
    print(stock_hist_df.head())
    print("----------------------")

    # Test MFI calculation directly on raw data
    print("--- 测试 talib.MFI 直接计算 ---")
    try:
        test_high = np.array(stock_hist_df['最高'].values)
        test_low = np.array(stock_hist_df['最低'].values)
        test_close = np.array(stock_hist_df['收盘'].values)
        test_volume = np.array(stock_hist_df['成交量'].values)
        
        # Ensure data types are float for talib
        test_high = test_high.astype(float)
        test_low = test_low.astype(float)
        test_close = test_close.astype(float)
        test_volume = test_volume.astype(float)

        test_mfi = talib.MFI(test_high, test_low, test_close, test_volume, timeperiod=14)
        print("talib.MFI 测试结果 (前10个):")
        print(test_mfi[:10])
        print("talib.MFI 测试结果 (后10个):")
        print(test_mfi[-10:])
        print(f"talib.MFI 测试结果中 NaN 数量: {np.isnan(test_mfi).sum()}")
    except Exception as e:
        print(f"talib.MFI 直接计算时发生错误: {e}")
    print("----------------------")

    stock_hist_df['日期'] = pd.to_datetime(stock_hist_df['日期'])
    stock_hist_df.set_index('日期', inplace=True)
    stock_hist_df.rename(columns={'开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
    return bt.feeds.PandasData(dataname=stock_hist_df, fromdate=datetime.datetime.strptime(start_date, '%Y%m%d'))

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(VolumeMFIStrategy)

    data = get_stock_data(symbol='600570', start_date='20230101') # Longer start date for 200-day SMA
    cerebro.adddata(data)

    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95) # Use 95% of portfolio for each trade

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    print('--- 开始回测 ---')
    results = cerebro.run()
    
    print('--- 回测结束 ---')
    final_value = cerebro.broker.getvalue()
    print(f'最终组合价值: {final_value:.2f}')

    # Debugging prints
    print(f"Type of results: {type(results)}")
    print(f"Content of results: {results}")
    print(f"Type of results[0]: {type(results[0])}")
    print(f"Content of results[0]: {results[0]}")

    # Print analysis results
    # strat = results[0][0] # Corrected access
    sharpe_ratio = results[0].analyzers.sharpe.get_analysis()
    drawdown = results[0].analyzers.drawdown.get_analysis()
    returns = results[0].analyzers.returns.get_analysis()

    print(f"夏普比率: {sharpe_ratio['sharperatio']:.2f}")
    print(f"最大回撤: {drawdown.max.drawdown:.2%}")
    print(f"年化收益率: {returns['rnorm100']:.2f}%")

    cerebro.plot()