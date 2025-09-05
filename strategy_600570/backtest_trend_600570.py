from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import backtrader as bt
import pandas as pd
import baostock as bs
import datetime
import numpy as np

# --- 纯粹OBV能量潮跟随策略 for sh.600570 ---

def fetch_data_for_stock(symbol, start_date='2018-01-01'):
    """使用 baostock 获取股票历史数据，并手动计算OBV"""
    print(f"开始获取 {symbol} 的历史数据...")
    lg = bs.login()
    if lg.error_code != '0': return None
    try:
        fields = "date,code,open,high,low,close,volume,amount"
        rs = bs.query_history_k_data_plus(
            symbol, fields, start_date=start_date, frequency="d", adjustflag="2"
        )
        if rs.error_code != '0': return None
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)
        if df.empty: return None

        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col])
        df.set_index('date', inplace=True)
        df.rename(columns={'amount': 'turnover'}, inplace=True)

        daily_change = df['close'].diff()
        direction = np.sign(daily_change.fillna(0))
        directional_volume = direction * df['volume']
        df['obv'] = directional_volume.cumsum()
        
        print(f"数据获取与OBV计算成功，共 {len(df)} 条记录。")
        return df
    finally:
        bs.logout()
        print("baostock 已登出。")

class PandasDataWithOBV(bt.feeds.PandasData):
    lines = ('obv',)
    params = (('obv', -1),)

class OBVFollowStrategy(bt.Strategy):
    """纯粹的OBV跟随策略"""
    params = (
        ('obv_period', 120), # OBV的移动平均周期
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        
        self.obv = self.datas[0].obv
        self.obv_sma = bt.indicators.SimpleMovingAverage(
            self.obv, period=self.p.obv_period)
        self.crossover = bt.indicators.CrossOver(self.obv, self.obv_sma)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status in [order.Completed]:
            if order.isbuy(): self.log(f'买入成交, 价格: {order.executed.price:.2f}')
            else: self.log(f'卖出成交, 价格: {order.executed.price:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单 Canceled/Margin/Rejected')
        self.order = None

    def next(self):
        if self.order: return

        if not self.position:
            if self.crossover > 0: # OBV上穿其均线
                self.log(f'发出买入信号 (OBV上穿), 收盘价: {self.dataclose[0]:.2f}')
                self.order = self.buy()
        else:
            if self.crossover < 0: # OBV下穿其均线
                self.log(f'发出卖出信号 (OBV下穿), 收盘价: {self.dataclose[0]:.2f}')
                self.order = self.sell()

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(OBVFollowStrategy)

    stock_symbol = 'sh.600570'
    dataframe = fetch_data_for_stock(stock_symbol)
    if dataframe is None or dataframe.empty:
        print("数据未能加载，回测终止。")
    else:
        data = PandasDataWithOBV(dataname=dataframe)
        cerebro.adddata(data)
        cerebro.broker.setcash(100000.0)
        cerebro.broker.setcommission(commission=0.001)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        print('--- 开始回测 (纯粹OBV跟随策略) --- ')
        start_value = cerebro.broker.getvalue()
        print(f'初始资金: {start_value:.2f}')
        
        results = cerebro.run()
        
        print('--- 回测结束 --- ')
        end_value = cerebro.broker.getvalue()
        print(f'最终资金: {end_value:.2f}')
        print(f'总收益率: {(end_value - start_value) / start_value * 100:.2f}%')

        strat = results[0]
        analysis = strat.analyzers
        sharpe_ratio = analysis.sharpe_ratio.get_analysis().get('sharperatio', 0)
        if sharpe_ratio is None: sharpe_ratio = 0.0
        
        print("")
        print("--- 性能指标 ---")
        print(f"夏普比率 (Sharpe Ratio): {sharpe_ratio:.4f}")
        print(f"年化收益率 (Annualized Return): {analysis.returns.get_analysis().get('rnorm100', 0):.2f}%")
        print(f"最大回撤 (Max Drawdown): {analysis.drawdown.get_analysis().get('max', {}).get('drawdown', 0):.2f}%")