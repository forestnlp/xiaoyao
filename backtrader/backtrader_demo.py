#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backtrader Demo - 3天连续涨跌策略
测试股票：600570 恒生电子
策略：3天连续上涨买入，3天连续下跌卖出
数据源：baostock
"""

import backtrader as bt
import baostock as bs
import pandas as pd
import datetime
import os
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False


def fetch_stock_data(stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    获取股票数据（后复权）
    
    Args:
        stock_code: 股票代码，如'sh.600570'
        start_date: 开始日期，格式'YYYY-MM-DD'
        end_date: 结束日期，格式'YYYY-MM-DD'
    
    Returns:
        包含股票数据的DataFrame或None
    """
    # 登录baostock
    lg = bs.login()
    if lg.error_code != '0':
        print(f'baostock登录失败: {lg.error_msg}')
        return None
    
    try:
        # 获取日K线数据（后复权）
        rs = bs.query_history_k_data_plus(
            stock_code,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="1"  # 1：后复权， 2：前复权，3：不复权
        )
        
        if rs.error_code != '0':
            print(f"获取数据失败: {rs.error_msg}")
            return None
        
        # 转换为DataFrame
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            print("未获取到数据")
            return None
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 数据类型转换
        df['date'] = pd.to_datetime(df['date'])
        numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 设置日期为索引
        df.set_index('date', inplace=True)
        
        # 过滤掉停牌日期
        df = df[df['tradestatus'] == '1']
        
        print(f"获取到 {len(df)} 条数据")
        return df
        
    except Exception as e:
        print(f'获取数据异常: {e}')
        return None
    finally:
        bs.logout()


def get_or_update_stock_data(stock_code: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """
    获取或更新股票全量数据（后复权）
    
    Args:
        stock_code: 股票代码，如'sh.600570'
        start_date: 开始日期，默认为股票上市日期
        end_date: 结束日期，默认为当前日期
    
    Returns:
        包含股票数据的DataFrame或None
    """
    csv_filename = f'{stock_code.replace(".", "_")}_full_data.csv'
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # 检查CSV文件是否存在
    if os.path.exists(csv_filename):
        try:
            # 读取现有数据
            existing_df = pd.read_csv(csv_filename, index_col='date', parse_dates=True)
            last_date = existing_df.index.max().strftime('%Y-%m-%d')
            
            # 检查数据是否为最新
            if last_date >= current_date:
                print(f"CSV数据已是最新，从 {csv_filename} 加载了 {len(existing_df)} 条记录")
                print(f"数据范围: {existing_df.index.min().date()} 到 {existing_df.index.max().date()}")
                return existing_df
            else:
                # 需要更新数据
                next_date = (existing_df.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                print(f"数据需要更新，从 {next_date} 开始获取新数据...")
                
                # 获取增量数据
                new_data = fetch_stock_data(stock_code, next_date, current_date)
                if new_data is not None and not new_data.empty:
                    # 合并数据
                    combined_df = pd.concat([existing_df, new_data])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df.sort_index(inplace=True)
                    
                    # 保存更新后的数据
                    combined_df.to_csv(csv_filename)
                    print(f"数据已更新，新增 {len(new_data)} 条记录，总计 {len(combined_df)} 条记录")
                    return combined_df
                else:
                    print("无新数据需要更新")
                    return existing_df
                    
        except Exception as e:
            print(f'读取CSV文件异常: {e}，重新获取全量数据...')
    
    # 获取全量数据
    if start_date is None:
        # 获取股票上市日期
        lg = bs.login()
        if lg.error_code == '0':
            try:
                rs = bs.query_stock_basic(code=stock_code)
                if rs.error_code == '0':
                    stock_info = rs.get_data()
                    if not stock_info.empty:
                        start_date = stock_info.iloc[0]['ipoDate']
                        print(f"股票 {stock_code} 上市日期: {start_date}")
            except:
                pass
            finally:
                bs.logout()
        
        if start_date is None:
            start_date = '2020-01-01'  # 默认开始日期
    
    if end_date is None:
        end_date = current_date
    
    print(f"正在获取 {stock_code} 从 {start_date} 到 {end_date} 的全量数据...")
    df = fetch_stock_data(stock_code, start_date, end_date)
    
    if df is not None and not df.empty:
        # 保存为CSV文件
        df.to_csv(csv_filename)
        print(f"数据已保存到 {csv_filename}，共 {len(df)} 条记录")
        print(f"数据范围: {df.index.min().date()} 到 {df.index.max().date()}")
    
    return df


class MovingAverageCrossStrategy(bt.Strategy):
    """
    双均线交叉策略
    - 短期均线上穿长期均线时买入
    - 短期均线下穿长期均线时卖出
    """
    
    params = (
        ('ma_short', 2),     # 短期均线周期
        ('ma_long', 8),      # 长期均线周期
        ('stop_loss', 0.05), # 止损比例 5%
        ('take_profit', 0.10), # 止盈比例 10%
        ('printlog', True),  # 是否打印日志
    )
    
    def __init__(self):
        # 记录交易信号
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        
        # 添加数据引用
        self.dataclose = self.datas[0].close
        
        # 计算移动平均线
        self.ma_short = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.ma_short)
        self.ma_long = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.ma_long)
        
        # 交叉信号
        self.crossover = bt.indicators.CrossOver(self.ma_short, self.ma_long)
        
        # 记录交易信号用于绘图
        self.buy_signals = []
        self.sell_signals = []
        
    def log(self, txt, dt=None):
        """日志记录函数"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行: 价格={order.executed.price:.2f}, 数量={order.executed.size}, 佣金={order.executed.comm:.2f}')
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
                # 记录买入信号
                self.buy_signals.append({
                    'date': self.datas[0].datetime.date(0),
                    'price': order.executed.price
                })
            else:
                self.log(f'卖出执行: 价格={order.executed.price:.2f}, 数量={order.executed.size}, 佣金={order.executed.comm:.2f}')
                profit = (order.executed.price - self.buy_price) * order.executed.size - order.executed.comm - self.buy_comm
                self.log(f'交易盈亏: {profit:.2f}')
                # 记录卖出信号
                self.sell_signals.append({
                    'date': self.datas[0].datetime.date(0),
                    'price': order.executed.price
                })
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')
        
        self.order = None
    
    def notify_trade(self, trade):
        """交易完成通知"""
        if not trade.isclosed:
            return
        
        self.log(f'交易完成: 毛利润={trade.pnl:.2f}, 净利润={trade.pnlcomm:.2f}')
    
    def next(self):
        """策略主逻辑"""
        # 当前价格信息
        current_close = self.dataclose[0]
        
        # 记录当前状态
        self.log(f'收盘={current_close:.2f}, 短期均线={self.ma_short[0]:.2f}, 长期均线={self.ma_long[0]:.2f}')
        
        # 如果有未完成订单，跳过
        if self.order:
            return
        
        # 买入信号：短期均线上穿长期均线且当前无持仓
        if not self.position and self.crossover[0] > 0:
            self.log(f'买入信号: 短期均线上穿长期均线')
            # 计算买入数量（按可用资金的95%买入，A股最小100股）
            cash = self.broker.getcash()
            size = int((cash * 0.95) / current_close / 100) * 100
            if size >= 100:
                self.order = self.buy(size=size)
                self.buy_price = current_close
                self.log(f'创建买入订单: 数量={size}股')
        
        # 卖出信号：检查止损止盈或均线信号
        elif self.position:
            # 检查止损止盈
            if self.buy_price:
                # 止损：亏损超过5%
                if current_close <= self.buy_price * (1 - self.params.stop_loss):
                    self.log(f'止损卖出: 价格={current_close:.2f}, 买入价={self.buy_price:.2f}')
                    self.order = self.sell(size=self.position.size)
                # 止盈：盈利超过10%
                elif current_close >= self.buy_price * (1 + self.params.take_profit):
                    self.log(f'止盈卖出: 价格={current_close:.2f}, 买入价={self.buy_price:.2f}')
                    self.order = self.sell(size=self.position.size)
                # 死叉卖出信号
                elif self.crossover[0] < 0:
                    self.log(f'卖出信号: 短期均线下穿长期均线')
                    self.order = self.sell(size=self.position.size)
                    self.log(f'创建卖出订单: 数量={self.position.size}股')
    
    def stop(self):
        """策略结束时调用"""
        self.log(f'策略结束，最终资产: {self.broker.getvalue():.2f}')


class TradeAnalyzer(bt.Analyzer):
    """交易分析器 - 记录详细交易信息"""
    
    def __init__(self):
        self.trades = []
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                'date': self.strategy.datetime.date(0),
                'size': trade.size,
                'price': trade.price,
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm
            })
    
    def get_analysis(self):
        return {'trades': self.trades}


def run_backtest(stock_code: str = 'sh.600570', start_date: str = '2005-01-01', end_date: str = None):
    """
    运行回测
    
    Args:
        stock_code: 股票代码
        start_date: 回测开始日期
        end_date: 回测结束日期，默认为当前日期
    """
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    print("=" * 60)
    print("Backtrader 股票策略回测")
    print(f"测试股票: {stock_code.split('.')[1]} {stock_code}")
    print("策略参数: 2日均线 vs 8日均线 + 止损止盈")
    print(f"回测时间范围: {start_date} 到 {end_date}")
    print("=" * 60)
    
    print(f"正在获取 {stock_code} 的数据...")
    
    # 获取或更新全量数据
    print("步骤1: 获取或更新股票数据...")
    full_df = get_or_update_stock_data(stock_code)
    
    if full_df is None or full_df.empty:
        print("获取股票数据失败")
        return
    
    # 筛选回测时间范围的数据
    print("步骤2: 筛选回测时间范围数据...")
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    df = full_df[(full_df.index >= start_dt) & (full_df.index <= end_dt)]
    
    if df.empty:
        print(f"在指定时间范围 {start_date} 到 {end_date} 内没有数据")
        return
    
    print(f"回测数据范围: {df.index.min().date()} 到 {df.index.max().date()}，共 {len(df)} 条记录")
    
    print("步骤3: 开始回测...")
    
    if df is None or len(df) < 10:
        print("加载回测数据失败或数据量不足")
        return
    
    # 创建Cerebro引擎
    cerebro = bt.Cerebro()
    
    # 添加策略
    cerebro.addstrategy(MovingAverageCrossStrategy)
    
    # 准备数据
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # 使用索引作为日期
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )
    
    # 添加数据到引擎
    cerebro.adddata(data)
    
    # 设置初始资金
    cerebro.broker.setcash(10000000.0)
    
    # 设置A股佣金（万分之三，最低5元）
    cerebro.broker.setcommission(
        commission=0.0003,
        mult=1.0,
        margin=None,
        percabs=False,
        commtype=bt.CommInfoBase.COMM_PERC,
        stocklike=True,
        leverage=1.0
    )
    
    # 设置A股T+1规则（当日收盘价成交）
    cerebro.broker.set_coc(True)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(TradeAnalyzer, _name='trades')
    
    print(f"初始资金: {cerebro.broker.getvalue():.2f}")
    
    # 运行回测
    print("\n开始回测...")
    print("-" * 60)
    results = cerebro.run()
    
    print("-" * 60)
    
    # 获取分析结果
    strat = results[0]
    initial_cash = 100000.0
    final_value = cerebro.broker.getvalue()
    total_return = final_value - initial_cash
    return_rate = (total_return / initial_cash) * 100
    
    # 统计交易次数
    total_trades = len(strat.buy_signals) + len(strat.sell_signals)
    
    print(f"\n{'='*60}")
    print(f"回测完成! 📊")
    print(f"{'='*60}")
    print(f"💰 资金情况:")
    print(f"   初始资金: {initial_cash:,.2f} 元")
    print(f"   最终资产: {final_value:,.2f} 元")
    print(f"   总收益: {total_return:,.2f} 元")
    print(f"")
    print(f"📈 关键指标:")
    
    # 夏普比率
    if hasattr(strat.analyzers.sharpe, 'get_analysis'):
        sharpe = strat.analyzers.sharpe.get_analysis()
        if 'sharperatio' in sharpe and sharpe['sharperatio'] is not None:
            sharpe_ratio = sharpe['sharperatio']
            performance_desc = '(优秀表现)' if sharpe_ratio > 1.0 else '(一般表现)' if sharpe_ratio > 0.5 else '(较差表现)'
            print(f"   - 夏普比率: {sharpe_ratio:.4f} {performance_desc}")
    
    print(f"   - 总交易次数: {total_trades}笔")
    
    # 最大回撤
    drawdown = strat.analyzers.drawdown.get_analysis()
    print(f"   - 最大回撤: {drawdown['max']['drawdown']:.2f}%")
    
    print(f"   - 总收益率: {return_rate:.2f}%")
    # 计算年化收益率
    years = (df.index.max() - df.index.min()).days / 365.25
    annual_return = return_rate / years if years > 0 else 0
    print(f"   - 年化收益率: {annual_return:.2f}% (基于{years:.1f}年数据)")
    print(f"{'='*60}")
    
    # 交易明细
    trades = strat.analyzers.trades.get_analysis()['trades']
    if trades:
        print(f"\n=== 交易明细 (共{len(trades)}笔) ===")
        for i, trade in enumerate(trades, 1):
            print(f"{i:2d}. {trade['date']} | 数量:{trade['size']:4d} | 价格:{trade['price']:6.2f} | 盈亏:{trade['pnlcomm']:8.2f}")
    else:
        print("\n未产生交易")
    
    print("\n回测完成！")
    
    # 绘制K线图和交易信号
    plot_trading_signals(df, strat.buy_signals, strat.sell_signals)


def plot_trading_signals(data, buy_signals, sell_signals):
    """绘制K线图和交易信号"""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 重置索引以便访问日期
    data_plot = data.reset_index()
    
    # 绘制K线图
    for i in range(len(data_plot)):
        date = data_plot.iloc[i]['date']
        open_price = data_plot.iloc[i]['open']
        high_price = data_plot.iloc[i]['high']
        low_price = data_plot.iloc[i]['low']
        close_price = data_plot.iloc[i]['close']
        
        # K线颜色：红涨绿跌
        color = 'red' if close_price >= open_price else 'green'
        
        # 绘制影线
        ax.plot([date, date], [low_price, high_price], color='black', linewidth=0.5)
        
        # 绘制实体
        height = abs(close_price - open_price)
        bottom = min(open_price, close_price)
        rect = Rectangle((mdates.date2num(date) - 0.3, bottom), 0.6, height, 
                        facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
    
    # 绘制买入信号
    if buy_signals:
        buy_dates = [signal['date'] for signal in buy_signals]
        buy_prices = [signal['price'] for signal in buy_signals]
        ax.scatter(buy_dates, buy_prices, color='red', marker='^', s=100, 
                  label='买入信号', zorder=5)
    
    # 绘制卖出信号
    if sell_signals:
        sell_dates = [signal['date'] for signal in sell_signals]
        sell_prices = [signal['price'] for signal in sell_signals]
        ax.scatter(sell_dates, sell_prices, color='blue', marker='v', s=100, 
                  label='卖出信号', zorder=5)
    
    # 设置图表格式
    ax.set_title('600570 恒生电子 - 双均线交叉策略交易信号', fontsize=16, fontweight='bold')
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('价格 (元)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 格式化x轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('trading_signals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("交易信号图表已保存为 trading_signals.png")

if __name__ == '__main__':
    run_backtest()