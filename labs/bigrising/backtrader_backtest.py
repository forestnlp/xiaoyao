import backtrader as bt
import pandas as pd
import os

# --- Constants ---
DATA_DIR = r'D:\workspace\xiaoyao\data'
PRICE_DATA_FILENAME = 'stock_daily_price.parquet'
HOLDING_PERIOD = 10
INVESTMENT_AMOUNT = 10000.0

# --- Backtrader Strategy Definition ---
class MLStrategy(bt.Strategy):
    params = (('holding_period', HOLDING_PERIOD),)

    def __init__(self):
        # Indicators
        self.macd = bt.indicators.MACD(self.data.close)
        self.bband = bt.indicators.BollingerBands()
        self.atr = bt.indicators.ATR()
        
        # To keep track of open trades
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = 0

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        # print(f'{dt.isoformat()}, {txt}') # Commented out for performance

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        # self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def next(self):
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Rule 1: Strong Momentum
            rule1 = self.macd.macd[0] > 0.67
            
            # Rule 2: Building Momentum
            rule2 = (self.macd.macd[0] > 0) and (self.macd.macd[0] <= 0.67) and (self.bband.lines.bot[0] <= 37.65)

            # Rule 3: Volatility Contraction with Support
            rule3 = (self.macd.macd[0] <= 0) and (self.atr.atr[0] > 1.36) and (self.bband.lines.bot[0] <= 56.71)

            if rule1 or rule2 or rule3:
                # Buy signal
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}')
                amount_to_invest = INVESTMENT_AMOUNT
                self.size = amount_to_invest / self.data.close[0]
                self.order = self.buy(size=self.size)
        else:
            # Already in the market, check for sell condition
            if len(self) >= (self.bar_executed + self.p.holding_period):
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}')
                self.order = self.sell(size=self.size)

# --- Main Backtest Execution ---
def run_backtest():
    print("--- Starting Backtrader Backtest ---")
    
    # Load all data
    price_path = os.path.join(DATA_DIR, PRICE_DATA_FILENAME)
    all_prices = pd.read_parquet(price_path)
    all_prices['date'] = pd.to_datetime(all_prices['date'])
    all_prices.set_index('date', inplace=True)
    
    stock_codes = all_prices['stock_code'].unique()
    print(f"- Found {len(stock_codes)} unique stocks.")

    total_trades = 0
    winning_trades = 0
    total_pnl = 0.0
    total_profit = 0.0
    total_loss = 0.0

    # Loop through each stock and run a backtest
    for i, stock_code in enumerate(stock_codes):
        if (i + 1) % 100 == 0:
            print(f"  - Processing stock {i+1}/{len(stock_codes)}: {stock_code}")
            
        stock_data = all_prices[all_prices['stock_code'] == stock_code].copy()
        if len(stock_data) < 50: # Need enough data for indicators
            continue

        data_feed = bt.feeds.PandasData(dataname=stock_data, open='open', high='high', low='low', close='close', volume='volume', openinterest=-1)
        
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.addstrategy(MLStrategy)
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(INVESTMENT_AMOUNT * 100) # High initial cash
        cerebro.broker.setcommission(commission=0.001) # 0.1% commission
        
        # Add analyzer for trades
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        
        # Run backtest
        results = cerebro.run()
        
        # Analyze results
        trade_analyzer = results[0].analyzers.trade_analyzer.get_analysis()
        
        if trade_analyzer and trade_analyzer.total and trade_analyzer.total.total > 0:
            total_trades += trade_analyzer.total.total

            if trade_analyzer.pnl and trade_analyzer.pnl.net:
                total_pnl += trade_analyzer.pnl.net.total

            if trade_analyzer.won and trade_analyzer.won.total > 0:
                winning_trades += trade_analyzer.won.total
                if trade_analyzer.won.pnl:
                    total_profit += trade_analyzer.won.pnl.total
            
            if trade_analyzer.lost and trade_analyzer.lost.total > 0:
                if trade_analyzer.lost.pnl:
                    total_loss += abs(trade_analyzer.lost.pnl.total)

    # --- Final Performance Report ---
    print("\n--- Backtrader Performance Report ---")
    if total_trades > 0:
        win_rate = (winning_trades / total_trades) * 100
        # Profit/Loss Ratio (Payoff Ratio)
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / (total_trades - winning_trades) if (total_trades - winning_trades) > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        print(f"Total Trades Executed: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Win: {avg_win:.2f}")
        print(f"Average Loss: {avg_loss:.2f}")
        print(f"Profit/Loss Ratio (Payoff Ratio): {profit_loss_ratio:.2f}")
        print(f"Total Net Profit/Loss: {total_pnl:.2f}")
    else:
        print("No trades were executed across all stocks.")

if __name__ == '__main__':
    run_backtest()