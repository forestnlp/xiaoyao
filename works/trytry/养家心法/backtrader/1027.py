# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\trytry\养家心法\backtrader\1027.ipynb



# ----------------------------------------------------------------------import pandas as pd
import backtrader as bt
import numpy as np

import pandas as pd
import backtrader as bt
import numpy as np

# 1. 读取数据+时间过滤（优化：指定引擎加速parquet读取，提前过滤无效列）
# 只保留策略必需字段，减少内存占用
needed_cols = [
    "date", "stock_code", "open", "high", "low", "close", "volume", 
    "pre_close", "high_limit", "low_limit", "paused", "sw_l1_industry_code"
]
df = pd.read_parquet(
    path="D:\\workspace\\xiaoyao\\data\\widetable.parquet",
    columns=needed_cols,  # 只读必需字段，减少IO和内存
    engine="pyarrow"  # 用pyarrow引擎加速parquet读取（需安装pyarrow：pip install pyarrow）
)
df["date"] = pd.to_datetime(df["date"])
df = df[(df["date"] >= "2023-01-01") & (df["date"] <= "2025-10-27")]

# 2. 数据类型优化（关键：减少内存占用，加速后续计算）
df["stock_code"] = df["stock_code"].astype("category")  # 股票代码转为分类类型
df["sw_l1_industry_code"] = df["sw_l1_industry_code"].astype("category")  # 行业代码转分类
df["paused"] = df["paused"].astype("float32")  # 停牌标识用float32（原float64冗余）
df[["open", "close", "volume"]] = df[["open", "close", "volume"]].astype("float32")  # 价格/成交量降精度

# 3. 过滤无效数据（提前移除无法交易的样本，减少后续计算量）
df = df[
    (df["paused"] == 0.0) &  # 非停牌
    (df["volume"] > 0) &      # 有成交（避免无成交量的死股）
    (df["open"].notna()) &    # 开盘价非空
    (df["close"].notna())     # 收盘价非空
].reset_index(drop=True)

# 4. 计算衍生指标（优化：用transform替代merge，减少中间DataFrame）
# 涨跌幅
df["price_change"] = (df["close"] - df["pre_close"]) / df["pre_close"] * 100
# 涨停标识
df["is_limit_up"] = (df["close"] == df["high_limit"]) & (df["price_change"] >= 9.8)
# 量比（5日滚动均值，用transform直接添加到原表，避免索引错位）
df["vol_5d_avg"] = df.groupby("stock_code")["volume"].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
df["volume_ratio"] = df["volume"] / df["vol_5d_avg"]
df.drop("vol_5d_avg", axis=1, inplace=True)  # 删除中间列，节省内存

# 5. 统计涨停数（优化：用transform替代merge，避免2次merge耗时）
# 全市场每日涨停数
df["daily_limit_up_count"] = df.groupby("date")["is_limit_up"].transform("sum")
# 申万一级行业每日涨停数
df["industry_limit_up_count"] = df.groupby(["date", "sw_l1_industry_code"])["is_limit_up"].transform("sum")

# 6. 自定义数据源类（保持原功能，移除冗余参数）
class CustomPandasData(bt.feeds.PandasData):
    lines = (
        "paused", "sw_l1_industry_code", "is_limit_up", 
        "daily_limit_up_count", "industry_limit_up_count", "volume_ratio", "price_change"
    )
    params = (
        ("paused", "paused"),
        ("sw_l1_industry_code", "sw_l1_industry_code"),
        ("is_limit_up", "is_limit_up"),
        ("daily_limit_up_count", "daily_limit_up_count"),
        ("industry_limit_up_count", "industry_limit_up_count"),
        ("volume_ratio", "volume_ratio"),
        ("price_change", "price_change"),
        ("openinterest", -1),
    )

# 7. 生成数据源（优化：批量处理+过滤小市值/低流动性股票，减少循环次数）
# 只保留有足够交易天数的股票（如≥60天，避免垃圾股）
stock_trade_days = df.groupby("stock_code").size()
valid_stocks = stock_trade_days[stock_trade_days >= 60].index.tolist()
df_valid = df[df["stock_code"].isin(valid_stocks)]

# 循环生成数据源（只遍历有效股票，减少循环次数）
data_feeds = []
for code in df_valid["stock_code"].unique():
    stock_df = df_valid[df_valid["stock_code"] == code].sort_values("date").set_index("date")
    # 直接创建数据源，无冗余参数
    data = CustomPandasData(dataname=stock_df)
    data_feeds.append(data)

# 释放内存（删除大DataFrame）
del df, df_valid, stock_trade_days
print(f"数据源生成完成！共{len(data_feeds)}只有效股票")

class ShortTermYangJiaStrategy(bt.Strategy):
    def __init__(self):
        self.trade_count = 0  # 记录买入次数（区分启动期/正常期）
        self.main_industries = []  # T日主流热点行业（申万一级前3）
        self.buy_date_map = {}  # 记录买入日期：{股票名: 买入日}

    def next(self):
        today = self.data.datetime.date(0)
        # 1. T日统计主流行业（仅情绪达标时：全市场涨停数>20）
        if self.data.daily_limit_up_count[0] > 20:
            # 筛选当日行业涨停数据，取前3
            daily_industry = df[df["date"] == today].groupby("sw_l1_industry_code")["industry_limit_up_count"].max()
            self.main_industries = daily_industry.nlargest(3).index.tolist()
        else:
            self.main_industries = []
            return  # 情绪差，不选股

        # 2. 筛选符合条件的标的（4只上限）
        eligible = []
        for data in self.datas:
            # 检查：主流行业+非停牌+量比>1.2+强势（涨停或3%-9.8%涨幅）
            if (str(data.sw_l1_industry_code[0]) in self.main_industries  # 行业匹配（转字符串避免类型问题）
                and data.paused[0] == 0.0 
                and data.volume_ratio[0] > 1.2 
                and (data.is_limit_up[0] or (3 < data.price_change[0] < 9.8))):
                eligible.append(data)
        eligible = eligible[:4]  # 控制4只

        # 3. T+1日买入（无持仓时执行）
        if eligible and not self.positions:
            self.trade_count += 1
            total_cash = self.broker.getcash()
            # 启动期（第1次）用50%资金，后续用100%可用资金
            buy_cash = total_cash * 0.5 if self.trade_count == 1 else total_cash
            per_cash = buy_cash / len(eligible)  # 单只标的资金

            for data in eligible:
                # 100股整数倍（A股最小交易单位）
                size = int(per_cash // (data.open[0] * 100) * 100)
                if size > 0:
                    self.buy(data=data, price=data.open[0], size=size)
                    self.buy_date_map[data._name] = today  # 记录买入日

        # 4. T+2日卖出（持仓满2天）
        for data in self.datas:
            if self.getposition(data).size > 0:
                buy_date = self.buy_date_map.get(data._name)
                if buy_date and (today - buy_date).days == 2:
                    self.sell(data=data, price=data.close[0], size=self.getposition(data).size)

# 1. 创建回测引擎
cerebro = bt.Cerebro(stdstats=True)

# 2. 添加所有股票数据源（修正名称获取方式）
for data in data_feeds:
    # 从数据源参数中获取原始DataFrame，再提取股票代码
    # data.p.dataname 才是传入的stock_df（原始DataFrame）
    stock_code = data.p.dataname["stock_code"].iloc[0]
    cerebro.adddata(data, name=str(stock_code))  # 转字符串避免特殊字符问题

# 3. 添加策略
cerebro.addstrategy(ShortTermYangJiaStrategy)

# 4. 配置回测参数
cerebro.broker.setcash(1000000.0)  # 初始资金100万
cerebro.broker.setcommission(commission=0.0003)  # 手续费0.03%（双边）
cerebro.broker.set_slippage_perc(perc=0.0005)  # 滑点0.05%（双边）

# 5. 添加分析器
cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

# 6. 运行回测
print(f"回测开始 | 初始资金：{cerebro.broker.getvalue():.2f}元")
results = cerebro.run()
print(f"回测结束 | 最终资金：{cerebro.broker.getvalue():.2f}元")

# 提取分析结果
strat = results[0]
returns = strat.analyzers.returns.get_analysis()
drawdown = strat.analyzers.drawdown.get_analysis()
trades = strat.analyzers.trades.get_analysis()

# 输出核心指标
print("="*50)
print("回测核心指标（2023-01-01至2025-10-27）")
print("="*50)
print(f"累计收益率：{returns['rtot']*100:.2f}%")
print(f"年化收益率：{returns['rnorm100']:.2f}%")
print(f"最大回撤：{drawdown['max']['drawdown']:.2f}%")
print(f"总交易次数：{trades['total']['total']}")
print(f"盈利交易次数：{trades['won']['total']}")
print(f"亏损交易次数：{trades['lost']['total']}")
if trades['total']['total'] > 0:
    print(f"胜率：{trades['won']['total']/trades['total']['total']*100:.2f}%")
    print(f"平均盈亏比：{trades['won']['pnl']['total']/abs(trades['lost']['pnl']['total']):.2f}")

# 可选：绘制回测净值曲线（需安装matplotlib）
cerebro.plot(style="candlestick", iplot=True)

