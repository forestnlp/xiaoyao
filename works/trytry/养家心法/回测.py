# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\trytry\养家心法\回测.ipynb



# ----------------------------------------------------------------------import pandas as pd
import numpy as np
import os
from datetime import datetime

# --------------------------
# 1. 配置参数（不变）
# --------------------------
CONFIG = {
    "selection_input_path": r'./yangjia_selection_result.csv',
    "widetable_input_path": r'D:\workspace\xiaoyao\data\widetable.parquet',
    "backtest_detail_path": r'./yangjia_backtest_detail.csv',
    "fund_growth_path": r'./yangjia_fund_growth.csv',
    "backtest_summary_path": r'./yangjia_backtest_summary.txt',
    "log_path": r'./yangjia_backtest_log.txt',
    "trade_rules": {
        "buy_delay": 1,        # T日选股→T+1买入
        "sell_delay": 5,       # T+1买入→T+5卖出
        "stop_loss_ratio": 0.05,
        "stop_profit_ratio": 0.15,
        "initial_position": 0.5,
        "add_position_threshold": 0.03
    },
    "initial_fund": 100000  # 初始资金（整数，后续转为浮点数）
}

# --------------------------
# 2. 工具函数（不变）
# --------------------------
def init_log():
    with open(CONFIG["log_path"], 'w', encoding='utf-8') as f:
        f.write(f"【养家心法回测启动】{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def log_msg(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {msg}"
    print(log_line)
    with open(CONFIG["log_path"], 'a', encoding='utf-8') as f:
        f.write(log_line + "\n")

# --------------------------
# 3. 加载回测数据（不变，已统一日期类型）
# --------------------------
def load_backtest_data():
    log_msg("开始加载回测数据（选股结果+原始宽表）...")
    
    # 加载选股结果（统一t_date为datetime）
    selection_df = pd.read_csv(CONFIG["selection_input_path"])
    selection_df["date"] = pd.to_datetime(selection_df["date"])
    selection_df["t_date"] = pd.to_datetime(selection_df["selection_date"])  # 转为datetime
    selection_df = selection_df.drop(columns=["selection_date"])
    log_msg(f"✅ 选股结果加载：{len(selection_df)}条记录，{selection_df['t_date'].nunique()}个选股日")
    
    # 加载宽表价格数据
    price_cols = ['date', 'stock_code', 'open', 'close', 'paused']
    widetable_df = pd.read_parquet(CONFIG["widetable_input_path"], columns=price_cols)
    widetable_df["date"] = pd.to_datetime(widetable_df["date"])
    widetable_df = widetable_df.sort_values(by=["stock_code", "date"]).reset_index(drop=True)
    widetable_df["trade_seq"] = widetable_df.groupby("stock_code").cumcount()
    log_msg(f"✅ 宽表价格数据加载：{len(widetable_df)}条记录，{widetable_df['stock_code'].nunique()}只股票")
    
    # 合并选股结果与交易序列
    selection_df = pd.merge(
        selection_df,
        widetable_df[["stock_code", "date", "trade_seq"]].rename(columns={"date": "t_date"}),
        on=["stock_code", "t_date"],
        how="left"
    ).dropna(subset=["trade_seq"])
    selection_df["trade_seq"] = selection_df["trade_seq"].astype(int)
    
    log_msg(f"✅ 数据合并完成：{len(selection_df)}条有效选股记录")
    return selection_df, widetable_df

# --------------------------
# 4. 核心回测逻辑（不变）
# --------------------------
def run_backtest(selection_df, widetable_df):
    log_msg("开始执行养家心法回测...")
    rules = CONFIG["trade_rules"]
    
    price_seq_map = widetable_df.set_index(["stock_code", "trade_seq"])[["open", "close", "paused", "date"]].to_dict('index')
    
    def calc_trade_info(row):
        stock_code = row["stock_code"]
        t_seq = row["trade_seq"]
        
        buy_seq = t_seq + rules["buy_delay"]
        sell_seq = buy_seq + rules["sell_delay"]
        
        buy_data = price_seq_map.get((stock_code, buy_seq), {})
        buy_price = buy_data.get("open", np.nan)
        buy_date = buy_data.get("date", np.nan)
        is_buy_paused = buy_data.get("paused", 1.0)
        
        sell_data = price_seq_map.get((stock_code, sell_seq), {})
        sell_price = sell_data.get("close", np.nan)
        sell_date = sell_data.get("date", np.nan)
        is_sell_paused = sell_data.get("paused", 1.0)
        
        if pd.isna(buy_price) or pd.isna(sell_price) or is_buy_paused == 1.0 or is_sell_paused == 1.0:
            return pd.Series({
                "buy_date": np.nan, "sell_date": np.nan,
                "buy_price": np.nan, "sell_price": np.nan,
                "return_rate": np.nan, "position": np.nan,
                "contribution_return": np.nan, "is_valid": False
            })
        
        return_rate = (sell_price - buy_price) / buy_price * 100
        position = rules["initial_position"]
        if return_rate >= 3:
            position = 1.0
        contribution_return = return_rate * position
        
        return pd.Series({
            "buy_date": buy_date, "sell_date": sell_date,
            "buy_price": buy_price, "sell_price": sell_price,
            "return_rate": return_rate, "position": position,
            "contribution_return": contribution_return, "is_valid": True
        })
    
    # 执行计算并返回完整backtest_df（包含有效+无效交易）
    trade_info = selection_df.apply(calc_trade_info, axis=1)
    backtest_df = pd.concat([selection_df, trade_info], axis=1)
    valid_backtest_df = backtest_df[backtest_df["is_valid"]].copy()
    invalid_count = len(backtest_df) - len(valid_backtest_df)
    
    log_msg(f"✅ 交易计算完成：有效交易{len(valid_backtest_df)}条，无效交易{invalid_count}条")
    
    # 止损止盈修正
    def apply_stop_rule(row):
        return_rate = row["return_rate"]
        if return_rate <= -rules["stop_loss_ratio"] * 100:
            return -rules["stop_loss_ratio"] * 100
        elif return_rate >= rules["stop_profit_ratio"] * 100:
            return rules["stop_profit_ratio"] * 100
        else:
            return return_rate
    
    valid_backtest_df["adjusted_return"] = valid_backtest_df.apply(apply_stop_rule, axis=1)
    valid_backtest_df["adjusted_contribution"] = valid_backtest_df["adjusted_return"] * valid_backtest_df["position"]
    
    return backtest_df, valid_backtest_df  # 同时返回完整backtest_df和有效交易df

# --------------------------
# 5. 资金增长计算（修复dtype警告）
# --------------------------
def calculate_fund_growth(valid_backtest_df):
    log_msg("开始计算资金增长（按日平均收益连乘）...")
    
    daily_return = valid_backtest_df.groupby("sell_date").agg({
        "adjusted_contribution": ["mean", "count"],
        "stock_code": "nunique"
    }).reset_index()
    daily_return.columns = ["sell_date", "daily_avg_return", "daily_trade_count", "daily_stock_count"]
    daily_return = daily_return[daily_return["daily_trade_count"] >= 2].sort_values("sell_date")
    
    # 修复dtype警告：初始化为浮点数
    daily_return["cumulative_fund"] = float(CONFIG["initial_fund"])  # 直接用浮点数初始化
    daily_return["daily_growth_rate"] = 1 + 0.5 * (daily_return["daily_avg_return"] / 100)
    
    # 计算累计资金（保持浮点数类型）
    for i in range(len(daily_return)):
        if i == 0:
            daily_return.iloc[i, daily_return.columns.get_loc("cumulative_fund")] = float(CONFIG["initial_fund"]) * daily_return.iloc[i]["daily_growth_rate"]
        else:
            daily_return.iloc[i, daily_return.columns.get_loc("cumulative_fund")] = daily_return.iloc[i-1]["cumulative_fund"] * daily_return.iloc[i]["daily_growth_rate"]
    
    # 格式化数值（保持浮点数）
    daily_return["daily_avg_return"] = np.round(daily_return["daily_avg_return"], 2)
    daily_return["daily_growth_rate"] = np.round(daily_return["daily_growth_rate"], 4)
    daily_return["cumulative_fund"] = np.round(daily_return["cumulative_fund"], 2)
    
    log_msg(f"✅ 资金增长计算完成：{len(daily_return)}个有效收益日")
    return daily_return

# --------------------------
# 6. 回测结果统计与保存（修复backtest_df传递）
# --------------------------
def summarize_and_save(backtest_df, valid_backtest_df, daily_return):
    log_msg("开始生成回测汇总报告...")
    
    total_trades = len(valid_backtest_df)
    total_selection = len(backtest_df)  # 用完整backtest_df获取总选股记录数
    total_return_rate = (daily_return["cumulative_fund"].iloc[-1] / CONFIG["initial_fund"] - 1) * 100 if len(daily_return) > 0 else 0
    trading_days = len(daily_return)
    annual_return = total_return_rate / (trading_days / 250) if trading_days > 0 else 0
    positive_days = len(daily_return[daily_return["daily_avg_return"] > 0])
    positive_day_ratio = positive_days / trading_days * 100 if trading_days > 0 else 0
    max_drawdown = 0
    
    # 计算最大回撤
    if len(daily_return) > 0:
        cumulative_fund = daily_return["cumulative_fund"].values
        peak = np.maximum.accumulate(cumulative_fund)
        drawdown = (cumulative_fund - peak) / peak * 100
        max_drawdown = np.min(drawdown)
    
    # 强势板块统计
    industry_return = valid_backtest_df.groupby("sw_l1_industry_name").agg({
        "adjusted_return": ["mean", "count", lambda x: np.round((x>0).mean()*100, 2)],
        "stock_code": "nunique"
    }).reset_index()
    industry_return.columns = ["industry", "avg_return(%)", "trade_count", "positive_ratio(%)", "stock_count"]
    industry_return = industry_return.sort_values("avg_return(%)", ascending=False).head(10)
    
    # 保存结果
    valid_backtest_df.to_csv(CONFIG["backtest_detail_path"], index=False, encoding='utf-8-sig')
    daily_return.to_csv(CONFIG["fund_growth_path"], index=False, encoding='utf-8-sig')
    
    # 生成汇总报告（使用backtest_df计算总选股记录）
    summary_content = f"""
【养家心法回测汇总报告】
==========================
回测规则：T日选股→T+{CONFIG['trade_rules']['buy_delay']}买入→T+{CONFIG['trade_rules']['sell_delay']}卖出
         初始半仓，盈利≥3%加仓至满仓；止损5%，止盈15%
==========================
1. 基础交易统计
   - 总选股记录：{total_selection}条
   - 有效交易记录：{total_trades}条
   - 无效交易记录：{total_selection - total_trades}条
   - 有效交易天数：{trading_days}天
   - 平均每日交易：{total_trades/trading_days:.1f}只（若trading_days>0）

2. 收益表现
   - 初始资金：{CONFIG['initial_fund']:.2f}元
   - 最终资金：{daily_return['cumulative_fund'].iloc[-1]:.2f}元（若trading_days>0）
   - 累计收益率：{total_return_rate:.2f}%
   - 年化收益率：{annual_return:.2f}%（按250个交易日/年）
   - 正收益日占比：{positive_day_ratio:.2f}%（{positive_days}/{trading_days}）
   - 最大回撤：{max_drawdown:.2f}%

3. 风险控制
   - 止损触发次数：{len(valid_backtest_df[valid_backtest_df['adjusted_return'] <= -5])}次
   - 止盈触发次数：{len(valid_backtest_df[valid_backtest_df['adjusted_return'] >= 15])}次
   - 平均单票收益：{valid_backtest_df['adjusted_return'].mean():.2f}%
   - 收益标准差：{valid_backtest_df['adjusted_return'].std():.2f}%

4. 强势板块TOP10
{industry_return.to_string(index=False, float_format=lambda x: f"{x:.2f}")}
==========================
回测完成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    with open(CONFIG["backtest_summary_path"], 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    # 打印核心结果
    print("\n" + "="*60)
    print("🎉 养家心法回测核心结果")
    print("="*60)
    print(f"📊 累计收益率：{total_return_rate:.2f}% | 年化收益率：{annual_return:.2f}%")
    print(f"💰 初始资金：{CONFIG['initial_fund']}元 → 最终资金：{daily_return['cumulative_fund'].iloc[-1]:.2f}元")
    print(f"🎯 正收益日占比：{positive_day_ratio:.2f}% | 最大回撤：{max_drawdown:.2f}%")
    print(f"📁 回测明细：{CONFIG['backtest_detail_path']}")
    print(f"📁 资金曲线：{CONFIG['fund_growth_path']}")
    print(f"📁 汇总报告：{CONFIG['backtest_summary_path']}")
    print("="*60)
    
    return valid_backtest_df, daily_return

# --------------------------
# 主函数（修复backtest_df传递）
# --------------------------
def main_yangjia_backtest():
    try:
        init_log()
        # 步骤1：加载数据
        selection_df, widetable_df = load_backtest_data()
        # 步骤2：执行回测（获取完整backtest_df和有效交易df）
        backtest_df, valid_backtest_df = run_backtest(selection_df, widetable_df)
        if len(valid_backtest_df) == 0:
            log_msg("❌ 无有效交易记录，回测终止")
            return None, None
        # 步骤3：计算资金增长
        daily_return = calculate_fund_growth(valid_backtest_df)
        # 步骤4：生成报告（传递backtest_df）
        valid_backtest_df, daily_return = summarize_and_save(backtest_df, valid_backtest_df, daily_return)
        return valid_backtest_df, daily_return
    except Exception as e:
        log_msg(f"❌ 回测失败：{str(e)}")
        raise

# 执行回测
if __name__ == "__main__":
    backtest_detail, fund_growth = main_yangjia_backtest()

