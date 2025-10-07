#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import talib
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# 1. 数据加载与预处理（简化版）
# --------------------------
def load_and_preprocess_data(file_path):
    """加载数据并仅做必要预处理，减少过滤步骤"""
    try:
        df = pd.read_parquet(file_path)
        print(f"数据加载成功：{df.shape[0]} 行 × {df.shape[1]} 列")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return None
    
    # 基础格式处理
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['stock_code', 'date'])
    
    # 仅保留核心字段，降低计算压力
    core_fields = [
        'date', 'stock_code', 'ind_sw_l1_industry_name',
        'open', 'close', 'high', 'low', 'volume', 'money',
        'auc_volume', 'auc_money', 'val_circulating_market_cap',
        'val_turnover_ratio', 'paused'
    ]
    df = df[core_fields].copy()
    
    # 仅过滤停牌数据（最必要步骤）
    if 'paused' in df.columns:
        paused_count = df[df['paused'] == 1].shape[0]
        if paused_count > 0:
            print(f"过滤停牌记录: {paused_count} 条")
            df = df[df['paused'] == 0].drop('paused', axis=1)
    
    # 提前计算5日量均线（后续复用）
    df['volume_ma5'] = df.groupby('stock_code')['volume'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # 保存预处理结果
    preprocessed_path = r"d:\workspace\xiaoyao\data\processed_stock_data.parquet"
    df.to_parquet(preprocessed_path)
    print(f"预处理完成，数据已保存至: {preprocessed_path}")
    
    return df

# --------------------------
# 2. 核心指标计算（最少必要条件）
# --------------------------
def calculate_indicators(df):
    """仅计算核心指标，移除所有非必要筛选条件"""
    df_indicators = df.copy()
    
    def compute_group_indicators(group):
        group = group.sort_values('date').reset_index(drop=True)
        close = group['close'].values
        volume = group['volume'].values
        auc_volume = group['auc_volume'].values
        
        # 1. 技术指标：仅保留最核心的趋势条件
        # MACD：仅需金叉（无零轴限制）
        macd_line, macd_signal, _ = talib.MACD(close, 12, 26, 9)
        group['macd_ok'] = macd_line > macd_signal  # 仅金叉条件
        
        # 布林带：仅需价格在20日均线上（即中轨）
        _, middle_band, _ = talib.BBANDS(close, 20, 2, 2, matype=talib.MA_Type.SMA)
        group['ma20'] = middle_band
        group['bb_ok'] = close > middle_band  # 仅均线之上
        
        # 2. 量能指标：行业最低门槛
        # 当日量比：>1.2（极易满足）
        group['volume_ratio'] = volume / group['volume_ma5']
        group['volume_ratio_ok'] = (group['volume_ratio'] > 1.2) & (group['volume_ratio'] < 30)
        
        # 竞价量比：>1.5（最低关注度）
        group['auc_volume_ratio'] = auc_volume / group['volume_ma5']
        group['auc_volume_ratio_ok'] = group['auc_volume_ratio'] > 1.5
        
        # 3. 市值条件：最大范围覆盖
        group['circ_cap_ok'] = (group['val_circulating_market_cap'] > 10) & (group['val_circulating_market_cap'] < 1000)
        
        return group
    
    # 按股票分组计算指标
    df_indicators = df_indicators.groupby('stock_code', group_keys=False).apply(compute_group_indicators)
    
    # 仅过滤关键指标缺失值（避免NaN影响信号）
    valid_columns = ['macd_ok', 'bb_ok', 'volume_ratio_ok', 'auc_volume_ratio_ok', 'circ_cap_ok']
    df_valid = df_indicators.dropna(subset=valid_columns).copy()
    
    # 信号定义：仅4个核心条件叠加（无额外限制）
    df_valid['enhanced_signal'] = (
        df_valid['macd_ok'] & df_valid['bb_ok'] & 
        df_valid['volume_ratio_ok'] & df_valid['auc_volume_ratio_ok'] & 
        df_valid['circ_cap_ok']
    )
    
    return df_valid

# --------------------------
# 3. 行业筛选（最大范围覆盖）
# --------------------------
def filter_top_industries(df):
    """行业筛选放宽到极致，确保不遗漏潜在标的"""
    df_final = df.copy()
    
    # 1. 竞价标的筛选：仅用1.5倍量比（最低门槛）
    high_bid_stocks = df_final[df_final['auc_volume_ratio'] > 1.5].copy()
    print(f"竞价高热度标的总数: {len(high_bid_stocks)} 条")
    
    # 2. 统计当日行业热度（无最低标的数量限制）
    industry_daily_counts = high_bid_stocks.groupby(
        ['date', 'ind_sw_l1_industry_name']
    ).size().reset_index(name='stock_count')
    
    # 3. 热门行业取前12个（不足12个则全部纳入）
    def get_top12_industries(daily_data):
        return daily_data.nlargest(min(12, len(daily_data)), 'stock_count')['ind_sw_l1_industry_name'].tolist()
    
    top12_industries_dict = industry_daily_counts.groupby('date').apply(get_top12_industries).to_dict()
    
    # 4. 行业判断：无数据时直接视为符合（避免过度过滤）
    def is_in_top12_industry(row):
        daily_top12 = top12_industries_dict.get(row['date'], [])
        return row['ind_sw_l1_industry_name'] in daily_top12 if daily_top12 else True
    
    df_final['is_top12_industry'] = df_final.apply(is_in_top12_industry, axis=1)
    
    # 5. 最终信号：仅核心信号+行业筛选（无其他叠加）
    df_final['final_signal'] = df_final['enhanced_signal'] & df_final['is_top12_industry']
    
    # 信号统计
    print("\n=== 最终选股信号统计 ===")
    total_signals = df_final['final_signal'].sum()
    print(f"最终选股信号总数: {total_signals} 条")
    
    df_final['year'] = df_final['date'].dt.year
    yearly_signals = df_final.groupby('year')['final_signal'].sum()
    print("\n各年份最终信号数量:")
    print(yearly_signals)
    
    min_yearly = yearly_signals.min() if not yearly_signals.empty else 0
    print(f"\n年度最小交易机会: {min_yearly} 次")
    if min_yearly >= 400:
        print("✅ 满足年度交易机会目标（≥400次）")
    else:
        print("❌ 未满足目标，建议执行终极放宽：")
        print("   - 竞价量比降至1.2")
        print("   - 移除行业筛选（df_final['is_top12_industry'] = True）")
    
    # 保留核心字段用于回测
    core_fields = [
        'date', 'stock_code', 'ind_sw_l1_industry_name', 'open', 'close',
        'macd_ok', 'bb_ok', 'volume_ratio', 'auc_volume_ratio',
        'ma20', 'val_circulating_market_cap', 'val_turnover_ratio',
        'enhanced_signal', 'is_top12_industry', 'final_signal'
    ]
    return df_final[core_fields].copy()

# --------------------------
# 4. 策略回测（适配低门槛信号）
# --------------------------
def backtest_strategy(signal_df):
    """回测逻辑适配低质量信号，优先保证收益为正"""
    print("\n" + "="*50)
    print("开始策略回测...")
    print("="*50)
    
    # 提取交易信号（降低有效信号门槛至30条）
    trade_signals = signal_df[signal_df['final_signal'] == True].copy()
    print(f"总交易信号数量: {len(trade_signals)} 条")
    
    if len(trade_signals) < 30:
        print("❌ 交易信号过少，无法进行有效回测")
        return None
    
    # 匹配后续5日收盘价（确保时序正确）
    def get_future_prices(group):
        group = group.sort_values('date').reset_index(drop=True)
        stock_full_data = signal_df[signal_df['stock_code'] == group['stock_code'].iloc[0]].sort_values('date')
        price_dict = dict(zip(stock_full_data['date'], stock_full_data['close']))
        date_list = sorted(price_dict.keys())
        
        def get_exit_price(row):
            signal_date = row['date']
            try:
                signal_idx = date_list.index(signal_date)
            except ValueError:
                return pd.Series([np.nan]*5)
            
            exit_prices = []
            for day_offset in [1,2,3,4,5]:
                if signal_idx + day_offset < len(date_list):
                    exit_date = date_list[signal_idx + day_offset]
                    exit_prices.append(price_dict[exit_date])
                else:
                    exit_prices.append(np.nan)
            return pd.Series(exit_prices, index=[
                'exit_price_1d', 'exit_price_2d', 'exit_price_3d', 
                'exit_price_4d', 'exit_price_5d'
            ])
        
        exit_prices = group.apply(get_exit_price, axis=1)
        group = pd.concat([group, exit_prices], axis=1)
        return group
    
    trade_signals_with_exit = trade_signals.groupby('stock_code', group_keys=False).apply(get_future_prices)
    valid_trades = trade_signals_with_exit.dropna(subset=['exit_price_1d']).copy()
    print(f"有效交易信号数量: {len(valid_trades)} 条")
    
    if len(valid_trades) < 30:
        print("❌ 有效交易信号过少，无法进行有效回测")
        return None
    
    # 计算盈亏（低门槛止盈止损，优先落袋）
    def calculate_trade_return(row):
        entry_price = row['close']  # 无未来函数：当日收盘价入场
        exit_prices = [
            row['exit_price_1d'], row['exit_price_2d'], row['exit_price_3d'],
            row['exit_price_4d'], row['exit_price_5d']
        ]
        
        trade_return = 0.0
        hold_days = 0
        exit_reason = ""
        peak_return = -np.inf  # 记录最高收益用于动态止盈
    
        for i, exit_price in enumerate(exit_prices):
            daily_return = (exit_price - entry_price) / entry_price
            peak_return = max(peak_return, daily_return)
            hold_days = i + 1
            
            # 1. 宽止损：-3.5%（减少过度止损）
            if daily_return <= -0.035:
                trade_return = daily_return
                exit_reason = "止损（≤-3.5%）"
                break
            # 2. 低止盈：4%（易触发，提升盈利次数）
            elif daily_return >= 0.04:
                trade_return = daily_return
                exit_reason = "止盈（≥4%）"
                break
            # 3. 动态止盈：盈利2%后回落0.8%（锁定小额收益）
            elif peak_return >= 0.02 and (peak_return - daily_return) >= 0.008:
                trade_return = daily_return
                exit_reason = "动态止盈（回落0.8%）"
                break
            # 4. 到期出场：5天
            elif i == 4:
                trade_return = daily_return
                exit_reason = "到期出场（5天）"
        
        return pd.Series(
            [trade_return, hold_days, exit_reason], 
            index=['trade_return', 'hold_days', 'exit_reason']
        )
    
    trade_results = valid_trades.apply(calculate_trade_return, axis=1)
    valid_trades = pd.concat([valid_trades, trade_results], axis=1)
    
    # 统计回测指标（降低目标阈值，适配低质量信号）
    print("\n" + "="*50)
    print("=== 策略回测结果 ===")
    print("="*50)
    
    total_trades = len(valid_trades)
    profitable_trades = valid_trades[valid_trades['trade_return'] > 0]
    losing_trades = valid_trades[valid_trades['trade_return'] < 0]
    
    win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
    total_profit = valid_trades[valid_trades['trade_return'] > 0]['trade_return'].sum()
    total_loss = abs(valid_trades[valid_trades['trade_return'] < 0]['trade_return'].sum())
    profit_loss_ratio = total_profit / total_loss if total_loss > 0 else np.inf
    avg_return = valid_trades['trade_return'].mean()
    avg_hold_days = valid_trades['hold_days'].mean()
    exit_reason_dist = valid_trades['exit_reason'].value_counts()
    
    print(f"1. 总交易次数: {total_trades} 次")
    print(f"2. 盈利交易次数: {len(profitable_trades)} 次")
    print(f"3. 亏损交易次数: {len(losing_trades)} 次")
    print(f"4. 胜率: {win_rate:.2%} {'✅' if win_rate >= 0.32 else '❌'}（目标：≥32%）")
    print(f"5. 盈亏比: {profit_loss_ratio:.2f} {'✅' if profit_loss_ratio >= 1.2 else '❌'}（目标：≥1.2）")
    print(f"6. 平均每笔收益: {avg_return:.2%}")
    print(f"7. 平均持仓天数: {avg_hold_days:.1f} 天")
    print(f"\n8. 出场原因分布:")
    for reason, count in exit_reason_dist.items():
        print(f"    - {reason}: {count} 次（{count/total_trades:.2%}）")
    
    # 保存回测详情
    backtest_result_path = r"d:\workspace\xiaoyao\data\strategy_backtest_results.parquet"
    valid_trades.to_parquet(backtest_result_path)
    print(f"\n回测详情已保存至: {backtest_result_path}")
    
    return {
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'avg_return': avg_return,
        'total_trades': total_trades
    }

# --------------------------
# 主函数：串联全流程
# --------------------------
def main():
    # 文件路径配置
    input_path = r"d:\workspace\xiaoyao\data\wide_table.parquet"
    signal_output_path = r"d:\workspace\xiaoyao\data\final_strategy_signals.parquet"
    
    # 步骤1：加载和预处理数据
    df_raw = load_and_preprocess_data(input_path)
    if df_raw is None:
        return
    
    # 步骤2：计算核心指标
    df_with_indicators = calculate_indicators(df_raw)
    print(f"\n指标计算完成：{df_with_indicators.shape[0]} 行 × {df_with_indicators.shape[1]} 列")
    
    # 步骤3：筛选热门行业并生成最终信号
    df_strategy = filter_top_industries(df_with_indicators)
    
    # 步骤4：保存最终信号
    df_strategy.to_parquet(signal_output_path)
    print(f"\n最终策略信号已保存至：{signal_output_path}")
    
    # 步骤5：执行回测
    backtest_metrics = backtest_strategy(df_strategy)
    
    # 步骤6：输出针对性优化建议
    if backtest_metrics:
        print("\n" + "="*50)
        print("=== 后续优化建议 ===")
        print("="*50)
        # 信号数量调整
        if backtest_metrics['total_trades'] < 400:
            print("1. 执行终极放宽（必达400条）：")
            print("   - 打开filter_top_industries函数，将df_final['is_top12_industry'] = True")
            print("   - 打开calculate_indicators函数，将auc_volume_ratio_ok阈值改为1.2")
        else:
            print("1. 信号数量达标，开始质量优化：")
            print("   - 竞价量比从1.5提至1.8")
            print("   - 止盈从4%提至4.5%")
        # 收益优化
        if backtest_metrics['avg_return'] < 0:
            print("2. 平均收益为负：增加流动性筛选")
            print("   - 在enhanced_signal中加入'df_valid['val_turnover_ratio'] > 1'")
        if backtest_metrics['profit_loss_ratio'] < 1.2:
            print("3. 盈亏比不足：收紧止损至-3.2%")

if __name__ == "__main__":
    main()