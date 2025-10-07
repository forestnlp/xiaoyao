#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import talib
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# 1. 数据加载与预处理（仅保留一份实现）
# --------------------------
def load_and_preprocess_data(file_path):
    """加载并预处理数据，确保数据质量"""
    try:
        # 加载Parquet数据
        df = pd.read_parquet(file_path)
        print(f"数据加载成功：{df.shape[0]} 行 × {df.shape[1]} 列")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return None
    
    # 数据预处理
    df['date'] = pd.to_datetime(df['date'])  # 确保日期格式正确
    df = df.sort_values(by=['stock_code', 'date'])  # 按股票和日期排序
    
    # 处理重复记录
    duplicates = df.duplicated(subset=['stock_code', 'date'], keep=False)
    if duplicates.any():
        print(f"处理重复记录: {duplicates.sum()} 条")
        df = df.drop_duplicates(subset=['stock_code', 'date'], keep='first')
    
    # 过滤停牌数据
    if 'paused' in df.columns:
        paused_count = df[df['paused'] == 1].shape[0]
        if paused_count > 0:
            print(f"过滤停牌记录: {paused_count} 条")
            df = df[df['paused'] == 0]
    
    # 保存预处理结果（保留In[11]中的实用功能）
    preprocessed_path = r"d:\workspace\xiaoyao\data\processed_stock_data.parquet"
    df.to_parquet(preprocessed_path)
    print(f"预处理完成，数据已保存至: {preprocessed_path}")
    
    return df

# --------------------------
# 2. 核心指标计算（仅保留一份实现）
# --------------------------
def calculate_indicators(df):
    """计算所有核心指标：技术指标、量能指标、趋势指标等"""
    df_indicators = df.copy()
    
    def compute_group_indicators(group):
        """对单只股票计算指标"""
        group = group.sort_values('date').reset_index(drop=True)
        close = group['close'].values
        high = group['high'].values
        low = group['low'].values
        open_ = group['open'].values
        volume = group['volume'].values
        auc_volume = group['auc_volume'].values
        auc_money = group['auc_money'].values
        money = group['money'].values
        
        # 1. 技术指标
        macd_line, macd_signal, _ = talib.MACD(close, 12, 26, 9)
        group['macd_ok'] = (macd_line > macd_signal) & (macd_line > -0.5)
        
        upper_band, _, _ = talib.BBANDS(close, 20, 2, 2, matype=talib.MA_Type.SMA)
        group['bb_ok'] = close > (upper_band * 0.95)
        
        atr = talib.ATR(high, low, close, 14)
        group['atr_ok'] = atr / close < 0.07
        
        # 2. 量能与竞价指标
        group['volume_ma5'] = group['volume'].rolling(window=5, min_periods=1).mean()
        group['volume_ratio'] = volume / group['volume_ma5']
        group['volume_ratio_ok'] = (group['volume_ratio'] > 1.2) & (group['volume_ratio'] < 15)
        
        group['auc_volume_ratio'] = auc_volume / group['volume_ma5']
        group['auc_volume_ratio_ok'] = group['auc_volume_ratio'] > 3
        
        group['auc_money_ratio_ok'] = auc_money / money > 0.15
        
        group['total_bid'] = group[['auc_b1_v', 'auc_b2_v', 'auc_b3_v', 'auc_b4_v', 'auc_b5_v']].sum(axis=1)
        group['total_ask'] = group[['auc_a1_v', 'auc_a2_v', 'auc_a3_v', 'auc_a4_v', 'auc_a5_v']].sum(axis=1)
        group['bid_ask_ratio'] = np.where(group['total_ask'] > 0, group['total_bid'] / group['total_ask'], 0)
        group['bid_ask_ratio_ok'] = group['bid_ask_ratio'] > 1.1
        
        # 3. 趋势强度指标
        group['ma20'] = group['close'].rolling(window=20, min_periods=1).mean()
        group['ma_ok'] = group['close'] > group['ma20']
        
        # 4. 估值安全指标（行业PE分位）
        def calc_pe_quantile(subgroup):
            subgroup['pe_quantile'] = subgroup['val_pe_ratio'].rank(pct=True)
            return subgroup
        group = group.groupby('ind_sw_l1_industry_name', group_keys=False).apply(calc_pe_quantile)
        group['pe_quantile_ok'] = group['pe_quantile'] < 0.8
        
        # 5. 流动性指标
        group['circ_cap_ok'] = (group['val_circulating_market_cap'] > 50) & (group['val_circulating_market_cap'] < 500)
        group['turnover_ok'] = (group['val_turnover_ratio'] > 1) & (group['val_turnover_ratio'] < 15)
        
        return group
    
    # 按股票代码分组计算指标（仅计算一次）
    df_indicators = df_indicators.groupby('stock_code', group_keys=False).apply(compute_group_indicators)
    
    # 过滤关键指标缺失的记录
    valid_columns = ['macd_ok', 'bb_ok', 'atr_ok', 'volume_ratio_ok', 'ma_ok', 'circ_cap_ok']
    df_valid = df_indicators.dropna(subset=valid_columns).copy()
    
    # 定义信号层级
    df_valid['basic_signal'] = (
        df_valid['macd_ok'] & df_valid['bb_ok'] & 
        df_valid['atr_ok'] & df_valid['volume_ratio_ok']
    )
    
    df_valid['enhanced_signal'] = (
        df_valid['basic_signal'] & df_valid['ma_ok'] & 
        df_valid['circ_cap_ok'] & df_valid['turnover_ok']
    )
    
    return df_valid

# --------------------------
# 3. 行业筛选与最终信号生成
# --------------------------
def filter_top_industries(df):
    """筛选每日热门行业并生成最终选股信号"""
    df_final = df.copy()
    
    # 1. 筛选竞价高热度标的（竞价量比>3）
    high_bid_stocks = df_final[df_final['auc_volume_ratio'] > 3].copy()
    print(f"竞价高热度标的总数: {len(high_bid_stocks)} 条")
    
    # 2. 按日期统计各行业热度（标的数量）
    industry_daily_counts = high_bid_stocks.groupby(
        ['date', 'ind_sw_l1_industry_name']
    ).size().reset_index(name='stock_count')
    
    # 3. 生成每日前5大热门行业字典
    def get_top5_industries(daily_data):
        return daily_data.nlargest(5, 'stock_count')['ind_sw_l1_industry_name'].tolist()
    
    top5_industries_dict = industry_daily_counts.groupby('date').apply(
        get_top5_industries
    ).to_dict()
    
    # 4. 判断个股是否属于当日热门行业
    def is_in_top5_industry(row):
        daily_top5 = top5_industries_dict.get(row['date'], [])
        return row['ind_sw_l1_industry_name'] in daily_top5
    
    df_final['is_top5_industry'] = df_final.apply(is_in_top5_industry, axis=1)
    
    # 5. 生成最终选股信号
    df_final['final_signal'] = df_final['enhanced_signal'] & df_final['is_top5_industry']
    
    # 6. 信号统计
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
        print("❌ 未满足目标，建议调整：")
        print("1. 竞价量比阈值放宽至2.5")
        print("2. 热门行业数量增加至前8")
    
    # 保留核心字段
    core_fields = [
        'date', 'stock_code', 'ind_sw_l1_industry_name', 'open', 'close',
        'macd_ok', 'bb_ok', 'volume_ratio', 'auc_volume_ratio',
        'ma20', 'val_circulating_market_cap', 'val_turnover_ratio',
        'enhanced_signal', 'is_top5_industry', 'final_signal'
    ]
    return df_final[core_fields].copy()

# --------------------------
# 4. 策略回测
# --------------------------
def backtest_strategy(signal_df):
    """策略回测核心逻辑"""
    print("\n" + "="*50)
    print("开始策略回测...")
    print("="*50)
    
    # 1. 提取交易信号
    trade_signals = signal_df[signal_df['final_signal'] == True].copy()
    print(f"总交易信号数量: {len(trade_signals)} 条")
    
    if len(trade_signals) < 10:
        print("❌ 交易信号过少，无法进行有效回测")
        return None
    
    # 2. 匹配后续5日收盘价
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
            return pd.Series(exit_prices, index=['exit_price_1d', 'exit_price_2d', 'exit_price_3d', 'exit_price_4d', 'exit_price_5d'])
        
        exit_prices = group.apply(get_exit_price, axis=1)
        group = pd.concat([group, exit_prices], axis=1)
        return group
    
    trade_signals_with_exit = trade_signals.groupby('stock_code', group_keys=False).apply(get_future_prices)
    valid_trades = trade_signals_with_exit.dropna(subset=['exit_price_1d']).copy()
    print(f"有效交易信号数量: {len(valid_trades)} 条")
    
    if len(valid_trades) < 10:
        print("❌ 有效交易信号过少，无法进行有效回测")
        return None
    
    # 3. 计算每笔交易的盈亏（已修正未来函数：入场价=当日收盘价）
    def calculate_trade_return(row):
        entry_price = row['close']  # 核心修正：用当日收盘价作为入场价
        exit_prices = [row['exit_price_1d'], row['exit_price_2d'], row['exit_price_3d'], row['exit_price_4d'], row['exit_price_5d']]
        
        trade_return = 0.0
        hold_days = 0
        exit_reason = ""
        
        for i, exit_price in enumerate(exit_prices):
            daily_return = (exit_price - entry_price) / entry_price
            hold_days = i + 1
            
            if daily_return >= 0.06:
                trade_return = daily_return
                exit_reason = "止盈（≥6%）"
                break
            elif daily_return <= -0.03:
                trade_return = daily_return
                exit_reason = "止损（≤-3%）"
                break
            elif i == 4:
                trade_return = daily_return
                exit_reason = "到期出场（5天）"
        
        return pd.Series([trade_return, hold_days, exit_reason], index=['trade_return', 'hold_days', 'exit_reason'])
    
    trade_results = valid_trades.apply(calculate_trade_return, axis=1)
    valid_trades = pd.concat([valid_trades, trade_results], axis=1)
    
    # 4. 统计回测指标
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
    print(f"4. 胜率: {win_rate:.2%} {'✅' if win_rate > 0.5 else '❌'}（目标：＞50%）")
    print(f"5. 盈亏比: {profit_loss_ratio:.2f} {'✅' if profit_loss_ratio >= 1.9 else '❌'}（目标：≥1.9）")
    print(f"6. 平均每笔收益: {avg_return:.2%}")
    print(f"7. 平均持仓天数: {avg_hold_days:.1f} 天")
    print(f"\n8. 出场原因分布:")
    for reason, count in exit_reason_dist.items():
        print(f"    - {reason}: {count} 次（{count/total_trades:.2%}）")
    
    # 保存回测结果
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
    """股票策略全流程主函数"""
    # 文件路径配置
    input_path = r"d:\workspace\xiaoyao\data\wide_table.parquet"
    signal_output_path = r"d:\workspace\xiaoyao\data\final_strategy_signals.parquet"
    
    # 步骤1：加载和预处理数据
    df_raw = load_and_preprocess_data(input_path)
    if df_raw is None:
        return
    
    # 步骤2：计算核心指标（仅计算一次）
    df_with_indicators = calculate_indicators(df_raw)
    print(f"\n指标计算完成：{df_with_indicators.shape[0]} 行 × {df_with_indicators.shape[1]} 列")
    
    # 步骤3：筛选热门行业并生成最终信号
    df_strategy = filter_top_industries(df_with_indicators)
    
    # 步骤4：保存最终信号
    df_strategy.to_parquet(signal_output_path)
    print(f"\n最终策略信号已保存至：{signal_output_path}")
    
    # 步骤5：执行回测
    backtest_metrics = backtest_strategy(df_strategy)
    
    # 步骤6：输出优化建议
    if backtest_metrics:
        print("\n" + "="*50)
        print("=== 后续优化建议 ===")
        print("="*50)
        if backtest_metrics['win_rate'] <= 0.5:
            print("1. 胜率未达标：建议收紧技术指标条件（如MACD仅保留零轴上金叉）")
        if backtest_metrics['profit_loss_ratio'] < 1.9:
            print("2. 盈亏比未达标：建议调整止盈止损比例（如止盈7%/止损3%）")
        if backtest_metrics['avg_return'] < 0:
            print("3. 平均收益为负：建议增加估值筛选（如PE分位<60%）")
        print("4. 若需增加交易机会：可放宽竞价量比至2.5或热门行业增至前8")

if __name__ == "__main__":
    main()
