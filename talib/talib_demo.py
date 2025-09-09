#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TA-Lib技术分析库演示程序
重点演示MACD、RSI、BOLL、MA和成交量指标
仅使用真实股票数据，符合项目规范
"""

import baostock as bs
import pandas as pd
import talib
import numpy as np
import time
from typing import Optional, Dict, Any


def fetch_stock_data(symbol: str = "sh.600570", 
                    start_date: str = '2023-01-01', 
                    end_date: str = '2024-01-01') -> Optional[pd.DataFrame]:
    """从BaoStock获取真实股票数据"""
    print(f"\n=== 获取{symbol}真实数据 ===")
    
    lg = bs.login()
    if lg.error_code != '0':
        print(f"BaoStock登录失败: {lg.error_msg}")
        return None

    try:
        fields = "date,code,open,high,low,close,volume,amount"
        rs = bs.query_history_k_data_plus(
            symbol, fields, 
            start_date=start_date, 
            end_date=end_date,
            frequency="d", 
            adjustflag="2"  # 前复权
        )
        
        if rs.error_code != '0':
            print(f"查询数据失败: {rs.error_msg}")
            return None

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            print(f"未获取到{symbol}的数据")
            return None

        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 数据类型转换
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna()  # 删除无效数据
        
        print(f"成功获取{len(df)}天数据")
        return df
        
    except Exception as e:
        print(f"获取数据异常: {e}")
        return None
    finally:
        bs.logout()


def calculate_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算核心技术指标：MA、RSI、MACD、BOLL"""
    print("\n=== 计算核心技术指标 ===")
    
    if df is None or len(df) < 50:
        print("数据不足，无法计算指标")
        return df
    
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    
    try:
        # MA移动平均线
        df['MA_5'] = talib.SMA(close, timeperiod=5)
        df['MA_20'] = talib.SMA(close, timeperiod=20)
        
        # RSI相对强弱指标
        df['RSI_14'] = talib.RSI(close, timeperiod=14)
        
        # MACD指标
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        # 布林带(BOLL)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BOLL_Upper'] = bb_upper
        df['BOLL_Middle'] = bb_middle
        df['BOLL_Lower'] = bb_lower
        
        print("核心指标计算完成")
        
    except Exception as e:
        print(f"指标计算失败: {e}")
    
    return df


def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算成交量指标：OBV、AD、MFI"""
    print("\n=== 计算成交量指标 ===")
    
    if df is None or len(df) < 30:
        print("数据不足，无法计算成交量指标")
        return df
    
    try:
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)
        
        # OBV能量潮指标
        df['OBV'] = talib.OBV(close, volume)
        # AD累积/派发线
        df['AD'] = talib.AD(high, low, close, volume)
        # MFI资金流量指标
        df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        
        print("成交量指标计算完成")
        
    except Exception as e:
        print(f"成交量指标计算失败: {e}")
    
    return df


def analyze_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """分析交易信号"""
    print("\n=== 技术信号分析 ===")
    
    if df is None or len(df) == 0:
        return {}
    
    latest = df.iloc[-1]
    signals = {}
    
    try:
        # MA趋势信号
        if pd.notna(latest['MA_5']) and pd.notna(latest['MA_20']):
            if latest['MA_5'] > latest['MA_20']:
                signals['MA_trend'] = "看涨(短期均线上穿)"
            else:
                signals['MA_trend'] = "看跌(短期均线下穿)"
        
        # RSI信号
        if pd.notna(latest['RSI_14']):
            rsi = latest['RSI_14']
            if rsi > 70:
                signals['RSI'] = "超买区域"
            elif rsi < 30:
                signals['RSI'] = "超卖区域"
            else:
                signals['RSI'] = "中性区域"
        
        # MACD信号
        if pd.notna(latest['MACD_Hist']):
            if latest['MACD_Hist'] > 0:
                signals['MACD'] = "多头动能"
            else:
                signals['MACD'] = "空头动能"
        
        # 布林带信号
        if all(pd.notna(latest[col]) for col in ['BOLL_Upper', 'BOLL_Lower']):
            price = latest['close']
            if price > latest['BOLL_Upper']:
                signals['BOLL'] = "突破上轨"
            elif price < latest['BOLL_Lower']:
                signals['BOLL'] = "跌破下轨"
            else:
                signals['BOLL'] = "在轨道内"
        
        # 成交量信号
        if pd.notna(latest['MFI']):
            mfi = latest['MFI']
            if mfi > 80:
                signals['MFI'] = "资金流入过度"
            elif mfi < 20:
                signals['MFI'] = "资金流出过度"
            else:
                signals['MFI'] = "资金流动正常"
            
    except Exception as e:
        print(f"信号分析失败: {e}")
    
    return signals


def print_analysis_report(df: pd.DataFrame, signals: Dict[str, Any], symbol: str = ""):
    """打印分析报告"""
    print(f"\n{'='*50}")
    print(f"技术分析报告: {symbol}")
    print(f"{'='*50}")
    
    if df is None or len(df) == 0:
        print("无数据可分析")
        return
    
    latest = df.iloc[-1]
    
    print("\n--- 最新价格信息 ---")
    print(f"日期: {latest.get('date', 'N/A')}")
    print(f"收盘价: {latest['close']:.2f}")
    print(f"成交量: {latest['volume']:,.0f}")
    
    print("\n--- 技术指标 ---")
    if pd.notna(latest.get('SMA_10')):
        print(f"SMA(10): {latest['SMA_10']:.2f}")
    if pd.notna(latest.get('SMA_30')):
        print(f"SMA(30): {latest['SMA_30']:.2f}")
    if pd.notna(latest.get('RSI_14')):
        print(f"RSI(14): {latest['RSI_14']:.2f}")
    if pd.notna(latest.get('MACD_Hist')):
        print(f"MACD柱: {latest['MACD_Hist']:.4f}")
    
    print("\n--- 交易信号 ---")
    for key, value in signals.items():
        print(f"{key.upper()}: {value}")
    
    print("\n--- 最近5日数据 ---")
    display_cols = ['date', 'close', 'volume']
    if 'SMA_10' in df.columns:
        display_cols.extend(['SMA_10', 'RSI_14'])
    
    available_cols = [col for col in display_cols if col in df.columns]
    print(df[available_cols].tail())


def demo_core_indicators():
    """核心指标演示"""
    print("\n" + "="*50)
    print("TA-Lib 核心指标演示")
    print("="*50)
    
    # 使用真实股票数据
    df = fetch_stock_data('sz.000001', '2023-01-01', '2024-01-01')
    
    if df is not None and len(df) > 0:
        # 计算核心技术指标
        df = calculate_core_indicators(df)
        df = calculate_volume_indicators(df)
        
        # 分析信号
        signals = analyze_signals(df)
        
        # 打印分析报告
        print_analysis_report(df, signals, "平安银行(000001)")
    else:
        print("无法获取股票数据，请检查网络连接")


def demo_real_stock_analysis(symbol: str = "sh.600570"):
    """演示真实股票分析"""
    print("\n" + "="*60)
    print(f"真实股票分析演示: {symbol}")
    print("="*60)
    
    # 获取真实数据
    df = fetch_stock_data(symbol)
    if df is None:
        print("无法获取真实数据，跳过此演示")
        return
    
    # 计算指标
    df = calculate_core_indicators(df)
    df = calculate_volume_indicators(df)
    
    # 分析信号
    signals = analyze_signals(df)
    print_analysis_report(df, signals, symbol)


def batch_stock_analysis():
    """批量股票分析演示"""
    print("\n" + "="*60)
    print("批量股票分析演示")
    print("="*60)
    
    stocks = ["sh.600000", "sh.600036", "sh.600519"]
    
    for stock in stocks:
        print(f"\n{'='*20} 分析 {stock} {'='*20}")
        
        df = fetch_stock_data(stock)
        if df is None:
            print(f"{stock}: 数据获取失败")
            continue
            
        df = calculate_core_indicators(df)
        df = calculate_volume_indicators(df)
        signals = analyze_signals(df)
        
        # 简化输出
        latest = df.iloc[-1]
        print(f"收盘价: {latest['close']:.2f}")
        print(f"MA趋势: {signals.get('MA_trend', 'N/A')}")
        print(f"RSI: {signals.get('RSI', 'N/A')}")
        print(f"MACD: {signals.get('MACD', 'N/A')}")
        print(f"BOLL: {signals.get('BOLL', 'N/A')}")
        print(f"MFI: {signals.get('MFI', 'N/A')}")
        
        time.sleep(1)  # 避免请求过快


if __name__ == "__main__":
    start_time = time.time()
    print("TA-Lib技术分析演示程序启动")
    print(f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 演示1: 核心指标演示
        demo_core_indicators()
        
        # 演示2: 真实股票分析
        demo_real_stock_analysis("sh.600570")
        
        # 演示3: 批量分析
        batch_stock_analysis()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行异常: {e}")
    finally:
        end_time = time.time()
        print(f"\n程序结束，总耗时: {end_time - start_time:.2f}秒")
        print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")