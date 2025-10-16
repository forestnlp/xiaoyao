# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\trytry\年化50%~20251015.ipynb



# ----------------------------------------------------------------------import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.dates as mdates
from typing import List, Dict, Optional, Tuple

# 全局配置
pd.set_option('display.max_columns', None)
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams['axes.unicode_minus'] = False

# 核心字段定义
CORE_FIELDS = [
    "date", "stock_code", "stock_name", "open", "close", "pre_close",
    "auc_volume", "current", "auc_volume_ratio_vs_5d_avg",
    "volume", "volume_ratio_vs_5d_avg",
    "buy_total", "sell_total", "b1_v", "a1_v", "b1_p", "a1_p",
    "volatility", "amplitude",
    "sw_l1_industry_name"
]

class StockSelectionStrategy:
    def __init__(self, data_path: str, 
                 output_dir: str = "enhanced_selection_results",
                 eval_dir: str = "enhanced_strategy_evaluation"):
        """股票选股策略主类"""
        self.data_path = data_path
        self.output_dir = output_dir
        self.eval_dir = eval_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # 数据存储
        self.raw_data = None
        self.processed_data = None
        self.stock_daily_data = {}  # 按股票分组的数据
        self.trading_dates = []
        self.trading_dates_map = {}  # 日期→索引映射
        
        # 策略结果
        self.daily_selections = {}
        self.final_selections = {}
        self.trade_records = None
        self.performance = None

    def load_data(self) -> bool:
        """加载并预处理数据"""
        try:
            print("加载选股数据...")
            # 支持CSV和Parquet格式
            if self.data_path.endswith('.csv'):
                self.raw_data = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.parquet'):
                self.raw_data = pd.read_parquet(self.data_path)
            else:
                raise ValueError("仅支持CSV或Parquet格式数据")
            
            # 检查核心字段
            missing_fields = [f for f in CORE_FIELDS if f not in self.raw_data.columns]
            if missing_fields:
                raise ValueError(f"缺失核心字段: {missing_fields}")
            
            # 数据预处理
            df = self.raw_data[CORE_FIELDS].copy()
            # 日期格式处理
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dropna()
            # 数值字段转换
            numeric_fields = [f for f in CORE_FIELDS if f not in 
                             ["date", "stock_code", "stock_name", "sw_l1_industry_name"]]
            for field in numeric_fields:
                df[field] = pd.to_numeric(df[field], errors='coerce')
            
            # 过滤无效数据
            initial_count = len(df)
            df = df[
                (df['auc_volume'] > 0) &
                (df['volume'] > 0) &
                (df['open'] > 0) &
                (df['close'] > 0)
            ].dropna(subset=numeric_fields + ["date"])
            
            # 按股票分组存储
            self.processed_data = df
            for code, group in df.groupby('stock_code'):
                sorted_group = group.sort_values('date').reset_index(drop=True)
                sorted_group['date_idx'] = range(len(sorted_group))
                date_map = {row['date']: row['date_idx'] for _, row in sorted_group.iterrows()}
                self.stock_daily_data[code] = {
                    'data': sorted_group,
                    'date_map': date_map
                }
            
            # 生成交易日列表及映射
            self.trading_dates = sorted(df['date'].unique())
            self.trading_dates_map = {date: idx for idx, date in enumerate(self.trading_dates)}
            
            print(f"数据预处理完成: 保留 {len(self.processed_data)} 条记录，{len(self.stock_daily_data)} 只股票，{len(self.trading_dates)} 个交易日")
            return True
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            return False

    def get_t1_data(self, stock_code: str, t_date: datetime) -> Optional[pd.Series]:
        """获取单只股票的T+1日数据"""
        if stock_code not in self.stock_daily_data:
            return None
        
        stock_info = self.stock_daily_data[stock_code]
        stock_df = stock_info['data']
        date_map = stock_info['date_map']
        
        # 计算T+1日日期
        t1_date = t_date + timedelta(days=1)
        # 检查T+1日是否为交易日
        if t1_date not in date_map:
            t_idx = self.trading_dates_map.get(t_date, -1)
            if t_idx == -1 or t_idx + 1 >= len(self.trading_dates):
                return None
            t1_date = self.trading_dates[t_idx + 1]
        
        # 获取T+1日数据
        if t1_date not in date_map:
            return None
        t1_row_idx = date_map[t1_date]
        t1_data = stock_df.iloc[t1_row_idx]
        
        return t1_data

    # 辅助函数：处理单个数值的限制范围
    @staticmethod
    def clip_value(value: float, min_value: float, max_value: float) -> float:
        """将单个数值限制在[min_value, max_value]范围内"""
        return max(min_value, min(value, max_value))

    def calculate_t_daily_scores(self, date_data: pd.DataFrame) -> pd.DataFrame:
        """计算T日基础评分"""
        df = date_data.copy()
        
        # T日衍生指标
        df['t_open_pct'] = (df['open'] - df['pre_close']) / df['pre_close'] * 100
        df['t_order_book_ratio'] = df['buy_total'] / df['sell_total'].clip(lower=1)
        df['t_buy_sell_ratio'] = df['b1_v'] / df['a1_v'].clip(lower=1)
        
        # T日百分位评分（权重50分）
        t_fields = [
            ("auc_volume_ratio_vs_5d_avg", 1, 10),
            ("volume_ratio_vs_5d_avg", 1, 10),
            ("t_order_book_ratio", 1, 7),
            ("t_buy_sell_ratio", 1, 7),
            ("t_open_pct", 1, 6),
            ("volatility", -1, 5),
            ("amplitude", -1, 5)
        ]
        
        # 计算T日评分
        t_total = 0
        for field, direction, weight in t_fields:
            if field == "t_open_pct":
                clipped = df[field].clip(0.3, 5)
                df[f'{field}_pctile'] = clipped.rank(pct=True) * 100
            else:
                rank_pct = df[field].rank(pct=True)
                df[f'{field}_pctile'] = rank_pct * 100 if direction == 1 else (1 - rank_pct) * 100
            
            df[f'{field}_score'] = df[f'{field}_pctile'] * (weight / 100)
            t_total += df[f'{field}_score']
        
        df['t_score'] = t_total.round(1)  # T日基础分（满分50）
        return df

    def calculate_t1_confirm_scores(self, t_data: pd.Series, t1_data: pd.Series) -> Dict[str, float]:
        """计算T+1日确认评分"""
        t1_scores = {}
        
        # 计算T+1日关键指标
        t1_scores['t1_auc_strength'] = t1_data['auc_volume'] / self.clip_value(t_data['volume'], 1, float('inf'))
        t1_scores['t1_auc_premium'] = (t1_data['current'] - t_data['close']) / self.clip_value(t_data['close'], 1e-6, float('inf')) * 100
        t1_scores['t1_buy_sell_ratio'] = t1_data['b1_v'] / self.clip_value(t1_data['a1_v'], 1, float('inf'))
        t1_scores['t1_spread'] = (t1_data['a1_p'] - t1_data['b1_p']) / self.clip_value(t1_data['b1_p'], 1e-6, float('inf')) * 100
        
        # 计算T+1日确认评分（满分50）
        t1_total = 0
        # 竞价强度评分
        if t1_scores['t1_auc_strength'] >= 1.2:
            t1_total += 15
        elif t1_scores['t1_auc_strength'] >= 1.0:
            t1_total += 10
        else:
            t1_total += 5
        
        # 竞价溢价评分
        if 0.1 <= t1_scores['t1_auc_premium'] <= 3:
            t1_total += 10
        elif (0 <= t1_scores['t1_auc_premium'] < 0.1) or (3 < t1_scores['t1_auc_premium'] <= 5):
            t1_total += 5
        else:
            t1_total += 2
        
        # 盘口承接力评分
        if t1_scores['t1_buy_sell_ratio'] >= 1.5:
            t1_total += 15
        elif t1_scores['t1_buy_sell_ratio'] >= 1.2:
            t1_total += 10
        else:
            t1_total += 5
        
        # 买卖价差评分
        if t1_scores['t1_spread'] < 0.5:
            t1_total += 10
        elif t1_scores['t1_spread'] <= 1:
            t1_total += 5
        else:
            t1_total += 2
        
        t1_scores['t1_score'] = round(t1_total, 1)
        t1_scores['final_score'] = round(t_data['t_score'] + t1_scores['t1_score'], 1)
        return t1_scores

    def select_stocks_with_t1_confirm(self, date: datetime, 
                                   top_quality_pct: float = 0.2,
                                   top_industry_n: int = 3,
                                   top_stock_per_industry: int = 3,
                                   t1_confirm_threshold: float = 70) -> Optional[pd.DataFrame]:
        """双阶段选股：T日初选+T+1日确认"""
        # T日初选
        t_data = self.processed_data[self.processed_data['date'] == date].copy()
        if len(t_data) < 30:
            print(f"{date.strftime('%Y-%m-%d')} T日个股不足30只，跳过")
            return None
        
        # 计算T日评分并初选
        t_scored = self.calculate_t_daily_scores(t_data)
        t_quality_threshold = t_scored['t_score'].quantile(1 - top_quality_pct)
        t_high_quality = t_scored[t_scored['t_score'] >= t_quality_threshold].copy()
        if len(t_high_quality) < 10:
            print(f"{date.strftime('%Y-%m-%d')} T日优质股不足10只，跳过")
            return None
        
        # T日按行业初选
        industry_count = t_high_quality.groupby('sw_l1_industry_name').size().reset_index(name='count')
        top_industries = industry_count.nlargest(top_industry_n, 'count')['sw_l1_industry_name'].tolist()
        
        t_selected = []
        for industry in top_industries:
            industry_stocks = t_high_quality[t_high_quality['sw_l1_industry_name'] == industry]
            top_in_industry = industry_stocks.nlargest(min(top_stock_per_industry, len(industry_stocks)), 't_score')
            t_selected.append(top_in_industry)
        
        t_final = pd.concat(t_selected, ignore_index=True)
        t_candidates = t_final.nlargest(15, 't_score')
        self.daily_selections[date] = t_candidates
        
        # T+1日确认
        final_results = []
        t_idx = self.trading_dates_map.get(date, -1)
        if t_idx == -1 or t_idx + 1 >= len(self.trading_dates):
            print(f"{date.strftime('%Y-%m-%d')} 无T+1日交易日，跳过确认")
            return None
        t1_date = self.trading_dates[t_idx + 1]
        
        # 遍历T日候选股
        for _, t_stock in t_candidates.iterrows():
            stock_code = t_stock['stock_code']
            t1_stock = self.get_t1_data(stock_code, date)
            if t1_stock is None:
                continue
            
            try:
                t1_scores = self.calculate_t1_confirm_scores(t_stock, t1_stock)
                
                # 组装最终结果
                final_stock = {
                    'T日选股日期': date,
                    'T+1确认日期': t1_date,
                    '股票代码': stock_code,
                    '股票名称': t_stock['stock_name'],
                    '所属行业': t_stock['sw_l1_industry_name'],
                    'T日基础分(50分)': t_stock['t_score'],
                    'T+1确认分(50分)': t1_scores['t1_score'],
                    '最终得分(100分)': t1_scores['final_score'],
                    'T日高开幅度(%)': round(t_stock['t_open_pct'], 3),
                    'T日竞价量5日比': round(t_stock['auc_volume_ratio_vs_5d_avg'], 3),
                    'T日成交量5日比': round(t_stock['volume_ratio_vs_5d_avg'], 3),
                    'T+1竞价强度(竞价量/T日成交量)': round(t1_scores['t1_auc_strength'], 3),
                    'T+1竞价溢价(%)': round(t1_scores['t1_auc_premium'], 3),
                    'T+1盘口承接力(买一/卖一)': round(t1_scores['t1_buy_sell_ratio'], 3),
                    'T+1买卖价差(%)': round(t1_scores['t1_spread'], 3)
                }
                final_results.append(final_stock)
            except Exception as e:
                print(f"处理股票 {stock_code} 时出错: {str(e)}")
                continue
        
        if not final_results:
            print(f"{date.strftime('%Y-%m-%d')} 无通过T+1日确认的个股")
            return None
        
        final_df = pd.DataFrame(final_results)
        # 按行业控制数量
        industry_filtered = []
        for industry in top_industries:
            industry_stocks = final_df[final_df['所属行业'] == industry]
            if len(industry_stocks) > 0:
                top_in_industry = industry_stocks.nlargest(min(2, len(industry_stocks)), '最终得分(100分)')
                industry_filtered.append(top_in_industry)
        
        if not industry_filtered:
            print(f"{date.strftime('%Y-%m-%d')} 行业筛选后无个股")
            return None
        
        final_selected = pd.concat(industry_filtered, ignore_index=True)
        final_selected = final_selected.nlargest(min(6, len(final_selected)), '最终得分(100分)')
        
        # 保存最终结果
        self.final_selections[date] = {
            't1_date': t1_date,
            'stocks': final_selected
        }
        print(f"{date.strftime('%Y-%m-%d')} 双阶段选股完成：{len(final_selected)}只（经T+1日确认）")
        return final_selected

    def run_strategy(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> bool:
        """执行选股策略"""
        target_dates = self.trading_dates
        if start_date:
            start = pd.to_datetime(start_date)
            target_dates = [d for d in target_dates if d >= start]
        if end_date:
            end = pd.to_datetime(end_date)
            target_dates = [d for d in target_dates if d <= end]
        
        # 排除最后1个交易日（确保有T+1日）
        if len(target_dates) > 1:
            target_dates = target_dates[:-1]
        else:
            print("无足够交易日（需至少2个：T日+T+1日）")
            return False
        
        print(f"开始双阶段选股：{len(target_dates)} 个交易日")
        
        all_results = []
        for date in tqdm(target_dates, desc="选股进度"):
            try:
                daily_result = self.select_stocks_with_t1_confirm(date)
                if daily_result is not None and not daily_result.empty:
                    all_results.append(daily_result)
                    
                    # 保存每日结果
                    date_str = date.strftime("%Y%m%d")
                    daily_result.to_csv(
                        os.path.join(self.output_dir, f"final_selected_{date_str}.csv"),
                        index=False, encoding="utf-8-sig"
                    )
            except Exception as e:
                print(f"{date.strftime('%Y-%m-%d')} 选股出错: {str(e)}")
                continue
        
        if all_results:
            # 保存合并结果
            all_selected = pd.concat(all_results, ignore_index=True)
            all_selected.to_csv(
                os.path.join(self.output_dir, "all_final_selected.csv"),
                index=False, encoding="utf-8-sig"
            )
            print(f"双阶段选股完成：累计选出 {len(all_selected)} 只个股")
            return True
        else:
            print("未选出任何个股（所有日期均未通过T+1日确认）")
            return False

    def evaluate_strategy(self, hold_days: int = 1) -> bool:
        """评测策略收益"""
        if not self.final_selections:
            print("无最终选股结果，无法评测")
            return False
        
        print(f"开始策略评测：T+1日开盘买入→持有{hold_days}天卖出")
        records = []
        
        for select_date, data in tqdm(self.final_selections.items(), desc="计算收益"):
            t1_buy_date = data['t1_date']
            selections = data['stocks']
            
            # 计算卖出日期
            t1_idx = self.trading_dates_map.get(t1_buy_date, -1)
            if t1_idx == -1 or t1_idx + hold_days >= len(self.trading_dates):
                continue
            sell_date = self.trading_dates[t1_idx + hold_days]
            
            # 遍历每只选出的股票
            for _, stock in selections.iterrows():
                stock_code = stock['股票代码']
                
                # 获取T+1日买入价格
                t1_data = self.get_t1_data(stock_code, select_date)
                if t1_data is None:
                    continue
                buy_price = t1_data['open']
                
                # 获取卖出日价格
                sell_prev_date = sell_date - timedelta(days=1)
                sell_data = self.get_t1_data(stock_code, sell_prev_date)
                if sell_data is None:
                    continue
                sell_price = sell_data['close']
                
                # 计算收益
                profit_pct = (sell_price - buy_price) / self.clip_value(buy_price, 1e-6, float('inf')) * 100
                profit_ratio = round(profit_pct, 2)
                
                # 记录交易详情
                records.append({
                    'T日选股': select_date,
                    'T+1买入日': t1_buy_date,
                    '卖出日': sell_date,
                    '股票代码': stock_code,
                    '股票名称': stock['股票名称'],
                    '所属行业': stock['所属行业'],
                    '最终得分': stock['最终得分(100分)'],
                    'T+1竞价强度': stock['T+1竞价强度(竞价量/T日成交量)'],
                    'T+1盘口承接力': stock['T+1盘口承接力(买一/卖一)'],
                    '买入价(元)': round(buy_price, 2),
                    '卖出价(元)': round(sell_price, 2),
                    '收益率(%)': profit_ratio
                })
        
        if not records:
            print("无有效交易记录用于评测")
            return False
        
        # 整理交易记录与指标
        self.trade_records = pd.DataFrame(records)
        total_trades = len(self.trade_records)
        profitable = (self.trade_records['收益率(%)'] > 0).sum()
        win_rate = round(profitable / total_trades * 100, 2) if total_trades > 0 else 0
        
        # 验证T+1日指标有效性
        t1_factor_perf = {
            '竞价强度>1.2': self.trade_records[
                self.trade_records['T+1竞价强度'] > 1.2
            ]['收益率(%)'].mean() if len(self.trade_records[self.trade_records['T+1竞价强度'] > 1.2]) > 0 else 0,
            '竞价强度≤1.2': self.trade_records[
                self.trade_records['T+1竞价强度'] <= 1.2
            ]['收益率(%)'].mean() if len(self.trade_records[self.trade_records['T+1竞价强度'] <= 1.2]) > 0 else 0,
            '承接力>1.5': self.trade_records[
                self.trade_records['T+1盘口承接力'] > 1.5
            ]['收益率(%)'].mean() if len(self.trade_records[self.trade_records['T+1盘口承接力'] > 1.5]) > 0 else 0,
            '承接力≤1.5': self.trade_records[
                self.trade_records['T+1盘口承接力'] <= 1.5
            ]['收益率(%)'].mean() if len(self.trade_records[self.trade_records['T+1盘口承接力'] <= 1.5]) > 0 else 0
        }
        
        # 计算累计收益
        self.trade_records = self.trade_records.sort_values('T+1买入日').reset_index(drop=True)
        self.trade_records['累计收益(%)'] = ((1 + self.trade_records['收益率(%)']/100).cumprod() - 1) * 100
        
        # 整体表现统计
        self.performance = {
            '总交易次数': total_trades,
            '盈利次数': profitable,
            '胜率(%)': win_rate,
            '平均收益率(%)': round(self.trade_records['收益率(%)'].mean(), 2),
            '累计收益率(%)': round(self.trade_records['累计收益(%)'].iloc[-1], 2),
            'T+1指标有效性': {k: round(v, 2) for k, v in t1_factor_perf.items()}
        }
        
        # 保存评测结果
        self.trade_records.to_csv(
            os.path.join(self.eval_dir, "trade_records.csv"),
            index=False, encoding="utf-8-sig"
        )
        
        # 打印评测结果
        print("\n===== 策略评测结果 =====")
        print(f"交易规则: T日初选→T+1日确认→开盘买入→持有{hold_days}天卖出")
        print(f"总交易次数: {self.performance['总交易次数']}")
        print(f"胜率: {self.performance['胜率(%)']}%")
        print(f"平均收益率: {self.performance['平均收益率(%)']}%")
        print(f"累计收益率: {self.performance['累计收益率(%)']}%")
        print("\n----- T+1日指标有效性验证 -----")
        for k, v in self.performance['T+1指标有效性'].items():
            print(f"{k} 平均收益率: {v}%")
        
        return True

    def plot_performance(self) -> None:
        """可视化策略表现"""
        if self.trade_records is None or self.performance is None:
            print("无评测数据，无法绘图")
            return
        
        plt.figure(figsize=(15, 15))
        
        # 子图1：累计收益率走势
        plt.subplot(3, 1, 1)
        plt.plot(
            self.trade_records['T+1买入日'],
            self.trade_records['累计收益(%)'],
            'g-', linewidth=2.5,
            label=f'累计收益率: {self.performance["累计收益率(%)"]}%'
        )
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.6)
        plt.fill_between(
            self.trade_records['T+1买入日'],
            self.trade_records['累计收益(%)'],
            0,
            where=(self.trade_records['累计收益(%)'] >= 0),
            color='lightgreen', alpha=0.3
        )
        plt.title('策略累计收益率走势', fontsize=14)
        plt.ylabel('累计收益率(%)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gcf().autofmt_xdate()
        
        # 子图2：T+1日竞价强度与收益率关系
        plt.subplot(3, 1, 2)
        self.trade_records['竞价强度区间'] = pd.cut(
            self.trade_records['T+1竞价强度'],
            bins=[0, 0.5, 1.0, 1.2, 1.5, 3.0, 10.0],
            labels=['0-0.5', '0.5-1.0', '1.0-1.2', '1.2-1.5', '1.5-3.0', '3.0+']
        )
        auc_perf = self.trade_records.groupby('竞价强度区间')['收益率(%)'].agg(['mean', 'count']).reset_index()
        
        bars = plt.bar(
            auc_perf['竞价强度区间'],
            auc_perf['mean'],
            color='royalblue'
        )
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.6)
        plt.title('T+1日竞价强度与收益率关系', fontsize=14)
        plt.ylabel('平均收益率(%)', fontsize=12)
        for i, (_, row) in enumerate(auc_perf.iterrows()):
            plt.text(i, row['mean'] + 0.05,
                     f'{row["mean"]:.2f}%\n(N={row["count"]})',
                     ha='center', va='bottom')
        
        # 子图3：T+1日盘口承接力与收益率关系
        plt.subplot(3, 1, 3)
        self.trade_records['承接力区间'] = pd.cut(
            self.trade_records['T+1盘口承接力'],
            bins=[0, 0.8, 1.2, 1.5, 2.0, 10.0],
            labels=['0-0.8', '0.8-1.2', '1.2-1.5', '1.5-2.0', '2.0+']
        )
        buy_sell_perf = self.trade_records.groupby('承接力区间')['收益率(%)'].agg(['mean', 'count']).reset_index()
        
        bars = plt.bar(
            buy_sell_perf['承接力区间'],
            buy_sell_perf['mean'],
            color='seagreen'
        )
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.6)
        plt.title('T+1日盘口承接力与收益率关系', fontsize=14)
        plt.xlabel('T+1日买一/卖一挂单比区间', fontsize=12)
        plt.ylabel('平均收益率(%)', fontsize=12)
        for i, (_, row) in enumerate(buy_sell_perf.iterrows()):
            plt.text(i, row['mean'] + 0.05,
                     f'{row["mean"]:.2f}%\n(N={row["count"]})',
                     ha='center', va='bottom')
        
        plt.tight_layout(pad=3.0)
        plt.savefig(
            os.path.join(self.eval_dir, 'strategy_performance.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        print(f"策略表现图表已保存至: {self.eval_dir}/strategy_performance.png")

# 主函数入口
if __name__ == "__main__":
    # 配置参数（请替换为实际路径）
    DATA_PATH = r"D:\workspace\xiaoyao\data\factortable.parquet"
    START_DATE = "2025-01-01"
    END_DATE = "2025-09-20"
    HOLD_DAYS = 1  # 持有天数
    
    # 启动策略
    print("=== 启动股票选股策略 ===")
    strategy = StockSelectionStrategy(DATA_PATH)
    
    # 执行流程
    if strategy.load_data():
        if strategy.run_strategy(start_date=START_DATE, end_date=END_DATE):
            if strategy.evaluate_strategy(hold_days=HOLD_DAYS):
                strategy.plot_performance()
    
    print(f"\n所有流程完成!")
    print(f"最终选股结果目录: {strategy.output_dir}")
    print(f"评测结果目录: {strategy.eval_dir}")
    

import pandas as pd

# 读取交易记录文件
df = pd.read_csv(r'D:\workspace\xiaoyao\works\trytry\enhanced_strategy_evaluation/trade_records.csv')

# 按选股日期分组，计算每日平均收益率
sydf = df.groupby('T日选股')['收益率(%)'].mean().reset_index()

# 重命名列名
sydf = sydf.rename(columns={'T日选股': 'date', '收益率(%)': 'return'})

# 查看结果
print(sydf.head())

# 用1+0.5*return后 得到收益增长
sydf['return'] = 1 + 0.5 * sydf['return']/100
sydf

# 将return依次连乘，得到累计收益率
sydf['return'] = sydf['return'].cumprod()
sydf

