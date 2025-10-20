# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\trytry\基于竞价和量比\数据计算与验证.ipynb



# ----------------------------------------------------------------------import pandas as pd
import numpy as np
import os
from glob import glob
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# ---------------------- 工具函数：统一日期处理 ----------------------
def _str_to_date(date_str: str) -> datetime.date:
    """将字符串日期（如"2025-01-02"）转为datetime.date类型"""
    return datetime.strptime(date_str, "%Y-%m-%d").date()

# ---------------------- 模块1：DataLoader（保留缓存，高效加载） ----------------------
class DataLoader:
    def __init__(self, 
                 daily_data_path: str = r"D:\workspace\xiaoyao\data\widetable.parquet",
                 minutely_data_root: str = r"D:\workspace\xiaoyao\data\stock_minutely_price"):
        self.daily_data_path = daily_data_path
        self.minutely_data_root = minutely_data_root
        self.daily_df = None  # 日度数据缓存
        self.minutely_cache = {}  # 分钟K缓存（避免重复加载）
        
        # 预加载所有股票代码（从分钟K目录提取）
        self.all_stock_codes = [
            path.split("=")[-1] for path in glob(f"{minutely_data_root}/stock_code=*")
        ]
        print(f"[DataLoader] 初始化完成：宽表路径={daily_data_path}，检测到{len(self.all_stock_codes)}只股票")

    def load_daily_data(self, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """加载日度数据，支持按日期筛选"""
        if self.daily_df is None:
            self.daily_df = pd.read_parquet(self.daily_data_path)
            self.daily_df['date'] = self.daily_df['date'].apply(_str_to_date)  # 统一日期格式
            print(f"[DataLoader] 宽表加载完成：原始数据{len(self.daily_df)}行")
        
        # 按日期筛选
        filtered_df = self.daily_df.copy()
        if start_date:
            start = _str_to_date(start_date)
            filtered_df = filtered_df[filtered_df['date'] >= start]
        if end_date:
            end = _str_to_date(end_date)
            filtered_df = filtered_df[filtered_df['date'] <= end]
        
        print(f"[DataLoader] 筛选后数据：{start_date or '开始'}至{end_date or '结束'}，共{len(filtered_df)}行")
        return filtered_df

    def load_minutely_data(self, 
                         stock_codes: List[str], 
                         date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """按需加载指定股票的分钟K数据，支持按日期筛选"""
        result = {}
        target_date = _str_to_date(date) if date else None
        
        for code in stock_codes:
            # 优先从缓存读取
            if code in self.minutely_cache:
                df = self.minutely_cache[code]
            else:
                # 构建分钟K文件路径（适配你的目录格式：stock_code=XXX/data.parquet）
                code_path = os.path.join(self.minutely_data_root, f"stock_code={code}", "data.parquet")
                if not os.path.exists(code_path):
                    print(f"[DataLoader] 警告：{code}的分钟K不存在（路径：{code_path}）")
                    continue
                
                # 加载并预处理分钟K数据
                df = pd.read_parquet(code_path)
                df['date'] = df['date'].apply(_str_to_date)  # 统一日期
                df['time'] = pd.to_datetime(df['time']).dt.time  # 统一时间（时分秒）
                self.minutely_cache[code] = df
                print(f"[DataLoader] 加载分钟K：{code}，原始数据{len(df)}行")
            
            # 按日期筛选（若指定）
            if target_date:
                df_filtered = df[df['date'] == target_date].copy()
                print(f"[DataLoader] {code}在{date}的分钟K数据：{len(df_filtered)}行")
            else:
                df_filtered = df.copy()
            
            result[code] = df_filtered
        
        return result

    def clear_cache(self):
        """清空分钟K缓存，释放内存"""
        self.minutely_cache = {}
        print(f"[DataLoader] 分钟K缓存已清空")

# ---------------------- 模块2：HotspotAnalyzer（适配numpy.ndarray概念） ----------------------
class HotspotAnalyzer:
    def __init__(self, daily_df: pd.DataFrame):
        self.daily_df = daily_df.copy()
        
        # 1. 解析numpy.ndarray类型的概念数据
        print("\n[HotspotAnalyzer] 开始解析概念数据...")
        self.daily_df['concepts'] = self.daily_df['concept_name_list'].apply(self._parse_concepts)
        
        # 2. 验证概念解析效果
        valid_count = self.daily_df['concepts'].apply(lambda x: len(x) > 0).sum()
        total_count = len(self.daily_df)
        print(f"[HotspotAnalyzer] 概念解析结果：有效占比{valid_count/total_count:.2%}（{valid_count}/{total_count}行）")
        
        # 3. 计算个股涨跌幅（用close和pre_close）
        if 'pre_close' not in self.daily_df.columns:
            self.daily_df = self.daily_df.sort_values(['stock_code', 'date'])
            self.daily_df['pre_close'] = self.daily_df.groupby('stock_code')['close'].shift(1)
            print(f"[HotspotAnalyzer] 提示：无pre_close字段，已用前一日close填充")
        
        self.daily_df['pct_change'] = (self.daily_df['close'] / self.daily_df['pre_close']) - 1
        print(f"[HotspotAnalyzer] 初始化完成，数据日期范围：{self.daily_df['date'].min()}至{self.daily_df['date'].max()}")

    def _parse_concepts(self, concept_data) -> List[str]:
        """专门适配numpy.ndarray类型的概念解析"""
        # 情况1：numpy数组（核心适配）
        if isinstance(concept_data, np.ndarray):
            return [str(c).strip() for c in concept_data if str(c).strip()]
        # 情况2：普通列表（兼容）
        elif isinstance(concept_data, list):
            return [str(c).strip() for c in concept_data if str(c).strip()]
        # 情况3：其他类型（空值、字符串等）
        else:
            return []

    def get_hot_industries(self, date: str, top_n: int = 5) -> List[str]:
        """获取指定日期的热点行业（输入日期格式："2025-01-02"）"""
        date_obj = _str_to_date(date)
        df_day = self.daily_df[self.daily_df['date'] == date_obj]
        if df_day.empty:
            print(f"[HotspotAnalyzer] 警告：{date}无行业数据")
            return []
        
        # 统计行业指标（平均涨幅、总成交量、股票数量）
        industry_metrics = df_day.groupby('zjw_industry_name').agg(
            avg_pct_change=('pct_change', 'mean'),
            total_volume=('volume', 'sum'),
            stock_count=('stock_code', 'nunique')
        ).reset_index()
        
        # 过滤股票数过少的行业（至少3只）
        industry_metrics = industry_metrics[industry_metrics['stock_count'] >= 3]
        if industry_metrics.empty:
            return []
        
        # 计算热点分数（排名越低越热）
        industry_metrics['return_rank'] = industry_metrics['avg_pct_change'].rank(ascending=False)
        industry_metrics['volume_rank'] = industry_metrics['total_volume'].rank(ascending=False)
        industry_metrics['hot_score'] = industry_metrics['return_rank'] + industry_metrics['volume_rank']
        
        return industry_metrics.sort_values('hot_score').head(top_n)['zjw_industry_name'].tolist()

    def get_hot_concepts(self, date: str, top_n: int = 5) -> List[str]:
        """获取指定日期的热点概念（输入日期格式："2025-01-02"）"""
        date_obj = _str_to_date(date)
        df_day = self.daily_df[self.daily_df['date'] == date_obj]
        if df_day.empty:
            print(f"[HotspotAnalyzer] 警告：{date}无数据")
            return []
        
        # 收集概念-股票-涨跌幅数据（过滤停牌股和无概念股）
        concept_list = []
        for _, row in df_day.iterrows():
            if row.get('paused', 0) == 1:
                continue
            if len(row['concepts']) == 0:
                continue
            for concept in row['concepts']:
                concept_list.append({
                    'concept': concept,
                    'stock_code': row['stock_code'],
                    'pct_change': row['pct_change']
                })
        
        if not concept_list:
            print(f"[HotspotAnalyzer] 警告：{date}无有效概念数据")
            return []
        
        # 统计概念指标
        concept_df = pd.DataFrame(concept_list)
        concept_stats = concept_df.groupby('concept').agg(
            stock_count=('stock_code', 'nunique'),  # 去重股票数
            avg_pct_change=('pct_change', 'mean'),  # 平均涨幅
            total_occur=('stock_code', 'count')     # 总出现次数
        ).reset_index()
        
        # 筛选热点概念（3只股票+正涨幅）
        concept_stats = concept_stats[
            (concept_stats['stock_count'] >= 3) & 
            (concept_stats['avg_pct_change'] > 0)
        ]
        if concept_stats.empty:
            print(f"[HotspotAnalyzer] 警告：{date}无符合条件的热点概念")
            return []
        
        # 计算热点分数并排序
        concept_stats['count_rank'] = concept_stats['stock_count'].rank(ascending=False)
        concept_stats['return_rank'] = concept_stats['avg_pct_change'].rank(ascending=False)
        concept_stats['hot_score'] = concept_stats['count_rank'] + concept_stats['return_rank']
        
        # 输出候选概念Top10（便于调试）
        print(f"\n[HotspotAnalyzer] {date}候选概念Top10：")
        for _, row in concept_stats.sort_values('hot_score').head(10).iterrows():
            print(f"  {row['concept']}：{row['stock_count']}只股票，平均涨幅{row['avg_pct_change']:.2%}")
        
        return concept_stats.sort_values('hot_score').head(top_n)['concept'].tolist()

    def get_daily_hotspots(self, date: str, top_n: int = 5) -> Dict[str, List[str]]:
        """统一获取指定日期的热点行业和概念"""
        return {
            'hot_industries': self.get_hot_industries(date, top_n),
            'hot_concepts': self.get_hot_concepts(date, top_n)
        }

    def get_processed_daily_df(self) -> pd.DataFrame:
        """返回处理后的日度数据（含concepts和pct_change）"""
        return self.daily_df.copy()

# ---------------------- 完整测试代码（直接运行） ----------------------
if __name__ == "__main__":
    # 1. 初始化数据加载器并加载2025年1月数据
    data_loader = DataLoader()
    daily_df_raw = data_loader.load_daily_data(start_date="2025-01-01", end_date="2025-01-31")
    
    # 2. 初始化热点分析器
    hotspot_analyzer = HotspotAnalyzer(daily_df_raw)
    
    # 3. 测试2025-01-02的热点识别
    test_date = "2025-01-02"
    hotspots = hotspot_analyzer.get_daily_hotspots(test_date, top_n=5)
    
    # 4. 输出热点结果
    print(f"\n=== {test_date} 热点识别最终结果 ===")
    print(f"热点行业：{hotspots['hot_industries']}")
    print(f"热点概念：{hotspots['hot_concepts'] if hotspots['hot_concepts'] else '无'}")
    
    # 5. 验证零售业股票的概念和分钟K加载
    daily_df_processed = hotspot_analyzer.get_processed_daily_df()
    retail_stocks = daily_df_processed[
        (daily_df_processed['date'] == _str_to_date(test_date)) & 
        (daily_df_processed['zjw_industry_name'] == '零售业') &
        (daily_df_processed.get('paused', 0) == 0) &
        (daily_df_processed['concepts'].apply(len) > 0)
    ]['stock_code'].unique()[:3]  # 取前3只零售股测试
    
    if retail_stocks.size > 0:
        print(f"\n=== 零售业股票示例（前3只）===")
        for code in retail_stocks:
            stock_info = daily_df_processed[daily_df_processed['stock_code'] == code].iloc[0]
            print(f"股票{code}：概念={stock_info['concepts']}，涨幅={stock_info['pct_change']:.2%}")
        
        # 测试加载这3只股票的T+1日（2025-01-03）分钟K
        minutely_data = data_loader.load_minutely_data(
            stock_codes=retail_stocks.tolist(),
            date="2025-01-03"
        )
        if minutely_data:
            sample_code = next(iter(minutely_data.keys()))
            print(f"\n=== {sample_code}在2025-01-03的分钟K样例 ===")
            print(minutely_data[sample_code][['date', 'time', 'open', 'close', 'volume']].head(3))
    
    # 6. 清空缓存
    data_loader.clear_cache()

from typing import List, Tuple, Dict
from datetime import datetime, timedelta

class AuctionSelector:
    def __init__(self, data_loader: DataLoader, hotspot_analyzer: HotspotAnalyzer):
        """
        初始化竞价筛选器（基于宽表竞价字段）
        :param data_loader: 已初始化的DataLoader（加载宽表数据）
        :param hotspot_analyzer: 已初始化的HotspotAnalyzer（复用热点数据）
        """
        self.data_loader = data_loader
        self.hotspot_analyzer = hotspot_analyzer
        self.processed_daily_df = hotspot_analyzer.get_processed_daily_df()  # 含T日竞价/涨跌幅/概念
        # 确保宽表包含竞价字段
        required_auc_fields = ['auc_volume', 'auc_money']
        missing_fields = [f for f in required_auc_fields if f not in self.processed_daily_df.columns]
        if missing_fields:
            raise ValueError(f"宽表缺少必要竞价字段：{missing_fields}，请检查widetable.parquet")
        print(f"\n[AuctionSelector] 初始化完成，已确认宽表包含竞价字段：{required_auc_fields}")

    def _get_t_plus_1_date(self, t_date: str) -> str:
        """计算T+1日日期（自然日，后续可扩展交易日判断）"""
        t_date_obj = datetime.strptime(t_date, "%Y-%m-%d")
        t_plus_1_date_obj = t_date_obj + timedelta(days=1)
        return t_plus_1_date_obj.strftime("%Y-%m-%d")

    def _get_hot_candidate_stocks(self, t_date: str) -> List[str]:
        """筛选T日属于热点行业/概念的股票（候选池）"""
        hotspots = self.hotspot_analyzer.get_daily_hotspots(t_date)
        hot_industries = hotspots['hot_industries']
        hot_concepts = hotspots['hot_concepts']
        t_date_obj = _str_to_date(t_date)

        # 从宽表筛选热点相关股票
        df_t = self.processed_daily_df[self.processed_daily_df['date'] == t_date_obj]
        if df_t.empty:
            print(f"[AuctionSelector] 警告：{t_date}无日度数据，候选池为空")
            return []

        # 行业匹配 OR 概念匹配
        industry_mask = df_t['zjw_industry_name'].isin(hot_industries)
        concept_mask = df_t['concepts'].apply(lambda x: len(set(x) & set(hot_concepts)) > 0)
        candidate_stocks = df_t[industry_mask | concept_mask]['stock_code'].unique().tolist()
        
        print(f"[AuctionSelector] T日({t_date})热点候选股票：{len(candidate_stocks)}只")
        return candidate_stocks

    def _calc_auc_indicators(self, stock_code: str, t_date: str, t_plus_1_date: str) -> Dict:
        """
        基于宽表计算T+1日竞价指标
        核心指标：竞价涨幅、竞价量能放大倍数（对比T日）
        """
        t_date_obj = _str_to_date(t_date)
        t_plus_1_date_obj = _str_to_date(t_plus_1_date)

        # 1. 获取T日数据（用于计算量能对比和前收盘价）
        df_t = self.processed_daily_df[
            (self.processed_daily_df['stock_code'] == stock_code) &
            (self.processed_daily_df['date'] == t_date_obj)
        ]
        if df_t.empty:
            print(f"[AuctionSelector] 警告：{stock_code}在{t_date}无数据")
            return {}
        t_close = df_t['close'].iloc[0]  # T日前收盘价（用于计算竞价涨幅）
        t_auc_volume = df_t['auc_volume'].iloc[0]  # T日竞价成交量（用于计算量能倍数）
        if t_auc_volume == 0:
            print(f"[AuctionSelector] 警告：{stock_code}在{t_date}竞价成交量为0，无法计算量能倍数")
            return {}

        # 2. 获取T+1日竞价数据（宽表中）
        df_t1 = self.processed_daily_df[
            (self.processed_daily_df['stock_code'] == stock_code) &
            (self.processed_daily_df['date'] == t_plus_1_date_obj)
        ]
        if df_t1.empty:
            print(f"[AuctionSelector] 警告：{stock_code}在{t_plus_1_date}无竞价数据")
            return {}
        t1_auc_volume = df_t1['auc_volume'].iloc[0]  # T+1日竞价成交量
        t1_auc_money = df_t1['auc_money'].iloc[0]    # T+1日竞价成交额
        t1_open = df_t1['open'].iloc[0]              # T+1日开盘价（用开盘价近似竞价收盘价）

        # 3. 计算核心指标
        auction_pct = (t1_open / t_close) - 1  # 竞价涨幅（开盘价近似竞价收盘价）
        volume_multiple = t1_auc_volume / t_auc_volume  # 竞价量能倍数（对比T日）
        auction_avg_price = t1_auc_money / t1_auc_volume if t1_auc_volume > 0 else 0  # 竞价均价

        return {
            'auction_pct': auction_pct,
            'volume_multiple': volume_multiple,
            'auction_avg_price': auction_avg_price,
            't1_open': t1_open,
            't1_auc_volume': t1_auc_volume
        }

    def select_qualified_stocks(self, t_date: str, top_n: int = 5) -> List[Tuple[str, float, float]]:
        """
        核心筛选逻辑（基于宽表竞价字段）
        筛选条件：
        1. 竞价涨幅：1% ≤ 涨幅 ≤ 5%（用开盘价近似）
        2. 量能放大：竞价成交量 ≥ T日竞价成交量的2倍（资金介入）
        3. 竞价量不为0：避免无成交的无效数据
        返回：[(股票代码, 竞价涨幅, 量能倍数), ...]（按涨幅降序）
        """
        t_plus_1_date = self._get_t_plus_1_date(t_date)
        print(f"\n[AuctionSelector] 筛选T+1日({t_plus_1_date})竞价股票（基于宽表竞价字段）")

        # 步骤1：获取T日热点候选股票
        candidate_stocks = self._get_hot_candidate_stocks(t_date)
        if not candidate_stocks:
            return []

        # 步骤2：逐个验证竞价条件（最多处理200只，控制效率）
        qualified_stocks = []
        max_process = min(200, len(candidate_stocks))
        for i, stock_code in enumerate(candidate_stocks[:max_process]):
            if i % 50 == 0:
                print(f"[AuctionSelector] 处理第{i+1}/{max_process}只股票：{stock_code}")
            
            # 计算竞价指标
            indicators = self._calc_auc_indicators(stock_code, t_date, t_plus_1_date)
            if not indicators:
                continue
            
            # 验证筛选条件
            meet_pct = 0.01 <= indicators['auction_pct'] <= 0.05  # 涨幅1%-5%
            meet_volume = indicators['volume_multiple'] >= 2      # 量能≥2倍
            meet_auc_volume = indicators['t1_auc_volume'] > 0     # 竞价量不为0

            if meet_pct and meet_volume and meet_auc_volume:
                qualified_stocks.append((
                    stock_code,
                    round(indicators['auction_pct'], 4),
                    round(indicators['volume_multiple'], 2)
                ))
                print(f"[AuctionSelector] 筛选通过：{stock_code}，涨幅{indicators['auction_pct']:.2%}，量能{indicators['volume_multiple']:.1f}倍")

        # 步骤3：按涨幅降序，取前N只
        qualified_stocks.sort(key=lambda x: x[1], reverse=True)
        final_stocks = qualified_stocks[:top_n]

        # 输出结果
        print(f"\n[AuctionSelector] T+1日({t_plus_1_date})筛选结果：")
        if final_stocks:
            for i, (code, pct, vol_mult) in enumerate(final_stocks, 1):
                print(f"  {i}. {code}：竞价涨幅{pct:.2%}，量能倍数{vol_mult}倍")
        else:
            print(f"  无符合条件的股票（可放宽涨幅/量能条件）")

        return final_stocks


# ---------------------- 测试修正后的竞价筛选 ----------------------
if __name__ == "__main__":
    # 1. 初始化前序模块（数据加载+热点分析）
    data_loader = DataLoader()
    daily_df_raw = data_loader.load_daily_data(start_date="2025-01-01", end_date="2025-01-31")
    hotspot_analyzer = HotspotAnalyzer(daily_df_raw)
    
    # 2. 初始化竞价筛选器（基于宽表）
    try:
        auction_selector = AuctionSelector(data_loader, hotspot_analyzer)
    except ValueError as e:
        print(f"初始化失败：{e}")
        exit()
    
    # 3. 测试筛选：T日=2025-01-02，T+1日=2025-01-03
    t_date = "2025-01-02"
    qualified_stocks = auction_selector.select_qualified_stocks(t_date, top_n=5)
    
    # 4. 查看筛选股票的T+1日详细数据（验证）
    if qualified_stocks:
        sample_code = qualified_stocks[0][0]
        t_plus_1_date = auction_selector._get_t_plus_1_date(t_date)
        t_plus_1_date_obj = _str_to_date(t_plus_1_date)
        sample_data = daily_df_raw[
            (daily_df_raw['stock_code'] == sample_code) &
            (daily_df_raw['date'] == t_plus_1_date_obj)
        ][['date', 'stock_code', 'open', 'close', 'auc_volume', 'auc_money']]
        print(f"\n=== {sample_code}在{t_plus_1_date}的详细数据 ===")
        print(sample_data)
    
    # 5. 清空缓存
    data_loader.clear_cache()

from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

class MinuteTracker:
    def __init__(self, data_loader: DataLoader, qualified_stocks: List[Tuple[str, float, float]]):
        """
        初始化分钟K跟踪器
        :param data_loader: 已初始化的DataLoader（加载分钟K数据）
        :param qualified_stocks: 模块3筛选出的竞价合格股票列表
        """
        self.data_loader = data_loader
        self.qualified_stocks = qualified_stocks  # [(stock_code, 竞价涨幅, 量能倍数), ...]
        self.track_results = {}  # 跟踪结果：{stock_code: 买入信号详情}
        print(f"\n[MinuteTracker] 初始化完成，待跟踪股票数量：{len(qualified_stocks)}只")

    def _calc_minute_indicators(self, stock_code: str, date: str) -> Dict:
        """
        计算开盘后分钟K量能指标
        核心指标：开盘后30分钟量能、价格强度、量比
        """
        # 1. 加载该股票当日分钟K数据（9:30-10:00为重点跟踪时段）
        minutely_data = self.data_loader.load_minutely_data(
            stock_codes=[stock_code],
            date=date
        )
        if stock_code not in minutely_data or minutely_data[stock_code].empty:
            print(f"[MinuteTracker] 警告：{stock_code}在{date}无分钟K数据")
            return {}
        df_minute = minutely_data[stock_code].copy()

        # 2. 转换时间格式，筛选开盘后30分钟（9:30-10:00）
        df_minute['datetime_str'] = df_minute['date'].astype(str) + ' ' + df_minute['time'].astype(str)
        df_minute['datetime'] = pd.to_datetime(df_minute['datetime_str'])
        # 开盘后时段：9:30 ≤ 时间 < 10:00
        track_mask = (df_minute['datetime'].dt.hour == 9) & (df_minute['datetime'].dt.minute.between(30, 59)) | \
                     (df_minute['datetime'].dt.hour == 10) & (df_minute['datetime'].dt.minute == 0)
        df_track = df_minute[track_mask].sort_values('datetime')
        if len(df_track) < 5:  # 至少5条数据才可靠
            print(f"[MinuteTracker] 警告：{stock_code}开盘后数据不足（{len(df_track)}条）")
            return {}

        # 3. 计算量能指标
        # 3.1 开盘后30分钟总成交量
        track_volume = df_track['volume'].sum()
        # 3.2 前5日平均每分钟成交量（用于计算量比）
        # 先获取前5个交易日日期（简化版：取日期列表前5个，实际需排除非交易日）
        all_dates = sorted(self.data_loader.daily_df[self.data_loader.daily_df['stock_code'] == stock_code]['date'].unique())
        target_date_idx = all_dates.index(_str_to_date(date))
        if target_date_idx < 5:
            print(f"[MinuteTracker] 警告：{stock_code}历史数据不足5天，无法计算量比")
            return {}
        prev_5_dates = [d.strftime("%Y-%m-%d") for d in all_dates[target_date_idx-5:target_date_idx]]
        
        # 加载前5日分钟K，计算平均每分钟成交量
        prev_5_volume = []
        for d in prev_5_dates:
            prev_data = self.data_loader.load_minutely_data([stock_code], d)
            if stock_code in prev_data and not prev_data[stock_code].empty:
                prev_5_volume.append(prev_data[stock_code]['volume'].mean())
        if len(prev_5_volume) < 3:  # 至少3天有效数据
            print(f"[MinuteTracker] 警告：{stock_code}历史量能数据不足")
            return {}
        avg_prev_volume = np.mean(prev_5_volume)

        # 3.3 开盘后量比 = 开盘后平均每分钟成交量 / 前5日平均每分钟成交量
        track_minutes = len(df_track)  # 实际跟踪分钟数（约30）
        track_avg_volume = track_volume / track_minutes
        volume_ratio = track_avg_volume / avg_prev_volume  # 量比

        # 4. 计算价格强度（开盘后30分钟收盘价 ≥ 竞价收盘价）
        auction_close = df_track.iloc[0]['open']  # 用9:30开盘价近似竞价收盘价
        track_close = df_track.iloc[-1]['close']  # 10:00收盘价
        price_strength = track_close >= auction_close * 1.01  # 价格强度：≥1%涨幅

        return {
            'track_volume': track_volume,
            'volume_ratio': volume_ratio,
            'price_strength': price_strength,
            'track_close': track_close,
            'auction_close': auction_close
        }

    def generate_buy_signals(self, t_plus_1_date: str) -> Dict[str, Dict]:
        """
        生成买入信号：基于开盘后30分钟量能和价格强度
        买入条件：
        1. 量比 ≥ 2.5（开盘后量能持续放大）
        2. 价格强度达标（10:00收盘价 ≥ 竞价收盘价*1.01）
        3. 未大幅冲高回落（最高价 ≤ 收盘价*1.03）
        """
        print(f"\n[MinuteTracker] 开始跟踪T+1日({t_plus_1_date})开盘后量能...")

        for stock_code, auc_pct, vol_mult in self.qualified_stocks:
            print(f"\n[MinuteTracker] 跟踪股票：{stock_code}（竞价涨幅{auc_pct:.2%}）")
            
            # 计算分钟K指标
            indicators = self._calc_minute_indicators(stock_code, t_plus_1_date)
            if not indicators:
                continue

            # 验证买入条件
            meet_volume_ratio = indicators['volume_ratio'] >= 2.5  # 量比≥2.5
            meet_price_strength = indicators['price_strength']     # 价格强度达标
            # 检查是否冲高回落（最高价 ≤ 收盘价*1.03）
            df_minute = self.data_loader.load_minutely_data([stock_code], t_plus_1_date)[stock_code]
            track_high = df_minute[df_minute['time'].astype(str).str.contains('09:30|10:00')]['high'].max()
            no_drop_risk = track_high <= indicators['track_close'] * 1.03  # 允许3%以内波动

            # 生成买入信号
            if meet_volume_ratio and meet_price_strength and no_drop_risk:
                buy_signal = {
                    '竞价涨幅': f"{auc_pct:.2%}",
                    '竞价量能倍数': vol_mult,
                    '开盘后量比': f"{indicators['volume_ratio']:.2f}",
                    '价格强度': f"{(indicators['track_close']/indicators['auction_close']-1):.2%}",
                    '买入信号': True
                }
                self.track_results[stock_code] = buy_signal
                print(f"[MinuteTracker] 买入信号触发：{stock_code}，量比{indicators['volume_ratio']:.2f}，价格强度{buy_signal['价格强度']}")
            else:
                self.track_results[stock_code] = {
                    '买入信号': False,
                    '原因': f"量比不达标({indicators['volume_ratio']:.2f})" if not meet_volume_ratio else
                           f"价格强度不足" if not meet_price_strength else
                           f"存在冲高回落风险"
                }

        # 输出最终买入信号列表
        print(f"\n[MinuteTracker] T+1日({t_plus_1_date})买入信号汇总：")
        buy_stocks = {k: v for k, v in self.track_results.items() if v['买入信号']}
        if buy_stocks:
            for i, (code, info) in enumerate(buy_stocks.items(), 1):
                print(f"  {i}. {code}：{info}")
        else:
            print(f"  无符合条件的买入信号")

        return buy_stocks

# ---------------------- 完整流程测试（模块1→2→3→4） ----------------------
if __name__ == "__main__":
    # 1. 数据加载
    data_loader = DataLoader()
    daily_df_raw = data_loader.load_daily_data(start_date="2025-01-01", end_date="2025-01-31")
    
    # 2. 热点识别（T日=2025-01-02）
    hotspot_analyzer = HotspotAnalyzer(daily_df_raw)
    t_date = "2025-01-22"
    
    # 3. 竞价筛选（T+1日=2025-01-03）
    auction_selector = AuctionSelector(data_loader, hotspot_analyzer)
    qualified_stocks = auction_selector.select_qualified_stocks(t_date, top_n=5)
    
    # 4. 分钟K量能跟踪与买入信号（T+1日=2025-01-03）
    if qualified_stocks:
        minute_tracker = MinuteTracker(data_loader, qualified_stocks)
        t_plus_1_date = auction_selector._get_t_plus_1_date(t_date)
        buy_signals = minute_tracker.generate_buy_signals(t_plus_1_date)
    else:
        print("\n无竞价合格股票，跳过分钟跟踪")
    
    # 5. 清空缓存
    data_loader.clear_cache()

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class StrategyOptimizer:
    def __init__(self, data_loader: DataLoader):
        """
        初始化策略优化器
        :param data_loader: 已初始化的DataLoader（加载全量数据）
        """
        self.data_loader = data_loader
        self.hotspot_analyzer = None  # 延迟初始化，适配多日数据
        self.all_backtest_results = []  # 多日回测结果汇总
        print(f"[StrategyOptimizer] 初始化完成，数据覆盖日期：{data_loader.daily_df['date'].min()}至{data_loader.daily_df['date'].max()}")

    def _init_hotspot_analyzer(self, daily_df: pd.DataFrame):
        """延迟初始化热点分析器，避免重复加载数据"""
        if self.hotspot_analyzer is None:
            self.hotspot_analyzer = HotspotAnalyzer(daily_df)

    def single_day_backtest(self, 
                           t_date: str,
                           # 可调整参数：通过这些参数优化策略
                           auc_pct_min: float = 0.01,    # 竞价涨幅下限（原1%）
                           auc_pct_max: float = 0.04,    # 竞价涨幅上限（原5%，降低追高风险）
                           auc_vol_mult_min: float = 2.0, # 竞价量能倍数下限（原2倍）
                           minute_buy_minute: int = 32,  # 买入分钟（原9:40，改为9:32，更早介入）
                           minute_volume_ratio_min: float = 2.0, # 开盘后量比下限（原2.5倍）
                           minute_price_strength_min: float = 0.008 # 价格强度下限（原1%，改为0.8%）
                           ) -> Dict:
        """
        单日回测（支持参数调整）
        :param t_date: T日日期（如"2025-01-22"）
        :param 其他参数：可调整的策略阈值
        :return: 单日回测统计结果
        """
        t_plus_1_date = (datetime.strptime(t_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"\n[StrategyOptimizer] 开始单日回测：T日={t_date}，T+1日={t_plus_1_date}")

        # 1. 初始化热点分析器（复用全量数据）
        self._init_hotspot_analyzer(self.data_loader.daily_df)

        # 2. 调整版竞价筛选（基于新参数）
        auction_selector = AuctionSelector(self.data_loader, self.hotspot_analyzer)
        # 重写筛选条件：用传入的参数替代固定阈值
        qualified_stocks = []
        candidate_stocks = auction_selector._get_hot_candidate_stocks(t_date)
        if not candidate_stocks:
            return {'交易日期': t_plus_1_date, '状态': '无候选股票'}

        # 逐个验证竞价条件（用新参数）
        for stock_code in candidate_stocks[:200]:  # 限制处理数量
            indicators = auction_selector._calc_auc_indicators(stock_code, t_date, t_plus_1_date)
            if not indicators:
                continue
            # 新筛选条件：更严格的涨幅范围（1%-4%），避免过高追高
            meet_pct = auc_pct_min <= indicators['auction_pct'] <= auc_pct_max
            meet_volume = indicators['volume_multiple'] >= auc_vol_mult_min
            meet_auc_volume = indicators['t1_auc_volume'] > 0
            if meet_pct and meet_volume and meet_auc_volume:
                qualified_stocks.append((stock_code, indicators['auction_pct'], indicators['volume_multiple']))
        
        if not qualified_stocks:
            return {'交易日期': t_plus_1_date, '状态': '无合格竞价股票'}
        print(f"[StrategyOptimizer] 调整后合格竞价股票：{len(qualified_stocks)}只")

        # 3. 调整版分钟跟踪（更早买入+更低量比阈值）
        minute_tracker = MinuteTracker(self.data_loader, qualified_stocks)
        buy_signals = {}
        for stock_code, auc_pct, vol_mult in qualified_stocks:
            indicators = minute_tracker._calc_minute_indicators(stock_code, t_plus_1_date)
            if not indicators:
                continue
            # 新跟踪条件：量比≥2.0，价格强度≥0.8%
            meet_volume_ratio = indicators['volume_ratio'] >= minute_volume_ratio_min
            meet_price_strength = indicators['price_strength']  # 内部已用新参数判断
            # 检查是否冲高回落（放宽至5%波动）
            df_minute = self.data_loader.load_minutely_data([stock_code], t_plus_1_date)[stock_code]
            track_high = df_minute[df_minute['time'].astype(str).str.contains('09:30|09:35')]['high'].max()
            no_drop_risk = track_high <= indicators['track_close'] * 1.05

            if meet_volume_ratio and meet_price_strength and no_drop_risk:
                buy_signals[stock_code] = {
                    '竞价涨幅': f"{auc_pct:.2%}",
                    '竞价量能倍数': vol_mult,
                    '开盘后量比': f"{indicators['volume_ratio']:.2f}",
                    '价格强度': f"{(indicators['track_close']/indicators['auction_close']-1):.2%}",
                    '买入信号': True
                }

        if not buy_signals:
            return {'交易日期': t_plus_1_date, '状态': '无买入信号'}
        print(f"[StrategyOptimizer] 调整后买入信号：{len(buy_signals)}只股票")

        # 4. 调整版收益计算（更早买入价：9:32）
        profit_calculator = ProfitCalculator(self.data_loader, buy_signals)
        profit_results = []
        for stock_code in buy_signals.keys():
            # 重写买入价：取9:32收盘价（更早介入，避免冲高回落）
            minutely_data = self.data_loader.load_minutely_data([stock_code], t_plus_1_date)
            if stock_code not in minutely_data or minutely_data[stock_code].empty:
                continue
            df_minute = minutely_data[stock_code].copy()
            df_minute['datetime_str'] = df_minute['date'].astype(str) + ' ' + df_minute['time'].astype(str)
            df_minute['datetime'] = pd.to_datetime(df_minute['datetime_str'])
            # 买入时间改为9:32（原9:40）
            buy_mask = (df_minute['datetime'].dt.hour == 9) & (df_minute['datetime'].dt.minute == minute_buy_minute)
            if not df_minute[buy_mask].empty:
                buy_price = df_minute[buy_mask].iloc[0]['close']
            else:
                # 降级为9:31收盘价（最早可能的买入点）
                buy_price = df_minute[df_minute['datetime'].dt.minute == 31].iloc[0]['close']
            
            # 计算收益
            df_daily = self.data_loader.daily_df[
                (self.data_loader.daily_df['stock_code'] == stock_code) &
                (self.data_loader.daily_df['date'] == _str_to_date(t_plus_1_date))
            ]
            if df_daily.empty:
                continue
            close_price = df_daily['close'].iloc[0]
            relative_profit = (close_price / buy_price - 1) * 100
            profit_results.append({
                '股票代码': stock_code,
                '买入价': round(buy_price, 2),
                '收盘价': round(close_price, 2),
                '相对收益(%)': round(relative_profit, 2),
                '是否盈利': 1 if relative_profit > 0 else 0
            })

        # 5. 单日统计
        if not profit_results:
            return {'交易日期': t_plus_1_date, '状态': '无收益数据'}
        df_profit = pd.DataFrame(profit_results)
        total_count = len(df_profit)
        win_count = df_profit['是否盈利'].sum()
        single_stats = {
            '交易日期': t_plus_1_date,
            '状态': '完成',
            '总交易次数': total_count,
            '盈利次数': win_count,
            '亏损次数': total_count - win_count,
            '胜率(%)': round((win_count / total_count) * 100, 2) if total_count > 0 else 0,
            '平均收益(%)': round(df_profit['相对收益(%)'].mean(), 2) if total_count > 0 else 0,
            '最大收益(%)': round(df_profit['相对收益(%)'].max(), 2) if total_count > 0 else 0,
            '最大亏损(%)': round(df_profit['相对收益(%)'].min(), 2) if total_count > 0 else 0,
            '买入股票列表': df_profit['股票代码'].tolist()
        }

        # 输出单日结果
        print(f"[StrategyOptimizer] 单日回测结果：胜率{single_stats['胜率(%)']}%，平均收益{single_stats['平均收益(%)']}%")
        self.all_backtest_results.append(single_stats)
        return single_stats

    def multi_day_backtest(self, start_t_date: str, end_t_date: str, **kwargs) -> Tuple[List[Dict], Dict]:
        """
        多日批量回测（验证策略稳定性）
        :param start_t_date: 起始T日（如"2025-01-10"）
        :param end_t_date: 结束T日（如"2025-01-20"）
        :param kwargs: 传入的策略参数（如auc_pct_min=0.01）
        :return: 每日结果列表 + 整体统计
        """
        print(f"\n[StrategyOptimizer] 开始多日回测：T日范围={start_t_date}至{end_t_date}")

        # 生成T日列表（排除非交易日，简化版）
        all_dates = sorted(self.data_loader.daily_df['date'].unique())
        start_date_obj = _str_to_date(start_t_date)
        end_date_obj = _str_to_date(end_t_date)
        t_date_list = [d.strftime("%Y-%m-%d") for d in all_dates 
                      if start_date_obj <= d <= end_date_obj]

        if not t_date_list:
            return [], {'提示': '无符合条件的T日'}

        # 逐个T日回测
        for t_date in t_date_list:
            self.single_day_backtest(t_date, **kwargs)

        # 计算整体统计
        if not self.all_backtest_results:
            return self.all_backtest_results, {'提示': '无有效回测数据'}
        
        df_all = pd.DataFrame([r for r in self.all_backtest_results if r['状态'] == '完成'])
        if df_all.empty:
            return self.all_backtest_results, {'提示': '无完成的回测数据'}
        
        total_trade_days = len(df_all)
        total_trades = df_all['总交易次数'].sum()
        total_wins = df_all['盈利次数'].sum()
        
        overall_stats = {
            '回测T日范围': f"{start_t_date}至{end_t_date}",
            '有效交易天数': total_trade_days,
            '总交易次数': total_trades,
            '总盈利次数': total_wins,
            '整体胜率(%)': round((total_wins / total_trades) * 100, 2) if total_trades > 0 else 0,
            '日均收益(%)': round(df_all['平均收益(%)'].mean(), 2),
            '收益标准差(%)': round(df_all['平均收益(%)'].std(), 2),
            '最大单日收益(%)': round(df_all['平均收益(%)'].max(), 2),
            '最大单日亏损(%)': round(df_all['平均收益(%)'].min(), 2)
        }

        # 输出整体结果
        print(f"\n[StrategyOptimizer] 多日回测整体统计：")
        for key, value in overall_stats.items():
            print(f"  {key}：{value}")

        return self.all_backtest_results, overall_stats


# ---------------------- 测试策略优化与多日回测 ----------------------
if __name__ == "__main__":
    # 1. 加载全量数据（确保覆盖多日历史）
    data_loader = DataLoader()
    daily_df_raw = data_loader.load_daily_data(start_date="2025-01-10", end_date="2025-01-20")  # 10天T日范围
    
    # 2. 初始化优化器并执行多日回测（传入优化参数）
    optimizer = StrategyOptimizer(data_loader)
    # 优化参数：降低涨幅上限（1%-4%）、更早买入（9:32）、降低量比阈值（2.0）
    backtest_results, overall_stats = optimizer.multi_day_backtest(
        start_t_date="2025-01-10",
        end_t_date="2025-01-20",
        auc_pct_min=0.01,    # 竞价涨幅下限1%
        auc_pct_max=0.04,    # 竞价涨幅上限4%（原5%）
        auc_vol_mult_min=2.0, # 竞价量能倍数≥2倍
        minute_buy_minute=32, # 买入时间9:32（原9:40）
        minute_volume_ratio_min=2.0, # 开盘后量比≥2.0（原2.5）
        minute_price_strength_min=0.008 # 价格强度≥0.8%（原1%）
    )
    
    # 3. 保存回测结果（便于后续分析）
    if backtest_results:
        df_backtest = pd.DataFrame(backtest_results)
        print(f"\n=== 多日回测详情表 ===")
        print(df_backtest[['交易日期', '总交易次数', '胜率(%)', '平均收益(%)']])
    
    # 4. 清空缓存
    data_loader.clear_cache()

