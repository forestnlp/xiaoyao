#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime

# 读取数据
price_vol_df = pd.read_parquet(r"D:\workspace\xiaoyao\data\stock_daily_price.parquet")  # 量价数据
auction_df = pd.read_parquet(r"D:\workspace\xiaoyao\data\stock_daily_auction.parquet")    # 竞价数据
industry_df = pd.read_parquet(r"D:\workspace\xiaoyao\data\stock_daily_industry.parquet")  # 行业数据


# In[2]:


# 统一日期格式（假设原日期是字符串或带时区的格式）
price_vol_df["date"] = pd.to_datetime(price_vol_df["date"]).dt.date
auction_df["date"] = pd.to_datetime(auction_df["date"]).dt.date
industry_df["date"] = pd.to_datetime(industry_df["date"]).dt.date

# 日期区间过滤
start_date = datetime(2024, 10, 1).date()
end_date = datetime(2025, 9, 25).date()

# 去除停牌的股票 ，仅保留price_vol_df 里 paused==0 的
price_vol_df = price_vol_df[(price_vol_df["date"] >= start_date) & (price_vol_df["date"] <= end_date) & (price_vol_df["paused"] == 0)]
auction_df = auction_df[(auction_df["date"] >= start_date) & (auction_df["date"] <= end_date)]
industry_df = industry_df[(industry_df["date"] >= start_date) & (industry_df["date"] <= end_date)]


# In[3]:


# 竞价数据字段重命名（示例：前缀au_）
auction_df = auction_df.rename(columns={
    "volume": "au_volume", 
    "money": "au_money",
})


# In[4]:


# 1. 关联量价（带指标）和竞价（带指标）：按date和stock_code
merged_df = pd.merge(
    price_vol_df, 
    auction_df, 
    on=["date", "stock_code"], 
    how="inner"  # 取两者都有的日期和股票
)

# 2. 关联行业数据：按date和stock_code（行业数据可能是每日或历史快照，需确保关联逻辑）
final_df = pd.merge(
    merged_df, 
    industry_df[["date", "stock_code", "jq_l1_industry_name", "jq_l2_industry_name", "sw_l1_industry_name", "sw_l2_industry_name", "sw_l3_industry_name"]], 
    on=["date", "stock_code"], 
    how="left"
)


# In[5]:


len(final_df['sw_l3_industry_name'].unique())


# In[6]:


import pandas as pd
import numpy as np

# 假设final_df是之前合并好的完整数据
# 如果需要重新加载数据，请取消下面的注释并修改路径
# final_df = pd.read_parquet(r"D:\workspace\xiaoyao\data\merged_stock_data.parquet")

# ---------------------- 1. 计算个股级辅助指标 ----------------------
# 确保数据按股票代码和日期排序
final_df = final_df.sort_values(['stock_code', 'date'])

# 计算涨跌幅（如果尚未计算）
final_df['daily_return'] = (final_df['close'] / final_df['pre_close']) - 1

# 标记大涨（涨超5%）和大跌（跌超5%）
final_df['is_up_5pct'] = (final_df['daily_return'] >= 0.05).astype(int)
final_df['is_down_5pct'] = (final_df['daily_return'] <= -0.05).astype(int)

# 计算竞价量与昨日成交量的比值（需先获取昨日成交量）
final_df['prev_volume'] = final_df.groupby('stock_code')['volume'].shift(1)  # 昨日成交量
final_df['au_vol_ratio_prev'] = final_df['au_volume'] / final_df['prev_volume']  # 竞价量/昨日成交量


# ---------------------- 2. 按日期和行业分组计算指标 ----------------------
# 选择申万二级行业作为分析对象（可根据需要改为sw_l1或sw_l3）
industry_level = 'sw_l3_industry_name'

# 分组聚合计算行业指标
industry_indicators = final_df.groupby(['date', industry_level]).agg(
    # 基础规模指标
    stock_count=('stock_code', 'nunique'),  # 行业内股票总数
    
    # 涨跌家数指标
    up_count=('daily_return', lambda x: (x > 0).sum()),  # 上涨家数
    down_count=('daily_return', lambda x: (x < 0).sum()),  # 下跌家数
    flat_count=('daily_return', lambda x: (x == 0).sum()),  # 平盘家数
    up_ratio=('daily_return', lambda x: (x > 0).mean()),  # 上涨比例（上涨家数/总家数）
    
    # 极端行情指标
    limit_up_count=('high_limit', lambda x: (final_df.loc[x.index, 'close'] >= x).sum()),  # 涨停家数
    limit_down_count=('low_limit', lambda x: (final_df.loc[x.index, 'close'] <= x).sum()),  # 跌停家数
    up_5pct_count=('is_up_5pct', 'sum'),  # 涨超5%家数
    down_5pct_count=('is_down_5pct', 'sum'),  # 跌超5%家数
    
    # 竞价指标
    total_au_volume=('au_volume', 'sum'),  # 行业竞价总量
    total_au_money=('au_money', 'sum'),  # 行业竞价总金额
    avg_au_vol_ratio_prev=('au_vol_ratio_prev', 'mean'),  # 平均竞价量/昨日成交量比值
    
    # 量价指标
    total_volume=('volume', 'sum'),  # 行业总成交量
    total_money=('money', 'sum'),  # 行业总成交额
    avg_daily_return=('daily_return', 'mean'),  # 平均涨跌幅
    med_daily_return=('daily_return', 'median'),  # 中位数涨跌幅
    
    # 盘口指标
    avg_bid_ask_ratio=('b1_v', lambda x: (final_df.loc[x.index, 'b1_v'].sum() / 
                                          final_df.loc[x.index, 'a1_v'].sum()) 
                                          if final_df.loc[x.index, 'a1_v'].sum() > 0 else 0)  # 平均买一量/卖一量比
).reset_index()


# ---------------------- 3. 衍生更多有价值指标 ----------------------
# 计算上涨家数-下跌家数差值（反映行业整体情绪）
industry_indicators['net_up_count'] = industry_indicators['up_count'] - industry_indicators['down_count']

# 计算涨跌比（上涨家数/下跌家数，避免除零）
industry_indicators['up_down_ratio'] = np.where(
    industry_indicators['down_count'] > 0,
    industry_indicators['up_count'] / industry_indicators['down_count'],
    np.inf  # 下跌家数为0时设为无穷大
)

# 计算涨停率和跌停率
industry_indicators['limit_up_ratio'] = industry_indicators['limit_up_count'] / industry_indicators['stock_count']
industry_indicators['limit_down_ratio'] = industry_indicators['limit_down_count'] / industry_indicators['stock_count']

# 计算行业成交量占全市场比例
market_daily = final_df.groupby('date').agg(
    market_total_volume=('volume', 'sum'),
    market_total_money=('money', 'sum')
).reset_index()

industry_indicators = pd.merge(
    industry_indicators, 
    market_daily, 
    on='date', 
    how='left'
)

industry_indicators['volume_ratio'] = industry_indicators['total_volume'] / industry_indicators['market_total_volume']  # 成交量占比
industry_indicators['money_ratio'] = industry_indicators['total_money'] / industry_indicators['market_total_money']  # 成交额占比

# ---------------------- 4. 数据清洗与整理 ----------------------
# 重命名行业名称列，使其更简洁
industry_indicators = industry_indicators.rename(columns={industry_level: 'industry_name'})

# 按日期和行业排序
industry_indicators = industry_indicators.sort_values(['date', 'industry_name'])

# 重置索引
industry_indicators = industry_indicators.reset_index(drop=True)

# 查看结果前5行
print("行业指标计算结果预览：")
print(industry_indicators[['date', 'industry_name', 'stock_count', 'up_count', 
                          'limit_up_count', 'total_au_volume', 'avg_au_vol_ratio_prev']].head())


# ---------------------- 5. 保存为parquet文件 ----------------------
output_path = r"D:\workspace\xiaoyao\data\industry_indicators.parquet"
industry_indicators.to_parquet(output_path, index=False)
print(f"\n行业指标已保存至：{output_path}")


# In[7]:


# 读取parquet为df
df = pd.read_parquet(r"D:\workspace\xiaoyao\data\industry_indicators.parquet")

df.tail(5)


# In[8]:


# 查看df的列名，并翻译为中文
df.columns


# In[9]:


import pandas as pd
import numpy as np

def get_top_k_industries(
    industry_indicators: pd.DataFrame,
    start_date: str,
    end_date: str,
    k: int = 10,
    weights: dict = None
) -> pd.DataFrame:
    """
    获取指定时间段内最热门的前K个行业
    
    参数:
        industry_indicators: 包含行业指标的DataFrame（需包含'date'和'industry_name'列）
        start_date: 开始日期，格式如'2024-01-01'
        end_date: 结束日期，格式如'2024-12-31'
        k: 要返回的热门行业数量，默认10
        weights: 各指标权重字典，如{'limit_up_ratio':0.3, ...}，默认使用预设权重
    
    返回:
        包含前K个热门行业及其综合得分的DataFrame
    """
    # ---------------------- 1. 参数处理与数据过滤 ----------------------
    # 确保日期格式正确
    industry_indicators['date'] = pd.to_datetime(industry_indicators['date']).dt.date
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    
    # 过滤指定时间段的数据
    mask = (industry_indicators['date'] >= start_date) & (industry_indicators['date'] <= end_date)
    period_data = industry_indicators[mask].copy()
    
    if period_data.empty:
        raise ValueError(f"指定时间段内没有数据：{start_date} 至 {end_date}")
    
    # 检查必要字段是否存在
    required_cols = ['avg_daily_return', 'limit_up_ratio', 'volume_ratio', 'money_ratio']
    missing_cols = [col for col in required_cols if col not in period_data.columns]
    if missing_cols:
        raise ValueError(f"缺少必要指标列：{missing_cols}")
    
    # 设置默认权重（可根据业务调整）
    if weights is None:
        weights = {
            'avg_daily_return': 0.25,    # 平均涨跌幅
            'limit_up_ratio': 0.3,       # 涨停率（反映资金热度）
            'volume_ratio': 0.2,         # 成交量占比
            'money_ratio': 0.15,         # 成交额占比
            'up_down_ratio': 0.1         # 涨跌比（反映整体赚钱效应）
        }
    
    # 验证权重总和为1
    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"权重总和必须为1，当前总和：{weight_sum:.4f}")
    
    
    # ---------------------- 2. 计算每日热度得分 ----------------------
    # 标准化指标（消除量纲影响，将不同指标映射到0-1范围）
    for col in weights.keys():
        min_val = period_data[col].min()
        max_val = period_data[col].max()
        if max_val > min_val:  # 避免除零
            period_data[f'{col}_norm'] = (period_data[col] - min_val) / (max_val - min_val)
        else:
            period_data[f'{col}_norm'] = 0.5  # 无波动时赋中间值
    
    # 计算每日综合热度得分
    period_data['daily_heat_score'] = 0.0
    for col, w in weights.items():
        period_data['daily_heat_score'] += period_data[f'{col}_norm'] * w
    
    
    # ---------------------- 3. 计算时间段内的行业总热度 ----------------------
    # 按行业分组，计算时间段内的平均热度得分（也可使用总和，根据需求选择）
    industry_period_score = period_data.groupby('industry_name').agg(
        total_days=('date', 'nunique'),  # 时间段内包含的交易日数量
        avg_heat_score=('daily_heat_score', 'mean'),  # 平均每日热度得分
        avg_limit_up_ratio=('limit_up_ratio', 'mean'),  # 平均涨停率
        avg_return=('avg_daily_return', 'mean')  # 平均涨跌幅
    ).reset_index()
    
    # 按平均热度得分降序排序
    industry_period_score = industry_period_score.sort_values(
        'avg_heat_score', ascending=False
    ).reset_index(drop=True)
    
    
    # ---------------------- 4. 返回前K个行业 ----------------------
    top_k = industry_period_score.head(k).copy()
    # 添加排名列
    top_k['rank'] = range(1, k+1)
    
    # 调整列顺序，便于查看
    return top_k[['rank', 'industry_name', 'avg_heat_score', 
                  'avg_limit_up_ratio', 'avg_return', 'total_days']]


# ---------------------- 函数使用示例 ----------------------
if __name__ == "__main__":
    # 1. 加载行业指标数据
    industry_indicators = pd.read_parquet(
        r"D:\workspace\xiaoyao\data\industry_indicators.parquet"
    )
    
    # 2. 调用函数获取热门行业
    try:
        top_10 = get_top_k_industries(
            industry_indicators=industry_indicators,
            start_date='2025-06-01',
            end_date='2025-06-28',
            k=10  # 返回前10
        )
        
        # 3. 打印结果
        print(f"期间最热门的前10个行业：")
        print(top_10.round(4))  # 保留4位小数
        
        # 4. 保存结果
        top_10.to_csv("top_10_industries.csv", index=False)
        print("\n结果已保存至：top_10_industries.csv")
        
    except Exception as e:
        print(f"出错：{e}")
    


# In[ ]:




