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
start_date = datetime(2025, 1, 1).date()
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


def calculate_auction_indicators(df):
    """计算竞价相关衍生指标"""
    # 竞价量比 = 竞价成交量 / 最近N日平均竞价成交量（示例N=5，不包含当日，取最近N-1日）
    df["auction_volume_60d_ratio"] = df["au_volume"] / df.groupby("stock_code")["au_volume"].rolling(60, closed="left").mean().reset_index(0, drop=True)
    df["auction_volume_5d_ratio"] = df["au_volume"] / df.groupby("stock_code")["au_volume"].rolling(5, closed="left").mean().reset_index(0, drop=True)
    df["auction_volume_1d_ratio"] = df["au_volume"] / df.groupby("stock_code")["au_volume"].rolling(1, closed="left").mean().reset_index(0, drop=True)


    # 买一量 / 卖一量（假设买一量是a1_v，卖一量是b1_v）
    df["bid1_ask1_ratio"] = df["a1_v"] / df["b1_v"]
    
    # 五档买量和 / 五档卖量和（假设买一至买五量是a1_v~a5_v，卖一至卖五量是b1_v~b5_v）
    df["total_bid_vol"] = df["a1_v"] + df["a2_v"] + df["a3_v"] + df["a4_v"] + df["a5_v"]
    df["total_ask_vol"] = df["b1_v"] + df["b2_v"] + df["b3_v"] + df["b4_v"] + df["b5_v"]
    df["bid_ask_ratio"] = df["total_bid_vol"] / df["total_ask_vol"]
    
    return df

# 计算竞价衍生指标
auction_with_indicators = calculate_auction_indicators(auction_df)

# 仅保留需要的字段 auction_volume_60d_ratio auction_volume_5d_ratio auction_volume_1d_ratio bid1_ask1_ratio bid_ask_ratio
auction_with_indicators = auction_with_indicators[
    [
        "date",
        "stock_code",
        "au_volume",
        "au_money",
        "auction_volume_60d_ratio",
        "auction_volume_5d_ratio",
        "auction_volume_1d_ratio",
        "bid1_ask1_ratio",
        "bid_ask_ratio"
    ]
]
auction_with_indicators


# In[5]:


import pandas as pd
import numpy as np

def calculate_price_vol_indicators(price_vol_df):
    """
    计算无未来函数的技术指标（仅用昨日及之前数据，当日可使用open）
    新增：基于当日open计算的5日和20日布林带指标（包含当日数据）
    输入：包含date, stock_code, open, high, low, close, volume的量价数据
    输出：添加16个指标后的DataFrame（原10个+新增6个）
    """
    # 复制数据并按股票和日期排序（确保时间顺序正确）
    df = price_vol_df.copy()
    df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
    grouped = df.groupby('stock_code')
    
    # ----------------------
    # 一、趋势类指标（原4个 + 新增6个基于open的布林带指标）
    # ----------------------
    # 1. 短期移动平均线（MA5）：基于前5日收盘价（不含当日）
    df['MA5'] = grouped['close'].shift(1).rolling(window=5, min_periods=5).mean().reset_index(0, drop=True)
    
    # 2. 中期移动平均线（MA20）：基于前20日收盘价（不含当日）
    df['MA20'] = grouped['close'].shift(1).rolling(window=20, min_periods=20).mean().reset_index(0, drop=True)
    
    # 3. 平均差率（BIAS5）：(当日开盘价 - 前5日均价) / 前5日均价
    df['BIAS5'] = (df['open'] - df['MA5']) / df['MA5'].replace(0, 0.001) * 100
    
    # 4. 布林带带宽（BOLL_WIDTH）：基于前20日收盘价计算
    df['MB'] = grouped['close'].shift(1).rolling(window=20, min_periods=20).mean().reset_index(0, drop=True)  # 中轨
    df['STD20'] = grouped['close'].shift(1).rolling(window=20, min_periods=20).std().reset_index(0, drop=True)  # 标准差
    df['UB'] = df['MB'] + 2 * df['STD20']  # 上轨
    df['LB'] = df['MB'] - 2 * df['STD20']  # 下轨
    df['BOLL_WIDTH'] = (df['UB'] - df['LB']) / df['MB'].replace(0, 0.001) * 100  # 带宽
    df.drop(columns=['MB', 'STD20', 'UB', 'LB'], inplace=True)
    
    # 新增：基于当日open的5日布林带指标（包含当日数据）
    df['MA5_open'] = grouped['open'].rolling(window=5, min_periods=5).mean().reset_index(0, drop=True)  # 中轨：包含当日的5日开盘价均值
    df['STD5_open'] = grouped['open'].rolling(window=5, min_periods=5).std().reset_index(0, drop=True)  # 标准差
    df['UB5_open'] = df['MA5_open'] + 2 * df['STD5_open']  # 上轨
    df['LB5_open'] = df['MA5_open'] - 2 * df['STD5_open']  # 下轨
    
    # 新增：基于当日open的20日布林带指标（包含当日数据）
    df['MA20_open'] = grouped['open'].rolling(window=20, min_periods=20).mean().reset_index(0, drop=True)  # 中轨：包含当日的20日开盘价均值
    df['STD20_open'] = grouped['open'].rolling(window=20, min_periods=20).std().reset_index(0, drop=True)  # 标准差
    df['UB20_open'] = df['MA20_open'] + 2 * df['STD20_open']  # 上轨
    df['LB20_open'] = df['MA20_open'] - 2 * df['STD20_open']  # 下轨
    
    # ----------------------
    # 二、交易量类指标（3个）
    # ----------------------
    # 5. 成交量比（VOL_RATIO）：当日开盘前预估成交量 / 前5日平均成交量
    df['EST_VOLUME'] = grouped['volume'].shift(1)  # 模拟当日预估成交量
    df['VOL_RATIO'] = df['EST_VOLUME'] / grouped['volume'].shift(1).rolling(window=5, min_periods=5).mean().reset_index(0, drop=True)
    df.drop(columns=['EST_VOLUME'], inplace=True)
    
    # 6. 资金流向指数（MFI）：基于前14日数据计算
    df['PREV_CLOSE'] = grouped['close'].shift(1)  # 前日收盘价
    df['TP_PREV'] = (grouped['high'].shift(1) + grouped['low'].shift(1) + grouped['close'].shift(1)) / 3  # 前日典型价格
    df['MF_PREV'] = df['TP_PREV'] * grouped['volume'].shift(1)  # 前日资金流量
    df['POS_MF'] = np.where(grouped['close'].shift(1) > grouped['close'].shift(2), df['MF_PREV'], 0)  # 正资金流量
    df['NEG_MF'] = np.where(grouped['close'].shift(1) < grouped['close'].shift(2), df['MF_PREV'], 0)  # 负资金流量
    df['POS_MF_14'] = grouped['POS_MF'].rolling(window=14, min_periods=14).sum().reset_index(0, drop=True)
    df['NEG_MF_14'] = grouped['NEG_MF'].rolling(window=14, min_periods=14).sum().reset_index(0, drop=True)
    df['MFI'] = 100 - (100 / (1 + df['POS_MF_14'] / df['NEG_MF_14'].replace(0, 0.001)))
    df.drop(columns=['TP_PREV', 'MF_PREV', 'POS_MF', 'NEG_MF', 'POS_MF_14', 'NEG_MF_14'], inplace=True)
    
    # 7. 成交量波动率：基于前10日成交量计算
    df['VOL_MEAN_10'] = grouped['volume'].shift(1).rolling(window=10, min_periods=10).mean().reset_index(0, drop=True)
    df['VOL_STD_10'] = grouped['volume'].shift(1).rolling(window=10, min_periods=10).std().reset_index(0, drop=True)
    df['VOL_VOLATILITY'] = (grouped['volume'].shift(1) - df['VOL_MEAN_10']) / df['VOL_STD_10'].replace(0, 0.001)
    df.drop(columns=['VOL_MEAN_10', 'VOL_STD_10'], inplace=True)
    
    # ----------------------
    # 三、超买超卖类指标（3个）
    # ----------------------
    # 8. 相对强弱指数（RSI14）：基于前14日涨跌计算
    df['CHANGE_PREV'] = grouped['close'].shift(1) - grouped['close'].shift(2)  # 前日涨跌幅
    df['GAIN_PREV'] = np.where(df['CHANGE_PREV'] > 0, df['CHANGE_PREV'], 0)  # 前日上涨幅度
    df['LOSS_PREV'] = np.where(df['CHANGE_PREV'] < 0, -df['CHANGE_PREV'], 0)  # 前日下跌幅度
    df['AVG_GAIN_14'] = grouped['GAIN_PREV'].rolling(window=14, min_periods=14).mean().reset_index(0, drop=True)
    df['AVG_LOSS_14'] = grouped['LOSS_PREV'].rolling(window=14, min_periods=14).mean().reset_index(0, drop=True)
    df['RSI14'] = 100 - (100 / (1 + df['AVG_GAIN_14'] / df['AVG_LOSS_14'].replace(0, 0.001)))
    df.drop(columns=['CHANGE_PREV', 'GAIN_PREV', 'LOSS_PREV', 'AVG_GAIN_14', 'AVG_LOSS_14'], inplace=True)
    
    # 9. 随机指标（KDJ-K值）：基于前9日高低价和前日收盘价
    df['LOW9'] = grouped['low'].shift(1).rolling(window=9, min_periods=9).min().reset_index(0, drop=True)  # 前9日最低价
    df['HIGH9'] = grouped['high'].shift(1).rolling(window=9, min_periods=9).max().reset_index(0, drop=True)  # 前9日最高价
    df['KDJ_K'] = 100 * (grouped['close'].shift(1) - df['LOW9']) / (df['HIGH9'] - df['LOW9']).replace(0, 0.001)
    df.drop(columns=['LOW9', 'HIGH9'], inplace=True)
    
    # 10. 能量潮指标（OBV）：累计前日能量潮
    df['OBV_CHANGE_PREV'] = np.where(
        grouped['close'].shift(1) > grouped['close'].shift(2),
        grouped['volume'].shift(1),
        np.where(grouped['close'].shift(1) < grouped['close'].shift(2), -grouped['volume'].shift(1), 0)
    )
    df['OBV'] = grouped['OBV_CHANGE_PREV'].cumsum().reset_index(0, drop=True)
    df.drop(columns=['OBV_CHANGE_PREV'], inplace=True)
    
    return df

price_vol_with_indicators = calculate_price_vol_indicators(price_vol_df)


# In[6]:


# 1. 关联量价（带指标）和竞价（带指标）：按date和stock_code
merged_df = pd.merge(
    price_vol_with_indicators, 
    auction_with_indicators, 
    on=["date", "stock_code"], 
    how="inner"  # 取两者都有的日期和股票
)

# 2. 关联行业数据：按date和stock_code（行业数据可能是每日或历史快照，需确保关联逻辑）
final_df = pd.merge(
    merged_df, 
    industry_df[["date", "stock_code", "zjw_industry_code", "zjw_industry_name"]], 
    on=["date", "stock_code"], 
    how="left"  # 行业数据若缺失，填充为NaN
)


# In[7]:


final_df


# In[8]:


def calculate_future_returns(df):
    """计算未来N日收益率"""
    # 当日收盘价
    df["return_dt"] = (df["close"] - df["open"]) / df["close"] * 100

    # 未来1日收盘价（shift(-1)实现）
    df["close_next1"] = df.groupby("stock_code")["close"].shift(-1)
    df["return_next1"] = (df["close_next1"] - df["open"]) / df["open"] * 100

    # 未来3日收盘价（shift(-3)实现）
    df["close_next3"] = df.groupby("stock_code")["close"].shift(-3)
    df["return_next3"] = (df["close_next3"] - df["open"]) / df["open"] * 100

    # 未来5日收盘价（shift(-5)实现）
    df["close_next5"] = df.groupby("stock_code")["close"].shift(-5)
    df["return_next5"] = (df["close_next5"] - df["open"]) / df["open"] * 100

    return df

final_df = calculate_future_returns(final_df)


# In[9]:


# 保存为Parquet（压缩存储，适合大数据）
final_df.to_parquet("wide.parquet", index=False)

# 或保存为CSV（方便非编程工具查看）
# final_df.to_csv("融合后宽表.csv", index=False, encoding="utf-8")


# In[10]:


import pandas as pd
import numpy as np

def calculate_future_returns(df):
    """计算未来N日收益率（复用提供的收益计算逻辑）"""
    # 确保数据按股票代码和日期排序（计算shift时必需）
    df = df.sort_values(['stock_code', 'date']).copy()
    
    # 当日收益率（当日收盘-当日开盘)/当日收盘*100
    df["return_dt"] = (df["close"] - df["open"]) / df["close"] * 100

    # 未来1日收益率（下一日收盘-当日开盘)/当日开盘*100
    df["close_next1"] = df.groupby("stock_code")["close"].shift(-1)
    df["return_next1"] = (df["close_next1"] - df["open"]) / df["open"] * 100

    # 未来3日收益率
    df["close_next3"] = df.groupby("stock_code")["close"].shift(-3)
    df["return_next3"] = (df["close_next3"] - df["open"]) / df["open"] * 100

    # 未来5日收益率
    df["close_next5"] = df.groupby("stock_code")["close"].shift(-5)
    df["return_next5"] = (df["close_next5"] - df["open"]) / df["open"] * 100

    return df

def process_stock_selection(wide_table_path="wide.parquet", output_file="每日前2名股票_with_returns.csv"):
    """
    包含收益字段的股票筛选流程：
    1. 读取宽表数据并计算各类收益指标
    2. 按日筛选auction_volume_1d_ratio>10的股票（最多前300只）
    3. 统计前5行业并筛选出对应股票
    4. 取每日前2名并包含return_dt、return_next1等收益字段
    """
    # 1. 读取宽表数据
    try:
        if wide_table_path.endswith('.parquet'):
            df = pd.read_parquet(wide_table_path)
        else:
            df = pd.read_csv(wide_table_path, parse_dates=['date'])
        print(f"成功读取宽表数据，共 {len(df)} 条记录")
    except Exception as e:
        raise SystemExit(f"读取宽表失败：{str(e)}")
    
    # 验证必要字段（包含计算收益所需的open和close）
    required_fields = ['date', 'stock_code', 'zjw_industry_name', 'auction_volume_1d_ratio', 'open', 'close']
    missing_fields = [f for f in required_fields if f not in df.columns]
    if missing_fields:
        raise SystemExit(f"宽表缺少必要字段：{missing_fields}")
    
    # 数据预处理与收益计算
    df = df[
        df['date'].notna() &
        df['stock_code'].notna() &
        df['zjw_industry_name'].notna() &
        df['auction_volume_1d_ratio'].notna() &
        df['open'].notna() &
        df['close'].notna()
    ].copy()
    
    # 去除所有auction_volume_1d_ratio 是NAN 以及inf的
    df = df[~df['auction_volume_1d_ratio'].isna() & ~np.isinf(df['auction_volume_1d_ratio'])].copy()
    
    # 计算收益指标（包含当日及未来1/3/5日收益）
    df = calculate_future_returns(df)
    
    df['date'] = pd.to_datetime(df['date']).dt.date  # 统一日期格式
    if df.empty:
        raise SystemExit("数据清洗后无有效记录")
    
    # 2. 按日筛选auction_volume_1d_ratio>10的股票，最多前300只
    daily_results = []
    for date, daily_data in df.groupby('date'):
        # 筛选竞价量比>10的股票
        qualified = daily_data[daily_data['auction_volume_1d_ratio'] >20].copy()
        
        if qualified.empty:
            print(f"日期 {date}：无符合auction_volume_1d_ratio>10的股票，跳过")
            continue
        
        # 按竞价量比降序排序，最多取前300只
        top_stocks = qualified.sort_values(
            'auction_volume_1d_ratio', ascending=False
        ).head(300).copy()
        
        # 记录当日基本信息
        top_stocks['date'] = date
        top_stocks['total_qualified'] = len(qualified)
        top_stocks['selected_count'] = len(top_stocks)
        
        daily_results.append(top_stocks)
        print(f"日期 {date}：符合条件{len(qualified)}只，选取前{len(top_stocks)}只")
    
    if not daily_results:
        raise SystemExit("所有日期均无符合条件的股票")
    
    # 合并每日筛选结果
    top300_df = pd.concat(daily_results, ignore_index=True)
    
    # 3. 统计每日前5个行业（按股票数量）
    daily_top5_industries = []
    for date, data in top300_df.groupby('date'):
        # 统计行业股票数量
        industry_counts = data['zjw_industry_name'].value_counts().reset_index()
        industry_counts.columns = ['industry', 'count']
        
        # 取前5个行业
        top5 = industry_counts.head(5).copy()
        top5['date'] = date
        top5['industry_rank'] = range(1, len(top5) + 1)  # 1表示数量最多
        
        daily_top5_industries.append(top5)
    
    top5_industries_df = pd.concat(daily_top5_industries, ignore_index=True)
    
    # 4. 筛选出属于前5行业的股票
    merged_df = pd.merge(
        top300_df,
        top5_industries_df[['date', 'industry', 'industry_rank']],
        left_on=['date', 'zjw_industry_name'],
        right_on=['date', 'industry'],
        how='inner'
    ).drop(columns=['industry'])
    
    if merged_df.empty:
        raise SystemExit("未找到属于前5行业的股票")
    
    # 5. 按竞价量比排序，取每日前2名
    final_results = []
    for date, data in merged_df.groupby('date'):
        # 按竞价量比降序排序
        sorted_data = data.sort_values('auction_volume_1d_ratio', ascending=False)
        
        # 取前2名
        top2 = sorted_data.head(1).copy()
        top2['final_rank'] = range(1, len(top2) + 1)  # 1表示排名第一
        
        final_results.append(top2)
        print(f"日期 {date}：前5行业股票共{len(sorted_data)}只，选取前{len(top2)}名")
    
    if not final_results:
        raise SystemExit("所有日期均无法选出前2名股票")
    
    # 整理最终结果（包含所有收益字段）
    result_df = pd.concat(final_results, ignore_index=True)
    
    # 格式化输出字段（包含return_dt、return_next1等收益指标）
    result_df = result_df[
        ['date', 'final_rank', 'stock_code', 'zjw_industry_name',
         'auction_volume_1d_ratio', 'industry_rank', 
         'return_dt', 'return_next1', 'return_next3', 'return_next5',  # 新增收益字段
         'total_qualified', 'selected_count']
    ].rename(columns={
        'zjw_industry_name': 'industry_name',
        'auction_volume_1d_ratio': '1d_ratio',
        'industry_rank': '行业排名',
        'return_dt': '当日收益率(%)',
        'return_next1': '1日后收益率(%)',
        'return_next3': '3日后收益率(%)',
        'return_next5': '5日后收益率(%)',
        'total_qualified': '当日达标总数',
        'selected_count': '当日选取数'
    })
    
    # 6. 保存为CSV（保留收益字段）
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n最终结果已保存至 {output_file}")
    
    return result_df

# 主程序执行
if __name__ == "__main__":
    try:
        # 执行筛选流程
        result = process_stock_selection(
            wide_table_path="wide.parquet",  # 输入宽表路径
            output_file="每日前2名股票_with_returns.csv"    # 输出结果路径
        )
        
        # 显示部分结果（包含收益字段）
        print("\n最近10条筛选结果（含收益）：")
        print(result.sort_values('date', ascending=False).head(10).to_string(index=False))
        
    except Exception as e:
        print(f"执行出错：{str(e)}")


# In[11]:


df = pd.read_csv('每日前2名股票_with_returns.csv')
df['daily_return'] = 1+0.5*df['1日后收益率(%)']/100
# 计算累计收益率
df['cumulative_return'] = df['daily_return'].cumprod()


# In[12]:


df.to_csv('每日前2名股票_with_returns_3d.csv', index=False)

