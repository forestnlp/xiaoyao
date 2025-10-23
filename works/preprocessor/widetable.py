# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\preprocessor\widetable.ipynb



# ----------------------------------------------------------------------# 依次读取项目data目录下的parquet文件
import pandas as pd

# 读取股票日k线数据，行业数据，竞价数据，市值数据
price_df = pd.read_parquet(r'D:\workspace\xiaoyao\data\stock_daily_price.parquet')
industry_df = pd.read_parquet(r'D:\workspace\xiaoyao\data\stock_daily_industry.parquet')
auction_df = pd.read_parquet(r'D:\workspace\xiaoyao\data\stock_daily_auction.parquet')
marketcap_df = pd.read_parquet(r'D:\workspace\xiaoyao\data\stock_daily_marketcap.parquet')
concept_df = pd.read_parquet(r'D:\workspace\xiaoyao\data\stock_daily_concept.parquet')


# price_df 只取2025-01-01以后的数据
price_df = price_df[price_df['date'] >= '2025-01-01']
industry_df = industry_df[industry_df['date'] >= '2025-01-01']
marketcap_df = marketcap_df[marketcap_df['date'] >= '2025-01-01']
auction_df = auction_df[auction_df['date'] >= '2025-01-01']
concept_df = concept_df[concept_df['date'] >= '2025-01-01']

# 将date转换为字符串类型
price_df['date'] = price_df['date'].astype(str)
industry_df['date'] = industry_df['date'].astype(str)
marketcap_df['date'] = marketcap_df['date'].astype(str)
# 将auction的date只取日期部分
auction_df['date'] = auction_df['date'].astype(str).str[:10]
# 将概念信息的date转换为字符串
concept_df['date'] = concept_df['date'].astype(str)

# 增加stock_name字段，读取D:\workspace\xiaoyao\data\stocks_info.csv文件，用stock_code关联display_name，将display_name 重命名为stock_name
stocks_info = pd.read_csv(r"D:\workspace\xiaoyao\data\stocks_info.csv")
stocks_info = stocks_info[["stock_code", "display_name"]]
stocks_info.rename(columns={"display_name": "stock_name"}, inplace=True)

# 步骤1：按 date、stock_code、concept_code 排序
sorted_concept_df = concept_df.sort_values(by=['date', 'stock_code', 'concept_code'])

# 步骤2：按 date 和 stock_code 分组，聚合 concept_name 为列表
sorted_concept_df = sorted_concept_df.groupby(['date', 'stock_code'])['concept_name'].agg(list).reset_index()

# 重命名列名
sorted_concept_df = sorted_concept_df.rename(columns={'concept_name': 'concept_name_list'})

sorted_concept_df.head(3)

# 将四个df合并到price_df
merged_df = price_df.merge(stocks_info, on=['stock_code'], how='left')
merged_df = merged_df.merge(industry_df, on=['date', 'stock_code'], how='left')
merged_df = merged_df.merge(marketcap_df, on=['date', 'stock_code'], how='left')
merged_df = merged_df.merge(auction_df, on=['date', 'stock_code'], how='left')
merged_df = merged_df.merge(sorted_concept_df, on=['date', 'stock_code'], how='left')

# merged_df将字段重命名 其中volume重命名为volume_daily
merged_df = merged_df.rename(columns={'volume_x': 'volume'})
merged_df = merged_df.rename(columns={'volume_y': 'auc_volume'})
merged_df = merged_df.rename(columns={'money_x': 'money'})
merged_df = merged_df.rename(columns={'money_y': 'auc_money'})

merged_df.columns

# 保存merged_df到D:\workspace\xiaoyao\data下
merged_df.to_parquet(r'D:\workspace\xiaoyao\data\widetable.parquet', index=False)

# 将parquet读取后，随机采样5条数据，并导出为csv存放在本地目录
import pandas as pd

df = pd.read_parquet(r'D:\workspace\xiaoyao\data\widetable.parquet')
df.sample(5).to_csv('./widetable_sample.csv', index=False)


