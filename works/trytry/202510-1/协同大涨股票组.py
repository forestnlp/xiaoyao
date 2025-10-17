# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\trytry\202510-1\协同大涨股票组.ipynb



# ----------------------------------------------------------------------import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# --------------------------
# 步骤1：读取数据并筛选大涨股票（修复日期提取）
# --------------------------
try:
    df_wide = pd.read_parquet(r'D:\workspace\xiaoyao\data\stock_daily_price.parquet', engine='pyarrow')
    print(f"宽表数据读取成功，共{len(df_wide)}条记录")

    # 数据清洗
    df_valid = df_wide[
        (df_wide['paused'] == 0.0) & 
        (df_wide['open'] > 0) & 
        (df_wide['close'].notna()) & 
        (df_wide['open'].notna())
    ].copy()

    # 计算涨跌幅
    df_valid['dakepct'] = (df_valid['close'] - df_valid['open']) / df_valid['open']
    df_big_rise = df_valid[df_valid['dakepct'] > 0.07].copy()
    print(f"筛选出大涨股票记录：{len(df_big_rise)}条")

    # 【核心修复】彻底提取纯日期（兼容字符串/时间戳格式）
    def extract_pure_date(date_val):
        """
        处理多种日期格式，确保输出为'YYYY-MM-DD'字符串：
        - 若为datetime对象：直接格式化
        - 若为字符串：按空格分割取前半段，再按'-'截取前10位（避免特殊格式）
        """
        if pd.api.types.is_datetime64_any_dtype(date_val):
            return date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val).split(' ')[0]  # 去除时间部分
            return date_str[:10]  # 确保只保留'YYYY-MM-DD'（应对超长字符串）

    df_big_rise['pure_date'] = df_big_rise['date'].apply(extract_pure_date)
    
    # 校验日期格式（抽样检查前5条，避免后续解析错误）
    sample_dates = df_big_rise['pure_date'].head(5)
    for d in sample_dates:
        try:
            datetime.strptime(d, "%Y-%m-%d")
        except:
            raise ValueError(f"日期格式处理失败，错误格式：{d}（请检查原始date列格式）")

    # 保留必要字段
    required_fields = ['pure_date', 'stock_code']
    if not set(required_fields).issubset(df_big_rise.columns):
        missing = set(required_fields) - set(df_big_rise.columns)
        raise ValueError(f"宽表缺少必要字段：{missing}")
    df_big_rise = df_big_rise[required_fields]

except Exception as e:
    print(f"筛选大涨股票失败：{str(e)}")
    raise


# --------------------------
# 步骤2：按时间拆分训练集（2005-2024）和测试集（2025）
# --------------------------
def split_train_test(dates):
    """修复：解析纯日期“YYYY-MM-DD”，拆分训练/测试集"""
    try:
        train_dates = []
        test_dates = []
        for date_str in dates:
            # 跳过空字符串或无效格式（冗余处理）
            if not date_str or len(date_str) != 10:
                continue
            # 解析日期（此时date_str应为'YYYY-MM-DD'）
            date_dt = datetime.strptime(date_str, "%Y-%m-%d")
            if 2005 <= date_dt.year <= 2024:
                train_dates.append(date_str)
            elif date_dt.year == 2025:
                test_dates.append(date_str)
        return train_dates, test_dates
    except Exception as e:
        print(f"日期拆分错误（问题日期：{date_str}）：{str(e)}")
        return [], []

try:
    # 分组聚合大涨日期
    df_group = df_big_rise.groupby('stock_code').agg(
        大涨日期列表=('pure_date', list)
    ).reset_index()

    # 拆分训练/测试日期
    df_group[['训练期大涨日期', '测试期大涨日期']] = df_group['大涨日期列表'].apply(
        lambda x: pd.Series(split_train_test(x))
    )

    # 过滤训练期无数据的股票
    df_train = df_group[df_group['训练期大涨日期'].apply(len) > 0].copy()
    print(f"训练集有效股票数：{len(df_train)}（需≥2）")
    if len(df_train) < 2:
        # 检查年份分布（辅助调试）
        all_years = []
        for dates in df_group['大涨日期列表']:
            for d in dates:
                try:
                    if len(d) == 10:
                        all_years.append(datetime.strptime(d, "%Y-%m-%d").year)
                except:
                    continue
        print(f"数据中包含的有效年份：{sorted(list(set(all_years)))}")
        raise ValueError("训练集有效股票不足2只，可能是数据年份不在2005-2024范围内")

except Exception as e:
    print(f"数据拆分失败：{str(e)}")
    raise


# --------------------------
# 步骤3：计算股票对的惩罚后协同率
# --------------------------
MIN_COMMON_DATES = 3
PENALTY_THRESHOLD = 0.1

try:
    stock_dates = dict(zip(df_train['stock_code'], df_train['训练期大涨日期'].apply(set)))
    stocks = list(stock_dates.keys())

    co_pairs = []
    # 优化：若股票数量过多，限制组合数（避免内存溢出）
    max_pairs = 100000000  # 可根据内存调整
    if len(stocks) > int(np.sqrt(2 * max_pairs)):
        print(f"股票数量过多（{len(stocks)}只），仅计算前{int(np.sqrt(2 * max_pairs))}只股票的组合")
        stocks = stocks[:int(np.sqrt(2 * max_pairs))]

    for a, b in combinations(stocks, 2):
        dates_a = stock_dates[a]
        dates_b = stock_dates[b]
        len_a, len_b = len(dates_a), len(dates_b)
        common = len(dates_a & dates_b)

        if common < MIN_COMMON_DATES:
            continue

        raw_co = common / min(len_a, len_b)
        penalty = common / max(len_a, len_b)
        penalized_co = raw_co * penalty

        co_pairs.append({
            '股票A': a, '股票B': b,
            '股票A大涨次数': len_a, '股票B大涨次数': len_b,
            '共同大涨次数': common,
            '原始协同率': round(raw_co, 3),
            '惩罚后协同率': round(penalized_co, 3)
        })

    df_co = pd.DataFrame(co_pairs)
    high_co_pairs = df_co[df_co['惩罚后协同率'] >= PENALTY_THRESHOLD].copy()
    print(f"筛选出高协同股票对：{len(high_co_pairs)}对")
    if len(high_co_pairs) < 1:
        raise ValueError("高协同对不足，建议降低PENALTY_THRESHOLD至0.05")

except Exception as e:
    print(f"协同率计算失败：{str(e)}")
    raise


# --------------------------
# 步骤4：层次聚类生成股票组
# --------------------------
try:
    all_co_stocks = list(set(high_co_pairs['股票A']) | set(high_co_pairs['股票B']))
    if len(all_co_stocks) < 2:
        raise ValueError("参与聚类的股票不足2只")

    # 构建协同率矩阵
    co_matrix = pd.DataFrame(0.0, index=all_co_stocks, columns=all_co_stocks)
    for _, row in high_co_pairs.iterrows():
        a, b = row['股票A'], row['股票B']
        co_matrix.loc[a, b] = row['惩罚后协同率']
        co_matrix.loc[b, a] = row['惩罚后协同率']
    np.fill_diagonal(co_matrix.values, 0.0)

    # 转换为距离矩阵并聚类（优化：增加距离矩阵有效性校验）
    distance_matrix = 1 - co_matrix
    np.fill_diagonal(distance_matrix.values, 0.0)
    if not np.allclose(distance_matrix, distance_matrix.T):
        print("警告：距离矩阵非对称，可能存在计算误差，已自动修正")
        distance_matrix = (distance_matrix + distance_matrix.T) / 2  # 对称化处理

    linkage_matrix = linkage(squareform(distance_matrix), method='ward')
    cluster_labels = fcluster(linkage_matrix, t=1 - PENALTY_THRESHOLD, criterion='distance')

    # 生成股票组
    df_clusters = pd.DataFrame({
        'stock_code': all_co_stocks,
        '股票组编号': cluster_labels
    })
    cluster_result = df_clusters.groupby('股票组编号').agg({
        'stock_code': list
    }).rename(columns={'stock_code': '组内股票代码'})
    cluster_result = cluster_result[cluster_result['组内股票代码'].apply(len) >= 2]
    print(f"训练集生成有效股票组：{len(cluster_result)}组")
    if len(cluster_result) == 0:
        # 自动降低阈值重试（修复f-string语法）
        new_t = 0.95
        print(f"未生成有效组，尝试降低聚类阈值至t={new_t}")  # 修改此处语法
        cluster_labels = fcluster(linkage_matrix, t=new_t, criterion='distance')
        df_clusters = pd.DataFrame({'stock_code': all_co_stocks, '股票组编号': cluster_labels})
        cluster_result = df_clusters.groupby('股票组编号').agg({'stock_code': list}).rename(columns={'stock_code': '组内股票代码'})
        cluster_result = cluster_result[cluster_result['组内股票代码'].apply(len) >= 2]
        if len(cluster_result) == 0:
            raise ValueError("降低阈值后仍未生成有效组，建议进一步降低PENALTY_THRESHOLD")

except Exception as e:
    print(f"聚类失败：{str(e)}")
    raise


# --------------------------
# 步骤5：测试集验证（筛选稳定组）
# --------------------------
try:
    test_stock_dates = dict(zip(df_group['stock_code'], df_group['测试期大涨日期'].apply(set)))
    stable_clusters = []

    for cluster_id, group in cluster_result.iterrows():
        stocks_in_group = group['组内股票代码']
        test_valid = [s for s in stocks_in_group if len(test_stock_dates.get(s, [])) > 0]
        if len(test_valid) < 2:
            continue

        # 计算测试期协同率
        test_co_list = []
        for a, b in combinations(test_valid, 2):
            dates_a = test_stock_dates.get(a, set())
            dates_b = test_stock_dates.get(b, set())
            common = len(dates_a & dates_b)
            min_total = min(len(dates_a), len(dates_b))
            test_co = common / min_total if min_total > 0 else 0
            test_co_list.append(test_co)

        # 训练期平均协同率
        train_co_subset = high_co_pairs[
            (high_co_pairs['股票A'].isin(stocks_in_group)) &
            (high_co_pairs['股票B'].isin(stocks_in_group))
        ]['惩罚后协同率']
        avg_train_co = train_co_subset.mean() if not train_co_subset.empty else 0
        avg_test_co = np.mean(test_co_list) if test_co_list else 0

        # 验证稳定组（增加容错：若训练期协同率为0，直接保留测试期有协同的组）
        if (avg_train_co == 0 and avg_test_co > 0) or (avg_test_co >= avg_train_co * 0.5):
            stable_clusters.append({
                '股票组编号': cluster_id,
                '组内股票代码': stocks_in_group,
                '训练期平均协同率': round(avg_train_co, 3),
                '测试期平均协同率': round(avg_test_co, 3)
            })

    df_stable = pd.DataFrame(stable_clusters)
    print(f"2025测试期验证通过的稳定股票组：{len(df_stable)}组")
    df_stable.to_csv('稳定大涨股票组.csv', index=False, encoding='utf-8-sig')
    print("结果已保存至：稳定大涨股票组.csv")

except Exception as e:
    print(f"测试集验证失败：{str(e)}")
    raise


# 输出结果预览
if 'df_stable' in locals() and not df_stable.empty:
    print("\n稳定股票组预览：")
    print(df_stable[['股票组编号', '组内股票代码', '训练期平均协同率', '测试期平均协同率']].head())

df_stable


