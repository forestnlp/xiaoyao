# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\trytry\共涨股票\寻找共涨股票GPU版.ipynb



# ----------------------------------------------------------------------import pandas as pd
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ----------------------
# 原有功能函数（保持不变）
# ----------------------
def process_daily_data(raw_data):
    """处理日K数据，返回清洗后的涨跌幅数据"""
    df = raw_data[['date', 'stock_code', 'close', 'pre_close', 'paused']].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['paused'] != 1].drop(columns='paused')
    df = df[df['date'].dt.year < 2025]
    df = df.dropna(subset=['close', 'pre_close'])
    df['daily_return'] = (df['close'] / df['pre_close']) - 1
    return_df = df.pivot(index='date', columns='stock_code', values='daily_return')
    print(f"数据清洗完成：时间范围 {return_df.index.min()} 至 {return_df.index.max()}，包含 {return_df.shape[1]} 只股票")
    return return_df

def pytorch_gpu_corr(return_df):
    """用PyTorch在GPU上计算全量股票相关性矩阵"""
    # 新增：GPU不可用时自动降级为CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用{device}计算相关性矩阵")
    
    filled_returns = return_df.fillna(0).astype('float32')
    gpu_returns = torch.tensor(filled_returns.values, device=device)
    
    mean = torch.mean(gpu_returns, dim=0, keepdim=True)
    centered = gpu_returns - mean
    cov = torch.matmul(centered.T, centered) / (centered.shape[0] - 1)
    std = torch.sqrt(torch.diag(cov)).reshape(-1, 1)
    corr_matrix = cov / torch.matmul(std, std.T)
    
    corr_df = pd.DataFrame(
        corr_matrix.cpu().numpy(),
        index=return_df.columns,
        columns=return_df.columns
    )
    return corr_df

def hierarchical_corr_groups(corr_matrix, min_corr=0.3, min_group_size=3):
    """用层次聚类筛选正相关股票组"""
    stock_codes = corr_matrix.columns.tolist()
    n_stocks = len(stock_codes)
    if n_stocks < min_group_size:
        raise ValueError(f"股票总数（{n_stocks}）小于最小组规模（{min_group_size}），无法聚类")

    distance_matrix = 1 - corr_matrix.values
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.maximum(distance_matrix, 0)
    dist_array = squareform(distance_matrix)

    linkage_matrix = linkage(dist_array, method='complete')
    max_allowed_distance = 1 - min_corr
    cluster_labels = fcluster(linkage_matrix, t=max_allowed_distance, criterion='distance')

    label_to_stocks = {}
    for stock_idx, label in enumerate(cluster_labels):
        label_to_stocks.setdefault(label, []).append(stock_codes[stock_idx])
    
    high_corr_groups = [
        stocks for stocks in label_to_stocks.values() 
        if len(stocks) >= min_group_size
    ]
    high_corr_groups.sort(key=lambda x: len(x), reverse=True)
    return high_corr_groups

# ----------------------
# 新增功能：查找指定股票的高相关组
# ----------------------
def find_related_group(target_stock, high_corr_groups, corr_matrix):
    """
    查找指定股票所在的高相关组，并返回组内相关性详情
    
    参数：
        target_stock: str，指定股票代码（如"600570.XSHG"）
        high_corr_groups: list，层次聚类得到的高相关组列表
        corr_matrix: DataFrame，股票间相关性矩阵
    返回：
        result: dict，包含组信息和相关性排序（若找到）；None（若未找到）
    """
    # 1. 检查指定股票是否在相关性矩阵中（避免无效查询）
    if target_stock not in corr_matrix.columns:
        print(f"错误：股票代码 {target_stock} 不在相关性矩阵中（可能数据不足或代码错误）")
        return None
    
    # 2. 定位指定股票所在的高相关组
    target_group = None
    for group in high_corr_groups:
        if target_stock in group:
            target_group = group
            break
    
    if not target_group:
        print(f"未找到包含 {target_stock} 的高相关组（可能该股票与其他股票相关性均<0.3）")
        return None
    
    # 3. 计算组内所有股票与指定股票的相关性，并排序
    related_stocks = [stock for stock in target_group if stock != target_stock]  # 排除自身
    corr_with_target = [
        (stock, corr_matrix.loc[target_stock, stock]) 
        for stock in related_stocks
    ]
    # 按相关性从高到低排序
    corr_with_target.sort(key=lambda x: x[1], reverse=True)
    
    # 4. 计算组内整体相关性指标
    group_corr_matrix = corr_matrix.loc[target_group, target_group]
    upper_triangle = group_corr_matrix.values[np.triu_indices_from(group_corr_matrix.values, k=1)]
    avg_group_corr = round(upper_triangle.mean(), 4)  # 组内平均相关性
    target_avg_corr = round(np.mean([x[1] for x in corr_with_target]), 4)  # 指定股票与组内其他股票的平均相关性
    
    # 5. 整理结果
    result = {
        "目标股票": target_stock,
        "所在组规模": len(target_group),
        "组内平均相关性": avg_group_corr,
        "目标股票与组内平均相关性": target_avg_corr,
        "组内所有股票": target_group,
        "与目标股票的相关性排序": corr_with_target  # 列表：(股票代码, 相关性)
    }
    return result

# ----------------------
# 新增功能：打印指定股票的相关组详情
# ----------------------
def print_related_group_details(result):
    """格式化打印指定股票的高相关组详情"""
    if not result:
        return
    
    print(f"\n===== 目标股票 {result['目标股票']} 的高相关组详情 =====")
    print(f"1. 所在组规模：{result['所在组规模']} 只股票")
    print(f"2. 组内平均相关性：{result['组内平均相关性']:.4f}")
    print(f"3. 目标股票与组内其他股票的平均相关性：{result['目标股票与组内平均相关性']:.4f}")
    
    print(f"\n4. 组内股票与目标股票的相关性排序（从高到低）：")
    for i, (stock, corr) in enumerate(result['与目标股票的相关性排序'], 1):
        print(f"   第{i}名：{stock}，相关性 {corr:.4f}")
    
    print(f"\n5. 组内所有股票代码：")
    print(", ".join(result['组内所有股票']))

# ----------------------
# 主流程执行（含新增功能调用）
# ----------------------
if __name__ == "__main__":
    # 1. 加载并清洗数据
    raw_daily = pd.read_parquet(r'D:\workspace\xiaoyao\data\stock_daily_price.parquet')
    cleaned_returns = process_daily_data(raw_daily)
    
    # 2. 计算相关性矩阵
    corr_matrix = pytorch_gpu_corr(cleaned_returns)
    
    # 3. 聚类得到高相关组
    high_corr_groups = hierarchical_corr_groups(corr_matrix, min_corr=0.3, min_group_size=3)
    
    # 4. 保存聚类结果（原有功能）
    group_result = []
    for group_id, stock_group in enumerate(high_corr_groups, 1):
        group_corr_matrix = corr_matrix.loc[stock_group, stock_group]
        upper_triangle = group_corr_matrix.values[np.triu_indices_from(group_corr_matrix.values, k=1)]
        avg_corr = round(upper_triangle.mean(), 4)
        group_result.append({
            "组号": group_id,
            "股票数量": len(stock_group),
            "组内平均相关性": avg_corr,
            "组内股票代码": ",".join(stock_group)
        })
    result_df = pd.DataFrame(group_result)
    save_path = r"./high_corr_stock_groups_2024.csv"
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n高关联股票组结果已保存至：{save_path}")
    
    # 5. 调用新增功能：查找指定股票的高相关组（可修改股票代码）
    target_stock = "600570.XSHG"  # 此处替换为你想查询的股票代码
    related_group = find_related_group(target_stock, high_corr_groups, corr_matrix)
    print_related_group_details(related_group)

