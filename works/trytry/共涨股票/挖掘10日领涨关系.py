# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\trytry\共涨股票\挖掘10日领涨关系.ipynb



# ----------------------------------------------------------------------import pandas as pd
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# 设置全局设备（CPU/GPU自动切换）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")


# ----------------------
# 1. 数据预处理（2025年前训练集）
# ----------------------
def preprocess_data(raw_data):
    """预处理原始数据，转为张量格式（仅2025年前数据）"""
    df = raw_data[['date', 'stock_code', 'close', 'paused']].copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 筛选：非停牌、2025年前、无缺失值
    df = df[df['paused'] != 1].drop(columns='paused')
    df = df[df['date'].dt.year < 2025].dropna(subset=['close'])
    
    # 转为透视表（日期×股票，便于按周期计算）
    pivot_df = df.pivot(index='date', columns='stock_code', values='close')
    stock_codes = pivot_df.columns.tolist()
    dates = pivot_df.index.tolist()
    
    # 填充缺失值（前向+后向，避免周期计算断层）
    pivot_df = pivot_df.fillna(method='ffill').fillna(method='bfill')
    
    # 转为张量（形状[股票数, 日期数]，适配按股票处理）
    close_tensor = torch.tensor(pivot_df.values, dtype=torch.float32, device=device).T
    
    print(f"数据预处理完成：{len(stock_codes)}只股票，{len(dates)}个交易日（2025年前）")
    return close_tensor, stock_codes, dates, pivot_df.index  # 返回日期索引，便于后续周期对齐


# ----------------------
# 2. 周期特征计算（核心：生成滚动周期涨跌幅）
# ----------------------
def calculate_cycle_features(close_tensor, cycle_days=10):
    """计算所有股票的滚动周期涨跌幅（仅保留涨跌幅，用于领先关系计算）"""
    n_stocks, n_dates = close_tensor.shape
    valid_periods = n_dates - cycle_days + 1  # 有效周期数（如4858天→4858-10+1=4849个周期）
    
    # 边界判断：确保有效周期数≥1
    if valid_periods < 1:
        print(f"警告：{cycle_days}日周期下，仅{ n_dates }个交易日，无法生成有效周期")
        return torch.tensor([], device=device)
    
    # 计算滚动周期涨跌幅：(周期末收盘价/周期初收盘价) - 1
    # 矩阵运算：start_prices[股票数, 周期数] = 每个周期的起始价
    start_prices = close_tensor[:, :n_dates - cycle_days + 1]
    # end_prices[股票数, 周期数] = 每个周期的结束价
    end_prices = close_tensor[:, cycle_days - 1:]
    cycle_return = (end_prices / start_prices) - 1
    
    print(f"{cycle_days}日周期涨跌幅计算完成：形状 {cycle_return.shape}（股票数×周期数）")
    return cycle_return


# ----------------------
# 3. 周期协同矩阵计算（原有功能，用于聚类）
# ----------------------
def gpu_cycle_correlation(cycle_return, weight_return=1.0):
    """简化协同矩阵：仅用周期涨跌幅计算相关性（原上涨天数占比可按需添加）"""
    def normalize(x):
        mean = x.mean(dim=1, keepdim=True)  # 按股票维度算均值
        std = x.std(dim=1, keepdim=True) + 1e-8  # 避免除0
        return (x - mean) / std
    
    norm_return = normalize(cycle_return)
    n_periods = norm_return.shape[1]
    # 矩阵乘法计算相关性：(x·y)/(n-1)
    corr_matrix = torch.matmul(norm_return, norm_return.T) / (n_periods - 1)
    corr_matrix = torch.clamp(corr_matrix, -1, 1)  # 限制相关性范围在[-1,1]
    
    print(f"周期协同矩阵计算完成：形状 {corr_matrix.shape}")
    return corr_matrix


# ----------------------
# 4. 聚类与结果处理（原有功能，用于生成协同组）
# ----------------------
def hierarchical_cycle_groups(cycle_corr_matrix, stock_codes, min_corr=0.4, min_group_size=3):
    """从协同矩阵生成高协同股票组"""
    corr_np = cycle_corr_matrix.cpu().numpy()
    # 距离矩阵 = 1 - 相关性矩阵（相关性越高，距离越近）
    distance_matrix = 1 - corr_np
    np.fill_diagonal(distance_matrix, 0)  # 对角线距离为0（自身与自身）
    distance_matrix = np.maximum(distance_matrix, 0)  # 确保距离非负
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # 强制对称化，避免浮点误差
    
    # 层次聚类：complete linkage（全连接）
    dist_array = squareform(distance_matrix)  # 转为scipy聚类需要的1维距离数组
    linkage_matrix = linkage(dist_array, method='complete')
    # 按距离阈值聚类：最大允许距离=1-最小相关性（相关性≥min_corr的分为一组）
    max_allowed_distance = 1 - min_corr
    cluster_labels = fcluster(linkage_matrix, t=max_allowed_distance, criterion='distance')
    
    # 按聚类标签分组
    label_to_stocks = {}
    for stock_idx, label in enumerate(cluster_labels):
        label_to_stocks.setdefault(label, []).append(stock_codes[stock_idx])
    
    # 筛选股票数≥min_group_size的组，并按组大小排序
    high_cycle_groups = [
        stocks for stocks in label_to_stocks.values() 
        if len(stocks) >= min_group_size
    ]
    high_cycle_groups.sort(key=lambda x: len(x), reverse=True)
    return high_cycle_groups, corr_np


# ----------------------
# 5. 领先预测股票筛选（核心修改：按周期领先，而非天数）
# ----------------------
def find_leading_stocks(
    cycle_return, 
    stock_codes, 
    target_stock, 
    lead_cycles=1,        # 领先周期数（1=用前1个周期预判后1个周期）
    corr_threshold=0.4,   # 最小相关性阈值
    min_samples=20        # 最小有效样本量（避免小样本偶然误差）
):
    """
    筛选能“提前N个周期”预判目标股票的领先股票
    核心逻辑：领先股票的第t个周期 → 目标股票的第t+lead_cycles个周期
    """
    # 检查目标股票是否在股票池中
    if target_stock not in stock_codes:
        return pd.DataFrame(columns=['stock_code', 'leading_correlation', 'valid_samples'])
    
    target_idx = stock_codes.index(target_stock)
    n_stocks, n_total_periods = cycle_return.shape  # n_total_periods：总周期数
    
    # ----------------------
    # 核心：目标股票周期向前平移“领先周期数”（关键修改）
    # ----------------------
    target_return = cycle_return[target_idx]  # 目标股票原始周期序列：[C1, C2, ..., Cn]
    # 向前平移lead_cycles个周期：[C(1+lead), C(2+lead), ..., Cn, NaN, ...]
    target_shifted = torch.roll(target_return, shifts=-lead_cycles, dims=0)
    # 平移后，末尾lead_cycles个周期无对应数据，设为NaN
    target_shifted[-lead_cycles:] = float('nan')
    
    # 计算每个股票与“平移后目标周期”的相关性
    results = []
    for stock_idx in range(n_stocks):
        if stock_idx == target_idx:
            continue  # 跳过目标股票自身
        
        # 提取当前股票的周期序列
        stock_seq = cycle_return[stock_idx]
        # 有效数据掩码：排除NaN（目标股票末尾lead_cycles个周期是NaN）
        valid_mask = ~(torch.isnan(stock_seq) | torch.isnan(target_shifted))
        valid_samples = valid_mask.sum().item()
        
        # 有效样本量不足，跳过
        if valid_samples < min_samples:
            continue
        
        # 计算标准化后的相关性（避免量纲影响）
        s_valid = stock_seq[valid_mask]
        t_valid = target_shifted[valid_mask]
        
        # 计算均值和标准差（加1e-8避免除0）
        s_mean = s_valid.mean()
        s_std = s_valid.std() + 1e-8
        t_mean = t_valid.mean()
        t_std = t_valid.std() + 1e-8
        
        # 标准化后计算皮尔逊相关系数
        s_norm = (s_valid - s_mean) / s_std
        t_norm = (t_valid - t_mean) / t_std
        corr = (s_norm * t_norm).mean().item()  # 标准化后，均值即为相关性
        
        # 相关性达标，加入结果
        if corr >= corr_threshold:
            results.append({
                'stock_code': stock_codes[stock_idx],
                'leading_correlation': round(corr, 4),
                'valid_samples': valid_samples
            })
    
    # 结果排序（按相关性降序）
    leading_df = pd.DataFrame(results)
    if not leading_df.empty:
        leading_df = leading_df.sort_values(
            by='leading_correlation', 
            ascending=False
        ).reset_index(drop=True)
    
    return leading_df


# ----------------------
# 6. 主流程（训练集筛选领先组，输出可直接用于交易的结果）
# ----------------------
if __name__ == "__main__":
    # 1. 配置核心参数（可按需调整）
    PARAMS = {
        'target_stock': "600570.XSHG",    # 目标股票
        'cycle_days': 10,                 # 周期天数（10日周期，效果最优）
        'lead_cycles': 1,                 # 领先周期数（1=用前1个周期预判后1个周期）
        'corr_threshold': 0.4,            # 领先股票最小相关性阈值
        'min_samples': 20,                # 最小有效样本量
        'data_path': r'D:\workspace\xiaoyao\data\stock_daily_price.parquet',  # 数据路径
        'min_group_size': 3               # 协同组最小股票数
    }
    
    # 2. 加载原始数据并预处理（仅2025年前训练集）
    raw_daily = pd.read_parquet(PARAMS['data_path'])
    close_tensor, stock_codes, dates, date_index = preprocess_data(raw_daily)
    
    # 3. 计算周期涨跌幅（用于领先关系和协同矩阵）
    cycle_return = calculate_cycle_features(close_tensor, cycle_days=PARAMS['cycle_days'])
    if cycle_return.numel() == 0:
        print("周期特征计算失败，程序终止")
        exit()
    
    # 4. 计算协同矩阵并生成高协同组（原有功能，可选看）
    cycle_corr_matrix = gpu_cycle_correlation(cycle_return)
    high_cycle_groups, corr_np = hierarchical_cycle_groups(
        cycle_corr_matrix, 
        stock_codes, 
        min_corr=PARAMS['corr_threshold'], 
        min_group_size=PARAMS['min_group_size']
    )
    
    # 保存协同组结果（可选）
    corr_df = pd.DataFrame(corr_np, index=stock_codes, columns=stock_codes)
    group_result = []
    for group_id, group in enumerate(high_cycle_groups, 1):
        group_corr = corr_df.loc[group, group]
        upper_tri = group_corr.values[np.triu_indices_from(group_corr, k=1)]  # 上三角（排除自身）
        group_result.append({
            "周期天数": PARAMS['cycle_days'],
            "组号": group_id,
            "股票数": len(group),
            "平均协同得分": round(upper_tri.mean(), 4),
            "股票代码": ",".join(group)
        })
    pd.DataFrame(group_result).to_csv(
        f"./cycle_groups_{PARAMS['cycle_days']}d.csv", 
        index=False, 
        encoding="utf-8-sig"
    )
    print(f"\n{PARAMS['cycle_days']}日协同组结果已保存至：./cycle_groups_{PARAMS['cycle_days']}d.csv")
    
    # 5. 筛选领先股票（核心功能，输出可用于交易的领先组）
    leading_stocks = find_leading_stocks(
        cycle_return=cycle_return,
        stock_codes=stock_codes,
        target_stock=PARAMS['target_stock'],
        lead_cycles=PARAMS['lead_cycles'],
        corr_threshold=PARAMS['corr_threshold'],
        min_samples=PARAMS['min_samples']
    )
    
    # 保存领先股票结果（关键：后续交易用）
    save_path = f"./leading_stocks_{PARAMS['cycle_days']}d_lead{PARAMS['lead_cycles']}cycle.csv"
    leading_stocks.to_csv(save_path, index=False, encoding="utf-8-sig")
    
    # 6. 输出关键结果（便于快速查看）
    print(f"\n===== {PARAMS['cycle_days']}日周期-领先{PARAMS['lead_cycles']}个周期的领先股票结果 =====")
    if len(leading_stocks) > 0:
        print(f"共筛选出{len(leading_stocks)}只领先股票，前10只如下：")
        print(leading_stocks.head(10))
        print(f"\n领先组平均相关性：{leading_stocks['leading_correlation'].mean():.4f}")
        print(f"领先股票结果已保存至：{save_path}")
    else:
        print(f"未找到符合条件的领先股票（相关性≥{PARAMS['corr_threshold']}，有效样本≥{PARAMS['min_samples']}）")
    
    # 7. 额外输出：领先组股票列表（直接复制可用）
    if len(leading_stocks) > 0:
        leading_group = leading_stocks['stock_code'].tolist()
        print(f"\n可直接用于交易的领先组股票列表：")
        print(leading_group)