# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\build\find_intersect_stock.ipynb



# ----------------------------------------------------------------------import pandas as pd
import os

# -------------------------- 配置参数 --------------------------
WIDETABLE_PATH = "D:\\workspace\\xiaoyao\\data\\widetable.parquet"  # 原始数据路径
TARGET_DATE = "2025-10-30"  # 目标日期
TARGET_INDUSTRY = "航运港口II"  # 目标申万L2行业
TARGET_CONCEPT = "航运概念"  # 目标概念

# -------------------------- 筛选逻辑 --------------------------
def filter_stocks():
    # 1. 读取数据并筛选日期
    print(f"正在筛选 {TARGET_DATE} 符合条件的个股...")
    df = pd.read_parquet(
        WIDETABLE_PATH,
        engine="pyarrow",
        columns=[
            "date", "stock_code", "stock_name", 
            "sw_l2_industry_name", "concept_name_list"
        ]
    )
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")  # 统一日期格式
    
    # 2. 筛选目标日期和行业
    filtered = df[
        (df["date"] == TARGET_DATE) &
        (df["sw_l2_industry_name"] == TARGET_INDUSTRY)
    ].copy()
    
    if len(filtered) == 0:
        print(f"❌ 未找到 {TARGET_DATE} 属于 {TARGET_INDUSTRY} 行业的个股")
        return
    
    # 3. 筛选包含目标概念的个股（concept_name_list为列表格式）
    filtered["has_target_concept"] = filtered["concept_name_list"].apply(
        lambda x: TARGET_CONCEPT in x
    )
    result = filtered[filtered["has_target_concept"]]
    
    # 4. 输出结果
    print(f"\n✅ 共找到 {len(result)} 只符合条件的个股：")
    print(result[["stock_code", "stock_name", "sw_l2_industry_name", "concept_name_list"]].to_string(index=False))
    
    # 可选：保存结果
    output_path = f".\\{TARGET_DATE}_{TARGET_INDUSTRY}_{TARGET_CONCEPT}_stocks.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存至：{output_path}")

# -------------------------- 执行筛选 --------------------------
if __name__ == "__main__":
    filter_stocks()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -------------------------- 配置参数 --------------------------
WIDETABLE_PATH = "D:\\workspace\\xiaoyao\\data\\widetable.parquet"
CROSS_STOCKS_PATH = f".\\2025-10-30_航运港口II_航运概念_stocks.csv"  # 行业概念交叉结果
OUTPUT_PATH = ".\\yangjia_ranked_all_stocks.csv"

# 时间参数
MA5_DAYS = 5    # 5日均线
MA20_DAYS = 20  # 20日均线
LOOKBACK_DAYS = 10  # 风险评估周期
TARGET_DATE = "2025-10-30"


# -------------------------- 辅助函数 --------------------------
def calculate_slope(series):
    """计算均线斜率（趋势强度）"""
    x = np.arange(len(series))
    slope, _, _, _, _ = np.polyfit(x, series, 1)
    return slope


# -------------------------- 核心排序逻辑（仅排序不筛选） --------------------------
def yangjia_strategy_rank():
    # 1. 读取所有行业+概念交叉个股
    print(f"读取行业+概念交叉结果...")
    cross_df = pd.read_csv(CROSS_STOCKS_PATH)
    all_stock_codes = cross_df["stock_code"].unique()
    print(f"待排序个股总数：{len(all_stock_codes)} 只")
    
    # 2. 读取宽表完整数据
    df = pd.read_parquet(
        WIDETABLE_PATH,
        engine="pyarrow",
        columns=[
            "date", "stock_code", "stock_name", "close", "pre_close", "volume",
            "turnover_ratio", "paused", "auc_volume", "auc_money",
            "a1_p", "open"
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    latest_date = pd.to_datetime(TARGET_DATE)
    start_date = latest_date - timedelta(days=MA20_DAYS)
    
    # 筛选目标个股+日期范围（保留停牌股，后续标记）
    df = df[
        (df["stock_code"].isin(all_stock_codes)) &
        (df["date"] >= start_date) &
        (df["date"] <= latest_date)
    ].copy()
    df["is_target_date"] = (df["date"] == latest_date)
    
    # 3. 计算每只个股的养家心法指标（不筛选，仅计算）
    rank_data = []
    for stock_code in all_stock_codes:
        stock_data = df[df["stock_code"] == stock_code].sort_values("date")
        target_data = stock_data[stock_data["is_target_date"]].iloc[0] if any(stock_data["is_target_date"]) else None
        
        if target_data is None:
            continue  # 无目标日期数据，跳过
        
        # 基础信息
        is_paused = 1 if target_data["paused"] == 1.0 else 0
        stock_name = target_data["stock_name"]
        
        # （1）趋势指标（均线斜率）
        if len(stock_data) >= MA20_DAYS:
            stock_data["ma5"] = stock_data["close"].rolling(MA5_DAYS).mean()
            stock_data["ma20"] = stock_data["close"].rolling(MA20_DAYS).mean()
            ma5_slope = calculate_slope(stock_data["ma5"].tail(MA5_DAYS))
            ma20_slope = calculate_slope(stock_data["ma20"].tail(MA20_DAYS))
        else:
            ma5_slope = 0  # 数据不足，斜率记为0
            ma20_slope = 0
        
        # （2）量价配合指标
        if len(stock_data) >= 5:
            stock_data["pct_change"] = (stock_data["close"] / stock_data["pre_close"] - 1) * 100
            stock_data["vol_change"] = stock_data["volume"].pct_change()
            recent_5d = stock_data.tail(5)
            sync_days = sum((recent_5d["pct_change"] > 0) & (recent_5d["vol_change"] > 0))
        else:
            sync_days = 0
        
        # （3）风险指标（最大回撤）
        if len(stock_data) >= LOOKBACK_DAYS:
            recent_10d_close = stock_data.tail(LOOKBACK_DAYS)["close"]
            max_drawdown = (recent_10d_close.max() - recent_10d_close.min()) / recent_10d_close.max() * 100
        else:
            max_drawdown = 100  # 数据不足，记为高风险
        
        # （4）活跃度指标（换手率）
        turnover_ratio = target_data["turnover_ratio"] if target_data["turnover_ratio"] is not None else 0
        
        # （5）竞价指标
        if len(stock_data[stock_data["date"] < latest_date]) >= 5:
            avg_vol_5d = stock_data[stock_data["date"] < latest_date]["volume"].tail(5).mean()
            auc_vol_ratio = target_data["auc_volume"] / avg_vol_5d if avg_vol_5d != 0 else 0
        else:
            auc_vol_ratio = 0
        buy_accept = 1 if (target_data["a1_p"] >= target_data["open"]) and (is_paused == 0) else 0
        
        # 收集数据
        rank_data.append({
            "stock_code": stock_code,
            "stock_name": stock_name,
            "是否停牌": is_paused,
            "5日均线斜率": round(ma5_slope, 3),
            "20日均线斜率": round(ma20_slope, 3),
            "价涨量增天数": sync_days,
            "近10天最大回撤(%)": round(max_drawdown, 2),
            "换手率(%)": round(turnover_ratio, 2),
            "竞价量比": round(auc_vol_ratio, 2),
            "开盘承接力": buy_accept
        })
    
    # 4. 计算综合评分（仅排序，不淘汰）
    rank_df = pd.DataFrame(rank_data)
    
    # 标准化指标（统一到0-100分）
    # 趋势得分（30%）：斜率越大越好
    rank_df["趋势得分"] = (
        (rank_df["5日均线斜率"] / (rank_df["5日均线斜率"].max() + 1e-8)) * 0.6 +
        (rank_df["20日均线斜率"] / (rank_df["20日均线斜率"].max() + 1e-8)) * 0.4
    ) * 30
    
    # 量能得分（25%）：天数越多越好
    rank_df["量能得分"] = (rank_df["价涨量增天数"] / 5) * 25
    
    # 风险得分（20%）：回撤越小越好
    rank_df["风险得分"] = (1 - rank_df["近10天最大回撤(%)"] / 100) * 20
    
    # 活跃度得分（10%）：换手率5%-15%最佳
    rank_df["活跃度得分"] = np.where(
        (rank_df["换手率(%)"] >= 5) & (rank_df["换手率(%)"] <= 15),
        10,
        np.where(
            (rank_df["换手率(%)"] < 5),
            rank_df["换手率(%)"] / 5 * 10,
            (20 - rank_df["换手率(%)"]) / 5 * 10
        )
    )
    
    # 竞价得分（15%）：量比越大+承接越好得分越高
    rank_df["竞价得分"] = (
        (rank_df["竞价量比"] / (rank_df["竞价量比"].max() + 1e-8)) * 10 +
        rank_df["开盘承接力"] * 5
    )
    
    # 总得分（停牌股直接扣30分）
    rank_df["总得分"] = (
        rank_df["趋势得分"] + rank_df["量能得分"] + rank_df["风险得分"] +
        rank_df["活跃度得分"] + rank_df["竞价得分"] -
        rank_df["是否停牌"] * 30
    ).round(2)
    
    # 5. 按总得分降序排序
    final_ranked_df = rank_df.sort_values("总得分", ascending=False).reset_index(drop=True)
    final_ranked_df["排名"] = final_ranked_df.index + 1
    
    # 6. 输出结果
    print(f"\n🎉 所有个股按养家心法评分排序完成：")
    print(final_ranked_df[
        ["排名", "stock_code", "stock_name", "是否停牌", "总得分", 
         "趋势得分", "竞价得分", "近10天最大回撤(%)"]
    ].to_string(index=False))
    
    # 保存完整排序结果
    final_ranked_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n排序结果保存至：{OUTPUT_PATH}")


# -------------------------- 执行排序 --------------------------
if __name__ == "__main__":
    yangjia_strategy_rank()

