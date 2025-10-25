# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\works\trytry\养家心法\选股.ipynb



# ----------------------------------------------------------------------import pandas as pd
import numpy as np
import os
from datetime import datetime

# --------------------------
# 1. 配置参数（与因子模块联动）
# --------------------------
CONFIG = {
    "factor_input_path": r'./yangjia_factor_data.parquet',  # 上一步生成的养家因子数据
    "selection_output_path": r'./yangjia_selection_result.csv',  # 选股结果保存路径
    "daily_result_dir": r'./yangjia_daily_selection',  # 每日选股结果目录
    "log_path": r'./yangjia_selection_log.txt',  # 日志路径
    # 养家选股核心阈值（可按市场调整）
    "selection_thresholds": {
        "market_strength_min": 60,    # 市场强弱≥60分（可操作）
        "industry_strength_min": 70,  # 板块强弱≥70分（强势板块）
        "leader_score_min": 60,       # 龙头得分≥60分（有龙头潜质）
        "top_n": 20                   # 每日选前20只龙头（聚焦不分散）
    },
    # 额外过滤条件（贴合养家“资金安全”）
    "extra_filters": {
        "avoid_zt_open": True,        # 避免涨停开盘（防站岗）
        "max_daily_rise": 0.08        # 单日涨幅≤8%（防过度炒作）
    }
}

# --------------------------
# 2. 工具函数（保持风格一致）
# --------------------------
def init_environment():
    """创建每日结果目录+初始化日志"""
    os.makedirs(CONFIG["daily_result_dir"], exist_ok=True)
    with open(CONFIG["log_path"], 'w', encoding='utf-8') as f:
        f.write(f"【养家心法选股启动】{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_msg(f"✅ 选股环境初始化完成，每日结果目录：{CONFIG['daily_result_dir']}")

def log_msg(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {msg}"
    print(log_line)
    with open(CONFIG["log_path"], 'a', encoding='utf-8') as f:
        f.write(log_line + "\n")

# --------------------------
# 3. 加载因子数据（与上一步联动）
# --------------------------
def load_factor_data():
    log_msg("开始加载养家因子数据...")
    # 读取因子数据，保留选股必需字段
    factor_df = pd.read_parquet(CONFIG["factor_input_path"])
    factor_df["date"] = pd.to_datetime(factor_df["date"])
    # 按股票+日期排序，确保数据连续性
    factor_df = factor_df.sort_values(by=["date", "stock_code"]).reset_index(drop=True)
    
    # 验证核心因子字段（避免缺失）
    required_factor_cols = [
        'date', 'stock_code', 'stock_name', 'close', 'open',
        'market_strength_score', 'sw_l1_industry_name', 'industry_strength_score',
        'leader_total_score', 'consecutive_up_days', 'is_ma_bull'
    ]
    missing_cols = [col for col in required_factor_cols if col not in factor_df.columns]
    if missing_cols:
        raise ValueError(f"因子数据缺少必需字段：{missing_cols}")
    
    log_msg(f"✅ 因子数据加载完成：{len(factor_df)}条记录，{factor_df['date'].nunique()}个交易日")
    return factor_df

# --------------------------
# 4. 单日选股逻辑（核心：三步筛选）
# --------------------------
def select_single_day(daily_df, date):
    """单日选股：市场判断→板块筛选→龙头排序"""
    thresholds = CONFIG["selection_thresholds"]
    extra_filters = CONFIG["extra_filters"]
    daily_df = daily_df.copy()
    
    # --------------------------
    # 步骤1：市场强弱判断（养家“弱市不做”）
    # --------------------------
    # 取当日市场强弱分（所有股票当日市场分一致，取第一个值即可）
    daily_market_score = daily_df['market_strength_score'].iloc[0] if len(daily_df) > 0 else 0
    if daily_market_score < thresholds["market_strength_min"]:
        log_msg(f"📉 当日市场弱（评分：{daily_market_score:.1f}＜{thresholds['market_strength_min']}），空仓")
        return pd.DataFrame()
    log_msg(f"📈 当日市场强（评分：{daily_market_score:.1f}≥{thresholds['market_strength_min']}），进入选股")
    
    # --------------------------
    # 步骤2：筛选强势板块（养家“板块强势提供高成功率”）
    # --------------------------
    # 先获取当日强势板块列表（行业评分≥阈值）
    strong_industries = daily_df[
        daily_df['industry_strength_score'] >= thresholds["industry_strength_min"]
    ]['sw_l1_industry_name'].unique()
    if len(strong_industries) == 0:
        log_msg(f"⚠️ 当日无强势板块（行业评分≥{thresholds['industry_strength_min']}），空仓")
        return pd.DataFrame()
    log_msg(f"🔥 当日强势板块（{len(strong_industries)}个）：{', '.join(strong_industries[:3])}...")
    
    # 仅保留强势板块的股票
    daily_df = daily_df[daily_df['sw_l1_industry_name'].isin(strong_industries)].copy()
    
    # --------------------------
    # 步骤3：筛选龙头股（养家“龙头为王”）
    # --------------------------
    # 基础过滤：排除风险标的（贴合“资金安全”）
    # 1. 避免涨停开盘（防追高站岗）
    if extra_filters["avoid_zt_open"]:
        daily_df = daily_df[~((daily_df['open'] / daily_df['close'].shift(1) - 1) >= 0.095)].copy()
    # 2. 单日涨幅不过度（防炒作过度）
    daily_df['daily_rise'] = (daily_df['close'] / daily_df['open'] - 1)
    daily_df = daily_df[daily_df['daily_rise'] <= extra_filters["max_daily_rise"]].copy()
    # 3. 确保均线多头（趋势未破）
    daily_df = daily_df[daily_df['is_ma_bull'] == 1].copy()
    
    # 龙头排序：按龙头综合得分降序，取前N只
    daily_df = daily_df[daily_df['leader_total_score'] >= thresholds["leader_score_min"]].copy()
    if len(daily_df) == 0:
        log_msg(f"⚠️ 强势板块内无符合条件龙头（得分≥{thresholds['leader_score_min']}），空仓")
        return pd.DataFrame()
    
    # 按龙头得分排序，取前top_n
    selected_df = daily_df.sort_values(by='leader_total_score', ascending=False).head(thresholds["top_n"]).copy()
    
    # --------------------------
    # 步骤4：整理选股结果（保留关键字段）
    # --------------------------
    result_cols = [
        'date', 'stock_code', 'stock_name', 'sw_l1_industry_name',
        'close', 'open', 'volume', 'daily_rise',
        'market_strength_score', 'industry_strength_score', 'leader_total_score',
        'consecutive_up_days', 'is_ma_bull'
    ]
    selected_df = selected_df[result_cols].reset_index(drop=True)
    # 添加选股日期标签
    selected_df['selection_date'] = date.strftime('%Y-%m-%d')
    log_msg(f"✅ 当日选股完成：{len(selected_df)}只龙头股（来自{len(strong_industries)}个强势板块）")
    
    return selected_df

# --------------------------
# 5. 全周期选股（遍历所有交易日）
# --------------------------
def run_yangjia_selection(factor_df):
    log_msg("开始全周期养家选股...")
    # 获取所有交易日（按时间排序）
    trade_dates = sorted(factor_df['date'].dt.date.unique())
    all_selection_results = []
    
    for trade_date in trade_dates:
        log_msg(f"\n" + "="*50)
        log_msg(f"📅 处理交易日：{trade_date.strftime('%Y-%m-%d')}")
        log_msg("="*50)
        
        # 提取当日因子数据
        daily_factor_df = factor_df[factor_df['date'].dt.date == trade_date].copy()
        if len(daily_factor_df) < 500:  # 当日数据过少（如节假日后），跳过
            log_msg(f"⚠️ 当日数据异常（记录数{len(daily_factor_df)}<500），跳过")
            continue
        
        # 执行单日选股
        daily_selection = select_single_day(daily_factor_df, trade_date)
        if not daily_selection.empty:
            # 保存当日选股结果
            date_str = trade_date.strftime('%Y%m%d')
            daily_save_path = os.path.join(CONFIG["daily_result_dir"], f"yangjia_selection_{date_str}.csv")
            daily_selection.to_csv(daily_save_path, index=False, encoding='utf-8-sig')
            # 收集全周期结果
            all_selection_results.append(daily_selection)
    
    # --------------------------
    # 保存全周期选股结果
    # --------------------------
    if all_selection_results:
        final_selection = pd.concat(all_selection_results, ignore_index=True)
        final_selection.to_csv(CONFIG["selection_output_path"], index=False, encoding='utf-8-sig')
        # 统计核心信息
        total_days = len(trade_dates)
        trading_days = len(all_selection_results)
        avg_daily_selection = len(final_selection) / trading_days if trading_days > 0 else 0
        log_msg(f"\n" + "="*60)
        log_msg(f"🎉 全周期养家选股完成！核心统计：")
        log_msg(f"📊 总交易日：{total_days}天 | 可操作天数（市场强+有龙头）：{trading_days}天")
        log_msg(f"📈 累计选股：{len(final_selection)}条记录 | 平均每日选股：{avg_daily_selection:.1f}只")
        log_msg(f"📁 全周期结果路径：{CONFIG['selection_output_path']}")
        log_msg(f"📁 每日结果目录：{CONFIG['daily_result_dir']}")
        log_msg("="*60)
        return final_selection
    else:
        log_msg(f"\n⚠️ 全周期无选股结果（可能市场整体偏弱或无符合条件龙头）")
        return pd.DataFrame()

# --------------------------
# 主函数：选股全流程
# --------------------------
def main_yangjia_selection():
    try:
        init_environment()
        factor_df = load_factor_data()
        selection_result = run_yangjia_selection(factor_df)
        return selection_result
    except Exception as e:
        log_msg(f"❌ 选股失败：{str(e)}")
        raise

# 执行选股（Jupyter中运行）
if __name__ == "__main__":
    yangjia_selection_data = main_yangjia_selection()