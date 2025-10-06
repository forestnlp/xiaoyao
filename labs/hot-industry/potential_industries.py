#!/usr/bin/env python3
"""
读取 wide.parquet 为 df。

在 df 里计算：
- 当日收益 ret = (close - open) / open
- 隔日收益 ret_tomorrow = (close_tomorrow - open) / open，其中 close_tomorrow 为按股票分组后下一交易日的 close
- 竞价资金昨比 auction_money_ratio = auction_money / auction_money_yesterday，其中昨日值为按股票分组后上一交易日的 auction_money

最后统计最新交易日（当日）竞价昨比最大的 10 只股票，并打印结果。
"""

from pathlib import Path
import pandas as pd


def load_wide(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到宽表文件: {path}")
    df = pd.read_parquet(path)
    # 确保日期为可排序的时间戳
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # 基本字段检查
    required = ["stock_code", "date", "open", "close", "auction_money"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # 先按股票、日期排序，方便 shift 计算
    df = df.sort_values(["stock_code", "date"]).copy()

    # 当日收益 ret
    open_nonzero = df["open"].replace(0, pd.NA)
    df["ret"] = (df["close"] - df["open"]) / open_nonzero

    # 隔日收益 ret_tomorrow：组内下一交易日的 close
    df["close_tomorrow"] = df.groupby("stock_code")["close"].shift(-1)
    df["ret_tomorrow"] = (df["close_tomorrow"] - df["open"]) / open_nonzero

    # 竞价资金昨比：组内昨日的 auction_money
    df["auction_money_yesterday"] = df.groupby("stock_code")["auction_money"].shift(1)
    denom = df["auction_money_yesterday"].replace(0, pd.NA)
    df["auction_money_ratio"] = df["auction_money"] / denom
    return df


def top10_today(df: pd.DataFrame) -> pd.DataFrame:
    """返回最新有有效昨比数据的一日 Top 10（按 auction_money_ratio 降序）。"""
    if df["date"].isna().all():
        raise ValueError("日期列均为缺失，无法确定当日")
    latest_date = df["date"].max()
    today_df = df[df["date"] == latest_date].copy()
    today_df = today_df[pd.to_numeric(today_df["auction_money_ratio"], errors="coerce").notna()]
    if today_df.empty:
        valid_mask = pd.to_numeric(df["auction_money_ratio"], errors="coerce").notna()
        last_valid_date = df.loc[valid_mask, "date"].max()
        if pd.isna(last_valid_date):
            return today_df
        latest_date = last_valid_date
        today_df = df[df["date"] == latest_date].copy()
        today_df = today_df[pd.to_numeric(today_df["auction_money_ratio"], errors="coerce").notna()]
    cols = [
        "stock_code",
        "date",
        "auction_money_ratio",
        "auction_money",
        "auction_money_yesterday",
        "open",
        "close",
        "ret",
        "ret_tomorrow",
    ]
    cols = [c for c in cols if c in today_df.columns]
    top10 = today_df.sort_values("auction_money_ratio", ascending=False).head(10)
    return top10[cols]


def ratio_gt_one(df: pd.DataFrame) -> pd.DataFrame:
    """筛选竞价资金昨比>1的股票记录。"""
    if "auction_money_ratio" not in df.columns:
        raise ValueError("缺少列 auction_money_ratio，请先调用 compute_features。")
    return df[pd.to_numeric(df["auction_money_ratio"], errors="coerce") > 1].copy()


def daily_industry_counts(df: pd.DataFrame):
    """
    统计每日昨比>1的股票在不同申万行业层级上的数量：
    返回 (l1_counts, l2_counts, l3_counts)，各包含列：date, industry_name, count。
    并按每个日期内的 count 倒序排序。
    """
    required_cols = [
        "date",
        "sw_l1_industry_name",
        "sw_l2_industry_name",
        "sw_l3_industry_name",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要行业列: {missing}")

    sub = ratio_gt_one(df)

    def _count_by(col: str) -> pd.DataFrame:
        g = (
            sub.groupby(["date", col])
            .size()
            .reset_index(name="count")
            .sort_values(["date", "count"], ascending=[True, False])
        )
        g = g.rename(columns={col: "industry_name"})
        return g

    l1 = _count_by("sw_l1_industry_name")
    l2 = _count_by("sw_l2_industry_name")
    l3 = _count_by("sw_l3_industry_name")
    return l1, l2, l3


def most_frequent_industries_on(df: pd.DataFrame, date):
    """
    输入日期，返回该日昨比>1的股票中最频繁的三个行业层级名称：
    { 'l1': (name, count), 'l2': (name, count), 'l3': (name, count) }
    若该日无数据，则返回空字典。
    """
    # 统一日期类型
    date = pd.to_datetime(date)

    l1, l2, l3 = daily_industry_counts(df)
    result = {}

    def _top_one(g: pd.DataFrame):
        rows = g[g["date"] == date]
        if rows.empty:
            return None
        top = rows.sort_values("count", ascending=False).iloc[0]
        return str(top["industry_name"]), int(top["count"]) if not pd.isna(top["count"]) else 0

    t1 = _top_one(l1)
    t2 = _top_one(l2)
    t3 = _top_one(l3)
    if t1:
        result["l1"] = t1
    if t2:
        result["l2"] = t2
    if t3:
        result["l3"] = t3
    return result


def main():
    wide_path = Path(__file__).parent / "wide.parquet"
    df = load_wide(wide_path)
    df = compute_features(df)

    # Top10（最新存在有效昨比的日期）
    top10 = top10_today(df)
    if top10.empty:
        print("最新交易日以及回退日期均无有效竞价昨比数据。")
    else:
        chosen_date = top10["date"].iloc[0]
        print(f"竞价昨比 Top 10（日期: {chosen_date.date() if hasattr(chosen_date, 'date') else chosen_date}）：")
        print(top10.to_string(index=False))
    out_csv = Path(__file__).parent / "potential_industries_top10.csv"
    top10.to_csv(out_csv, index=False)
    print(f"已保存 Top10 到: {out_csv}")

    # 行业频次（昨比>1）
    latest_date = df["date"].max()
    l1, l2, l3 = daily_industry_counts(df)
    freq = most_frequent_industries_on(df, latest_date)
    print(f"\n昨比>1 最新日期行业Top（{latest_date.date() if hasattr(latest_date, 'date') else latest_date}）：")
    if freq:
        if "l1" in freq:
            print(f"  L1: {freq['l1'][0]} (count={freq['l1'][1]})")
        if "l2" in freq:
            print(f"  L2: {freq['l2'][0]} (count={freq['l2'][1]})")
        if "l3" in freq:
            print(f"  L3: {freq['l3'][0]} (count={freq['l3'][1]})")
    else:
        print("  当日无昨比>1的行业统计数据。")

    # 保存日度行业计数到CSV
    out_l1 = Path(__file__).parent / "ratio_gt1_daily_l1_counts.csv"
    out_l2 = Path(__file__).parent / "ratio_gt1_daily_l2_counts.csv"
    out_l3 = Path(__file__).parent / "ratio_gt1_daily_l3_counts.csv"
    l1.to_csv(out_l1, index=False)
    l2.to_csv(out_l2, index=False)
    l3.to_csv(out_l3, index=False)
    print(f"已保存行业日度计数到: {out_l1}, {out_l2}, {out_l3}")


if __name__ == "__main__":
    main()
