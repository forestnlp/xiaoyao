# 导入所需的包
#!/usr/bin/env python3
"""
读取数据字典 JSON，识别四个数据源（价格、竞价、行业、市值），
按 stock_code+date 进行表连接，处理重名列后生成宽表并保存在当前目录。

内置配置，不使用命令行参数：
- 数据字典：labs/data-dictionary/data_dictionary_llm.json
- 输入目录：data
- 输出文件：labs/hot-industry/wide.parquet
- 连接类型：left（以价格为基表）

字段重命名策略：
- 读取数据字典统计跨数据源重复列（排除连接键）。
- 对重复列按数据源前缀重命名：
  price_: stock_daily_price
  auction_: stock_daily_auction
  industry_: stock_daily_industry
  marketcap_: stock_daily_marketcap
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


JOIN_KEYS = ["stock_code", "date"]


def load_dictionary(dict_path: Path) -> List[Dict]:
    data = json.loads(dict_path.read_text(encoding="utf-8"))
    return data.get("data_sources", [])


def prefix_for(file_key: str) -> str:
    mapping = {
        "stock_daily_price": "price",
        "stock_daily_auction": "auction",
        "stock_daily_industry": "industry",
        "stock_daily_marketcap": "marketcap",
    }
    return mapping.get(file_key, file_key)


def compute_renames(sources: List[Dict]) -> Dict[str, Dict[str, str]]:
    name_to_files: Dict[str, List[str]] = {}
    for src in sources:
        file_key = src.get("file")
        for col in src.get("columns", []):
            name = col.get("name")
            if not name or name in JOIN_KEYS:
                continue
            name_to_files.setdefault(name, []).append(file_key)

    duplicates = {name for name, files in name_to_files.items() if len(files) > 1}

    renames: Dict[str, Dict[str, str]] = {}
    for src in sources:
        file_key = src.get("file")
        pf = prefix_for(file_key)
        m: Dict[str, str] = {}
        for col in src.get("columns", []):
            name = col.get("name")
            if not name or name in JOIN_KEYS:
                continue
            if name in duplicates:
                m[name] = f"{pf}_{name}"
        if m:
            renames[file_key] = m
    return renames


def read_parquet(input_dir: Path, file_name: str) -> pd.DataFrame:
    p = input_dir / file_name
    if not p.exists():
        raise FileNotFoundError(f"找不到输入文件: {p}")
    df = pd.read_parquet(p)
    # 规范日期类型
    if "date" in df.columns:
        # 转换为仅日期（去除时分秒），确保各源连接键一致
        try:
            dt = pd.to_datetime(df["date"], errors="coerce")
            # 仅保留日期部分，统一类型为 Python date 对象
            df["date"] = dt.dt.date
        except Exception:
            pass
    return df


def select_and_rename(df: pd.DataFrame, file_key: str, renames: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    # 保持原列顺序，不打乱
    cols = list(df.columns)
    sub = df[cols].copy()
    mapping = renames.get(file_key, {})
    if mapping:
        sub = sub.rename(columns=mapping)
    return sub


def merge_sources(sources: List[Dict], input_dir: Path, join_type: str) -> pd.DataFrame:
    renames = compute_renames(sources)
    col_orders: Dict[str, List[str]] = {}

    # 选择价格为基表
    base_src = next((s for s in sources if s.get("file") == "stock_daily_price"), sources[0])
    base_key = base_src.get("file")
    base_df = read_parquet(input_dir, base_src.get("file_name"))
    base_df = select_and_rename(base_df, base_key, renames)
    col_orders[base_key] = [c for c in base_df.columns if c not in JOIN_KEYS]

    for src in sources:
        if src.get("file") == base_key:
            continue
        df_i = read_parquet(input_dir, src.get("file_name"))
        df_i = select_and_rename(df_i, src.get("file"), renames)
        col_orders[src.get("file")] = [c for c in df_i.columns if c not in JOIN_KEYS]
        base_df = base_df.merge(df_i, on=JOIN_KEYS, how=join_type)

    # 输出列顺序：先键（date, stock_code），再按各源原始列顺序（基表优先，其余按数据字典顺序）
    final_order: List[str] = []
    # 键置前两列
    for k in ["date", "stock_code"]:
        if k in base_df.columns:
            final_order.append(k)
    # 依次追加各源的非键列，保持原顺序并去重
    join_sequence = [base_key] + [s.get("file") for s in sources if s.get("file") != base_key]
    for fk in join_sequence:
        for c in col_orders.get(fk, []):
            if c in base_df.columns and c not in JOIN_KEYS and c not in final_order:
                final_order.append(c)

    return base_df[final_order]


def main():
    dict_path = Path("labs/data-dictionary/data_dictionary_llm.json")
    input_dir = Path("data")
    out_path = Path(__file__).parent / "wide.parquet"
    join_type = "left"

    if not dict_path.exists():
        raise FileNotFoundError(f"未找到数据字典: {dict_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"未找到输入目录: {input_dir}")

    sources = load_dictionary(dict_path)
    target_files = {
        "stock_daily_price",
        "stock_daily_auction",
        "stock_daily_industry",
        "stock_daily_marketcap",
    }
    sources = [s for s in sources if s.get("file") in target_files]
    for s in sources:
        if not s.get("file_name"):
            raise ValueError(f"数据源缺少文件名: {s}")

    wide = merge_sources(sources, input_dir, join_type)

    # 列顺序已在 merge_sources 中对齐为：date、stock_code、各源原顺序

    # 诊断：a/b 系列竞价字段缺失情况
    ab_cols = [c for c in wide.columns if c.startswith("a") or c.startswith("b")]
    ab_cols = [c for c in ab_cols if ("_p" in c or "_v" in c)]
    if ab_cols:
        print("竞价字段缺失率：")
        for c in sorted(ab_cols):
            null_rate = float(wide[c].isna().mean())
            print(f"  {c}: 缺失率 {null_rate:.4%}")
        # 全部竞价字段缺失的行数（但价格存在）
        all_null_mask = wide[ab_cols].isna().all(axis=1)
        print(f"全部竞价字段缺失的行数: {int(all_null_mask.sum())}")
        # 展示这些行的示例键
        sample_keys = wide.loc[all_null_mask, ["date", "stock_code"]].head(10)
        if not sample_keys.empty:
            print("缺失示例前10行键:")
            print(sample_keys.to_string(index=False))

    wide.to_parquet(out_path)

    print(f"宽表已生成: {out_path}")
    print(f"行数: {len(wide)}, 列数: {wide.shape[1]}")
    print("前20列:", list(wide.columns)[:20])


if __name__ == "__main__":
    main()