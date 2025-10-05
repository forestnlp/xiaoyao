#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
import numpy as np
import pandas as pd


def build_dataset(
    source_parquet: str = r"D:\workspace\xiaoyao\data\industry_indicators.parquet",
    output_parquet: str = r"D:\workspace\xiaoyao\labs\industry_ml\data\industry_ml_dataset.parquet",
):
    # 读取行业日度指标
    df = pd.read_parquet(source_parquet)

    # 基础清洗与排序
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.sort_values(['industry_name', 'date']).reset_index(drop=True)

    # 构造 t+1 标签：下一日上涨比例是否大于 50%
    df['up_ratio_next'] = df.groupby('industry_name')['up_ratio'].shift(-1)
    df['y_next_up_gt50'] = (df['up_ratio_next'] > 0.5).astype('Int64')

    # 三分类标签：下一日上涨比例 >70% 为 1，<30% 为 -1，其余为 0
    def classify_ratio(r: float) -> int:
        try:
            if r > 0.7:
                return 1
            elif r < 0.3:
                return -1
            else:
                return 0
        except Exception:
            return np.nan

    df['y_next_up_ratio_cls'] = df['up_ratio_next'].apply(classify_ratio).astype('Int64')

    # 去除小样本行业-日（降低标签噪声）
    df = df[df['stock_count'] >= 10].copy()

    # 增强特征工程：滚动均值/标准差/日变动/z-score（以行业内时序为单位）
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    base_cols = ['avg_au_vol_ratio_prev', 'up_ratio', 'avg_daily_return', 'limit_up_ratio']
    windows = [3, 5, 10]
    for col in base_cols:
        # 分组保证同一行业内滚动
        data = data.sort_values(['industry_name', 'date'])
        # 一阶差分
        data[f'{col}_diff_1'] = data.groupby('industry_name')[col].transform(lambda s: s.diff(1))
        for n in windows:
            roll_mean = data.groupby('industry_name')[col].transform(lambda s: s.rolling(n).mean())
            roll_std = data.groupby('industry_name')[col].transform(lambda s: s.rolling(n).std())
            data[f'{col}_roll_mean_{n}'] = roll_mean
            data[f'{col}_roll_std_{n}'] = roll_std
            eps = 1e-8
            data[f'{col}_z_{n}'] = (data[col] - roll_mean) / (roll_std + eps)

    # 添加简单的时间特征
    data['day_of_week'] = data['date'].dt.weekday  # 0-6

    # Winsorize：对主要比例类特征按列剪裁极值(1%, 99%)
    clip_cols = [
        'avg_au_vol_ratio_prev', 'up_ratio', 'limit_up_ratio', 'limit_down_ratio',
        'volume_ratio', 'money_ratio', 'avg_daily_return', 'med_daily_return', 'up_down_ratio'
    ]
    for col in clip_cols:
        if col in data.columns:
            q01 = data[col].quantile(0.01)
            q99 = data[col].quantile(0.99)
            data[col] = data[col].clip(q01, q99)

    # 选择特征列（包含竞价量比及其派生特征）
    feature_cols = [
        'avg_au_vol_ratio_prev',  # 竞价量/昨日成交量
        'up_ratio',
        'limit_up_ratio',
        'limit_down_ratio',
        'volume_ratio',
        'money_ratio',
        'avg_daily_return',
        'med_daily_return',
        'up_down_ratio',
        'net_up_count',
        'stock_count',
        'day_of_week',
    ]
    # 动态加入新特征（存在即加入）
    for col in base_cols:
        feature_cols.append(f'{col}_diff_1')
        for n in windows:
            feature_cols.extend([f'{col}_roll_mean_{n}', f'{col}_roll_std_{n}', f'{col}_z_{n}'])
    feature_cols = [c for c in feature_cols if c in data.columns]

    # 保留必要列
    keep_cols = ['date', 'industry_name', 'y_next_up_gt50', 'y_next_up_ratio_cls'] + feature_cols
    data = data[keep_cols].copy()

    # 去除无法构造标签的最后一天样本（各行业最后一日没有 next）
    data = data.dropna(subset=['y_next_up_ratio_cls'])

    # 处理无穷与缺失
    for col in feature_cols:
        if col in data:
            # 将无穷替换为 NaN 后再用中位数填充
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
    data[feature_cols] = data[feature_cols].fillna(data[feature_cols].median())

    # 创建输出目录
    out_path = Path(output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存
    data.to_parquet(out_path.as_posix(), index=False)
    print(f"构建完成：{len(data)} 条样本，{len(feature_cols)} 个特征")
    print(f"已保存数据集至：{out_path}")


if __name__ == '__main__':
    build_dataset()