#!/usr/bin/env python
# coding: utf-8

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
import joblib


DATA_PARQUET = r"D:\workspace\xiaoyao\labs\industry_ml\data\industry_ml_dataset.parquet"
MODEL_PATH = r"D:\workspace\xiaoyao\labs\industry_ml\models\logreg_industry.pkl"
FEATURES_JSON = r"D:\workspace\xiaoyao\labs\industry_ml\models\features.json"
META_JSON = r"D:\workspace\xiaoyao\labs\industry_ml\models\meta.json"
OUTPUT_CSV = r"D:\workspace\xiaoyao\labs\industry_ml\output\predictions.csv"


def time_split(df: pd.DataFrame, test_ratio: float = 0.2):
    dates = sorted(df['date'].unique())
    split_idx = int((1 - test_ratio) * len(dates))
    test_dates = set(dates[split_idx:])
    test_df = df[df['date'].isin(test_dates)].copy()
    return test_df


def main():
    df = pd.read_parquet(DATA_PARQUET)
    df['date'] = pd.to_datetime(df['date']).dt.date

    with open(FEATURES_JSON, 'r', encoding='utf-8') as f:
        meta_f = json.load(f)
    feature_cols = meta_f['feature_cols']

    # 载入模型与阈值
    model = joblib.load(MODEL_PATH)
    t_pos, t_neg = 0.5, 0.5
    label_col = 'y_next_up_ratio_cls'
    try:
        with open(META_JSON, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            t_pos = float(meta.get('t_pos', 0.5))
            t_neg = float(meta.get('t_neg', 0.5))
            label_col = meta.get('label_col', label_col)
    except Exception:
        pass

    test_df = time_split(df, test_ratio=0.2)
    X_test = test_df[feature_cols].values
    y_test = test_df[label_col].astype(int).values

    proba = model.predict_proba(X_test)
    classes = model.named_steps['clf'].classes_
    idx_neg = int(np.where(classes == -1)[0][0])
    idx_zero = int(np.where(classes == 0)[0][0])
    idx_pos = int(np.where(classes == 1)[0][0])
    p_neg = proba[:, idx_neg]
    p_zero = proba[:, idx_zero]
    p_pos = proba[:, idx_pos]
    y_pred = np.zeros(len(proba), dtype=int)
    mask_pos = (p_pos >= t_pos) & (p_pos >= p_zero) & (p_pos >= p_neg)
    mask_neg = (p_neg >= t_neg) & (p_neg >= p_zero) & (p_neg >= p_pos)
    y_pred[mask_pos] = 1
    y_pred[mask_neg] = -1

    acc = accuracy_score(y_test, y_pred)
    # 极端类别macro F1
    y_true_pos = (y_test == 1).astype(int)
    y_pred_pos = (y_pred == 1).astype(int)
    y_true_neg = (y_test == -1).astype(int)
    y_pred_neg = (y_pred == -1).astype(int)
    f1_pos = f1_score(y_true_pos, y_pred_pos, zero_division=0)
    f1_neg = f1_score(y_true_neg, y_pred_neg, zero_division=0)
    extreme_macro_f1 = 0.5 * (f1_pos + f1_neg)
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])

    print(f"测试集：accuracy={acc:.4f} extreme_macro_f1={extreme_macro_f1:.4f} (t_pos={t_pos:.2f}, t_neg={t_neg:.2f})")
    print("分类报告：")
    print(classification_report(y_test, y_pred, labels=[-1, 0, 1], digits=4))
    print("混淆矩阵：")
    print(cm)

    # 保存预测明细
    out_dir = Path(OUTPUT_CSV).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = test_df[['date', 'industry_name']].copy()
    out['y_true'] = y_test
    out['p_neg'] = p_neg
    out['p_zero'] = p_zero
    out['p_pos'] = p_pos
    out['y_pred'] = y_pred
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"预测结果已保存：{OUTPUT_CSV}")


if __name__ == '__main__':
    main()