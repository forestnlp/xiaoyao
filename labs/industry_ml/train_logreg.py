#!/usr/bin/env python
# coding: utf-8

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
import joblib


DATA_PARQUET = r"D:\workspace\xiaoyao\labs\industry_ml\data\industry_ml_dataset.parquet"
MODEL_PATH = r"D:\workspace\xiaoyao\labs\industry_ml\models\logreg_industry.pkl"
FEATURES_JSON = r"D:\workspace\xiaoyao\labs\industry_ml\models\features.json"
META_JSON = r"D:\workspace\xiaoyao\labs\industry_ml\models\meta.json"


def time_split(df: pd.DataFrame, test_ratio: float = 0.2):
    dates = sorted(df['date'].unique())
    split_idx = int((1 - test_ratio) * len(dates))
    train_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])
    train_df = df[df['date'].isin(train_dates)].copy()
    test_df = df[df['date'].isin(test_dates)].copy()
    return train_df, test_df


def split_train_val_by_time(train_df: pd.DataFrame, val_ratio: float = 0.1):
    dates = sorted(train_df['date'].unique())
    split_idx = int((1 - val_ratio) * len(dates))
    tr_dates = set(dates[:split_idx])
    val_dates = set(dates[split_idx:])
    tr = train_df[train_df['date'].isin(tr_dates)].copy()
    val = train_df[train_df['date'].isin(val_dates)].copy()
    return tr, val


def select_features(df: pd.DataFrame) -> list:
    base_features = [
        'avg_au_vol_ratio_prev',
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
    dyn_cols = []
    for col in ['avg_au_vol_ratio_prev', 'up_ratio', 'avg_daily_return', 'limit_up_ratio']:
        dyn_cols.append(f'{col}_diff_1')
        for n in [3, 5, 10]:
            dyn_cols.extend([f'{col}_roll_mean_{n}', f'{col}_roll_std_{n}', f'{col}_z_{n}'])
    feature_cols = [c for c in base_features + dyn_cols if c in df.columns]
    return feature_cols


def make_predictions_with_thresholds(proba: np.ndarray, classes: np.ndarray, t_pos: float, t_neg: float) -> np.ndarray:
    # classes expected order like [-1, 0, 1]
    idx_neg = int(np.where(classes == -1)[0][0])
    idx_zero = int(np.where(classes == 0)[0][0])
    idx_pos = int(np.where(classes == 1)[0][0])
    p_neg = proba[:, idx_neg]
    p_zero = proba[:, idx_zero]
    p_pos = proba[:, idx_pos]
    y_pred = np.zeros(len(proba), dtype=int)
    # rule: predict extreme only if its prob >= threshold and is the max; otherwise 0
    mask_pos = (p_pos >= t_pos) & (p_pos >= p_zero) & (p_pos >= p_neg)
    mask_neg = (p_neg >= t_neg) & (p_neg >= p_zero) & (p_neg >= p_pos)
    y_pred[mask_pos] = 1
    y_pred[mask_neg] = -1
    return y_pred


def extreme_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Compute F1 for classes -1 and 1, then macro average
    y_true_pos = (y_true == 1).astype(int)
    y_pred_pos = (y_pred == 1).astype(int)
    y_true_neg = (y_true == -1).astype(int)
    y_pred_neg = (y_pred == -1).astype(int)
    f1_pos = f1_score(y_true_pos, y_pred_pos, zero_division=0)
    f1_neg = f1_score(y_true_neg, y_pred_neg, zero_division=0)
    return 0.5 * (f1_pos + f1_neg)


def main():
    df = pd.read_parquet(DATA_PARQUET)
    df['date'] = pd.to_datetime(df['date']).dt.date

    # 使用三分类标签
    label_col = 'y_next_up_ratio_cls'

    feature_cols = select_features(df)

    # 时间切分
    train_df, test_df = time_split(df, test_ratio=0.2)
    tr_df, val_df = split_train_val_by_time(train_df, val_ratio=0.1)

    X_train = tr_df[feature_cols].values
    y_train = tr_df[label_col].astype(int).values
    X_val = val_df[feature_cols].values
    y_val = val_df[label_col].astype(int).values
    X_test = test_df[feature_cols].values
    y_test = test_df[label_col].astype(int).values

    # 根据类别分布设置类权重，并加大极端类别权重
    vc = tr_df[label_col].value_counts().to_dict()
    # 基于频率的反比权重
    total = sum(vc.values())
    base_weights = {k: total / (3 * vc.get(k, 1)) for k in [-1, 0, 1]}
    # 提升极端类别权重
    boost = 1.5
    class_weight = {
        -1: base_weights.get(-1, 1.0) * boost,
        0: base_weights.get(0, 1.0),
        1: base_weights.get(1, 1.0) * boost,
    }

    pipe_template = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            multi_class='multinomial',
            class_weight=class_weight,
            C=1.0,
            random_state=42,
        ))
    ])

    # 样本权重：用行业覆盖数量增强稳定性
    sample_weight_train = np.sqrt(np.clip(tr_df['stock_count'].values, 1, None))

    # 网格搜索：C、t_pos、t_neg 以极端类别的 macro F1 为目标
    C_grid = [0.1, 0.5, 1.0, 2.0]
    t_grid = [0.45, 0.50, 0.55, 0.60]
    best = {
        'score': -np.inf,
        'C': None,
        't_pos': 0.5,
        't_neg': 0.5,
    }
    for C in C_grid:
        pipe = pipe_template.set_params(clf__C=C)
        pipe.fit(X_train, y_train, clf__sample_weight=sample_weight_train)
        proba_val = pipe.predict_proba(X_val)
        classes = pipe.named_steps['clf'].classes_
        for t_pos in t_grid:
            for t_neg in t_grid:
                y_val_pred = make_predictions_with_thresholds(proba_val, classes, t_pos, t_neg)
                score = extreme_macro_f1(y_val, y_val_pred)
                if score > best['score']:
                    best.update({'score': float(score), 'C': C, 't_pos': float(t_pos), 't_neg': float(t_neg)})

    # 使用最佳 C 在完整训练集重训
    X_train_full = train_df[feature_cols].values
    y_train_full = train_df[label_col].astype(int).values
    sample_weight_train_full = np.sqrt(np.clip(train_df['stock_count'].values, 1, None))
    pipe = pipe_template.set_params(clf__C=best['C'])
    pipe.fit(X_train_full, y_train_full, clf__sample_weight=sample_weight_train_full)

    # 测试集评估（含极端类别表现）
    proba_test = pipe.predict_proba(X_test)
    classes = pipe.named_steps['clf'].classes_
    y_pred_test = make_predictions_with_thresholds(proba_test, classes, best['t_pos'], best['t_neg'])

    acc = accuracy_score(y_test, y_pred_test)
    extreme_f1 = extreme_macro_f1(y_test, y_pred_test)
    print(f"测试集：accuracy={acc:.4f} extreme_macro_f1={extreme_f1:.4f} (C={best['C']}, t_pos={best['t_pos']:.2f}, t_neg={best['t_neg']:.2f}, val_extreme_macro_f1={best['score']:.4f})")
    print(classification_report(y_test, y_pred_test, labels=[-1, 0, 1], digits=4))

    # 保存模型与特征列、阈值/调参元数据
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    with open(FEATURES_JSON, 'w', encoding='utf-8') as f:
        json.dump({'feature_cols': feature_cols}, f, ensure_ascii=False, indent=2)
    with open(META_JSON, 'w', encoding='utf-8') as f:
        json.dump({
            'best_C': best['C'],
            't_pos': best['t_pos'],
            't_neg': best['t_neg'],
            'val_extreme_macro_f1': best['score'],
            'class_weight': class_weight,
            'label_col': label_col,
            'classes': [-1, 0, 1],
        }, f, ensure_ascii=False, indent=2)
    print(f"模型已保存：{MODEL_PATH}")
    print(f"特征列已保存：{FEATURES_JSON}")
    print(f"调参与阈值已保存：{META_JSON}")


if __name__ == '__main__':
    main()