# 项目进度报告 (PROGRESS.md)

本文档旨在记录和追踪 `xiaoyao` 量化项目的各个子模块的开发进度。

## 1. 数据源模块

数据源模块负责从不同的财经数据平台获取原始数据，是整个项目的基础。

### 1.1. `akshare`
- **状态**: `已完成`
- **功能**:
    - `akshare_demo.py` 脚本已实现以下功能：
        - 获取所有A股的实时行情列表 (`stock_zh_a_spot_em`)。
        - 获取单只股票的历史日线行情（前复权, `stock_zh_a_hist`）。
- **文档**: `usage.md` 提供了 `akshare` 的基本介绍、安装和常用示例。

### 1.2. `baostock`
- **状态**: `已完成`
- **功能**:
    - `baostock_demo.py` 脚本已实现以下功能：
        - 封装了 `login` 和 `logout` 流程。
        - 获取指定日期的所有A股列表 (`query_all_stock`)。
        - 获取单只股票的历史日线行情（前复权, `query_history_k_data_plus`）。
- **文档**: `usage.md` 提供了 `baostock` 的介绍、特点、安装和核心使用流程。

### 1.3. `tushare`
- **状态**: `已完成`
- **功能**:
    - `tushare_demo.py` 脚本已实现以下功能：
        - 从外部 `config.ini` 文件安全读取并初始化 API Token。
        - 获取基础的股票列表 (`stock_basic`)。
        - 获取单只股票的日线行情数据 (`daily`)。
        - 获取单只股票的主要财务指标 (`fina_indicator`)。
- **文档**: `tushare_usage.md` 提供了 `Tushare Pro` 的介绍、收费模式、Token获取方式和基本用法。

## 2. 技术分析模块

本模块专注于使用 `TA-Lib` 库进行各种技术指标的计算和分析。

### 2.1. `talib_learning`
- **状态**: `已完成`
- **功能**:
    - `z_talib_basic_intro.py`: 演示了如何基于模拟数据计算 SMA, RSI, MACD 等核心指标，是入门学习脚本。
    - `volume_indicators_demo.py`: 演示了如何获取真实股票数据，并计算交易量相关的指标，如 OBV, AD, MFI。
    - `stock_analyzer.py`: 一个更高级的综合性脚本，实现了：
        - 封装了数据获取、因子计算的函数。
        - 对单只股票进行批量技术分析。
        - 基于多个指标（SMA, RSI, MACD等）进行简单的多空信号解读和综合评估。
- **文档**: `usage.md` 详细说明了 `TA-Lib` 库的核心概念（需要输入NumPy数组）和常用函数示例。

---
*报告生成时间: 2025年9月2日*
