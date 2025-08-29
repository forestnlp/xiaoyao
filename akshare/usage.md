# AkShare 使用指南

## 1. 简介

AkShare 是一个完全免费、开源的 Python 金融数据接口库。它致力于为用户提供丰富、准确和及时的金融数据，是 Tushare 的一个强大替代品。

## 2. 核心特点

*   **完全免费**：无任何注册、登录或 Token 要求。
*   **数据全面**：覆盖中国和全球多个市场的股票、期货、基金、指数、债券、外汇和宏观经济数据。
*   **使用简单**：API 设计直观，返回 Pandas DataFrame 格式，方便进行数据分析。

## 3. 安装

```bash
pip install akshare
```

## 4. 常用示例

AkShare 的函数名通常很直观，直接描述了其获取的数据内容。

### 示例1：获取A股所有股票列表及实时行情

使用 `stock_zh_a_spot_em` 函数。

```python
import akshare as ak

stock_list_df = ak.stock_zh_a_spot_em()
print(stock_list_df)
```

### 示例2：获取单只股票的历史日线数据

使用 `stock_zh_a_hist` 函数。注意，返回的数据是**前复权**的。

```python
import akshare as ak

# 获取平安银行从2024年1月1日至今的日线数据
stock_daily_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20240101", adjust="qfq")
print(stock_daily_df)
```

*   `symbol`: 股票代码。
*   `period`: "daily", "weekly", "monthly".
*   `start_date`: 开始日期。
*   `adjust`: 复权选项, "" (不复权), "qfq" (前复权), "hfq" (后复权)。
