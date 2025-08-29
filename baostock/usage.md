# BaoStock 使用指南

## 1. 简介

BaoStock 是一个免费、开源的证券数据平台。它以提供稳定、高质量的 A 股历史数据而受到开发者的青睐，是进行量化回测的可靠数据来源。

## 2. 核心特点

*   **免费稳定**：数据服务免费，且有专业团队维护，数据质量和稳定性高。
*   **精确的复权数据**：提供准确的前复权和后复权历史 K 线数据，对量化回测至关重要。
*   **登录/登出模式**：使用时需要先登录 `login()`，所有操作完成后再登出 `logout()`，以释放资源。

## 3. 安装

```bash
pip install baostock
```

## 4. 常用示例

BaoStock 的核心使用流程是 `login -> query -> logout`。

### 示例1：登录和登出

```python
import baostock as bs

# 登录系统
lg = bs.login()
# 显示登录返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

# 此处执行数据查询操作

# 登出系统
bs.logout()
```

### 示例2：获取单只股票的历史日线数据

使用 `query_history_k_data_plus` 函数。

```python
# 省略登录代码...

# fields 中的字段可以按需选择
rs = bs.query_history_k_data_plus("sh.600519",
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
    start_date='2024-01-01', 
    frequency="d", 
    adjustflag="2") # 1:后复权, 2:前复权, 3:不复权

# data_list = []
# while (rs.error_code == '0') & rs.next():
#     data_list.append(rs.get_row_data())
# result = pd.DataFrame(data_list, columns=rs.fields)

# print(result)

# 省略登出代码...
```
**注意**：BaoStock 返回的是一个结果集（Result Set）对象，需要通过循环和 `rs.get_row_data()` 来提取数据，并手动组装成 Pandas DataFrame。
