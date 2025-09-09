# BaoStock 使用指南

## 1. 简介

BaoStock（证券宝）是一个免费、开源的证券数据平台（无需注册）<mcreference link="http://baostock.com/baostock/index.php/Python_API文档" index="1">1</mcreference>。它以提供稳定、高质量的 A 股历史数据而受到开发者的青睐，是进行量化回测的可靠数据来源。平台提供大量准确、完整的证券历史行情数据、上市公司财务数据等<mcreference link="https://blog.csdn.net/weixin_46277779/article/details/129821907" index="3">3</mcreference>。

## 2. 核心特点

*   **完全免费**：数据服务完全免费，无需注册或积分，且有专业团队维护<mcreference link="https://blog.csdn.net/m0_46603114/article/details/107869037" index="2">2</mcreference>
*   **数据覆盖全面**：涵盖股票、指数、基金、期货等多个金融领域<mcreference link="https://blog.csdn.net/weixin_45525272/article/details/135346948" index="3">3</mcreference>
*   **精确的复权数据**：提供准确的前复权和后复权历史 K 线数据，支持涨跌幅复权算法<mcreference link="https://klang.org.cn/docs/stockdata/" index="2">2</mcreference>
*   **多时间周期**：支持日线、周线、月线以及5分钟、15分钟、30分钟、60分钟K线数据<mcreference link="http://baostock.com/baostock/index.php/Python_API文档" index="1">1</mcreference>
*   **登录/登出模式**：使用时需要先登录 `login()`，所有操作完成后再登出 `logout()`，以释放资源
*   **DataFrame格式**：返回pandas DataFrame类型数据，便于数据分析和可视化<mcreference link="https://blog.csdn.net/qq_37944726/article/details/115268769" index="5">5</mcreference>

## 3. 安装与更新

### 基础安装
```bash
pip install baostock
```

### 国内镜像加速安装
```bash
pip install baostock -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 版本升级
```bash
pip install --upgrade baostock
```

### 系统要求
- Python 3.5及以上版本（不支持Python 2.x）<mcreference link="https://blog.csdn.net/qq_37944726/article/details/115268769" index="5">5</mcreference>
- 依赖pandas库：`pip install pandas`

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

### 示例3：获取分钟线数据

```python
import baostock as bs
import pandas as pd

# 登录系统
bs.login()

# 获取5分钟K线数据
rs = bs.query_history_k_data_plus("sh.600000",
    "date,time,code,open,high,low,close,volume,amount,adjustflag",
    start_date='2023-01-01', end_date='2023-01-31',
    frequency="5", adjustflag="3")

# 打印结果集
data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())

result = pd.DataFrame(data_list, columns=rs.fields)
print(result.head())

# 登出系统
bs.logout()
```

### 示例4：获取财务数据

```python
import baostock as bs
import pandas as pd

# 登录系统
bs.login()

# 获取季频盈利能力
rs = bs.query_profit_data(code="sh.600000", year=2023, quarter=1)

# 打印结果集
profit_list = []
while (rs.error_code == '0') & rs.next():
    profit_list.append(rs.get_row_data())

result = pd.DataFrame(profit_list, columns=rs.fields)
print(result)

# 登出系统
bs.logout()
```

### 示例5：获取宏观经济数据

```python
import baostock as bs
import pandas as pd

# 登录系统
bs.login()

# 获取存款利率数据
rs = bs.query_deposit_rate_data(start_date="2023-01-01", end_date="2023-12-31")

# 打印结果集
data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())

result = pd.DataFrame(data_list, columns=rs.fields)
print(result)

# 登出系统
bs.logout()
```

## 5. 数据覆盖范围

### 5.1 股票数据
- **历史K线数据**：日线、周线、月线、5分钟、15分钟、30分钟、60分钟<mcreference link="http://baostock.com/baostock/index.php/Python_API文档" index="1">1</mcreference>
- **复权数据**：前复权、后复权、不复权
- **股票基本信息**：股票代码、名称、上市日期、退市日期等
- **停牌信息**：停牌日期、复牌日期、停牌原因

### 5.2 财务数据
- **季频盈利能力**：ROE、ROA、销售毛利率、销售净利率等<mcreference link="http://baostock.com/baostock/index.php/Python_API文档" index="1">1</mcreference>
- **季频营运能力**：应收账款周转率、存货周转率等
- **季频成长能力**：营业收入增长率、净利润增长率等
- **季频偿债能力**：流动比率、速动比率、资产负债率等
- **季频现金流量**：经营活动现金流量净额等
- **季频杜邦指数**：权益乘数、权益净利率等
- **季频重点指标**：基本每股收益、稀释每股收益等

### 5.3 宏观经济数据
- **利率数据**：存款利率、贷款利率、银行间同业拆放利率(Shibor)<mcreference link="http://baostock.com/baostock/index.php/Python_API文档" index="1">1</mcreference>
- **货币政策**：存款准备金率、货币供应量
- **经济指标**：GDP、CPI、PPI等

### 5.4 板块数据
- **行业分类**：申万行业分类、证监会行业分类<mcreference link="http://baostock.com/baostock/index.php/Python_API文档" index="1">1</mcreference>
- **概念板块**：各类概念板块成分股
- **地域板块**：各省市地区板块

## 6. 使用注意事项

1. **登录登出**：每次使用前必须调用`bs.login()`，使用完毕后调用`bs.logout()`
2. **数据格式**：返回的是结果集对象，需要手动循环提取并转换为DataFrame
3. **频率参数**："d"=日k线、"w"=周、"m"=月、"5"=5分钟、"15"=15分钟、"30"=30分钟、"60"=60分钟<mcreference link="https://klang.org.cn/docs/stockdata/" index="2">2</mcreference>
4. **复权参数**："1"=后复权、"2"=前复权、"3"=不复权（默认）<mcreference link="https://klang.org.cn/docs/stockdata/" index="2">2</mcreference>
5. **数据时间范围**：支持1990年至今的历史数据<mcreference link="https://blog.csdn.net/m0_46603114/article/details/107869037" index="2">2</mcreference>

## 7. BaoStock vs AkShare 功能比较

### 7.1 BaoStock 独有优势

#### 数据质量与稳定性
- **专业复权算法**：提供涨跌幅复权算法，复权数据更加精确可靠
- **数据一致性**：历史数据经过严格校验，适合量化回测和学术研究
- **长期稳定**：接口变动较少，代码兼容性好，适合长期项目维护

#### 财务数据深度
- **季频财务指标体系**：提供完整的盈利能力、营运能力、成长能力、偿债能力分析
- **杜邦分析体系**：权益乘数、权益净利率等专业财务分析指标
- **现金流量数据**：详细的经营、投资、筹资活动现金流量数据

#### 宏观经济数据
- **央行政策数据**：存款准备金率、货币供应量等货币政策工具数据
- **利率体系**：存款利率、贷款利率、Shibor等完整利率曲线
- **经济指标**：GDP、CPI、PPI等宏观经济核心指标

#### 数据获取方式
- **批量下载**：支持大批量历史数据下载，适合建立本地数据库
- **离线分析**：下载后可离线分析，不依赖网络连接
- **数据缓存**：本地存储减少重复请求，提高分析效率

### 7.2 BaoStock 多出的功能

#### 多时间周期K线
- **分钟级数据**：5分钟、15分钟、30分钟、60分钟K线数据
- **复权支持**：所有时间周期均支持前复权、后复权处理
- **时间精度**：分钟线数据包含具体时间戳，便于高频分析

#### 板块分类数据
- **行业分类**：申万行业分类、证监会行业分类双重标准
- **概念板块**：各类主题概念板块成分股数据
- **地域分类**：按省市地区划分的板块数据

#### 停牌复牌信息
- **停牌数据**：详细的停牌日期、原因、预计复牌时间
- **交易状态**：实时交易状态标识，便于数据清洗
- **ST标识**：特殊处理股票标识，风险控制必备

#### 数据完整性
- **历史覆盖**：1990年至今完整历史数据，时间跨度更长
- **数据校验**：内置数据质量检查，异常数据标识
- **缺失处理**：明确的缺失数据标记，便于数据预处理

## 8. 相关资源

- **官方网站**：[http://baostock.com](http://baostock.com)
- **API文档**：[Python API文档](http://baostock.com/baostock/index.php/Python_API文档)<mcreference link="http://baostock.com/baostock/index.php/Python_API文档" index="1">1</mcreference>
- **GitHub仓库**：[https://github.com/BaoStock/baostock](https://github.com/BaoStock/baostock)
