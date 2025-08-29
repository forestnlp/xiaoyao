# Tushare Pro 使用指南

## 1. 简介

Tushare 是一个提供丰富金融数据的 Python 库。目前，其主要通过 Tushare Pro 接口提供服务，该接口覆盖了股票、基金、期货、外汇、债券、指数、数字货币等多种数据。

## 2. 收费模式

Tushare Pro 采用**积分制**来管理数据访问权限。

*   **免费积分**：新用户注册后会获得 120 免费积分，足以调用一些基础数据接口。
*   **积分获取**：更多积分可以通过在社区贡献（如提交代码、撰写文章）或付费捐赠来获得。
*   **接口权限**：不同的数据接口需要不同的积分才能调用。基础数据（如股票列表）所需积分较低，而高阶数据（如高频数据、财务指标）则需要更多积分。

总的来说，对于学习和个人量化研究，Tushare 提供了非常高性价比的选择。

## 3. 安装

通过 pip 可以轻松安装 Tushare 库：
```bash
pip install tushare
```

## 4. 获取 Token

所有 Tushare Pro 的数据都需要通过 Token 进行认证。

1.  访问 [Tushare Pro 官网](https://tushare.pro/) 并注册一个账户。
2.  登录后，在个人主页的 "接口TOKEN" 菜单下找到你的专属 Token。

## 5. 基本用法

使用时，你需要先设置你的 Token，然后初始化 API。

```python
import tushare as ts

# 1. 设置你的 Token
# 建议将 Token 设置为环境变量，避免硬编码在代码中
token = '在这里粘贴你的真实TOKEN'
ts.set_token(token)

# 2. 初始化 Pro 接口
pro = ts.pro_api()

# 3. 调用数据接口
# 例如，获取股票列表
df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
print(df.head())
```
