import tushare as ts
import configparser
import os

# --- Tushare Pro 数据读取演示 ---
#
# 重要提示:
# 1. 脚本现在会从同目录下的 `config.ini` 文件中读取 token。
# 2. 请确保 `config.ini` 文件存在，并且格式正确。

def initialize_api():
    """
    初始化 Tushare Pro API.
    从 config.ini 文件中读取 token 并设置。
    """
    try:
        config = configparser.ConfigParser()
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.ini')

        if not os.path.exists(config_path):
            print(f"错误：配置文件不存在于 {config_path}")
            return None

        config.read(config_path, encoding='utf-8')
        token = config.get('tushare', 'token')

        if not token or token == '在这里粘贴你的真实TOKEN':
            print("错误：请在 config.ini 文件中设置您的有效 token。")
            return None
            
        ts.set_token(token)
        print("Tushare token 从配置文件中加载成功。")
        return ts.pro_api()
    except Exception as e:
        print(f"初始化失败：{e}")
        return None

def get_stock_list(pro):
    """
    获取基础的股票列表 (上市状态为L)。
    """
    print("\n--- 正在获取股票列表 ---")
    try:
        df = pro.stock_basic(
            exchange='', 
            list_status='L', 
            fields='ts_code,symbol,name,area,industry,list_date'
        )
        print("成功获取 {} 支股票。".format(len(df)))
        print(df.head())
        return df
    except Exception as e:
        print(f"获取股票列表失败：{e}")
        return None

def get_daily_prices(pro, ts_code='000001.SZ'):
    """
    获取单只股票的日线行情数据。
    """
    print(f"\n--- 正在获取 {ts_code} 的日线行情 ---")
    try:
        df = pro.daily(
            ts_code=ts_code, 
            start_date='20240101', 
            end_date='20240131'
        )
        print(f"成功获取 {len(df)} 天的日线数据。")
        print(df.head())
        return df
    except Exception as e:
        print(f"获取日线行情失败：{e}")
        return None

def get_financial_indicators(pro, ts_code='600519.SH'):
    """
    获取单只股票的主要财务指标。
    """
    print(f"\n--- 正在获取 {ts_code} 的财务指标 ---")
    try:
        # ann_date（公告日期），end_date（报告期）
        df = pro.fina_indicator(ts_code=ts_code, period='20231231')
        print(f"成功获取 {ts_code} 在 2023 年末的财务指标。")
        print(df)
        return df
    except Exception as e:
        print(f"获取财务指标失败：{e}")
        return None

if __name__ == "__main__":
    pro_api = initialize_api()

    if pro_api:
        # 示例1: 获取股票列表
        get_stock_list(pro_api)

        # 示例2: 获取平安银行的日线行情
        get_daily_prices(pro_api, ts_code='000001.SZ')

        # 示例3: 获取贵州茅台的财务指标
        get_financial_indicators(pro_api, ts_code='600519.SH')
