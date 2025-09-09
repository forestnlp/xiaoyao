import akshare as ak

# --- AkShare 数据读取演示 ---

def get_stock_list():
    """
    获取所有A股的实时行情列表。
    """
    print("\n--- 正在获取A股所有股票列表 ---")
    try:
        stock_spot_df = ak.stock_zh_a_spot_em()
        print(f"成功获取 {len(stock_spot_df)} 支股票的实时行情。")
        print("前5支股票:")
        print(stock_spot_df.head())
        return stock_spot_df
    except Exception as e:
        print(f"获取股票列表失败：{e}")
        return None

def get_daily_prices(symbol='000001', start_date='20240101'):
    """
    获取单只股票的历史日线行情（前复权）。
    """
    print(f"\n--- 正在获取代码为 {symbol} 的股票日线行情 ---")
    try:
        stock_hist_df = ak.stock_zh_a_hist(
            symbol=symbol, 
            period="daily", 
            start_date=start_date, 
            adjust="qfq" # qfq: 前复权
        )
        if stock_hist_df.empty:
            print(f"未能获取到股票 {symbol} 的数据，请检查代码是否正确。")
            return None
            
        print(f"成功获取 {len(stock_hist_df)} 天的日线数据。")
        print(f"{symbol} 的近期行情:")
        print(f"数据列名: {list(stock_hist_df.columns)}")
        print(stock_hist_df.head())
        return stock_hist_df
    except Exception as e:
        print(f"获取日线行情失败：{e}")
        return None

def get_option_data():
    """
    获取ETF期权数据示例。
    """
    print("\n--- 正在获取ETF期权数据 ---")
    try:
        option_df = ak.option_finance_board(symbol="华夏上证50ETF期权", end_month="2412")
        print(f"成功获取 {len(option_df)} 条期权数据。")
        print("期权数据示例:")
        print(option_df.head())
        return option_df
    except Exception as e:
        print(f"获取期权数据失败：{e}")
        return None

def get_crypto_data():
    """
    获取加密货币数据示例。
    """
    print("\n--- 正在获取比特币持仓报告 ---")
    try:
        crypto_df = ak.crypto_bitcoin_hold_report()
        print(f"成功获取 {len(crypto_df)} 条比特币持仓数据。")
        print("比特币持仓数据示例:")
        print(crypto_df.head())
        return crypto_df
    except Exception as e:
        print(f"获取加密货币数据失败：{e}")
        return None

def get_futures_info():
    """
    获取期货基础信息示例。
    """
    print("\n--- 正在获取期货基础信息 ---")
    try:
        futures_df = ak.futures_comm_info(symbol="所有")
        print(f"成功获取 {len(futures_df)} 个期货品种信息。")
        print("期货品种信息示例:")
        print(futures_df.head())
        return futures_df
    except Exception as e:
        print(f"获取期货信息失败：{e}")
        return None

if __name__ == "__main__":
    print("--- AkShare Demo 开始 ---")
    
    # 示例1: 获取股票列表
    get_stock_list()
    
    # 示例2: 获取平安银行的日线数据
    get_daily_prices(symbol='000001')

    # 示例3: 获取贵州茅台的日线数据
    get_daily_prices(symbol='600519')
    
    # 示例4: 获取期权数据
    get_option_data()
    
    # 示例5: 获取加密货币数据
    get_crypto_data()
    
    # 示例6: 获取期货信息
    get_futures_info()
    
    print("\n--- AkShare Demo 结束 ---")
    print("\n提示: 如果某些接口报错，请尝试更新AkShare到最新版本：")
    print("pip install akshare --upgrade")
