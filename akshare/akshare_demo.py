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
        print(stock_hist_df.head())
        return stock_hist_df
    except Exception as e:
        print(f"获取日线行情失败：{e}")
        return None

if __name__ == "__main__":
    print("--- AkShare Demo 开始 ---")
    
    # 示例1: 获取股票列表
    get_stock_list()
    
    # 示例2: 获取平安银行的日线数据
    get_daily_prices(symbol='000001')

    # 示例3: 获取贵州茅台的日线数据
    get_daily_prices(symbol='600519')
    
    print("\n--- AkShare Demo 结束 ---")
