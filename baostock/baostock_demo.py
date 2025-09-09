import baostock as bs
import pandas as pd

# --- BaoStock 数据读取演示 ---

def login():
    """登录BaoStock系统"""
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return False
    print("BaoStock 登录成功。")
    return True

def logout():
    """登出BaoStock系统"""
    bs.logout()
    print("BaoStock 已登出。")

def get_all_stocks(date='2024-01-05'):
    """
    获取指定日期的所有A股列表。
    """
    print(f"\n--- 正在获取 {date} 的所有A股列表 ---")
    try:
        rs = bs.query_all_stock(day=date)
        if rs.error_code != '0':
            print(f"查询失败: {rs.error_msg}")
            return None
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        result = pd.DataFrame(data_list, columns=rs.fields)
        print(f"成功获取 {len(result)} 支股票。")
        print(result.head())
        return result
    except Exception as e:
        print(f"获取股票列表时发生错误: {e}")
        return None

def get_daily_prices(symbol='sh.600519', start_date='2024-01-01'):
    """
    获取单只股票的历史日线行情（前复权）。
    """
    print(f"\n--- 正在获取代码为 {symbol} 的股票日线行情 ---")
    try:
        # 查询字段可以按需调整
        fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,pctChg"
        rs = bs.query_history_k_data_plus(
            symbol, 
            fields, 
            start_date=start_date, 
            frequency="d", # d=日线, w=周线, m=月线
            adjustflag="2" # 1:后复权, 2:前复权, 3:不复权
        )
        if rs.error_code != '0':
            print(f"查询K线数据失败: {rs.error_msg}")
            return None

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        result = pd.DataFrame(data_list, columns=rs.fields)
        print(f"成功获取 {len(result)} 天的日线数据。")
        print(result.head())
        return result
    except Exception as e:
        print(f"获取日线行情时发生错误: {e}")
        return None

def get_minute_data(code, start_date, end_date, frequency="5"):
    """获取分钟线数据"""
    try:
        rs = bs.query_history_k_data_plus(code,
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date=start_date, end_date=end_date,
            frequency=frequency, adjustflag="3")
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if data_list:
            result = pd.DataFrame(data_list, columns=rs.fields)
            print(f"成功获取 {code} 的{frequency}分钟线数据")
            return result
        else:
            print(f"未获取到 {code} 的分钟线数据")
            return pd.DataFrame()
    except Exception as e:
        print(f"获取分钟线数据时出错: {e}")
        return pd.DataFrame()

def get_financial_data(code, year, quarter):
    """获取财务数据"""
    try:
        rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if data_list:
            result = pd.DataFrame(data_list, columns=rs.fields)
            print(f"成功获取 {code} 的{year}年Q{quarter}财务数据")
            return result
        else:
            print(f"未获取到 {code} 的财务数据")
            return pd.DataFrame()
    except Exception as e:
        print(f"获取财务数据时出错: {e}")
        return pd.DataFrame()

def get_macro_data(start_date, end_date):
    """获取宏观经济数据"""
    try:
        rs = bs.query_deposit_rate_data(start_date=start_date, end_date=end_date)
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if data_list:
            result = pd.DataFrame(data_list, columns=rs.fields)
            print(f"成功获取存款利率数据")
            return result
        else:
            print(f"未获取到宏观经济数据")
            return pd.DataFrame()
    except Exception as e:
        print(f"获取宏观经济数据时出错: {e}")
        return pd.DataFrame()

def main():
    """主函数：演示 BaoStock 的基本使用"""
    print("=== BaoStock 演示开始 ===")
    
    # 登录
    if not login():
        return
    
    # 获取股票列表
    print("\n1. 获取股票列表：")
    stocks = get_all_stocks('2023-12-29')
    if stocks is not None:
        print(f"获取到 {len(stocks)} 只股票")
        print(stocks.head())
    
    # 获取历史数据
    print("\n2. 获取历史数据：")
    hist_data = get_daily_prices('sh.600000', '2023-01-01')
    if hist_data is not None:
        print(f"获取到 {len(hist_data)} 条数据")
        print(f"数据列名: {list(hist_data.columns)}")
        print(hist_data.head())
    
    # 获取分钟线数据
    print("\n3. 获取分钟线数据：")
    minute_data = get_minute_data('sh.600000', '2023-12-01', '2023-12-05', '5')
    if not minute_data.empty:
        print(f"获取到 {len(minute_data)} 条5分钟线数据")
        print(minute_data.head())
    
    # 获取财务数据
    print("\n4. 获取财务数据：")
    financial_data = get_financial_data('sh.600000', 2023, 3)
    if not financial_data.empty:
        print(financial_data)
    
    # 获取宏观经济数据
    print("\n5. 获取宏观经济数据：")
    macro_data = get_macro_data('2023-01-01', '2023-12-31')
    if not macro_data.empty:
        print(f"获取到 {len(macro_data)} 条宏观数据")
        print(macro_data.head())
    
    # 登出
    logout()
    
    print("\n=== BaoStock 演示结束 ===")
    print("\n提示：如需更新到最新版本，请运行：")
    print("pip install --upgrade baostock")

if __name__ == "__main__":
    main()
