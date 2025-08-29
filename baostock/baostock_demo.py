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

if __name__ == "__main__":
    print("--- BaoStock Demo 开始 ---")
    if login():
        # 示例1: 获取所有股票列表
        get_all_stocks()
        
        # 示例2: 获取贵州茅台的日线数据
        # 注意BaoStock的代码格式为 `sh.xxxxxx` 或 `sz.xxxxxx`
        get_daily_prices(symbol='sh.600519')
        
        # 操作结束后登出
        logout()
    print("\n--- BaoStock Demo 结束 ---")
