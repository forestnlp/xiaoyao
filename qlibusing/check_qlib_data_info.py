import qlib
from qlib.data import D
import pandas as pd
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    # 初始化 Qlib
    # 请确保 'd:\qlib_data' 是您Qlib数据的实际路径
    qlib.init(provider_uri='d:\\qlib_data', region='cn')

    # 获取所有股票代码
    all_instruments = D.list_instruments(D.instruments('all'))
    print(f"\n当前数据目录中包含的股票数量：{len(all_instruments)}")

    # 检查 all_instruments 是否为 DataFrame
    if isinstance(all_instruments, pd.DataFrame):
        if len(all_instruments) > 0:
            # 获取最早和最新的数据日期
            if 'start_time' in all_instruments.columns and 'end_time' in all_instruments.columns:
                earliest_date = all_instruments['start_time'].min()
                latest_date = all_instruments['end_time'].max()
                print(f"最早的数据日期：{earliest_date.strftime('%Y-%m-%d')}")
                print(f"最新的数据日期：{latest_date.strftime('%Y-%m-%d')}")
            else:
                print("DataFrame 中没有 'start_time' 或 'end_time' 列，尝试使用 D.calendar() 获取日期范围。")
                full_calendar = sorted(list(D.calendar()))
                if full_calendar:
                    earliest_date = full_calendar[0]
                    latest_date = full_calendar[-1]
                    print(f"最早的数据日期：{earliest_date.strftime('%Y-%m-%d')}")
                    print(f"最新的数据日期：{latest_date.strftime('%Y-%m-%d')}")
                else:
                    print("无法获取数据日期范围，可能数据为空。")

            # 获取可用的字段
            if not all_instruments.empty:
                # 确保 sample_stock 是一个有效的股票代码字符串
                sample_stock = str(all_instruments.index[0])
                if sample_stock.isdigit(): # 如果是数字，尝试从 D.instruments('all') 获取
                    all_instruments_dict = D.instruments('all')
                    if all_instruments_dict:
                        sample_stock = str(list(all_instruments_dict.keys())[0])
                    else:
                        sample_stock = None
            else:
                sample_stock = None # 无法获取样本股票
    else: # all_instruments is a dict
        print("D.list_instruments 返回的不是 DataFrame，将回退到使用 D.calendar() 获取日期范围。")
        if len(all_instruments) > 0: # all_instruments is a dict here
            full_calendar = sorted(list(D.calendar()))
            if full_calendar:
                earliest_date = full_calendar[0]
                latest_date = full_calendar[-1]
                print(f"最早的数据日期：{earliest_date.strftime('%Y-%m-%d')}")
                print(f"最新的数据日期：{latest_date.strftime('%Y-%m-%d')}")
            else:
                print("无法获取数据日期范围，可能数据为空。")

            # 获取可用的字段
            if all_instruments: # all_instruments is a dict here
                # 确保 sample_stock 是一个有效的股票代码字符串
                sample_stock = str(list(all_instruments.keys())[0])
                if sample_stock.isdigit(): # 如果是数字，尝试从 D.instruments('all') 获取
                    all_instruments_dict = D.instruments('all')
                    if all_instruments_dict:
                        sample_stock = str(list(all_instruments_dict.keys())[0])
                    else:
                        sample_stock = None
            else:
                sample_stock = None # 无法获取样本股票
        else:
            print("当前数据目录中没有股票数据。")
            sample_stock = None

    if sample_stock:
        # 尝试获取该股票的最新一天数据，以获取所有字段
        try:
            # 获取该股票的最新一天数据，以获取所有字段
            # 注意：这里假设数据至少包含一天，并且freq='day'是合适的
            df_sample = D.features([sample_stock], fields=['$open', '$close', '$high', '$low', '$volume', '$amount'], start_time=latest_date, end_time=latest_date, freq='day')
            if not df_sample.empty:
                print(f"\n可用的字段：{df_sample.columns.tolist()}")
            else:
                print(f"无法获取股票 {sample_stock} 的数据，可能没有最新日期的数据。")
        except Exception as e:
            print(f"获取股票 {sample_stock} 字段时发生错误：{e}")
    else:
        print("无法获取样本股票，无法查询字段信息。")