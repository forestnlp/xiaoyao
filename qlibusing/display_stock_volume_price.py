import qlib
from qlib.data import D
import pandas as pd
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    # 初始化 Qlib
    # 请确保 'd:\qlib_data' 是您Qlib数据的实际路径
    qlib.init(provider_uri='d:\\qlib_data', region='cn')

    # 获取股票数据
    # 这里以 'sh600000' (浦发银行) 为例，获取2020年的日线数据
    # 您可以根据需要修改股票代码和日期范围
    start_date = '2020-01-01'
    end_date = '2020-12-21'

    all_instruments = D.list_instruments(D.instruments('all'))
    print(f"\nQlib数据中所有可用的股票数量：{len(all_instruments)}")
    print(f"\nQlib数据中所有可用的股票（前10个）：{list(all_instruments)[:10]}")

    # 用户指定的股票代码
    target_stock_code = 'SH600000' # 默认使用SH600570进行测试

    # 检查目标股票是否存在于所有股票列表中
    if target_stock_code not in all_instruments:
        print(f"\n错误：Qlib数据中不存在股票代码 {target_stock_code}。")
        # 如果不存在，可以考虑退出或处理其他逻辑
        # exit()

    df_stock = D.features([target_stock_code], fields=['$open', '$high', '$low', '$close', '$volume'],
                          start_time=start_date, end_time=end_date, freq='day')

    # 打印数据前几行，检查数据是否正确加载
    print(f"\n所有股票数据前5行：")

    # 确保索引是日期时间类型
    df_stock.index = df_stock.index.set_names(['instrument', 'datetime'])
    df_stock = df_stock.reset_index(level=[0])
    # 打印DataFrame的列名和前几行，以便调试
    print(f"\nDataFrame列名：{df_stock.columns.tolist()}")
    print(f"\nDataFrame前5行：\n{df_stock.head().to_string()}")

    # 打印数据前几行，检查数据是否正确加载
    print("\n数据已成功加载并显示。")
    print(f"\n所有股票代码：{df_stock['instrument'].unique().tolist()}")
