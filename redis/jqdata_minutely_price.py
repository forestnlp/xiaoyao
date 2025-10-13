# 从Jupyter Notebook转换而来的Python代码
# 原始文件：D:\workspace\xiaoyao\redis\jqdata_minutely_price.ipynb



# ----------------------------------------------------------------------#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 /d:/workspace/xiaoyao/redis/ 目录下的所有 stock_***.csv 文件合并为一个 parquet 文件
确保与现有 /d:/workspace/xiaoyao/data/stock_daily_price.parquet 保持字段、压缩方式一致
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import glob


def merge_stock_csv_to_parquet(csv_dir,stock_code, output_parquet_file):
    """
    合并指定目录下的所有 stock_***.csv 文件到单个 parquet 文件
    
    Args:
        csv_dir: CSV 文件所在目录
        output_parquet_file: 输出的 parquet 文件路径
    """
    print(f"📁 开始处理目录: {csv_dir}")
    
    # 获取所有 stock_***.csv 文件
    csv_pattern = os.path.join(csv_dir, f"stock_minutely_price_{stock_code}*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"❌ 未找到 stock_minutely_price_{stock_code}*.csv 文件")
        return False
    
    print(f"📊 找到 {len(csv_files)} 个 CSV 文件")
    
    # 按文件名排序（确保按日期顺序处理）
    csv_files.sort()
    
    # 读取并合并所有 CSV 文件
    all_dataframes = []
    total_records = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        filename = os.path.basename(csv_file)
        print(f"正在处理 ({i}/{len(csv_files)}): {filename}")
        
        try:
            # 读取 CSV 文件
            df = pd.read_csv(csv_file)
            
            # 数据验证和清洗
            # 确保 date 列是 datetime 类型
            df['date'] = pd.to_datetime(df['date'])
            
            # 确保数值列的数据类型正确
            numeric_columns = ['open', 'close', 'low', 'high', 'volume']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 删除无效数据
            df = df.dropna(subset=['date', 'stock_code','time'])
            
            all_dataframes.append(df)
            total_records += len(df)
            print(f"  ✅ 成功读取 {len(df)} 条记录")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            continue
    
    if not all_dataframes:
        print("❌ 没有成功读取任何数据")
        return False
    
    print(f"\n📊 合并所有数据...")
    # 合并所有数据框
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 去重（按 date + stock_code + time）
    combined_df = combined_df.drop_duplicates(subset=['date', 'stock_code','time'])
    
    # 按日期和股票代码排序
    combined_df = combined_df.sort_values(['date', 'stock_code','time']).reset_index(drop=True)
    
    print(f"📈 总计 {len(combined_df)} 条记录（去重后）")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_parquet_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")
    
    # 转换为 pyarrow Table
    table = pa.Table.from_pandas(combined_df)
    
    # 使用与目标文件相同的压缩方式 (snappy) 和格式写入 parquet
    try:
        pq.write_table(
            table, 
            output_parquet_file,
            compression='snappy',
            version='2.6',  # 使用较新的 parquet 版本
            use_dictionary=True,
            write_batch_size=64 * 1024 * 1024  # 64MB batch size for better performance
        )
        
        print(f"✅ 成功保存到: {output_parquet_file}")
        print(f"📊 文件大小: {os.path.getsize(output_parquet_file) / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存 parquet 文件失败: {e}")
        return False

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将新生成的 merged_stock_data.parquet 与现有的 stock_daily_price.parquet 合并
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import shutil

def merge_parquet_files(stock_code):
    """
    合并两个 parquet 文件
    
    Args:
        existing_file: 现有的 parquet 文件路径
        new_file: 新的 parquet 文件路径  
        output_file: 输出的合并文件路径
    """

    existing_file = f"d:/workspace/xiaoyao/data/stock_minutely_price/{stock_code}.parquet"
    new_file = f"d:/workspace/xiaoyao/redis/minutely/{stock_code}_merged.parquet"
    output_file = f"d:/workspace/xiaoyao/data/stock_minutely_price/{stock_code}.parquet"


    # 检查文件是否存在
    if not os.path.exists(existing_file):
        print(f"❌ 现有文件不存在: {existing_file}")
        #直接用先用文件去覆盖
        shutil.move(new_file, existing_file)
        return
    
    if not os.path.exists(new_file):
        print(f"❌ 新文件不存在: {new_file}")
        return

    print("📊 开始合并 parquet 文件...")
    
    try:
        # 读取现有数据
        print(f"📖 读取现有文件: {existing_file}")
        existing_df = pd.read_parquet(existing_file)
        print(f"   现有数据行数: {len(existing_df)}")
        
        # 读取新数据
        print(f"📖 读取新文件: {new_file}")
        new_df = pd.read_parquet(new_file)
        print(f"   新数据行数: {len(new_df)}")
        
        # 合并数据
        print("🔄 合并数据中...")
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # 去重（按 date + stock_code）
        print("🧹 去重处理...")
        combined_df = combined_df.drop_duplicates(subset=['date', 'stock_code', 'time'])
        
        # 排序
        print("📅 按日期排序...")
        combined_df = combined_df.sort_values(['date', 'stock_code', 'time']).reset_index(drop=True)
        
        print(f"📈 合并后总行数: {len(combined_df)}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 转换为 pyarrow Table
        table = pa.Table.from_pandas(combined_df)
        
        # 写入 parquet（使用与源文件相同的格式）
        print(f"💾 保存合并结果: {output_file}")
        pq.write_table(
            table,
            output_file,
            compression='snappy',
            version='2.6',
            use_dictionary=True,
            write_batch_size=64 * 1024 * 1024
        )
        
        print(f"✅ 合并完成！文件大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        # 验证结果
        print("\n📋 验证结果:")
        result_df = pd.read_parquet(output_file)
        print(f"   最终行数: {len(result_df)}")
        print(f"   日期范围: {result_df['date'].min()} 到 {result_df['date'].max()}")
        print(f"   股票数量: {result_df['stock_code'].nunique()}")
        

        # 移动文件，清理
        # # 检查源文件是否存在
        # source_file_path = f'./minutely/{stock_code}.parquet'
        # target_file_path = f'D:/workspace/xiaoyao/redis/data/minutely/{stock_code}.parquet'
        # if os.path.exists(source_file_path):
        #     # 移动文件
        #     shutil.move(source_file_path, target_file_path)
        #     print(f"Moved: {file_to_move} to {target_dir}")
        # else:
        #     print(f"File not found: {source_file_path}")

        return True
        
    except Exception as e:
        print(f"❌ 合并失败: {e}")
        return False

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除指定目录下的 stock_***.csv 文件
"""

import os
import glob
from pathlib import Path


def delete_stock_csv_files(stock_code,target_directory=r'D:\workspace\xiaoyao\redis\minutely'):
    #删除满足模式的所有文件
    pattern = f"stock_minutely_price_{stock_code}*.csv"
    files = glob.glob(os.path.join(target_directory, pattern))
    for file in files:
        try:
            os.remove(file)
            print(f"已删除：{file}")
        except Exception as e:
            print(f"删除 {file} 失败：{e}")

import redis
import pickle
import time
import uuid
import pandas as pd
from io import StringIO
from typing import Any, Optional
from datetime import datetime, timedelta

class RemoteSender:
    def __init__(self, host='*', port=6379, password='*'):
        self.redis = redis.Redis(
            host=host, port=port, password=password,
            decode_responses=False
        )
        self.task_queue = 'function_calls'
        self.result_queue = 'function_results'
        self._test_connection()
        print(f"✅ 发送端pandas版本：{pd.__version__}")

    def _test_connection(self):
        try:
            self.redis.ping()
            print("✅ 发送端：Redis连接成功")
        except Exception as e:
            print(f"❌ 发送端：连接失败 - {e}")
            raise

    def call_remote_function(self, func_name: str, *args, **kwargs) -> Any:
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task = {
            'func_name': func_name,
            'args': args,
            'kwargs': kwargs,
            'task_id': task_id
        }
        self.redis.rpush(self.task_queue, pickle.dumps(task))
        print(f"📤 已调用远程函数：{func_name}（任务ID：{task_id}）")
        return self._get_result(task_id)

    def _get_result(self, task_id: str, timeout=300) -> Any:
        start_time = time.time()
        while time.time() - start_time < timeout:
            result_data = self.redis.blpop(self.result_queue, timeout=10)
            if not result_data:
                continue

            _, res_bytes = result_data
            result = pickle.loads(res_bytes)
            if result['task_id'] == task_id:
                if result['status'] == 'success':
                    return result['result']  # 返回CSV字符串
                else:
                    raise Exception(f"远程执行失败：{result['error']}")
            self.redis.rpush(self.result_queue, res_bytes)
        raise TimeoutError("任务超时")

    def save_to_csv(self, csv_str: Optional[str], filename: str) -> bool:
        """将CSV字符串保存为本地CSV文件（替代Parquet）"""
        if not csv_str:
            print("⚠️ 数据为空，不保存")
            return False
        try:
            # 从CSV字符串恢复DataFrame（兼容所有pandas版本）
            df = pd.read_csv(StringIO(csv_str))
            # 保存为CSV文件
            df.to_csv(filename, index=False)
            print(f"✅ 保存成功：{filename}（{len(df)}条记录）")
            return True
        except Exception as e:
            print(f"❌ 保存失败：{e}")
            return False

def generate_date_range(start_date_str: str, end_date_str: str) -> list:
    """生成从开始日期到结束日期的所有日期字符串（YYYYMMDD格式）"""
    dates = []
    try:
        start_date = datetime.strptime(start_date_str, '%Y%m%d')
        end_date = datetime.strptime(end_date_str, '%Y%m%d')
        
        if start_date > end_date:
            raise ValueError("开始日期晚于结束日期")
            
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date.strftime('%Y%m%d'))
            current_date += timedelta(days=1)
    except Exception as e:
        print(f"日期处理错误：{e}")
    return dates

from tqdm import tqdm


if __name__ == "__main__":

    # 从配置文件读取Redis连接信息
    with open('redis.conf', 'r') as f:
        for line in f:
            if line.startswith('host='):
                host = line.split('=')[1].strip()
            elif line.startswith('port='):
                port = int(line.split('=')[1].strip())
            elif line.startswith('password='):
                password = line.split('=')[1].strip()
    # 初始化Redis发送端
    sender = RemoteSender(host=host, port=port, password=password)
    
    # 定义日期范围：从20250516到20250923
    start_date = '20250922'#(df['date'].max() + timedelta(days=1)).strftime('%Y%m%d')
    # 获取当日日期-1，是end_date
    end_date = '20250926'#(datetime.today() - timedelta(days=1)).strftime('%Y%m%d')
    
    
    daily_df = pd.read_parquet('../data/stock_daily_price.parquet')
    stock_code_list = daily_df['stock_code'].unique()

    # 循环调用获取每日数据
    for stock_code in tqdm(stock_code_list):
        # 读取已存在的parquet文件
        start_date2 = start_date

        existing_file = f"d:/workspace/xiaoyao/redis/minutely/{stock_code}.parquet"
        # if os.path.exists(existing_file):
        #     existing_df = pd.read_parquet(existing_file)
        #     # 读取existing_df的date列的最大date
        #     max_date = existing_df['date'].max()
        #     start_date2 = (datetime.strptime(max_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
        
        date_list = generate_date_range(start_date2, end_date)
        print(f"=== 共需获取 {len(date_list)} 天的数据 ===")

        for i, date in enumerate(date_list, 1):
            try:
                # 调用远程函数获取当日数据
                csv_data = sender.call_remote_function('fetch_minute_stock_data', date,[stock_code])
                # 保存为CSV文件，文件名包含日期
                sender.save_to_csv(csv_data, f'./minutely/stock_minutely_price_{stock_code}_{date}.csv')
                # 适当延迟，避免请求过于频繁
                time.sleep(0.05)
            except Exception as e:
                print(f"❌ {date} 处理失败：{e}")
                # 失败后也延迟一下，避免快速重试导致的问题
                time.sleep(1)
        # 下载完成后对csv进行合并
        success = merge_stock_csv_to_parquet('./minutely/', stock_code, f'./minutely/{stock_code}_merged.parquet')
        
        if not success:
            print(f"❌ {stock_code} 合并失败")
        
        # 将生成的parquet进行合并
        merge_parquet_files(stock_code)


        # 删除指定的parquet文件
        file_to_delete = f'./minutely/{stock_code}_merged.parquet'
        if os.path.exists(file_to_delete):
            os.remove(file_to_delete)


        file_to_delete = f'./minutely/{stock_code}.parquet'
        if os.path.exists(file_to_delete):
            os.remove(file_to_delete)

        # 删除无用的csv
        delete_stock_csv_files(stock_code)

    print("\n=== 所有日期处理完成 ===")

