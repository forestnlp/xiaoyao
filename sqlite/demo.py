#!/usr/bin/env python3
"""
Parquet转SQLite简化演示
功能：将单个Parquet文件转换为SQLite数据库，支持指定索引列
"""

import pandas as pd
import sqlite3
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parquet_to_sqlite(input_file, output_file, index_columns=None):
    """
    将Parquet文件转换为SQLite数据库
    
    Args:
        input_file: 输入Parquet文件路径
        output_file: 输出SQLite文件路径
        index_columns: 需要创建索引的列名列表，默认为None
    
    Returns:
        bool: 转换是否成功
    """
    try:
        # 读取Parquet文件
        logger.info(f"正在读取Parquet文件: {input_file}")
        df = pd.read_parquet(input_file)
        logger.info(f"数据读取完成，共{len(df)}条记录，{len(df.columns)}个字段")
        
        # 创建SQLite数据库
        logger.info(f"正在创建SQLite数据库: {output_file}")
        conn = sqlite3.connect(output_file)
        
        # 写入数据表（表名固定为'data'）
        table_name = 'data'
        logger.info(f"正在写入数据表: {table_name}")
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # 创建索引
        if index_columns:
            logger.info(f"正在创建索引，列: {index_columns}")
            for col in index_columns:
                if col in df.columns:
                    try:
                        index_name = f"idx_{table_name}_{col}"
                        conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({col})")
                        logger.info(f"成功创建索引: {index_name}")
                    except Exception as e:
                        logger.warning(f"创建索引 {col} 失败: {e}")
                else:
                    logger.warning(f"列 {col} 不存在，跳过索引创建")
        
        # 获取数据库统计信息
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        
        logger.info(f"转换完成!")
        logger.info(f"  记录数: {count:,}")
        logger.info(f"  字段数: {len(columns)}")
        logger.info(f"  字段列表: {columns}")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"转换失败: {e}")
        return False


def main():
    """主函数：测试转换功能"""
    
    # 设置输入输出文件
    input_file = "../data/stock_daily_price.parquet"  # 输入Parquet文件
    output_file = "../data/stock_daily_price_demo.db"  # 输出SQLite文件
    
    # 设置索引列
    index_columns = ["date", "stock_code"]  # 需要创建索引的列
    
    logger.info("=" * 50)
    logger.info("Parquet转SQLite转换测试")
    logger.info("=" * 50)
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"索引列: {index_columns}")
    logger.info("-" * 50)
    
    # 执行转换
    success = parquet_to_sqlite(input_file, output_file, index_columns)
    
    if success:
        logger.info("✅ 转换成功!")
        
        # 验证转换结果
        try:
            conn = sqlite3.connect(output_file)
            cursor = conn.cursor()
            
            # 查询记录数
            cursor.execute("SELECT COUNT(*) FROM data")
            count = cursor.fetchone()[0]
            
            # 查询前5条数据
            cursor.execute("SELECT * FROM data LIMIT 5")
            sample_data = cursor.fetchall()
            
            # 获取字段名
            cursor.execute("PRAGMA table_info(data)")
            columns = [row[1] for row in cursor.fetchall()]
            
            logger.info("\n📊 转换结果验证:")
            logger.info(f"  总记录数: {count:,}")
            logger.info(f"  字段列表: {columns}")
            logger.info(f"  前5条数据:")
            for i, row in enumerate(sample_data, 1):
                logger.info(f"    {i}: {dict(zip(columns, row))}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"验证失败: {e}")
    else:
        logger.error("❌ 转换失败!")


if __name__ == "__main__":
    main()