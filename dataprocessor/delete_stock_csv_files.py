#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除指定目录下的 stock_***.csv 文件
"""

import os
import glob
from pathlib import Path


def delete_stock_csv_files(target_directory, pattern="stock_*.csv", dry_run=True):
    """
    删除指定目录下的 stock_***.csv 文件
    
    Args:
        target_directory: 目标目录路径
        pattern: 文件匹配模式，默认为 "stock_*.csv"
        dry_run: 是否模拟运行（默认True，只显示要删除的文件但不实际删除）
    
    Returns:
        tuple: (成功删除的文件数, 失败数, 总文件大小)
    """
    print(f"扫描目录: {target_directory}")
    
    # 获取所有匹配的文件
    search_pattern = os.path.join(target_directory, pattern)
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print("未找到匹配的 CSV 文件")
        return 0, 0, 0
    
    print(f"找到 {len(csv_files)} 个文件")
    
    # 计算总文件大小
    total_size = sum(os.path.getsize(f) for f in csv_files)
    
    if dry_run:
        print("\n模拟运行模式（dry-run），以下文件将被删除：")
        for i, file_path in enumerate(sorted(csv_files), 1):
            file_size = os.path.getsize(file_path)
            print(f"  {i:3d}. {os.path.basename(file_path)} ({file_size / 1024:.1f} KB)")
        print(f"\n总计: {len(csv_files)} 个文件, {total_size / (1024*1024):.2f} MB")
        print("\n注意：当前为模拟模式，文件未被实际删除")
        print("   如需实际删除，请设置 dry_run=False")
        return len(csv_files), 0, total_size
    
    # 实际删除模式
    print(f"\n开始删除文件...")
    success_count = 0
    fail_count = 0
    
    for i, file_path in enumerate(sorted(csv_files), 1):
        try:
            file_size = os.path.getsize(file_path)
            os.remove(file_path)
            print(f"  已删除: {os.path.basename(file_path)} ({file_size / 1024:.1f} KB)")
            success_count += 1
            
        except Exception as e:
            print(f"  删除失败: {os.path.basename(file_path)} - {e}")
            fail_count += 1
    
    print(f"\n删除完成:")
    print(f"   成功删除: {success_count} 个文件")
    print(f"   失败: {fail_count} 个文件")
    print(f"   释放空间: {total_size / (1024*1024):.2f} MB")
    
    return success_count, fail_count, total_size


def confirm_delete(target_directory, pattern="stock_*.csv"):
    """
    确认删除操作
    """
    print("警告：此操作将永久删除以下文件！")
    print("=" * 50)
    
    # 先进行模拟运行显示要删除的文件
    delete_stock_csv_files(target_directory, pattern, dry_run=True)
    
    print("\n" + "=" * 50)
    response = input("确定要删除这些文件吗？输入 'YES' 确认删除: ")
    
    if response.upper() == 'YES':
        print("\n执行删除操作...")
        success, fail, size = delete_stock_csv_files(target_directory, pattern, dry_run=False)
        return success, fail, size
    else:
        print("\n操作已取消")
        return 0, 0, 0


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='删除 stock_***.csv 文件')
    parser.add_argument('--dir', '-d', 
                       default='d:/workspace/xiaoyao/redis',
                       help='目标目录路径 (默认: d:/workspace/xiaoyao/redis)')
    parser.add_argument('--pattern', '-p',
                       default='stock_*.csv',
                       help='文件匹配模式 (默认: stock_*.csv)')
    parser.add_argument('--force', '-f',
                       action='store_true',
                       help='强制删除，不确认')
    parser.add_argument('--dry-run',
                       action='store_true',
                       help='模拟运行，显示要删除的文件但不实际删除')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stock CSV 文件删除工具")
    print("=" * 60)
    
    # 检查目录是否存在
    if not os.path.exists(args.dir):
        print(f"目录不存在: {args.dir}")
        return
    
    # 执行操作
    if args.dry_run:
        # 模拟模式
        delete_stock_csv_files(args.dir, args.pattern, dry_run=True)
    elif args.force:
        # 强制删除模式
        print("强制删除模式，无需确认")
        delete_stock_csv_files(args.dir, args.pattern, dry_run=False)
    else:
        # 正常模式（需要确认）
        confirm_delete(args.dir, args.pattern)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()