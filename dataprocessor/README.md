# 股票数据合并工具

本目录包含用于合并股票 CSV 数据的 Python 脚本。

## 文件说明

### 1. merge_stock_csv_to_parquet.py
**功能**: 将指定目录下的所有 `stock_***.csv` 文件合并为一个 parquet 文件

**特点**:
- 自动扫描目录中的 `stock_***.csv` 文件
- 按文件名排序（按日期顺序处理）
- 数据验证和清洗（确保数据类型正确）
- 去重处理（按 date + stock_code）
- 使用 snappy 压缩，与现有 parquet 文件保持一致
- 详细的处理日志和统计信息

**使用方法**:
```bash
python merge_stock_csv_to_parquet.py
```

**输出**: `merged_stock_data.parquet`

### 2. merge_with_existing_parquet.py
**功能**: 将新生成的 parquet 文件与现有的 parquet 文件合并

**特点**:
- 读取两个 parquet 文件
- 合并数据并去重
- 按日期和股票代码排序
- 保持与源文件相同的格式和压缩方式
- 生成详细的合并统计信息

**使用方法**:
```bash
python merge_with_existing_parquet.py
```

**输入**:
- 现有文件: `d:/workspace/xiaoyao/data/stock_daily_price.parquet`
- 新文件: `d:/workspace/xiaoyao/dataprocessor/merged_stock_data.parquet`

**输出**: `merged_combined_stock_data.parquet`

## 数据一致性

两个脚本确保：
- ✅ 字段名称和顺序完全一致
- ✅ 数据类型完全一致（datetime64[ns] 和 float64）
- ✅ 使用相同的压缩方式（snappy）
- ✅ 支持增量合并，避免重复数据

## 最新合并结果

- **合并文件**: `merged_combined_stock_data.parquet`
- **总行数**: 15,023,585 条记录
- **日期范围**: 2005-01-04 到 2025-09-22
- **股票数量**: 5,441 只股票
- **文件大小**: 518.82 MB