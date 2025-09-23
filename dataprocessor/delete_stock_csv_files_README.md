# Stock CSV 文件删除工具

## 功能说明

此工具用于删除指定目录下的 `stock_***.csv` 文件，支持以下功能：

- ✅ 自动扫描目标目录中的 `stock_***.csv` 文件
- ✅ 模拟运行模式（dry-run）- 预览要删除的文件
- ✅ 确认删除模式 - 需要用户输入 'YES' 确认
- ✅ 强制删除模式 - 无需确认直接删除
- ✅ 显示详细的删除统计信息

## 使用方法

### 1. 模拟运行（推荐先执行）
```bash
python delete_stock_csv_files.py --dry-run
```
显示将要删除的文件列表，但不会实际删除。

### 2. 正常删除（需要确认）
```bash
python delete_stock_csv_files.py
```
显示要删除的文件列表，需要输入 'YES' 确认后才执行删除。

### 3. 强制删除（无需确认）
```bash
python delete_stock_csv_files.py --force
```
直接删除文件，无需用户确认。

### 4. 指定其他目录
```bash
python delete_stock_csv_files.py --dir "其他目录路径"
```

### 5. 使用不同的文件模式
```bash
python delete_stock_csv_files.py --pattern "data_*.csv"
```

## 默认设置

- **默认目录**: `d:/workspace/xiaoyao/redis`
- **默认文件模式**: `stock_*.csv`
- **默认模式**: 需要确认删除

## 安全提示

⚠️ **重要提醒**:
1. 删除的文件无法恢复，请先使用 `--dry-run` 确认
2. 强制删除模式（`--force`）会直接删除文件，请谨慎使用
3. 确保已备份重要数据

## 当前状态

根据模拟运行结果，将要删除：
- **90 个文件**
- **日期范围**: 2025-05-19 到 2025-09-22
- **总大小**: 约 47.44 MB
- **文件格式**: `stock_YYYYMMDD.csv`