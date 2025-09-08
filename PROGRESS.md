## Qlib 数据信息检查脚本 `check_qlib_data_info.py` 修复记录

### 初始问题与诊断

- **问题描述**：在尝试获取 Qlib 数据目录中的股票数量、最早和最新的数据日期以及可用的数据字段时，脚本 `check_qlib_data_info.py` 遇到 `KeyError`。
- **首次尝试修复**：在访问 `start_time` 和 `end_time` 列之前，增加了打印 `all_instruments` 列名的调试语句，并增加了回退逻辑：如果这些列不存在，则尝试使用 `D.calendar()` 来获取日期范围。
- **结果**：文件修改成功，但脚本执行失败，日志显示 `AttributeError: 'dict' object has no attribute 'columns'`，表明 `all_instruments` 变量是一个字典而不是 DataFrame。

### 解决 `AttributeError`

- **问题描述**：`AttributeError: 'dict' object has no attribute 'columns'` 错误，因为 `all_instruments` 是字典类型，无法访问 `columns` 属性。
- **第二次尝试修复**：
    1. 确保 `all_instruments` 是一个 DataFrame。
    2. 移除了调试打印语句。
    3. 增加了回退逻辑：如果 `D.list_instruments` 返回的不是 DataFrame，则回退到使用 `D.calendar()` 获取日期范围。
- **结果**：文件修改成功，但脚本执行再次失败，`all_instruments` 仍然是字典类型，再次出现 `AttributeError`。

### 进一步解决 `AttributeError` 和 `TypeError`

- **问题描述**：`AttributeError` 持续存在，因为在回退逻辑中 `all_instruments` 变为字典后，仍然尝试访问其 `columns` 属性。同时，`D.calendar()` 函数在接收到字典类型输入时导致 `TypeError: Cannot convert input [{'market': 'all', 'filter_pipe': []}] of type <class 'dict'> to Timestamp`。
- **第三次尝试修复**：
    1. 增加对 `all_instruments` 类型的检查，确保在它是 DataFrame 时才访问 `columns` 属性。
    2. 在它是字典时，直接使用 `D.calendar()` 获取日期范围。
    3. 调整了 `sample_stock` 的获取方式。
    4. 移除了冗余的 `all_instruments = D.instruments('all')` 赋值。
    5. 修正了 `D.calendar()` 的调用方式，使其不再接收参数。
- **结果**：文件修改成功，脚本执行成功，输出了股票数量、最早和最新的数据日期，但提示 `DataFrame 中没有 'start_time' 或 'end_time' 列，尝试使用 D.calendar() 获取日期范围`，并且在获取股票字段时发生错误 `fields cannot be empty`。

### 解决 `fields cannot be empty` 和 `NameError`

- **问题描述**：`fields cannot be empty` 错误，因为 `D.features` 函数的 `fields` 参数为空。同时，出现 `NameError: name 'amount' is not defined` 和 `'int' object has no attribute 'lower'` 错误。
- **第四次尝试修复**：
    1. 在调用 `D.features` 时提供非空的 `fields` 参数（`['$open', '$close', '$high', '$low', '$volume', '$amount']`），并用 `$` 符号包裹字段名以符合 Qlib 的字段引用规范。
    2. 将获取 `sample_stock` 的逻辑修改为在赋值前将其转换为字符串类型，并增加逻辑确保其是一个有效的股票代码字符串。
    3. 重新添加 `import qlib` 语句。
- **结果**：文件修改成功，脚本执行成功，能够正确获取股票数量、最早和最新的数据日期，并成功获取可用的字段信息。

### 最终状态

`check_qlib_data_info.py` 脚本现在能够稳定运行，并提供 Qlib 数据目录的概览信息，包括股票数量、数据日期范围以及可用的数据字段。脚本已具备健壮性，能够处理 `D.list_instruments` 返回不同类型数据的情况，并正确回退到 `D.calendar()` 获取日期信息。

---

## Qlib 框架学习与实战项目计划

### 项目目标

通过系统性学习和实践，掌握 Qlib 量化投资框架的核心功能，成为能够进行实战开发的专家级用户。项目将通过一系列 Jupyter Notebook 来逐步学习和实现各个功能模块。

### 项目结构

所有代码和数据文件存放在 `/d:/workspace/xiaoyao/qlibusing/` 目录下，定期更新推送到 GitHub。

### 学习阶段计划

#### 阶段 1：数据导入与处理
- **目标**：掌握将外部数据源转换为 Qlib 格式的完整流程
- **内容**：
  - Parquet 文件解析与处理
  - 数据过滤与清洗（时间范围、股票代码、字段选择）
  - CSV 格式转换（按股票代码分文件）
  - Qlib 二进制格式转换（使用 dump_bin 脚本）
  - 性能优化与进度监控
- **交付物**：`01_data_import.ipynb`

#### 阶段 2：现有因子测试
- **目标**：熟悉 Qlib 内置因子库，学会因子计算与验证
- **内容**：
  - Qlib 内置因子库探索
  - 技术指标因子计算
  - 基本面因子计算
  - 因子数据可视化
  - 因子有效性初步验证
- **交付物**：`02_existing_factors.ipynb`

#### 阶段 3：自定义因子开发
- **目标**：学会创建和实现自定义量化因子
- **内容**：
  - 因子表达式语法学习
  - 自定义技术因子开发
  - 自定义基本面因子开发
  - 复合因子构建
  - 因子缓存与优化
- **交付物**：`03_custom_factors.ipynb`

#### 阶段 4：因子评测与分析
- **目标**：掌握因子质量评估的完整方法论
- **内容**：
  - IC（信息系数）分析
  - 因子收益率分析
  - 因子稳定性测试
  - 因子相关性分析
  - 因子衰减分析
  - 分层回测分析
- **交付物**：`04_factor_evaluation.ipynb`

#### 阶段 5：策略开发与回测
- **目标**：构建完整的量化交易策略并进行回测
- **内容**：
  - 策略框架设计
  - 信号生成逻辑
  - 组合构建方法
  - 风险管理模块
  - 回测引擎使用
  - 策略性能评估
- **交付物**：`05_strategy_backtest.ipynb`

#### 阶段 6：高级功能与优化
- **目标**：掌握 Qlib 的高级功能和性能优化技巧
- **内容**：
  - 机器学习模型集成
  - 在线学习与模型更新
  - 高频数据处理
  - 分布式计算优化
  - 实盘交易接口
- **交付物**：`06_advanced_features.ipynb`

### 开发规范

#### Notebook 命名规范
- 格式：`{序号}_{功能描述}.ipynb`
- 示例：`01_data_import.ipynb`、`02_existing_factors.ipynb`

#### Notebook 结构规范
每个 Notebook 应包含以下标准结构：
1. **项目标题与目标**
2. **环境设置与依赖导入**
3. **核心功能实现**
4. **示例与测试**
5. **性能分析与优化**
6. **总结与下一步计划**

#### 代码质量要求
- 函数化编程，避免重复代码
- 完整的参数验证和错误处理
- 详细的文档字符串和注释
- 进度条显示长时间运行的操作
- 性能优化（使用向量化操作，避免循环）

### 当前进度

- [x] 项目计划制定 ✅ (2025-01-08)
- [x] 阶段 1：数据导入功能开发 ✅ (2025-01-08)
  - 创建了完整的项目目录结构
  - 开发了数据导入notebook (`01_data_import.ipynb`)
  - 实现了数据处理工具模块 (`data_utils.py`)
  - 实现了Qlib格式转换器 (`qlib_converter.py`)
  - 创建了测试脚本验证功能完整性
  - 生成了示例数据并成功转换为Qlib格式
- [ ] 阶段 2：现有因子测试
- [ ] 阶段 3：自定义因子开发
- [ ] 阶段 4：因子评测与分析
- [ ] 阶段 5：策略开发与回测
- [ ] 阶段 6：高级功能与优化

### 阶段 1 详细完成情况

#### 已创建的文件和功能：
1. **项目结构**：
   - `notebooks/` - 存放学习notebook
   - `data/raw/` - 原始数据存储
   - `data/processed/` - 处理后数据存储
   - `data/qlib_data/` - Qlib格式数据存储
   - `utils/` - 工具函数模块

2. **核心文件**：
   - `notebooks/01_data_import.ipynb` - 数据导入主notebook
   - `utils/data_utils.py` - 数据处理工具类
   - `utils/qlib_converter.py` - Qlib格式转换器
   - `test_data_import.py` - 功能测试脚本
   - `README.md` - 项目说明文档

3. **实现的功能**：
   - Parquet文件读取和验证
   - 数据清洗和优化（内存使用优化66.3%）
   - 按股票代码分组导出CSV文件
   - CSV到Qlib格式的转换
   - 完整的数据处理流水线
   - 进度条显示和性能监控
   - 错误处理和日志记录

4. **测试验证**：
   - 生成了20只股票、2年期间的示例数据（10,400条记录）
   - 成功转换为3个CSV文件（测试用）
   - 验证了完整的数据转换流程
   - 所有功能测试通过 ✅
