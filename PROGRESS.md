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
