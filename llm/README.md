# 本地LLM客户端

基于Java示例实现的Python版本地LLM客户端，兼容OpenAI API格式，支持访问本地部署的大模型服务。

## 文件说明

### 1. `client.py` - 一次性对话客户端
- **功能**: 单次对话，无上下文保持
- **特点**: 简单快速，适合独立问题
- **计时功能**: 内置TimeInterval类似功能

### 2. `multi_round_client.py` - 多轮对话客户端
- **功能**: 支持上下文保持的多轮对话
- **特点**: 对话历史管理、上下文长度控制
- **额外功能**: 对话导出、历史查看

## 配置参数

两个客户端都使用相同的API配置：
- **Base URL**: `http://192.168.7.88:11434/v1`
- **模型名称**: `qwen3:30b-a3b-instruct-2507-q4_K_M`
- **超时时间**: 30秒
- **日志记录**: 支持请求和响应日志

## 快速开始

### 一次性对话
```python
from client import create_client

# 创建客户端
client = create_client()

# 发送消息
result = client.single_chat("你好，请介绍你自己。")
print(f"响应: {result['response']}")
print(f"耗时: {result['elapsed_time_ms']:.2f}毫秒")
```

### 多轮对话
```python
from multi_round_client import create_multi_round_client

# 创建客户端
client = create_multi_round_client()

# 开始对话会话
client.start_new_conversation("你是一个专业的量化分析师。")

# 多轮对话
result1 = client.send_message("什么是夏普比率？")
result2 = client.send_message("它在实际交易中如何应用？")
result3 = client.send_message("能否举个具体的例子？")

# 查看历史
history = client.get_conversation_history()
```

## API参考

### 一次性对话客户端

#### `single_chat(user_message, system_prompt=None, **kwargs)`
- **user_message**: 用户消息内容
- **system_prompt**: 系统提示词（可选）
- **返回值**: 包含响应、耗时、状态的字典

#### `chat_completion(messages, **kwargs)`
- **messages**: 消息列表
- **temperature**: 温度参数（默认0.7）
- **max_tokens**: 最大token数（默认1000）
- **stream**: 是否流式返回（默认False）

### 多轮对话客户端

#### `start_new_conversation(system_prompt=None)`
- 开始新的对话会话
- **system_prompt**: 系统提示词（可选）

#### `send_message(user_message, **kwargs)`
- 发送消息（自动维护上下文）
- **user_message**: 用户消息内容
- **返回值**: 包含响应、耗时、对话轮数的字典

#### `get_conversation_history()`
- 获取当前对话历史
- **返回值**: 消息列表

#### `clear_history()`
- 清空对话历史（保留系统提示词）

#### `export_conversation(filename=None)`
- 导出对话到JSON文件
- **filename**: 文件名（可选，自动生成时间戳）

#### `get_last_exchange()`
- 获取最后一次对话交换
- **返回值**: 包含用户消息和助手响应的字典

## 高级用法

### 自定义配置
```python
from client import LocalLLMClient

client = LocalLLMClient(
    base_url="http://your-api-url:11434/v1",
    model_name="your-model-name",
    timeout=60,
    log_requests=True,
    log_responses=True
)
```

### 流式响应
```python
# 启用流式响应
response = client.chat_completion(
    messages=messages,
    stream=True,
    temperature=0.8
)
```

### 温度控制
```python
# 高温度（更随机）
result = client.single_chat("写一首诗", temperature=0.9)

# 低温度（更确定）
result = client.single_chat("解释概念", temperature=0.3)
```

## 错误处理

客户端包含完整的错误处理机制：
- **网络异常**: 连接超时、请求失败
- **API错误**: 响应格式错误、解析失败
- **参数验证**: 输入参数检查

所有错误都会返回包含错误信息的字典：
```python
{
    "response": None,
    "status": "error",
    "error": "具体的错误信息",
    "elapsed_time_ms": 耗时,
    "timestamp": "时间戳"
}
```

## 性能优化

### 上下文管理（多轮对话）
- 自动截断过长的对话历史
- 保留系统消息和最近的相关对话
- 可配置的最大上下文长度

### 连接复用
- 使用requests.Session保持连接
- 减少重复连接开销

## 日志记录

支持详细的日志记录：
- **请求日志**: 模型名称、用户消息、参数
- **响应日志**: 响应内容（前200字符）、耗时
- **错误日志**: 异常信息和堆栈跟踪

## 与Java示例对比

| 功能 | Java版本 | Python版本 |
|------|----------|------------|
| API配置 | ✅ | ✅ |
| 计时功能 | TimeInterval | time.time() |
| 日志记录 | LangChain4j内置 | logging模块 |
| 流式响应 | ❌ | ✅ |
| 多轮对话 | ❌ | ✅ |
| 上下文管理 | ❌ | ✅ |
| 对话导出 | ❌ | ✅ |

## 注意事项

1. **API地址**: 确保本地LLM服务运行在配置地址
2. **模型名称**: 使用正确的模型名称
3. **超时设置**: 根据网络状况调整超时时间
4. **上下文长度**: 多轮对话注意上下文长度限制
5. **日志级别**: 生产环境可适当调整日志级别

## 依赖要求

```bash
pip install requests
```

## 运行环境

- Python 3.7+
- 网络连接（访问本地LLM服务）
- 足够的内存（处理大模型响应）