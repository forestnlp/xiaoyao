import requests
import json
import time
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

class MultiRoundLLMClient:
    """
    多轮对话LLM客户端
    支持上下文保持的多轮对话，兼容OpenAI API格式
    """
    
    def __init__(self, 
                 base_url: str = "http://192.168.7.88:11434/v1",
                 model_name: str = "qwen3:30b-a3b-instruct-2507-q4_K_M",
                 timeout: int = 30,
                 log_requests: bool = True,
                 log_responses: bool = True,
                 max_context_length: int = 4000):
        """
        初始化多轮对话LLM客户端
        
        Args:
            base_url: API基础地址
            model_name: 模型名称
            timeout: 请求超时时间（秒）
            log_requests: 是否记录请求日志
            log_responses: 是否记录响应日志
            max_context_length: 最大上下文长度
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.max_context_length = max_context_length
        self.session = requests.Session()
        
        # 对话历史
        self.conversation_history = []
        self.current_system_prompt = None
        
        # 设置日志
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _log_request(self, messages: List[Dict], **kwargs):
        """记录请求日志"""
        if self.log_requests:
            self.logger.info(f"请求模型: {self.model_name}")
            self.logger.info(f"消息数量: {len(messages)}")
            self.logger.info(f"最后消息: {messages[-1]['content'][:100]}..." if messages else '无内容')
            self.logger.info(f"参数: {kwargs}")
    
    def _log_response(self, response: str, elapsed_time: float):
        """记录响应日志"""
        if self.log_responses:
            self.logger.info(f"响应内容: {response[:200]}...")  # 只记录前200字符
            self.logger.info(f"响应时间: {elapsed_time:.2f}毫秒")
    
    def start_new_conversation(self, system_prompt: Optional[str] = None):
        """
        开始新的对话会话
        
        Args:
            system_prompt: 系统提示词（可选）
        """
        self.conversation_history = []
        self.current_system_prompt = system_prompt
        
        if system_prompt:
            self.conversation_history.append({
                "role": "system", 
                "content": system_prompt
            })
        
        self.logger.info("开始新的对话会话")
    
    def add_message(self, role: str, content: str):
        """
        添加消息到对话历史
        
        Args:
            role: 消息角色 (user/assistant/system)
            content: 消息内容
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # 检查上下文长度，必要时进行截断
        self._manage_context_length()
    
    def _manage_context_length(self):
        """管理上下文长度，避免超出限制"""
        total_length = sum(len(msg["content"]) for msg in self.conversation_history)
        
        # 如果超出最大长度，保留系统消息和最近的对话
        while total_length > self.max_context_length and len(self.conversation_history) > 2:
            # 移除最旧的用户/助手消息对
            if len(self.conversation_history) > 2:
                # 保留系统消息（如果有）
                if self.conversation_history[0]["role"] == "system":
                    removed = self.conversation_history.pop(1)  # 移除第一个用户消息
                    if self.conversation_history and self.conversation_history[1]["role"] == "assistant":
                        removed = self.conversation_history.pop(1)  # 移除对应的助手消息
                else:
                    removed = self.conversation_history.pop(0)  # 移除最旧的消息
                
                total_length = sum(len(msg["content"]) for msg in self.conversation_history)
                self.logger.info(f"截断对话历史，移除消息: {removed['content'][:50]}...")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        获取当前对话历史
        
        Returns:
            对话历史列表
        """
        return self.conversation_history.copy()
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        if self.current_system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": self.current_system_prompt
            })
        self.logger.info("清空对话历史")
    
    def chat_completion(self,
                       messages: List[Dict[str, str]],
                       model: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       stream: bool = False,
                       **kwargs) -> str:
        """
        发送聊天完成请求
        
        Args:
            messages: 消息列表
            model: 模型名称（使用默认模型如果为None）
            temperature: 温度参数
            max_tokens: 最大返回token数
            stream: 是否流式返回
            **kwargs: 其他参数
            
        Returns:
            模型响应内容
        """
        url = f"{self.base_url}/chat/completions"
        
        # 使用指定模型或默认模型
        current_model = model or self.model_name
        
        payload = {
            "model": current_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # 添加其他参数
        payload.update(kwargs)
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_stream_response(response)
            else:
                result = response.json()
                return result['choices'][0]['message']['content']
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"响应解析失败: {str(e)}")
        except KeyError as e:
            raise Exception(f"响应格式错误: {str(e)}")
    
    def _handle_stream_response(self, response) -> str:
        """处理流式响应"""
        content = ""
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    data_str = line_text[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content += delta['content']
                    except json.JSONDecodeError:
                        continue
        return content
    
    def send_message(self, 
                    user_message: str, 
                    temperature: float = 0.7,
                    max_tokens: int = 1000,
                    **kwargs) -> Dict[str, Union[str, float]]:
        """
        发送消息（多轮对话）
        
        Args:
            user_message: 用户消息
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数
            
        Returns:
            包含响应内容和耗时的字典
        """
        start_time = time.time() * 1000  # 转换为毫秒
        
        # 添加用户消息到历史
        self.add_message("user", user_message)
        
        # 记录请求
        self._log_request(self.conversation_history, temperature=temperature, max_tokens=max_tokens, **kwargs)
        
        try:
            # 发送请求（包含完整对话历史）
            response = self.chat_completion(
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # 添加助手响应到历史
            self.add_message("assistant", response)
            
            # 计算耗时
            elapsed_time = (time.time() * 1000) - start_time
            
            # 记录响应
            self._log_response(response, elapsed_time)
            
            return {
                "response": response,
                "elapsed_time_ms": elapsed_time,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "conversation_turns": len(self.conversation_history)
            }
            
        except Exception as e:
            elapsed_time = (time.time() * 1000) - start_time
            self.logger.error(f"请求失败: {str(e)}")
            return {
                "response": None,
                "elapsed_time_ms": elapsed_time,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "conversation_turns": len(self.conversation_history)
            }
    
    def get_last_exchange(self) -> Optional[Dict[str, str]]:
        """
        获取最后一次对话交换
        
        Returns:
            包含用户消息和助手响应的字典
        """
        if len(self.conversation_history) >= 2:
            # 找到最后一条用户消息和对应的助手响应
            for i in range(len(self.conversation_history) - 1, 0, -1):
                if self.conversation_history[i-1]["role"] == "user" and \
                   self.conversation_history[i]["role"] == "assistant":
                    return {
                        "user": self.conversation_history[i-1]["content"],
                        "assistant": self.conversation_history[i]["content"]
                    }
        return None
    
    def export_conversation(self, filename: Optional[str] = None) -> str:
        """
        导出对话历史到文件
        
        Args:
            filename: 文件名（可选，自动生成如果为None）
            
        Returns:
            导出的文件名
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
        
        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "total_turns": len(self.conversation_history),
            "conversation": self.conversation_history
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"对话历史已导出到: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"导出对话历史失败: {str(e)}")
            raise


# 便捷函数
def create_multi_round_client(**kwargs) -> MultiRoundLLMClient:
    """
    创建多轮对话LLM客户端的便捷函数
    
    Returns:
        MultiRoundLLMClient实例
    """
    return MultiRoundLLMClient(**kwargs)


if __name__ == "__main__":
    # 演示使用
    print("=== 多轮对话LLM客户端演示 ===\n")
    
    # 创建客户端
    client = create_multi_round_client()
    
    # 1. 开始新的对话会话
    print("1. 开始新的对话会话:")
    system_prompt = "你是一个专业的量化交易分析师，请用专业的角度回答问题。"
    client.start_new_conversation(system_prompt)
    print("系统提示词已设置\n")
    
    # 2. 第一轮对话
    print("2. 第一轮对话:")
    result = client.send_message("请解释什么是夏普比率")
    print(f"用户: 请解释什么是夏普比率")
    print(f"助手: {result['response']}")
    print(f"耗时: {result['elapsed_time_ms']:.2f}毫秒\n")
    
    # 3. 第二轮对话（基于上下文）
    print("3. 第二轮对话（基于上下文）:")
    result = client.send_message("它在实际交易中如何应用？")
    print(f"用户: 它在实际交易中如何应用？")
    print(f"助手: {result['response']}")
    print(f"耗时: {result['elapsed_time_ms']:.2f}毫秒\n")
    
    # 4. 第三轮对话
    print("4. 第三轮对话:")
    result = client.send_message("能否举个具体的例子？")
    print(f"用户: 能否举个具体的例子？")
    print(f"助手: {result['response']}")
    print(f"耗时: {result['elapsed_time_ms']:.2f}毫秒\n")
    
    # 5. 查看对话历史
    print("5. 查看对话历史:")
    history = client.get_conversation_history()
    print(f"总对话轮数: {len(history)}")
    for i, msg in enumerate(history):
        print(f"{i+1}. {msg['role']}: {msg['content'][:50]}...")
    
    # 6. 获取最后一次对话
    print("\n6. 最后一次对话:")
    last_exchange = client.get_last_exchange()
    if last_exchange:
        print(f"用户: {last_exchange['user']}")
        print(f"助手: {last_exchange['assistant']}")
    
    # 7. 导出对话历史
    print("\n7. 导出对话历史:")
    filename = client.export_conversation()
    print(f"对话历史已导出到: {filename}")
    
    # 8. 开始新的对话（清空历史）
    print("\n8. 开始新的对话会话:")
    client.start_new_conversation("你是一个友好的AI助手。")
    result = client.send_message("你好，请介绍你自己。")
    print(f"助手: {result['response']}")
    print(f"当前对话轮数: {len(client.get_conversation_history())}")