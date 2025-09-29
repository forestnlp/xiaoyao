import requests
import json
import time
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

class LocalLLMClient:
    """
    本地LLM客户端 - 一次性对话版本
    兼容OpenAI API格式，支持计时功能
    """
    
    def __init__(self, 
                 base_url: str = "http://192.168.7.88:11434/v1",
                 model_name: str = "qwen3:30b-a3b-instruct-2507-q4_K_M",
                 timeout: int = 30,
                 log_requests: bool = True,
                 log_responses: bool = True):
        """
        初始化LLM客户端
        
        Args:
            base_url: API基础地址
            model_name: 模型名称
            timeout: 请求超时时间（秒）
            log_requests: 是否记录请求日志
            log_responses: 是否记录响应日志
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.session = requests.Session()
        
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
            self.logger.info(f"用户消息: {messages[-1]['content'] if messages else '无内容'}")
            self.logger.info(f"参数: {kwargs}")
    
    def _log_response(self, response: str, elapsed_time: float):
        """记录响应日志"""
        if self.log_responses:
            self.logger.info(f"响应内容: {response[:200]}...")  # 只记录前200字符
            self.logger.info(f"响应时间: {elapsed_time:.2f}毫秒")
    
    def chat_completion(self,
                       messages: List[Dict[str, str]],
                       model: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None,
                       stream: bool = False,
                       **kwargs) -> str:
        """
        发送聊天完成请求
        
        Args:
            messages: 消息列表
            model: 模型名称（使用默认模型如果为None）
            temperature: 温度参数
            max_tokens: 最大返回token数（None表示使用模型默认的最大token数）
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
            "stream": stream
        }
        
        # 只在指定max_tokens时添加到payload
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
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
    
    def single_chat(self, 
                   user_message: str, 
                   system_prompt: Optional[str] = None,
                   temperature: float = 0.7,
                   max_tokens: Optional[int] = None,
                   **kwargs) -> Dict[str, Union[str, float]]:
        """
        一次性对话功能（带计时）
        
        Args:
            user_message: 用户消息
            system_prompt: 系统提示词（可选）
            temperature: 温度参数
            max_tokens: 最大token数（None表示使用模型默认的最大token数）
            **kwargs: 其他参数
            
        Returns:
            包含响应内容和耗时的字典
        """
        start_time = time.time() * 1000  # 转换为毫秒
        
        # 构建消息列表
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        
        # 记录请求
        log_params = {"temperature": temperature}
        if max_tokens is not None:
            log_params["max_tokens"] = max_tokens
        log_params.update(kwargs)
        self._log_request(messages, **log_params)
        
        try:
            # 发送请求
            chat_params = {
                "messages": messages,
                "temperature": temperature
            }
            if max_tokens is not None:
                chat_params["max_tokens"] = max_tokens
            chat_params.update(kwargs)
            
            response = self.chat_completion(**chat_params)
            
            # 计算耗时
            elapsed_time = (time.time() * 1000) - start_time
            
            # 记录响应
            self._log_response(response, elapsed_time)
            
            return {
                "response": response,
                "elapsed_time_ms": elapsed_time,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            elapsed_time = (time.time() * 1000) - start_time
            self.logger.error(f"请求失败: {str(e)}")
            return {
                "response": None,
                "elapsed_time_ms": elapsed_time,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# 便捷函数
def create_client(**kwargs) -> LocalLLMClient:
    """
    创建LLM客户端的便捷函数
    
    Returns:
        LocalLLMClient实例
    """
    return LocalLLMClient(**kwargs)


def quick_chat(message: str, 
               system_prompt: Optional[str] = None,
               max_tokens: Optional[int] = None,
               **kwargs) -> str:
    """
    快速聊天函数
    
    Args:
        message: 用户消息
        system_prompt: 系统提示词
            max_tokens: 最大token数（None表示使用模型默认的最大token数）
            **kwargs: 其他参数
        
    Returns:
        模型响应
    """
    client = create_client()
    result = client.single_chat(
        user_message=message,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        **kwargs
    )
    return result["response"] if result["status"] == "success" else f"请求失败: {result['error']}"


if __name__ == "__main__":
    # 演示使用
    print("=== 本地LLM客户端演示 ===\n")
    
    # 创建客户端
    client = create_client()
    
    # # 1. 基本聊天功能
    # print("1. 基本聊天测试:")
    # result = client.single_chat("草莓的英文里有几个r")
    # print(f"响应: {result['response']}")
    # print(f"耗时: {result['elapsed_time_ms']:.2f}毫秒\n")
    
    # # 2. 带系统提示词的聊天
    # print("2. 带系统提示词的聊天:")
    # system_prompt = "你是一个专业的量化交易分析师，请用专业的角度回答问题。"
    # result = client.single_chat("请解释什么是夏普比率", system_prompt=system_prompt)
    # print(f"响应: {result['response']}")
    # print(f"耗时: {result['elapsed_time_ms']:.2f}毫秒\n")
    
    # # 3. 使用便捷函数
    # print("3. 使用便捷函数:")
    # response = quick_chat("你好，请介绍你自己。")
    # print(f"响应: {response}\n")

    system_prompt = "你是一个专业的量化交易分析师，请用专业的角度回答问题，请解释以下新闻对那些行业有利好或者利空，以及你的信念。"
    news = '''央视网消息
            （新闻联播）：西藏、宁夏认真贯彻落实习近平总书记在全国两会上的重要讲话精神，立足自身资源优势，坚持生态优先、绿色发展，加快绿色低碳产业布局，推动经济高质量发展。
            西藏坚持生态优先 推动高原经济绿色低碳发展
            这几天，拉萨南北山绿化工程正在进行新一轮植树造林。这个绿化工程是西藏首个规模化山体生态修复工程，计划用十年时间完成营造林206.72万亩，进一步筑牢生态安全屏障。
            全力保护高原生态，持续构建绿色发展方式。今年，西藏将启动林草碳汇项目开发交易试点，建设农业产业化联合体，发展智慧农牧业，持续在工业、城乡、交通、公共机构等重点领域节能降碳。同时，充分利用水、风、光等资源，发展清洁能源产业，推进120万千瓦风电和75万千瓦光热项目建设，构建西藏新型电力系统示范区和清洁可再生能源利用示范区，为西藏高质量发展注入新动力。
            宁夏积极培育绿色生产力 加快推动可持续发展
            近日，宁夏闽宁镇重点规划配套项目——“绿电小镇”共享储能电站进入后期调试。这个储能电站相当于一个超大型“充电宝”，通过转化来自周边的光伏、风能等新能源发电，运行后将满足闽宁镇6万多人生产生活用电，预计年消纳绿电6.47亿千瓦时，减少二氧化碳排放22.03万吨。
            眼下，宁夏绿电园区提速建设，绿色生产生活方式加快形成。在牧区，探索智能化养殖，形成新的生态养殖方式。在老工业城市石嘴山，昔日的煤渣山已成为市民热门旅游地。宁夏以新能源开发为引擎，带动产业转型升级，加速打造风能、光伏板、新型储能等千亿级新兴产业集群。'''
    result = client.single_chat(news, system_prompt=system_prompt)
    print(f"响应: {result['response']}")
    print(f"耗时: {result['elapsed_time_ms']:.2f}毫秒\n")