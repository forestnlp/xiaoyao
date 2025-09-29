#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试max_tokens参数的不同设置
"""

from client import LocalLLMClient

def test_max_tokens():
    """测试不同max_tokens设置的效果"""
    
    client = LocalLLMClient()
    
    # 测试用的长文本
    long_text = """
    请详细解释量化交易中的多因子模型。量化交易是一种利用数学模型和计算机算法进行投资决策的交易方式。
    多因子模型是量化投资中的核心工具之一，它通过分析多个影响股票收益的因素来预测未来收益。
    
    多因子模型的基本思想是，股票的收益可以由一系列共同因子和特异因子来解释。
    共同因子包括市场风险因子、规模因子、价值因子、动量因子等。
    这些因子代表了不同风格的系统性风险，对大多数股票都有影响。
    
    在实际应用中，我们需要考虑因子的选择、因子的有效性检验、因子权重的确定、模型的风险控制等多个方面。
    同时，还需要考虑交易成本、市场冲击、流动性风险等实际交易中的问题。
    
    请从理论基础和实际应用两个角度，详细阐述多因子模型的构建过程、优化方法和风险管理策略。
    """
    
    system_prompt = "你是一个专业的量化交易分析师，请用专业的角度回答问题，回答要详细全面。"
    
    print("=== 测试max_tokens参数的不同设置 ===\n")
    
    # 测试1: max_tokens=None (使用模型默认)
    print("1. max_tokens=None (使用模型默认最大token数):")
    result1 = client.single_chat(
        user_message=long_text,
        system_prompt=system_prompt,
        max_tokens=None,
        temperature=0.7
    )
    print(f"响应长度: {len(result1['response']) if result1['response'] else 0} 字符")
    print(f"耗时: {result1['elapsed_time_ms']:.2f} 毫秒")
    print(f"状态: {result1['status']}")
    if result1['response']:
        print(f"前200字符: {result1['response'][:200]}...")
    print()
    
    # 测试2: max_tokens=500 (限制响应长度)
    print("2. max_tokens=500 (限制响应长度):")
    result2 = client.single_chat(
        user_message=long_text,
        system_prompt=system_prompt,
        max_tokens=500,
        temperature=0.7
    )
    print(f"响应长度: {len(result2['response']) if result2['response'] else 0} 字符")
    print(f"耗时: {result2['elapsed_time_ms']:.2f} 毫秒")
    print(f"状态: {result2['status']}")
    if result2['response']:
        print(f"前200字符: {result2['response'][:200]}...")
    print()
    
    # 测试3: max_tokens=2000 (较大的token限制)
    print("3. max_tokens=2000 (较大的token限制):")
    result3 = client.single_chat(
        user_message=long_text,
        system_prompt=system_prompt,
        max_tokens=2000,
        temperature=0.7
    )
    print(f"响应长度: {len(result3['response']) if result3['response'] else 0} 字符")
    print(f"耗时: {result3['elapsed_time_ms']:.2f} 毫秒")
    print(f"状态: {result3['status']}")
    if result3['response']:
        print(f"前200字符: {result3['response'][:200]}...")
    print()
    
    print("=== 测试完成 ===")
    print("\n总结:")
    print("- max_tokens=None: 使用模型默认的最大token数，适合需要详细回答的场景")
    print("- max_tokens=具体数值: 限制响应长度，适合需要简短回答或控制成本的场景")
    print("- 设置为None可以避免因token限制导致的回答截断问题")

if __name__ == "__main__":
    test_max_tokens()