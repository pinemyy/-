#!/usr/bin/env python3
"""
测试概率显示功能
"""

import numpy as np
from qa_system import qa_system

def test_qa_system():
    """测试智能问答系统"""
    print("=== 智能问答系统测试 ===")
    
    # 测试各种意图识别
    test_messages = [
        "你好",
        "如何识别手写数字？",
        "银行卡号怎么识别？",
        "哪个模型准确率最高？",
        "识别不准确怎么办？",
        "如何上传图片？",
        "再见"
    ]
    
    for message in test_messages:
        print(f"\n用户: {message}")
        response = qa_system.process_message(message)
        print(f"AI: {response['response']}")
        print(f"意图: {response['intent']}")
        print(f"对话状态: {response['dialogue_state']}")

def test_probability_calculation():
    """测试概率计算"""
    print("\n=== 概率计算测试 ===")
    
    # 模拟概率数据
    probabilities = {
        '0': 0.05,
        '1': 0.02,
        '2': 0.15,
        '3': 0.08,
        '4': 0.12,
        '5': 0.03,
        '6': 0.25,
        '7': 0.10,
        '8': 0.15,
        '9': 0.05
    }
    
    print("模拟概率分布:")
    for digit, prob in probabilities.items():
        print(f"数字 {digit}: {prob:.1%}")
    
    # 找到最高概率的数字
    max_digit = max(probabilities, key=probabilities.get)
    max_prob = probabilities[max_digit]
    print(f"\n最高概率: 数字 {max_digit} ({max_prob:.1%})")

if __name__ == "__main__":
    test_qa_system()
    test_probability_calculation()
    print("\n=== 测试完成 ===")
