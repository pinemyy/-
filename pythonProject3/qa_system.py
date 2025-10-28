"""
智能问答系统 - 数字识别任务型问答智能体
实现管道架构：NLU -> DST -> Policy -> NLG
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

class Intent(Enum):
    """用户意图枚举"""
    GREETING = "greeting"
    HOW_TO_USE = "how_to_use"
    RECOGNIZE_NUMBER = "recognize_number"
    RECOGNIZE_CARD = "recognize_card"
    RECOGNIZE_ID = "recognize_id"
    TROUBLESHOOT = "troubleshoot"
    MODEL_COMPARISON = "model_comparison"
    UPLOAD_HELP = "upload_help"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"

class DialogueState(Enum):
    """对话状态枚举"""
    INITIAL = "initial"
    ASKING_IMAGE_TYPE = "asking_image_type"
    ASKING_UPLOAD_METHOD = "asking_upload_method"
    PROVIDING_GUIDANCE = "providing_guidance"
    TROUBLESHOOTING = "troubleshooting"
    COMPARING_MODELS = "comparing_models"
    COMPLETED = "completed"

class QASystem:
    """智能问答系统主类"""
    
    def __init__(self):
        self.dialogue_state = DialogueState.INITIAL
        self.context = {
            "user_id": None,
            "session_id": None,
            "current_task": None,
            "image_type": None,
            "upload_method": None,
            "troubleshooting_step": 0,
            "conversation_history": []
        }
        
        # 意图识别关键词
        self.intent_keywords = {
            Intent.GREETING: ["你好", "hi", "hello", "您好", "开始", "帮助", "hi", "hey", "早上好", "下午好", "晚上好"],
            Intent.HOW_TO_USE: ["怎么用", "如何使用", "怎么识别", "怎么上传", "操作", "步骤", "教程", "指导", "使用方法", "怎么开始"],
            Intent.RECOGNIZE_NUMBER: ["数字", "手写数字", "0-9", "单个数字", "数字识别", "手写", "数字0", "数字1", "数字2", "数字3", "数字4", "数字5", "数字6", "数字7", "数字8", "数字9"],
            Intent.RECOGNIZE_CARD: ["银行卡", "卡号", "银行卡号", "信用卡", "卡号识别", "银行", "卡片", "卡", "借记卡", "储蓄卡"],
            Intent.RECOGNIZE_ID: ["身份证", "身份证号", "证件号", "身份证识别", "证件", "身份", "id", "证件识别"],
            Intent.TROUBLESHOOT: ["问题", "错误", "失败", "不准确", "识别不了", "怎么办", "出错", "异常", "bug", "故障", "不工作", "无法识别", "识别错误"],
            Intent.MODEL_COMPARISON: ["哪个好", "区别", "比较", "准确率", "模型", "哪个模型", "模型对比", "性能", "效果", "哪个准确", "推荐"],
            Intent.UPLOAD_HELP: ["上传", "文件", "图片", "格式", "大小", "如何上传", "上传图片", "文件格式", "图片格式", "支持格式"],
            Intent.GOODBYE: ["再见", "bye", "结束", "退出", "谢谢", "拜拜", "结束对话", "关闭", "退出系统"]
        }
        
        # 对话策略模板
        self.response_templates = {
            Intent.GREETING: "您好！我是数字识别系统的智能助手。我可以帮助您了解如何使用我们的数字识别功能，包括手写数字、银行卡号、身份证号等识别任务。请告诉我您想要识别什么类型的数字？",
            Intent.HOW_TO_USE: "我来为您详细介绍数字识别系统的使用方法。我们支持多种识别模式，包括在线手写、文件上传等。首先，请告诉我您想要识别什么类型的数字？",
            Intent.RECOGNIZE_NUMBER: "好的！手写数字识别是我们的核心功能。我来指导您完成识别过程。请告诉我您准备如何上传图片？",
            Intent.RECOGNIZE_CARD: "银行卡号识别需要特殊处理。由于银行卡号通常较长且格式特殊，建议使用CNN模型获得最佳效果。请告诉我您遇到的具体问题是什么？",
            Intent.RECOGNIZE_ID: "身份证号识别比较复杂。需要确保图片清晰、光线充足，建议使用高分辨率拍摄。请告诉我您当前的操作步骤。",
            Intent.TROUBLESHOOT: "我来帮您解决问题。请告诉我具体遇到了什么错误或问题？我会根据您的情况提供针对性的解决方案。",
            Intent.MODEL_COMPARISON: "我们系统有三种识别模型，各有特色。我来为您详细介绍它们的区别和适用场景。",
            Intent.UPLOAD_HELP: "关于图片上传，我来为您提供详细指导。我们支持多种格式和上传方式。",
            Intent.GOODBYE: "感谢使用数字识别系统！如有其他问题，随时欢迎咨询。祝您使用愉快！"
        }

    def process_message(self, user_input: str, user_id: str = None) -> Dict[str, Any]:
        """处理用户消息的主入口"""
        # 更新上下文
        if user_id:
            self.context["user_id"] = user_id
        if not self.context["session_id"]:
            self.context["session_id"] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 记录对话历史
        self.context["conversation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "dialogue_state": self.dialogue_state.value
        })
        
        # 管道处理
        intent = self._nlu(user_input)
        self._dst(intent, user_input)
        response_strategy = self._policy(intent)
        response = self._nlg(response_strategy, user_input)
        
        return {
            "response": response,
            "intent": intent.value,
            "dialogue_state": self.dialogue_state.value,
            "context": self.context.copy()
        }

    def _nlu(self, user_input: str) -> Intent:
        """自然语言理解模块"""
        user_input_lower = user_input.lower()
        
        # 计算每个意图的匹配分数
        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in user_input_lower:
                    score += 1
            intent_scores[intent] = score
        
        # 返回得分最高的意图
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            if intent_scores[best_intent] > 0:
                return best_intent
        
        return Intent.UNKNOWN

    def _dst(self, intent: Intent, user_input: str):
        """对话状态跟踪模块"""
        if intent == Intent.GREETING:
            self.dialogue_state = DialogueState.INITIAL
        elif intent == Intent.HOW_TO_USE:
            self.dialogue_state = DialogueState.ASKING_IMAGE_TYPE
        elif intent in [Intent.RECOGNIZE_NUMBER, Intent.RECOGNIZE_CARD, Intent.RECOGNIZE_ID]:
            self.dialogue_state = DialogueState.ASKING_UPLOAD_METHOD
            self.context["current_task"] = intent.value
        elif intent == Intent.TROUBLESHOOT:
            self.dialogue_state = DialogueState.TROUBLESHOOTING
        elif intent == Intent.MODEL_COMPARISON:
            self.dialogue_state = DialogueState.COMPARING_MODELS
        elif intent == Intent.GOODBYE:
            self.dialogue_state = DialogueState.COMPLETED

    def _policy(self, intent: Intent) -> Dict[str, Any]:
        """对话策略模块"""
        strategy = {
            "intent": intent,
            "action": "provide_guidance",
            "parameters": {}
        }
        
        if intent == Intent.RECOGNIZE_NUMBER:
            strategy["action"] = "guide_number_recognition"
            strategy["parameters"] = {
                "steps": [
                    "准备清晰的手写数字图片",
                    "选择上传方式（文件上传或在线绘制）",
                    "选择合适的识别模型",
                    "查看识别结果"
                ]
            }
        elif intent == Intent.RECOGNIZE_CARD:
            strategy["action"] = "guide_card_recognition"
            strategy["parameters"] = {
                "steps": [
                    "确保银行卡号清晰可见",
                    "避免反光和阴影",
                    "使用高分辨率拍摄",
                    "选择CNN模型进行识别"
                ]
            }
        elif intent == Intent.MODEL_COMPARISON:
            strategy["action"] = "compare_models"
            strategy["parameters"] = {
                "models": {
                    "perceptron": "多层感知机：适合简单数字识别，速度快",
                    "naive_bayes": "朴素贝叶斯：适合二值化图像，计算简单",
                    "cnn": "卷积神经网络：准确率最高，适合复杂图像"
                }
            }
        elif intent == Intent.TROUBLESHOOT:
            strategy["action"] = "troubleshoot"
            strategy["parameters"] = {
                "common_issues": [
                    "图片不清晰 - 建议重新拍摄",
                    "数字太小 - 建议放大后识别",
                    "背景复杂 - 建议使用纯色背景",
                    "识别不准确 - 建议尝试不同模型"
                ]
            }
        
        return strategy

    def _nlg(self, strategy: Dict[str, Any], user_input: str) -> str:
        """自然语言生成模块"""
        intent = strategy["intent"]
        action = strategy["action"]
        parameters = strategy["parameters"]
        
        # 基础回复
        base_response = self.response_templates.get(intent, "我理解您的需求，让我为您提供帮助。")
        
        # 根据策略生成具体回复
        if action == "guide_number_recognition":
            steps = parameters.get("steps", [])
            response = f"{base_response}\n\n 手写数字识别步骤：\n"
            for i, step in enumerate(steps, 1):
                response += f"{i}. {step}\n"
            response += "\n 小贴士：\n"
            response += "• 确保数字清晰可见，避免模糊\n"
            response += "• 建议使用纯色背景\n"
            response += "• 数字大小适中，不要太小\n"
            response += "• 可以尝试不同的识别模型\n\n"
            response += "您想了解哪个步骤的详细信息？"
            
        elif action == "guide_card_recognition":
            steps = parameters.get("steps", [])
            response = f"{base_response}\n\n 银行卡号识别要点：\n"
            for i, step in enumerate(steps, 1):
                response += f"{i}. {step}\n"
            response += "\n 推荐设置：\n"
            response += "• 使用CNN模型（准确率最高）\n"
            response += "• 图片分辨率不低于800x600\n"
            response += "• 避免反光和阴影\n"
            response += "• 确保卡号完整可见\n\n"
            response += "需要我帮您解决具体问题吗？"
            
        elif action == "compare_models":
            models = parameters.get("models", {})
            response = f"{base_response}\n\n️ 模型详细对比：\n\n"
            for model_name, description in models.items():
                response += f"🔹 {model_name.upper()}模型：\n"
                response += f"   {description}\n\n"
            
            response += " 性能对比：\n"
            response += "• 准确率：CNN > 朴素贝叶斯 > 感知机\n"
            response += "• 速度：感知机 > 朴素贝叶斯 > CNN\n"
            response += "• 适用场景：\n"
            response += "  - 简单数字：感知机\n"
            response += "  - 一般图像：朴素贝叶斯\n"
            response += "  - 复杂图像：CNN\n\n"
            response += "建议根据您的具体需求选择合适的模型。"
            
        elif action == "troubleshoot":
            issues = parameters.get("common_issues", [])
            response = f"{base_response}\n\n 常见问题解决方案：\n\n"
            for i, issue in enumerate(issues, 1):
                response += f"{i}. {issue}\n"
            
            response += "\n 如果问题仍未解决：\n"
            response += "• 检查图片格式（支持JPG、PNG）\n"
            response += "• 确认图片大小不超过10MB\n"
            response += "• 尝试重新上传图片\n"
            response += "• 更换不同的识别模型\n"
            response += "• 提供更清晰的图片\n\n"
            response += "请告诉我具体遇到了什么问题，我会提供更详细的帮助。"
            
        else:
            response = base_response
        
        return response

    def reset_session(self):
        """重置对话会话"""
        self.dialogue_state = DialogueState.INITIAL
        self.context = {
            "user_id": None,
            "session_id": None,
            "current_task": None,
            "image_type": None,
            "upload_method": None,
            "troubleshooting_step": 0,
            "conversation_history": []
        }

# 全局问答系统实例
qa_system = QASystem()
