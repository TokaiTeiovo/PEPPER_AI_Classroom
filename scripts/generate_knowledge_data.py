#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
教育知识图谱数据生成器
生成大量的教育领域知识点和关系数据
"""

import json
import logging
import os
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ai_service.knowledge_graph.knowledge_graph import KnowledgeGraph

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KG_DATA_GENERATOR")


class KnowledgeGraphDataGenerator:
    def __init__(self, kg_uri="bolt://localhost:7687", kg_user="neo4j", kg_password="admin123"):
        self.knowledge_graph = KnowledgeGraph(kg_uri, kg_user, kg_password)

    def generate_programming_knowledge(self):
        """生成编程相关知识"""
        programming_data = []

        # Python编程知识
        python_concepts = [
            {"name": "Python", "description": "一种高级编程语言，语法简洁易读", "category": "编程语言",
             "difficulty": "入门"},
            {"name": "变量", "description": "存储数据值的容器", "category": "基础概念", "difficulty": "入门"},
            {"name": "数据类型", "description": "定义变量可以存储的数据种类", "category": "基础概念",
             "difficulty": "入门"},
            {"name": "整数", "description": "不带小数点的数字", "category": "数据类型", "difficulty": "入门"},
            {"name": "浮点数", "description": "带小数点的数字", "category": "数据类型", "difficulty": "入门"},
            {"name": "字符串", "description": "文本数据类型", "category": "数据类型", "difficulty": "入门"},
            {"name": "布尔值", "description": "True或False的逻辑值", "category": "数据类型", "difficulty": "入门"},
            {"name": "列表", "description": "有序的可变数据集合", "category": "数据结构", "difficulty": "初级"},
            {"name": "元组", "description": "有序的不可变数据集合", "category": "数据结构", "difficulty": "初级"},
            {"name": "字典", "description": "键值对形式的数据结构", "category": "数据结构", "difficulty": "初级"},
            {"name": "集合", "description": "无序不重复元素的集合", "category": "数据结构", "difficulty": "初级"},
            {"name": "if语句", "description": "条件判断语句", "category": "控制结构", "difficulty": "入门"},
            {"name": "for循环", "description": "遍历序列的循环结构", "category": "控制结构", "difficulty": "入门"},
            {"name": "while循环", "description": "基于条件的循环结构", "category": "控制结构", "difficulty": "入门"},
            {"name": "函数", "description": "执行特定任务的可重用代码块", "category": "函数", "difficulty": "初级"},
            {"name": "参数", "description": "传递给函数的输入值", "category": "函数", "difficulty": "初级"},
            {"name": "返回值", "description": "函数执行后返回的结果", "category": "函数", "difficulty": "初级"},
            {"name": "类", "description": "面向对象编程的基本单位", "category": "面向对象", "difficulty": "中级"},
            {"name": "对象", "description": "类的实例", "category": "面向对象", "difficulty": "中级"},
            {"name": "继承", "description": "子类获得父类属性和方法的机制", "category": "面向对象",
             "difficulty": "中级"},
            {"name": "封装", "description": "隐藏内部实现细节的特性", "category": "面向对象", "difficulty": "中级"},
            {"name": "多态", "description": "同一接口的不同实现", "category": "面向对象", "difficulty": "中级"},
            {"name": "模块", "description": "包含Python定义和语句的文件", "category": "模块化", "difficulty": "初级"},
            {"name": "包", "description": "包含多个模块的目录", "category": "模块化", "difficulty": "初级"},
            {"name": "异常处理", "description": "处理程序运行时错误的机制", "category": "错误处理",
             "difficulty": "初级"},
            {"name": "try-except", "description": "异常处理语句块", "category": "错误处理", "difficulty": "初级"},
            {"name": "文件操作", "description": "读写文件的操作", "category": "文件处理", "difficulty": "初级"},
            {"name": "正则表达式", "description": "用于匹配字符串模式的工具", "category": "文本处理",
             "difficulty": "中级"},
        ]

        # 创建Python概念节点
        for concept in python_concepts:
            programming_data.append({
                "type": "node",
                "label": "Concept",
                "properties": concept
            })

        # 创建关系
        relationships = [
            ("Python", "包含", "变量", "Python语言包含变量概念"),
            ("Python", "包含", "数据类型", "Python语言包含多种数据类型"),
            ("Python", "包含", "控制结构", "Python语言包含控制结构"),
            ("数据类型", "包括", "整数", "数据类型包括整数类型"),
            ("数据类型", "包括", "浮点数", "数据类型包括浮点数类型"),
            ("数据类型", "包括", "字符串", "数据类型包括字符串类型"),
            ("数据类型", "包括", "布尔值", "数据类型包括布尔值类型"),
            ("Python", "支持", "列表", "Python支持列表数据结构"),
            ("Python", "支持", "元组", "Python支持元组数据结构"),
            ("Python", "支持", "字典", "Python支持字典数据结构"),
            ("Python", "支持", "集合", "Python支持集合数据结构"),
            ("控制结构", "包括", "if语句", "控制结构包括条件语句"),
            ("控制结构", "包括", "for循环", "控制结构包括for循环"),
            ("控制结构", "包括", "while循环", "控制结构包括while循环"),
            ("for循环", "用于", "遍历", "for循环用于遍历序列"),
            ("while循环", "基于", "条件判断", "while循环基于条件判断"),
            ("函数", "接受", "参数", "函数可以接受参数"),
            ("函数", "返回", "返回值", "函数可以返回值"),
            ("类", "创建", "对象", "类用于创建对象"),
            ("类", "支持", "继承", "类支持继承机制"),
            ("面向对象", "特性", "封装", "面向对象具有封装特性"),
            ("面向对象", "特性", "多态", "面向对象具有多态特性"),
            ("Python", "支持", "模块", "Python支持模块化编程"),
            ("模块", "组成", "包", "多个模块组成包"),
            ("Python", "提供", "异常处理", "Python提供异常处理机制"),
            ("异常处理", "使用", "try-except", "异常处理使用try-except语句"),
        ]

        for start, relation, end, desc in relationships:
            programming_data.append({
                "type": "relationship",
                "start_node": start,
                "end_node": end,
                "relationship": relation,
                "properties": {"description": desc}
            })

        return programming_data

    def generate_ai_knowledge(self):
        """生成人工智能相关知识"""
        ai_data = []

        # 人工智能概念
        ai_concepts = [
            {"name": "人工智能", "description": "模拟人类智能的计算机系统", "category": "AI基础", "difficulty": "入门"},
            {"name": "机器学习", "description": "让计算机从数据中学习的方法", "category": "AI基础",
             "difficulty": "初级"},
            {"name": "深度学习", "description": "基于神经网络的机器学习方法", "category": "AI基础",
             "difficulty": "中级"},
            {"name": "神经网络", "description": "模拟大脑神经元的计算模型", "category": "模型架构",
             "difficulty": "中级"},
            {"name": "卷积神经网络", "description": "用于图像处理的神经网络", "category": "模型架构",
             "difficulty": "中级"},
            {"name": "循环神经网络", "description": "处理序列数据的神经网络", "category": "模型架构",
             "difficulty": "中级"},
            {"name": "Transformer", "description": "基于注意力机制的神经网络架构", "category": "模型架构",
             "difficulty": "高级"},
            {"name": "监督学习", "description": "使用标记数据进行学习", "category": "学习方式", "difficulty": "初级"},
            {"name": "无监督学习", "description": "从无标记数据中发现模式", "category": "学习方式",
             "difficulty": "初级"},
            {"name": "强化学习", "description": "通过试错和奖励进行学习", "category": "学习方式", "difficulty": "中级"},
            {"name": "自然语言处理", "description": "处理和理解人类语言的技术", "category": "AI应用",
             "difficulty": "中级"},
            {"name": "计算机视觉", "description": "让计算机理解和分析图像", "category": "AI应用", "difficulty": "中级"},
            {"name": "语音识别", "description": "将语音转换为文本", "category": "AI应用", "difficulty": "初级"},
            {"name": "语音合成", "description": "将文本转换为语音", "category": "AI应用", "difficulty": "初级"},
            {"name": "推荐系统", "description": "为用户推荐相关内容的系统", "category": "AI应用", "difficulty": "初级"},
            {"name": "聊天机器人", "description": "能与人对话的AI系统", "category": "AI应用", "difficulty": "初级"},
            {"name": "大语言模型", "description": "参数规模巨大的语言模型", "category": "模型类型",
             "difficulty": "高级"},
            {"name": "GPT", "description": "生成式预训练Transformer模型", "category": "模型类型", "difficulty": "高级"},
            {"name": "BERT", "description": "双向编码器表示的Transformer", "category": "模型类型",
             "difficulty": "高级"},
            {"name": "训练数据", "description": "用于训练模型的数据集", "category": "数据", "difficulty": "入门"},
            {"name": "测试数据", "description": "用于评估模型性能的数据", "category": "数据", "difficulty": "入门"},
            {"name": "特征工程", "description": "设计和选择模型输入特征", "category": "数据处理", "difficulty": "初级"},
            {"name": "数据预处理", "description": "清理和准备训练数据", "category": "数据处理", "difficulty": "入门"},
            {"name": "过拟合", "description": "模型在训练数据上表现好但泛化差", "category": "模型问题",
             "difficulty": "初级"},
            {"name": "欠拟合", "description": "模型过于简单无法捕捉数据模式", "category": "模型问题",
             "difficulty": "初级"},
        ]

        # 创建AI概念节点
        for concept in ai_concepts:
            ai_data.append({
                "type": "node",
                "label": "Concept",
                "properties": concept
            })

        # AI领域关系
        ai_relationships = [
            ("人工智能", "包含", "机器学习", "人工智能包含机器学习"),
            ("机器学习", "包含", "深度学习", "机器学习包含深度学习"),
            ("深度学习", "基于", "神经网络", "深度学习基于神经网络"),
            ("神经网络", "类型", "卷积神经网络", "卷积神经网络是神经网络的一种"),
            ("神经网络", "类型", "循环神经网络", "循环神经网络是神经网络的一种"),
            ("Transformer", "是", "神经网络", "Transformer是一种神经网络架构"),
            ("机器学习", "方式", "监督学习", "监督学习是机器学习方式之一"),
            ("机器学习", "方式", "无监督学习", "无监督学习是机器学习方式之一"),
            ("机器学习", "方式", "强化学习", "强化学习是机器学习方式之一"),
            ("自然语言处理", "应用", "聊天机器人", "自然语言处理应用于聊天机器人"),
            ("自然语言处理", "使用", "Transformer", "自然语言处理常使用Transformer"),
            ("计算机视觉", "使用", "卷积神经网络", "计算机视觉常使用卷积神经网络"),
            ("语音识别", "属于", "自然语言处理", "语音识别属于自然语言处理"),
            ("语音合成", "属于", "自然语言处理", "语音合成属于自然语言处理"),
            ("大语言模型", "基于", "Transformer", "大语言模型基于Transformer架构"),
            ("GPT", "是", "大语言模型", "GPT是一种大语言模型"),
            ("BERT", "是", "大语言模型", "BERT是一种大语言模型"),
            ("训练数据", "用于", "监督学习", "训练数据用于监督学习"),
            ("测试数据", "评估", "模型性能", "测试数据用于评估模型性能"),
            ("特征工程", "影响", "模型性能", "特征工程影响模型性能"),
            ("数据预处理", "先于", "模型训练", "数据预处理在模型训练之前"),
            ("过拟合", "问题", "泛化能力差", "过拟合导致泛化能力差"),
            ("欠拟合", "问题", "学习能力不足", "欠拟合说明学习能力不足"),
        ]

        for start, relation, end, desc in ai_relationships:
            ai_data.append({
                "type": "relationship",
                "start_node": start,
                "end_node": end,
                "relationship": relation,
                "properties": {"description": desc}
            })

        return ai_data

    def generate_math_knowledge(self):
        """生成数学相关知识"""
        math_data = []

        # 数学概念
        math_concepts = [
            {"name": "数学", "description": "研究数量、结构、变化的学科", "category": "基础学科", "difficulty": "入门"},
            {"name": "代数", "description": "研究数的运算规律和方程的数学分支", "category": "数学分支",
             "difficulty": "初级"},
            {"name": "几何", "description": "研究图形性质和空间关系的数学分支", "category": "数学分支",
             "difficulty": "初级"},
            {"name": "微积分", "description": "研究变化和积累的数学分支", "category": "数学分支", "difficulty": "中级"},
            {"name": "统计学", "description": "收集、分析和解释数据的数学分支", "category": "数学分支",
             "difficulty": "初级"},
            {"name": "概率论", "description": "研究随机现象规律的数学分支", "category": "数学分支",
             "difficulty": "中级"},
            {"name": "线性代数", "description": "研究向量和线性变换的数学分支", "category": "数学分支",
             "difficulty": "中级"},
            {"name": "函数", "description": "输入和输出之间的对应关系", "category": "基本概念", "difficulty": "初级"},
            {"name": "方程", "description": "含有未知数的等式", "category": "基本概念", "difficulty": "初级"},
            {"name": "不等式", "description": "表示数量大小关系的式子", "category": "基本概念", "difficulty": "初级"},
            {"name": "导数", "description": "函数在某点的变化率", "category": "微积分", "difficulty": "中级"},
            {"name": "积分", "description": "求曲线下面积的运算", "category": "微积分", "difficulty": "中级"},
            {"name": "极限", "description": "变量无限接近某值的概念", "category": "微积分", "difficulty": "中级"},
            {"name": "矩阵", "description": "按矩形排列的数的集合", "category": "线性代数", "difficulty": "中级"},
            {"name": "向量", "description": "有大小和方向的量", "category": "线性代数", "difficulty": "初级"},
            {"name": "概率", "description": "事件发生可能性的数值度量", "category": "概率论", "difficulty": "初级"},
            {"name": "随机变量", "description": "随机试验结果的数值函数", "category": "概率论", "difficulty": "中级"},
            {"name": "正态分布", "description": "重要的连续概率分布", "category": "概率论", "difficulty": "中级"},
            {"name": "平均数", "description": "一组数据的算术平均值", "category": "统计学", "difficulty": "入门"},
            {"name": "标准差", "description": "数据离散程度的度量", "category": "统计学", "difficulty": "初级"},
            {"name": "相关性", "description": "变量间关系的强度", "category": "统计学", "difficulty": "初级"},
        ]

        # 创建数学概念节点
        for concept in math_concepts:
            math_data.append({
                "type": "node",
                "label": "Concept",
                "properties": concept
            })

        # 数学关系
        math_relationships = [
            ("数学", "包含", "代数", "数学包含代数分支"),
            ("数学", "包含", "几何", "数学包含几何分支"),
            ("数学", "包含", "微积分", "数学包含微积分分支"),
            ("数学", "包含", "统计学", "数学包含统计学分支"),
            ("数学", "包含", "概率论", "数学包含概率论分支"),
            ("数学", "包含", "线性代数", "数学包含线性代数分支"),
            ("代数", "研究", "方程", "代数研究方程"),
            ("代数", "研究", "不等式", "代数研究不等式"),
            ("微积分", "包含", "导数", "微积分包含导数概念"),
            ("微积分", "包含", "积分", "微积分包含积分概念"),
            ("微积分", "基于", "极限", "微积分基于极限概念"),
            ("导数", "表示", "变化率", "导数表示函数变化率"),
            ("积分", "计算", "面积", "积分用于计算面积"),
            ("线性代数", "研究", "矩阵", "线性代数研究矩阵"),
            ("线性代数", "研究", "向量", "线性代数研究向量"),
            ("概率论", "研究", "概率", "概率论研究概率"),
            ("概率论", "研究", "随机变量", "概率论研究随机变量"),
            ("正态分布", "是", "概率分布", "正态分布是一种概率分布"),
            ("统计学", "计算", "平均数", "统计学计算平均数"),
            ("统计学", "计算", "标准差", "统计学计算标准差"),
            ("统计学", "分析", "相关性", "统计学分析相关性"),
            ("函数", "可以", "求导", "函数可以求导数"),
            ("函数", "可以", "积分", "函数可以求积分"),
        ]

        for start, relation, end, desc in math_relationships:
            math_data.append({
                "type": "relationship",
                "start_node": start,
                "end_node": end,
                "relationship": relation,
                "properties": {"description": desc}
            })

        return math_data

    def generate_education_knowledge(self):
        """生成教育相关知识"""
        edu_data = []

        # 教育概念
        edu_concepts = [
            {"name": "教育", "description": "培养人的社会活动", "category": "基础概念", "difficulty": "入门"},
            {"name": "学习", "description": "获取知识和技能的过程", "category": "基础概念", "difficulty": "入门"},
            {"name": "教学", "description": "传授知识和技能的活动", "category": "基础概念", "difficulty": "入门"},
            {"name": "个性化学习", "description": "根据学习者特点定制的学习方式", "category": "学习方式",
             "difficulty": "初级"},
            {"name": "协作学习", "description": "学习者共同完成学习任务", "category": "学习方式", "difficulty": "初级"},
            {"name": "自主学习", "description": "学习者独立规划和管理学习", "category": "学习方式",
             "difficulty": "初级"},
            {"name": "项目式学习", "description": "通过项目实践获得学习", "category": "学习方式", "difficulty": "初级"},
            {"name": "翻转课堂", "description": "课前学习课上实践的教学模式", "category": "教学模式",
             "difficulty": "初级"},
            {"name": "混合式学习", "description": "线上线下结合的学习方式", "category": "教学模式",
             "difficulty": "初级"},
            {"name": "在线教育", "description": "通过网络进行的教育", "category": "教学模式", "difficulty": "入门"},
            {"name": "智慧教育", "description": "运用信息技术的现代教育", "category": "教学模式", "difficulty": "初级"},
            {"name": "学习分析", "description": "分析学习过程和结果的技术", "category": "教育技术",
             "difficulty": "中级"},
            {"name": "教育数据挖掘", "description": "从教育数据中发现模式", "category": "教育技术",
             "difficulty": "中级"},
            {"name": "自适应学习", "description": "根据学习表现调整内容", "category": "教育技术", "difficulty": "中级"},
            {"name": "虚拟现实教学", "description": "使用VR技术的教学方式", "category": "教育技术",
             "difficulty": "中级"},
            {"name": "增强现实教学", "description": "使用AR技术的教学方式", "category": "教育技术",
             "difficulty": "中级"},
            {"name": "多模态交互", "description": "多种感官参与的交互方式", "category": "交互技术",
             "difficulty": "中级"},
            {"name": "语音交互", "description": "通过语音进行的人机交互", "category": "交互技术", "difficulty": "初级"},
            {"name": "手势识别", "description": "识别和理解手势的技术", "category": "交互技术", "difficulty": "中级"},
            {"name": "情感计算", "description": "识别和处理情感信息", "category": "交互技术", "difficulty": "中级"},
            {"name": "知识图谱", "description": "结构化的知识表示方法", "category": "知识表示", "difficulty": "中级"},
            {"name": "本体论", "description": "概念及其关系的正式表示", "category": "知识表示", "difficulty": "高级"},
            {"name": "语义网", "description": "机器可理解的网络", "category": "知识表示", "difficulty": "高级"},
            {"name": "学习路径", "description": "学习内容的组织顺序", "category": "学习设计", "difficulty": "初级"},
            {"name": "学习目标", "description": "学习要达到的预期结果", "category": "学习设计", "difficulty": "入门"},
        ]

        # 创建教育概念节点
        for concept in edu_concepts:
            edu_data.append({
                "type": "node",
                "label": "Concept",
                "properties": concept
            })

        # 教育关系
        edu_relationships = [
            ("教育", "包含", "学习", "教育活动包含学习过程"),
            ("教育", "包含", "教学", "教育活动包含教学过程"),
            ("学习", "方式", "个性化学习", "个性化学习是一种学习方式"),
            ("学习", "方式", "协作学习", "协作学习是一种学习方式"),
            ("学习", "方式", "自主学习", "自主学习是一种学习方式"),
            ("学习", "方式", "项目式学习", "项目式学习是一种学习方式"),
            ("教学", "模式", "翻转课堂", "翻转课堂是一种教学模式"),
            ("教学", "模式", "混合式学习", "混合式学习是一种教学模式"),
            ("教学", "模式", "在线教育", "在线教育是一种教学模式"),
            ("智慧教育", "使用", "人工智能", "智慧教育使用人工智能技术"),
            ("学习分析", "属于", "教育技术", "学习分析属于教育技术"),
            ("教育数据挖掘", "属于", "教育技术", "教育数据挖掘属于教育技术"),
            ("自适应学习", "基于", "学习分析", "自适应学习基于学习分析"),
            ("虚拟现实教学", "提供", "沉浸体验", "VR教学提供沉浸式体验"),
            ("增强现实教学", "增强", "现实体验", "AR教学增强现实体验"),
            ("多模态交互", "包含", "语音交互", "多模态交互包含语音交互"),
            ("多模态交互", "包含", "手势识别", "多模态交互包含手势识别"),
            ("情感计算", "分析", "学习情绪", "情感计算分析学习情绪"),
            ("知识图谱", "支持", "个性化学习", "知识图谱支持个性化学习"),
            ("知识图谱", "基于", "本体论", "知识图谱基于本体论"),
            ("语义网", "实现", "知识共享", "语义网实现知识共享"),
            ("学习路径", "达成", "学习目标", "学习路径帮助达成学习目标"),
            ("PEPPER机器人", "应用", "多模态交互", "PEPPER机器人应用多模态交互"),
            ("PEPPER机器人", "支持", "个性化学习", "PEPPER机器人支持个性化学习"),
        ]

        for start, relation, end, desc in edu_relationships:
            edu_data.append({
                "type": "relationship",
                "start_node": start,
                "end_node": end,
                "relationship": relation,
                "properties": {"description": desc}
            })

        return edu_data

    def generate_technology_knowledge(self):
        """生成技术相关知识"""
        tech_data = []

        # 技术概念
        tech_concepts = [
            {"name": "计算机科学", "description": "研究计算机和计算系统的学科", "category": "基础学科",
             "difficulty": "入门"},
            {"name": "软件工程", "description": "系统化开发软件的工程学科", "category": "工程学科",
             "difficulty": "初级"},
            {"name": "数据库", "description": "存储和管理数据的系统", "category": "数据管理", "difficulty": "初级"},
            {"name": "网络技术", "description": "计算机网络相关技术", "category": "网络通信", "difficulty": "初级"},
            {"name": "云计算", "description": "通过网络提供计算服务", "category": "分布式计算", "difficulty": "中级"},
            {"name": "大数据", "description": "超大规模数据集合及处理技术", "category": "数据处理",
             "difficulty": "中级"},
            {"name": "物联网", "description": "物品通过网络互联的系统", "category": "网络技术", "difficulty": "中级"},
            {"name": "区块链", "description": "分布式账本技术", "category": "分布式技术", "difficulty": "高级"},
            {"name": "Web开发", "description": "开发网站和Web应用", "category": "软件开发", "difficulty": "初级"},
            {"name": "移动开发", "description": "开发移动应用程序", "category": "软件开发", "difficulty": "初级"},
            {"name": "前端开发", "description": "开发用户界面和交互", "category": "软件开发", "difficulty": "初级"},
            {"name": "后端开发", "description": "开发服务器端逻辑", "category": "软件开发", "difficulty": "初级"},
            {"name": "全栈开发", "description": "同时进行前端和后端开发", "category": "软件开发", "difficulty": "中级"},
            {"name": "DevOps", "description": "开发和运维的协作模式", "category": "软件工程", "difficulty": "中级"},
            {"name": "敏捷开发", "description": "迭代式软件开发方法", "category": "开发方法", "difficulty": "初级"},
            {"name": "版本控制", "description": "管理代码版本的系统", "category": "开发工具", "difficulty": "入门"},
            {"name": "Git", "description": "分布式版本控制系统", "category": "开发工具", "difficulty": "入门"},
            {"name": "Docker", "description": "容器化技术平台", "category": "部署技术", "difficulty": "中级"},
            {"name": "Kubernetes", "description": "容器编排平台", "category": "部署技术", "difficulty": "高级"},
            {"name": "微服务", "description": "分布式系统架构模式", "category": "系统架构", "difficulty": "高级"},
            {"name": "API", "description": "应用程序编程接口", "category": "接口技术", "difficulty": "初级"},
            {"name": "RESTful", "description": "Web服务架构风格", "category": "接口技术", "difficulty": "初级"},
            {"name": "GraphQL", "description": "数据查询和操作语言", "category": "接口技术", "difficulty": "中级"},
            {"name": "NoSQL", "description": "非关系型数据库", "category": "数据库", "difficulty": "中级"},
            {"name": "SQL", "description": "结构化查询语言", "category": "数据库", "difficulty": "初级"},
        ]

        # 创建技术概念节点
        for concept in tech_concepts:
            tech_data.append({
                "type": "node",
                "label": "Concept",
                "properties": concept
            })

        # 技术关系
        tech_relationships = [
            ("计算机科学", "包含", "软件工程", "计算机科学包含软件工程"),
            ("软件工程", "使用", "版本控制", "软件工程使用版本控制"),
            ("版本控制", "工具", "Git", "Git是版本控制工具"),
            ("软件开发", "包含", "Web开发", "软件开发包含Web开发"),
            ("软件开发", "包含", "移动开发", "软件开发包含移动开发"),
            ("Web开发", "分为", "前端开发", "Web开发分为前端开发"),
            ("Web开发", "分为", "后端开发", "Web开发分为后端开发"),
            ("全栈开发", "结合", "前端开发", "全栈开发结合前端开发"),
            ("全栈开发", "结合", "后端开发", "全栈开发结合后端开发"),
            ("软件工程", "方法", "敏捷开发", "敏捷开发是软件工程方法"),
            ("软件工程", "实践", "DevOps", "DevOps是软件工程实践"),
            ("云计算", "支持", "大数据", "云计算支持大数据处理"),
            ("大数据", "需要", "分布式计算", "大数据需要分布式计算"),
            ("微服务", "属于", "分布式系统", "微服务属于分布式系统"),
            ("Docker", "支持", "微服务", "Docker支持微服务部署"),
            ("Kubernetes", "管理", "Docker", "Kubernetes管理Docker容器"),
            ("API", "实现", "系统集成", "API实现系统集成"),
            ("RESTful", "是", "API设计", "RESTful是API设计风格"),
            ("GraphQL", "是", "API技术", "GraphQL是API技术"),
            ("数据库", "类型", "SQL", "SQL是关系型数据库"),
            ("数据库", "类型", "NoSQL", "NoSQL是非关系型数据库"),
            ("物联网", "依赖", "网络技术", "物联网依赖网络技术"),
            ("区块链", "应用", "分布式技术", "区块链应用分布式技术"),
        ]

        for start, relation, end, desc in tech_relationships:
            tech_data.append({
                "type": "relationship",
                "start_node": start,
                "end_node": end,
                "relationship": relation,
                "properties": {"description": desc}
            })

        return tech_data

    def import_data_to_neo4j(self, data_list):
        """将数据导入Neo4j"""
        node_count = 0
        relationship_count = 0

        try:
            # 先创建所有节点
            for item in data_list:
                if item["type"] == "node":
                    try:
                        self.knowledge_graph.create_node(item["label"], item["properties"])
                        node_count += 1
                    except Exception as e:
                        logger.warning(f"创建节点失败: {e}")

            # 再创建所有关系
            for item in data_list:
                if item["type"] == "relationship":
                    try:
                        self.knowledge_graph.create_relationship(
                            "Concept", {"name": item["start_node"]},
                            "Concept", {"name": item["end_node"]},
                            item["relationship"],
                            item.get("properties", {})
                        )
                        relationship_count += 1
                    except Exception as e:
                        logger.warning(f"创建关系失败: {e}")

            logger.info(f"数据导入完成: {node_count}个节点, {relationship_count}个关系")
            return node_count, relationship_count

        except Exception as e:
            logger.error(f"数据导入失败: {e}")
            return node_count, relationship_count

    def generate_all_knowledge(self):
        """生成所有领域的知识数据"""
        logger.info("开始生成知识图谱数据...")

        all_data = []

        # 生成各领域知识
        programming_data = self.generate_programming_knowledge()
        ai_data = self.generate_ai_knowledge()
        math_data = self.generate_math_knowledge()
        edu_data = self.generate_education_knowledge()
        tech_data = self.generate_technology_knowledge()

        # 合并所有数据
        all_data.extend(programming_data)
        all_data.extend(ai_data)
        all_data.extend(math_data)
        all_data.extend(edu_data)
        all_data.extend(tech_data)

        logger.info(f"总共生成 {len(all_data)} 条数据")

        # 保存到JSON文件
        output_file = "data/knowledge_graph/comprehensive_knowledge.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        logger.info(f"知识数据已保存到: {output_file}")

        return all_data

    def run_data_generation(self):
        """运行完整的数据生成流程"""
        try:
            # 生成所有知识数据
            all_data = self.generate_all_knowledge()

            # 导入到Neo4j
            logger.info("开始导入数据到Neo4j...")
            node_count, relationship_count = self.import_data_to_neo4j(all_data)

            logger.info("=" * 50)
            logger.info("知识图谱数据生成完成!")
            logger.info(f"创建节点数: {node_count}")
            logger.info(f"创建关系数: {relationship_count}")
            logger.info("=" * 50)

            return True

        except Exception as e:
            logger.error(f"数据生成失败: {e}")
            return False

    def close(self):
        """关闭数据库连接"""
        self.knowledge_graph.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='教育知识图谱数据生成器')
    parser.add_argument('--uri', default='bolt://localhost:7687', help='Neo4j连接URI')
    parser.add_argument('--user', default='neo4j', help='Neo4j用户名')
    parser.add_argument('--password', default='admin123', help='Neo4j密码')
    parser.add_argument('--generate-only', action='store_true', help='仅生成JSON文件，不导入数据库')

    args = parser.parse_args()

    generator = KnowledgeGraphDataGenerator(args.uri, args.user, args.password)

    try:
        if args.generate_only:
            # 仅生成JSON文件
            logger.info("仅生成知识数据文件...")
            generator.generate_all_knowledge()
        else:
            # 生成并导入数据库
            generator.run_data_generation()

    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error(f"执行失败: {e}")
    finally:
        generator.close()


if __name__ == "__main__":
    main()