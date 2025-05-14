# 在ai_service/llm_module/kg_llm_integration.py中创建新模块

import logging

from ai_service.knowledge_graph.knowledge_graph import KnowledgeGraph
from ai_service.llm_module.llm_interface import LLMService


class KnowledgeLLMIntegration:
    """知识图谱与大语言模型的协同集成"""

    def __init__(self, kg_uri="bolt://localhost:7687", kg_user="neo4j", kg_password="password"):
        self.knowledge_graph = KnowledgeGraph(kg_uri, kg_user, kg_password)
        self.llm_service = LLMService()
        self.logger = logging.getLogger("KG_LLM_INTEGRATION")

    def answer_with_knowledge_graph(self, question, keywords=None):
        """使用知识图谱增强的回答"""
        try:
            # 如果没有提供关键词，从问题中提取
            if not keywords:
                # 假设我们有个简单的关键词提取函数
                keywords = question.split()
                keywords = [k for k in keywords if len(k) > 2]

            # 从知识图谱查询相关知识
            knowledge_items = []
            for keyword in keywords:
                items = self.knowledge_graph.find_related_knowledge(keyword)
                if items:
                    knowledge_items.extend(items)

            # 使用LLM结合知识图谱生成回答
            response = self.llm_service.answer_with_knowledge(question, knowledge_items)
            return {
                "response": response,
                "knowledge_items": knowledge_items,
                "keywords": keywords
            }

        except Exception as e:
            self.logger.error(f"知识图谱增强回答失败: {e}")
            # 退回到基本LLM回答
            return {
                "response": self.llm_service.generate_response(question),
                "knowledge_items": [],
                "keywords": keywords,
                "error": str(e)
            }