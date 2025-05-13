import re
import jieba
import numpy as np


class TextProcessor:
    def __init__(self):
        # 可以加载自定义词典
        # jieba.load_userdict("path/to/dict.txt")
        pass

    def preprocess_text(self, text):
        """文本预处理：去除特殊字符，分词等"""
        # 去除特殊字符
        text = re.sub(r'[^\w\s]', '', text)
        # 分词
        words = jieba.cut(text)
        return list(words)

    def extract_keywords(self, text, top_k=5):
        """提取文本中的关键词"""
        import jieba.analyse
        keywords = jieba.analyse.extract_tags(text, topK=top_k)
        return keywords

    def classify_question(self, text):
        """简单分类问题类型"""
        if "什么" in text or "是什么" in text:
            return "定义型问题"
        elif "如何" in text or "怎么" in text:
            return "方法型问题"
        elif "为什么" in text:
            return "原因型问题"
        elif "区别" in text or "比较" in text:
            return "比较型问题"
        else:
            return "其他问题"


# 测试文本处理
if __name__ == "__main__":
    processor = TextProcessor()
    text = "人工智能在教育领域有什么应用？"
    words = processor.preprocess_text(text)
    keywords = processor.extract_keywords(text)
    question_type = processor.classify_question(text)

    print(f"分词结果: {words}")
    print(f"关键词: {keywords}")
    print(f"问题类型: {question_type}")
