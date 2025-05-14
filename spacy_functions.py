# spacy_functions.py
"""
存放所有依赖spacy的功能
"""
import json
import sys

from ai_service.multimodal.text_processor import TextProcessor


def process_text(text):
    """处理文本，返回分词和关键词"""
    processor = TextProcessor()
    words = processor.preprocess_text(text)
    keywords = processor.extract_keywords(text)
    question_type = processor.classify_question(text)

    result = {
        "words": words,
        "keywords": keywords,
        "question_type": question_type
    }

    return json.dumps(result)


if __name__ == "__main__":
    # 接收命令行参数
    text = sys.argv[1] if len(sys.argv) > 1 else ""
    result = process_text(text)
    print(result)  # 输出结果供主程序读取
