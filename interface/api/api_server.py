import os
import sys

from flask import Flask, request, jsonify
from flask_cors import CORS

# 添加父目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入各个模块
from ai_service.multimodal.speech_recognition import SpeechRecognizer
from ai_service.multimodal.image_recognition import ImageRecognizer
from ai_service.multimodal.text_processor import TextProcessor
from ai_service.knowledge_graph.knowledge_graph import KnowledgeGraph
from ai_service.llm_module.llm_interface import LLMService
from ai_service.llm_module.langchain_integration import LangChainIntegration

app = Flask(__name__)
CORS(app)  # 启用CORS

# 初始化各个模块
speech_recognizer = SpeechRecognizer()
image_recognizer = ImageRecognizer()
text_processor = TextProcessor()
knowledge_graph = KnowledgeGraph()
llm_service = LLMService()
langchain = LangChainIntegration()


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "ok", "message": "API服务运行正常"})


@app.route('/speech_recognition', methods=['POST'])
def speech_recognition_api():
    """语音识别API"""
    if 'audio_file' in request.files:
        audio_file = request.files['audio_file']
        audio_path = '/tmp/audio_file.wav'
        audio_file.save(audio_path)

        language = request.form.get('language', 'zh-CN')
        text = speech_recognizer.recognize_from_file(audio_path, language)

        return jsonify({"status": "success", "text": text})
    else:
        return jsonify({"status": "error", "message": "未找到音频文件"}), 400


@app.route('/image_recognition', methods=['POST'])
def image_recognition_api():
    """图像识别API"""
    if 'image_file' in request.files:
        image_file = request.files['image_file']
        image_path = '/tmp/image_file.jpg'
        image_file.save(image_path)

        result = image_recognizer.recognize_image(image_path)

        return jsonify({"status": "success", "result": result})
    else:
        return jsonify({"status": "error", "message": "未找到图像文件"}), 400


@app.route('/text_processing', methods=['POST'])
def text_processing_api():
    """文本处理API"""
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"status": "error", "message": "未找到文本数据"}), 400

    text = data['text']
    words = text_processor.preprocess_text(text)
    keywords = text_processor.extract_keywords(text)
    question_type = text_processor.classify_question(text)

    return jsonify({
        "status": "success",
        "words": words,
        "keywords": keywords,
        "question_type": question_type
    })


@app.route('/knowledge_graph/query', methods=['POST'])
def knowledge_graph_query_api():
    """知识图谱查询API"""
    data = request.json
    if not data or 'keyword' not in data:
        return jsonify({"status": "error", "message": "未找到关键词"}), 400

    keyword = data['keyword']
    results = knowledge_graph.find_related_knowledge(keyword)

    return jsonify({"status": "success", "results": results})


@app.route('/llm/query', methods=['POST'])
def llm_query_api():
    """大语言模型查询API"""
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"status": "error", "message": "未找到查询文本"}), 400

    query = data['query']
    response = llm_service.generate_response(query)

    return jsonify({"status": "success", "response": response})


@app.route('/llm/query_with_knowledge', methods=['POST'])
def llm_query_with_knowledge_api():
    """结合知识图谱的大语言模型查询API"""
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"status": "error", "message": "未找到查询文本"}), 400

    query = data['query']

    # 提取关键词
    keywords = text_processor.extract_keywords(query)

    # 查询知识图谱
    knowledge = []
    for keyword in keywords:
        results = knowledge_graph.find_related_knowledge(keyword)
        knowledge.extend(results)

    # 使用知识图谱结果生成回答
    response = llm_service.answer_with_knowledge(query, knowledge)

    return jsonify({
        "status": "success",
        "response": response,
        "keywords": keywords,
        "knowledge": knowledge
    })


@app.route('/langchain/query', methods=['POST'])
def langchain_query_api():
    """LangChain查询API"""
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"status": "error", "message": "未找到查询文本"}), 400

    query = data['query']
    response = langchain.query(query)

    return jsonify({"status": "success", "response": response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
