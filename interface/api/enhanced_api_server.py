#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PEPPER智能教学系统 - 增强API服务器
支持大语言模型、知识图谱、多模态交互、智能教学四大功能模块
"""

import logging
import os
import sys
import time
import uuid
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入各个模块
from ai_service.llm_module.llm_interface import LLMService
from ai_service.llm_module.lora_fine_tuning import LoRAFineTuner
from ai_service.knowledge_graph.knowledge_graph import KnowledgeGraph
from ai_service.knowledge_graph.education_knowledge_processor import EducationKnowledgeProcessor
from ai_service.multimodal.speech_recognition import SpeechRecognizer
from ai_service.multimodal.image_recognition import ImageRecognizer
from ai_service.multimodal.text_processor import TextProcessor
from ai_service.teaching_module.personalized_teaching import PersonalizedTeaching

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PEPPER_API")

app = Flask(__name__)
CORS(app)

# 全局变量存储服务实例
services = {
    'llm': None,
    'fine_tuner': None,
    'knowledge_graph': None,
    'knowledge_processor': None,
    'speech_recognizer': None,
    'image_recognizer': None,
    'text_processor': None,
    'teaching': None
}

# 系统状态
system_status = {
    'model_loaded': False,
    'neo4j_connected': False,
    'training_progress': 0,
    'training_active': False
}

# 配置
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


def initialize_services():
    """初始化服务"""
    try:
        # 初始化文本处理器
        services['text_processor'] = TextProcessor()

        # 初始化语音识别器
        services['speech_recognizer'] = SpeechRecognizer()

        # 初始化图像识别器
        services['image_recognizer'] = ImageRecognizer()

        logger.info("基础服务初始化完成")
    except Exception as e:
        logger.error(f"服务初始化失败: {e}")


# ======================== 大语言模型API ========================

@app.route('/api/load_model', methods=['POST'])
def load_model():
    """加载大语言模型"""
    try:
        data = request.json
        model_path = data.get('model_path', 'models/deepseek-coder-1.3b-base')

        logger.info(f"正在加载模型: {model_path}")

        if not os.path.exists(model_path):
            return jsonify({
                "status": "error",
                "message": f"模型路径不存在: {model_path}"
            })

        services['llm'] = LLMService(model_path)
        system_status['model_loaded'] = True

        logger.info("模型加载成功")
        return jsonify({
            "status": "success",
            "message": "模型加载成功"
        })

    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/unload_model', methods=['POST'])
def unload_model():
    """卸载大语言模型"""
    try:
        services['llm'] = None
        system_status['model_loaded'] = False

        logger.info("模型已卸载")
        return jsonify({
            "status": "success",
            "message": "模型已卸载"
        })

    except Exception as e:
        logger.error(f"模型卸载失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/test_model', methods=['POST'])
def test_model():
    """测试大语言模型"""
    try:
        if not services['llm']:
            return jsonify({
                "status": "error",
                "message": "模型未加载"
            })

        data = request.json
        question = data.get('question', '')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 512)

        if not question.strip():
            return jsonify({
                "status": "error",
                "message": "问题不能为空"
            })

        # 生成回答
        response = services['llm'].generate_response(question, max_tokens)

        return jsonify({
            "status": "success",
            "response": response
        })

    except Exception as e:
        logger.error(f"模型测试失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/start_finetuning', methods=['POST'])
def start_finetuning():
    """开始LoRA微调"""
    try:
        if 'training_data' not in request.files:
            return jsonify({
                "status": "error",
                "message": "未找到训练数据文件"
            })

        file = request.files['training_data']
        epochs = int(request.form.get('epochs', 3))
        learning_rate = float(request.form.get('learning_rate', 0.0002))

        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "未选择文件"
            })

        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 创建微调器
        services['fine_tuner'] = LoRAFineTuner()

        # 开始微调（这里简化为模拟过程）
        system_status['training_active'] = True
        system_status['training_progress'] = 0

        # 实际项目中这里应该启动异步训练任务
        # 这里只是模拟进度更新
        import threading
        def simulate_training():
            for i in range(101):
                if not system_status['training_active']:
                    break
                system_status['training_progress'] = i
                time.sleep(0.1)
            system_status['training_active'] = False

        thread = threading.Thread(target=simulate_training)
        thread.daemon = True
        thread.start()

        logger.info("LoRA微调已开始")
        return jsonify({
            "status": "success",
            "message": "微调已开始"
        })

    except Exception as e:
        logger.error(f"微调启动失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/training_progress', methods=['GET'])
def get_training_progress():
    """获取训练进度"""
    return jsonify({
        "status": "success",
        "progress": system_status['training_progress'],
        "active": system_status['training_active']
    })


# ======================== 知识图谱API ========================

@app.route('/api/connect_neo4j', methods=['POST'])
def connect_neo4j():
    """连接Neo4j数据库"""
    try:
        data = request.json
        uri = data.get('uri', 'bolt://localhost:7687')
        user = data.get('user', 'neo4j')
        password = data.get('password', 'password')

        services['knowledge_graph'] = KnowledgeGraph(uri, user, password)
        services['knowledge_processor'] = EducationKnowledgeProcessor(uri, user, password)

        # 测试连接
        test_result = services['knowledge_graph'].query("RETURN 1 as test")
        if test_result:
            system_status['neo4j_connected'] = True
            logger.info("Neo4j连接成功")
            return jsonify({
                "status": "success",
                "message": "数据库连接成功"
            })
        else:
            raise Exception("连接测试失败")

    except Exception as e:
        logger.error(f"Neo4j连接失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/disconnect_neo4j', methods=['POST'])
def disconnect_neo4j():
    """断开Neo4j连接"""
    try:
        if services['knowledge_graph']:
            services['knowledge_graph'].close()

        services['knowledge_graph'] = None
        services['knowledge_processor'] = None
        system_status['neo4j_connected'] = False

        logger.info("Neo4j连接已断开")
        return jsonify({
            "status": "success",
            "message": "数据库连接已断开"
        })

    except Exception as e:
        logger.error(f"断开连接失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/import_knowledge', methods=['POST'])
def import_knowledge():
    """导入知识"""
    try:
        if not services['knowledge_processor']:
            return jsonify({
                "status": "error",
                "message": "未连接知识图谱数据库"
            })

        if 'knowledge_file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "未找到知识文件"
            })

        file = request.files['knowledge_file']
        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "未选择文件"
            })

        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 导入知识
        knowledge_items = services['knowledge_processor'].extract_from_file(filepath)
        count = services['knowledge_processor'].build_knowledge_graph(knowledge_items)

        logger.info(f"知识导入成功，共导入{count}条")
        return jsonify({
            "status": "success",
            "message": "知识导入成功",
            "count": count
        })

    except Exception as e:
        logger.error(f"知识导入失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/generate_sample_knowledge', methods=['POST'])
def generate_sample_knowledge():
    """生成示例知识"""
    try:
        if not services['knowledge_processor']:
            return jsonify({
                "status": "error",
                "message": "未连接知识图谱数据库"
            })

        count = services['knowledge_processor'].create_educational_knowledge_base()

        logger.info("示例知识生成成功")
        return jsonify({
            "status": "success",
            "message": "示例知识生成成功",
            "count": count
        })

    except Exception as e:
        logger.error(f"示例知识生成失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/execute_cypher', methods=['POST'])
def execute_cypher():
    """执行Cypher查询"""
    try:
        if not services['knowledge_graph']:
            return jsonify({
                "status": "error",
                "message": "未连接Neo4j数据库"
            })

        data = request.json
        query = data.get('query', '')

        if not query.strip():
            return jsonify({
                "status": "error",
                "message": "查询语句不能为空"
            })

        results = services['knowledge_graph'].query(query)

        # 将结果转换为JSON可序列化的格式
        serializable_results = []
        for record in results:
            serializable_record = {}
            for key, value in record.items():
                if hasattr(value, '_properties'):
                    # Neo4j节点或关系
                    serializable_record[key] = dict(value)
                else:
                    serializable_record[key] = value
            serializable_results.append(serializable_record)

        return jsonify({
            "status": "success",
            "data": serializable_results
        })

    except Exception as e:
        logger.error(f"Cypher查询失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/get_graph_stats', methods=['GET'])
def get_graph_stats():
    """获取图谱统计信息"""
    try:
        if not services['knowledge_graph']:
            return jsonify({
                "status": "error",
                "message": "未连接Neo4j数据库"
            })

        # 获取节点数量
        node_result = services['knowledge_graph'].query("MATCH (n) RETURN count(n) as count")
        node_count = node_result[0]['count'] if node_result else 0

        # 获取关系数量
        rel_result = services['knowledge_graph'].query("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = rel_result[0]['count'] if rel_result else 0

        # 获取域数量（假设通过domain属性区分）
        domain_result = services['knowledge_graph'].query(
            "MATCH (n) WHERE n.domain IS NOT NULL RETURN count(DISTINCT n.domain) as count"
        )
        domain_count = domain_result[0]['count'] if domain_result else 0

        return jsonify({
            "status": "success",
            "stats": {
                "nodes": node_count,
                "relationships": rel_count,
                "domains": domain_count
            }
        })

    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


# ======================== 多模态交互API ========================

@app.route('/api/recognize_speech', methods=['POST'])
def recognize_speech():
    """语音识别"""
    try:
        if not services['speech_recognizer']:
            return jsonify({
                "status": "error",
                "message": "语音识别服务未初始化"
            })

        if 'audio_file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "未找到音频文件"
            })

        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "未选择文件"
            })

        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 识别语音
        text = services['speech_recognizer'].recognize_from_file(filepath)

        # 清理临时文件
        os.remove(filepath)

        return jsonify({
            "status": "success",
            "text": text
        })

    except Exception as e:
        logger.error(f"语音识别失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/recognize_image', methods=['POST'])
def recognize_image():
    """图像识别"""
    try:
        if not services['image_recognizer']:
            return jsonify({
                "status": "error",
                "message": "图像识别服务未初始化"
            })

        if 'image_file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "未找到图像文件"
            })

        file = request.files['image_file']
        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "未选择文件"
            })

        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 识别图像
        result = services['image_recognizer'].recognize_image(filepath)

        # 清理临时文件
        os.remove(filepath)

        # 添加置信度（示例）
        result['confidence'] = 0.85

        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:
        logger.error(f"图像识别失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/process_text', methods=['POST'])
def process_text():
    """文本处理"""
    try:
        if not services['text_processor']:
            return jsonify({
                "status": "error",
                "message": "文本处理服务未初始化"
            })

        data = request.json
        text = data.get('text', '')

        if not text.strip():
            return jsonify({
                "status": "error",
                "message": "文本不能为空"
            })

        # 处理文本
        tokens = services['text_processor'].preprocess_text(text)
        keywords = services['text_processor'].extract_keywords(text)
        question_type = services['text_processor'].classify_question(text)

        return jsonify({
            "status": "success",
            "tokens": tokens,
            "keywords": keywords,
            "question_type": question_type
        })

    except Exception as e:
        logger.error(f"文本处理失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/extract_keywords', methods=['POST'])
def extract_keywords():
    """提取关键词"""
    try:
        if not services['text_processor']:
            return jsonify({
                "status": "error",
                "message": "文本处理服务未初始化"
            })

        data = request.json
        text = data.get('text', '')

        if not text.strip():
            return jsonify({
                "status": "error",
                "message": "文本不能为空"
            })

        keywords = services['text_processor'].extract_keywords(text, top_k=10)

        return jsonify({
            "status": "success",
            "keywords": keywords
        })

    except Exception as e:
        logger.error(f"关键词提取失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


# ======================== 智能教学API ========================

@app.route('/api/add_student', methods=['POST'])
def add_student():
    """添加学生"""
    try:
        # 初始化教学服务（如果尚未初始化）
        if not services['teaching']:
            services['teaching'] = PersonalizedTeaching()

        data = request.json
        name = data.get('name', '')
        learning_style = data.get('learning_style', 'visual')

        if not name.strip():
            return jsonify({
                "status": "error",
                "message": "学生姓名不能为空"
            })

        # 生成学生ID
        student_id = str(uuid.uuid4())[:8]

        # 添加学生档案
        success = services['teaching'].add_student_profile(student_id, name)

        if success:
            # 设置学习风格
            profile = services['teaching'].get_student_profile(student_id)
            profile.set_learning_style(learning_style)

            logger.info(f"添加学生成功: {name} ({student_id})")
            return jsonify({
                "status": "success",
                "student_id": student_id,
                "message": "学生添加成功"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "学生添加失败"
            })

    except Exception as e:
        logger.error(f"添加学生失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/get_students', methods=['GET'])
def get_students():
    """获取学生列表"""
    try:
        if not services['teaching']:
            services['teaching'] = PersonalizedTeaching()
            # 创建示例学生档案
            services['teaching'].create_demo_student_profiles()

        students = []
        for student_id, profile in services['teaching'].student_profiles.items():
            students.append({
                "id": student_id,
                "name": profile.name,
                "learning_style": profile.learning_style
            })

        return jsonify({
            "status": "success",
            "students": students
        })

    except Exception as e:
        logger.error(f"获取学生列表失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/get_student_profile', methods=['POST'])
def get_student_profile():
    """获取学生档案"""
    try:
        if not services['teaching']:
            return jsonify({
                "status": "error",
                "message": "教学服务未初始化"
            })

        data = request.json
        student_id = data.get('student_id', '')

        profile = services['teaching'].get_student_profile(student_id)

        if profile:
            return jsonify({
                "status": "success",
                "profile": {
                    "student_id": profile.student_id,
                    "name": profile.name,
                    "learning_style": profile.learning_style,
                    "strengths": profile.get_top_strengths(5),
                    "weaknesses": profile.get_top_weaknesses(5),
                    "preferences": profile.get_top_preferences(5)
                }
            })
        else:
            return jsonify({
                "status": "error",
                "message": "学生档案不存在"
            })

    except Exception as e:
        logger.error(f"获取学生档案失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/generate_learning_path', methods=['POST'])
def generate_learning_path():
    """生成学习路径"""
    try:
        if not services['teaching']:
            return jsonify({
                "status": "error",
                "message": "教学服务未初始化"
            })

        data = request.json
        student_id = data.get('student_id', '')
        goal = data.get('goal', '')

        if not student_id or not goal:
            return jsonify({
                "status": "error",
                "message": "学生ID和学习目标不能为空"
            })

        learning_path = services['teaching'].generate_learning_path(student_id, goal)

        if learning_path:
            return jsonify({
                "status": "success",
                "learning_path": learning_path['learning_path']
            })
        else:
            return jsonify({
                "status": "error",
                "message": "学习路径生成失败"
            })

    except Exception as e:
        logger.error(f"学习路径生成失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/chat', methods=['POST'])
def chat():
    """智能对话"""
    try:
        if not services['teaching']:
            return jsonify({
                "status": "error",
                "message": "教学服务未初始化"
            })

        data = request.json
        message = data.get('message', '')
        student_id = data.get('student_id', '')

        if not message.strip():
            return jsonify({
                "status": "error",
                "message": "消息不能为空"
            })

        if student_id:
            # 个性化回答
            response = services['teaching'].generate_personalized_answer(student_id, message)

            # 记录学习交互
            services['teaching'].add_learning_interaction(student_id, "general", message)
        else:
            # 通用回答
            if services['llm']:
                response = services['llm'].generate_response(message)
            else:
                response = "抱歉，我现在无法回答这个问题。请确保已加载语言模型。"

        return jsonify({
            "status": "success",
            "response": response
        })

    except Exception as e:
        logger.error(f"智能对话失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/get_recommendations', methods=['POST'])
def get_recommendations():
    """获取资源推荐"""
    try:
        if not services['teaching']:
            return jsonify({
                "status": "error",
                "message": "教学服务未初始化"
            })

        data = request.json
        topic = data.get('topic', '')
        student_id = data.get('student_id', '')

        if not topic:
            return jsonify({
                "status": "error",
                "message": "主题不能为空"
            })

        if student_id:
            resources = services['teaching'].recommend_learning_resources(student_id, topic, count=5)
        else:
            # 返回通用推荐
            resources = [
                {
                    "title": f"{topic}入门教程",
                    "type": "video",
                    "description": f"适合初学者的{topic}视频教程"
                },
                {
                    "title": f"{topic}实践指南",
                    "type": "article",
                    "description": f"{topic}的实际应用案例和练习"
                },
                {
                    "title": f"{topic}进阶课程",
                    "type": "course",
                    "description": f"深入学习{topic}的高级课程"
                }
            ]

        return jsonify({
            "status": "success",
            "resources": resources
        })

    except Exception as e:
        logger.error(f"获取推荐失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """生成学习报告"""
    try:
        if not services['teaching']:
            return jsonify({
                "status": "error",
                "message": "教学服务未初始化"
            })

        data = request.json
        student_id = data.get('student_id', '')
        report_type = data.get('report_type', 'progress')
        time_range = data.get('time_range', 'week')

        if not student_id:
            return jsonify({
                "status": "error",
                "message": "学生ID不能为空"
            })

        profile = services['teaching'].get_student_profile(student_id)
        if not profile:
            return jsonify({
                "status": "error",
                "message": "学生档案不存在"
            })

        # 生成报告HTML
        report_html = f"""
        <div class="report">
            <h3>{profile.name}的学习报告</h3>
            <p><strong>报告类型:</strong> {report_type}</p>
            <p><strong>时间范围:</strong> {time_range}</p>
            <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h4>学习优势</h4>
            <ul>
        """

        strengths = profile.get_top_strengths(3)
        for topic, score in strengths:
            report_html += f"<li>{topic}: {score:.1f}分</li>"

        report_html += """
            </ul>

            <h4>需要改进的领域</h4>
            <ul>
        """

        weaknesses = profile.get_top_weaknesses(3)
        for topic, score in weaknesses:
            report_html += f"<li>{topic}: 需要加强</li>"

        report_html += """
            </ul>

            <h4>学习建议</h4>
            <p>建议继续保持在优势领域的学习，同时加强薄弱环节的练习。</p>
        </div>
        """

        return jsonify({
            "status": "success",
            "report_html": report_html
        })

    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/api/export_report', methods=['POST'])
def export_report():
    """导出报告"""
    try:
        data = request.json
        content = data.get('content', '')
        format_type = data.get('format', 'html')

        if not content:
            return jsonify({
                "status": "error",
                "message": "报告内容不能为空"
            })

        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"learning_report_{timestamp}.html"
        filepath = os.path.join(REPORTS_FOLDER, filename)

        # 创建完整的HTML文件
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>学习报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .report {{ max-width: 800px; margin: 0 auto; }}
                h3 {{ color: #333; }}
                h4 {{ color: #666; }}
                ul {{ margin: 10px 0; }}
                li {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            {content}
        </body>
        </html>
        """

        # 保存文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return jsonify({
            "status": "success",
            "download_url": f"/download_report/{filename}",
            "filename": filename
        })

    except Exception as e:
        logger.error(f"报告导出失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/download_report/<filename>')
def download_report(filename):
    """下载报告文件"""
    try:
        filepath = os.path.join(REPORTS_FOLDER, filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name=filename)
        else:
            return jsonify({
                "status": "error",
                "message": "文件不存在"
            }), 404
    except Exception as e:
        logger.error(f"文件下载失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ======================== 系统API ========================

@app.route('/api/system_status', methods=['GET'])
def get_system_status():
    """获取系统状态"""
    return jsonify({
        "status": "success",
        "system_status": system_status,
        "services": {
            "llm": services['llm'] is not None,
            "knowledge_graph": services['knowledge_graph'] is not None,
            "teaching": services['teaching'] is not None,
            "speech_recognizer": services['speech_recognizer'] is not None,
            "image_recognizer": services['image_recognizer'] is not None,
            "text_processor": services['text_processor'] is not None
        }
    })


@app.route('/')
def index():
    """主页"""
    try:
        # 尝试加载HTML模板
        template_path = os.path.join('interface', 'web_console', 'templates', 'index.html')
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return html_content
        else:
            # 返回简单的HTML页面
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>PEPPER智能教学系统</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .header { background: #4a86e8; color: white; padding: 20px; border-radius: 5px; text-align: center; margin-bottom: 20px; }
                    .panel { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; }
                    button { background: #4a86e8; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 5px; }
                    input, textarea { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
                    .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
                    .status.success { background: #d4edcf; color: #2ed573; }
                    .status.error { background: #ffd3d8; color: #ff4757; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🤖 PEPPER智能教学系统</h1>
                        <p>基于DeepSeek模型的个性化AI教学助手</p>
                    </div>

                    <div class="panel">
                        <h2>🧠 大语言模型测试</h2>
                        <div id="model-status" class="status">模型未加载</div>
                        <button onclick="loadModel()">加载模型</button>
                        <button onclick="testModel()">测试模型</button>
                        <div style="margin: 20px 0;">
                            <textarea id="test-question" rows="3" placeholder="输入测试问题，例如：什么是Python循环？"></textarea>
                            <div id="model-response" style="background: #f8f9ff; padding: 15px; border-radius: 4px; min-height: 100px; margin-top: 10px;">
                                等待测试...
                            </div>
                        </div>
                    </div>

                    <div class="panel">
                        <h2>🗂️ 知识图谱</h2>
                        <div id="neo4j-status" class="status">数据库未连接</div>
                        <button onclick="connectNeo4j()">连接Neo4j</button>
                        <button onclick="generateSampleKnowledge()">生成示例知识</button>
                    </div>

                    <div class="panel">
                        <h2>🎯 文本处理</h2>
                        <textarea id="text-input" rows="3" placeholder="输入要处理的文本..."></textarea>
                        <button onclick="processText()">处理文本</button>
                        <div id="text-result" style="background: #f8f9ff; padding: 15px; border-radius: 4px; margin-top: 10px; min-height: 80px;">
                            等待处理...
                        </div>
                    </div>

                    <div class="panel">
                        <h2>📚 智能对话</h2>
                        <div id="chat-container" style="border: 1px solid #ddd; height: 300px; overflow-y: auto; padding: 10px; background: #fafafa; margin-bottom: 10px;">
                            <div style="background: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                <strong>PEPPER:</strong> 你好！我是你的AI教学助手，有什么问题吗？
                            </div>
                        </div>
                        <div style="display: flex; gap: 10px;">
                            <textarea id="chat-input" rows="2" placeholder="输入你的问题..." style="flex: 1;"></textarea>
                            <button onclick="sendMessage()" style="width: 80px;">发送</button>
                        </div>
                    </div>
                </div>

                <script>
                    // API调用函数
                    async function callAPI(endpoint, method = 'GET', data = null) {
                        try {
                            const options = { method, headers: { 'Content-Type': 'application/json' } };
                            if (data) options.body = JSON.stringify(data);
                            const response = await fetch('/api/' + endpoint, options);
                            return await response.json();
                        } catch (error) {
                            return { status: 'error', message: '网络错误: ' + error.message };
                        }
                    }

                    // 更新状态显示
                    function updateStatus(elementId, message, isSuccess = false) {
                        const element = document.getElementById(elementId);
                        element.textContent = message;
                        element.className = isSuccess ? 'status success' : 'status error';
                    }

                    // 加载模型
                    async function loadModel() {
                        updateStatus('model-status', '正在加载模型...');
                        const result = await callAPI('load_model', 'POST', { 
                            model_path: 'models/deepseek-coder-1.3b-base' 
                        });

                        if (result.status === 'success') {
                            updateStatus('model-status', '✅ 模型已加载', true);
                        } else {
                            updateStatus('model-status', '❌ 模型加载失败: ' + result.message);
                        }
                    }

                    // 测试模型
                    async function testModel() {
                        const question = document.getElementById('test-question').value;
                        if (!question.trim()) {
                            alert('请输入测试问题');
                            return;
                        }

                        document.getElementById('model-response').textContent = '正在思考...';
                        const result = await callAPI('test_model', 'POST', { question });

                        if (result.status === 'success') {
                            document.getElementById('model-response').textContent = result.response;
                        } else {
                            document.getElementById('model-response').textContent = '❌ 错误: ' + result.message;
                        }
                    }

                    // 连接Neo4j
                    async function connectNeo4j() {
                        updateStatus('neo4j-status', '正在连接...');
                        const result = await callAPI('connect_neo4j', 'POST', {
                            uri: 'bolt://localhost:7687',
                            user: 'neo4j',
                            password: 'admin123'
                        });

                        if (result.status === 'success') {
                            updateStatus('neo4j-status', '✅ Neo4j已连接', true);
                        } else {
                            updateStatus('neo4j-status', '❌ 连接失败: ' + result.message);
                        }
                    }

                    // 生成示例知识
                    async function generateSampleKnowledge() {
                        const result = await callAPI('generate_sample_knowledge', 'POST');

                        if (result.status === 'success') {
                            alert('✅ 示例知识生成成功！共生成 ' + (result.count || 0) + ' 条知识');
                        } else {
                            alert('❌ 示例知识生成失败: ' + result.message);
                        }
                    }

                    // 处理文本
                    async function processText() {
                        const text = document.getElementById('text-input').value;
                        if (!text.trim()) {
                            alert('请输入要处理的文本');
                            return;
                        }

                        document.getElementById('text-result').textContent = '正在处理...';
                        const result = await callAPI('process_text', 'POST', { text });

                        if (result.status === 'success') {
                            document.getElementById('text-result').innerHTML = 
                                '<strong>分词结果:</strong> ' + result.tokens.join(', ') + '<br>' +
                                '<strong>关键词:</strong> ' + result.keywords.join(', ') + '<br>' +
                                '<strong>问题类型:</strong> ' + result.question_type;
                        } else {
                            document.getElementById('text-result').textContent = '❌ 错误: ' + result.message;
                        }
                    }

                    // 发送消息
                    async function sendMessage() {
                        const input = document.getElementById('chat-input');
                        const message = input.value.trim();

                        if (!message) return;

                        // 添加用户消息
                        const chatContainer = document.getElementById('chat-container');
                        const userMsg = document.createElement('div');
                        userMsg.innerHTML = '<strong>你:</strong> ' + message;
                        userMsg.style.cssText = 'background: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 10px; text-align: right;';
                        chatContainer.appendChild(userMsg);

                        input.value = '';

                        // 显示正在思考
                        const thinkingMsg = document.createElement('div');
                        thinkingMsg.innerHTML = '<strong>PEPPER:</strong> 正在思考...';
                        thinkingMsg.style.cssText = 'background: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;';
                        chatContainer.appendChild(thinkingMsg);
                        chatContainer.scrollTop = chatContainer.scrollHeight;

                        // 发送到AI助手
                        const result = await callAPI('chat', 'POST', { message });

                        // 移除思考消息
                        chatContainer.removeChild(thinkingMsg);

                        // 添加AI回复
                        const assistantMsg = document.createElement('div');
                        if (result.status === 'success') {
                            assistantMsg.innerHTML = '<strong>PEPPER:</strong> ' + result.response;
                        } else {
                            assistantMsg.innerHTML = '<strong>PEPPER:</strong> 抱歉，我现在无法回答。错误: ' + result.message;
                        }
                        assistantMsg.style.cssText = 'background: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;';
                        chatContainer.appendChild(assistantMsg);
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }

                    // 回车发送消息
                    document.getElementById('chat-input').addEventListener('keypress', function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            sendMessage();
                        }
                    });

                    // 页面加载完成后的初始化
                    document.addEventListener('DOMContentLoaded', function() {
                        console.log('PEPPER智能教学系统已加载');

                        // 检查系统状态
                        callAPI('system_status').then(result => {
                            if (result.status === 'success') {
                                console.log('系统状态:', result);
                            }
                        });
                    });
                </script>
            </body>
            </html>
            """
    except Exception as e:
        logger.error(f"主页路由出错: {e}")
        return f"<h1>系统错误</h1><p>{str(e)}</p>"


# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "API接口不存在"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "message": "服务器内部错误"
    }), 500


if __name__ == '__main__':
    # 初始化服务
    initialize_services()

    logger.info("PEPPER智能教学系统API服务器启动中...")
    logger.info("访问地址: http://localhost:5000")

    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=True)