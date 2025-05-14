# 在interface/web_console/app.py中创建Web控制台

import json
import logging
import os
import sys
import time

from flask import Flask, render_template, request, jsonify

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入集成系统
from integrated_system import PepperIntegratedSystem

app = Flask(__name__)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("WEB_CONSOLE")

# 全局变量
system = None
system_status = {
    "initialized": False,
    "running": False,
    "student_id": None,
    "topic": None,
    "log_messages": []
}


# 日志处理器，捕获日志消息
class WebLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage()
        }

        # 添加到状态中的日志消息
        system_status["log_messages"].append(log_entry)

        # 保持日志消息数量在合理范围内
        if len(system_status["log_messages"]) > 100:
            system_status["log_messages"] = system_status["log_messages"][-100:]


# 添加日志处理器
root_logger = logging.getLogger()
root_logger.addHandler(WebLogHandler())


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """获取系统状态"""
    global system_status
    return jsonify(system_status)


@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """初始化系统"""
    global system, system_status

    if system_status["initialized"]:
        return jsonify({"status": "error", "message": "系统已初始化"})

    try:
        config_path = request.json.get('config_path', 'config.json')

        # 加载配置
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            return jsonify({"status": "error", "message": f"配置文件不存在: {config_path}"})

        # 初始化系统
        system = PepperIntegratedSystem(config)
        system_status["initialized"] = True

        return jsonify({"status": "success", "message": "系统初始化成功"})

    except Exception as e:
        logger.error(f"初始化系统失败: {e}")
        return jsonify({"status": "error", "message": f"初始化系统失败: {str(e)}"})


@app.route('/api/start_session', methods=['POST'])
def start_session():
    """启动教学会话"""
    global system, system_status

    if not system_status["initialized"]:
        return jsonify({"status": "error", "message": "系统未初始化"})

    if system_status["running"]:
        return jsonify({"status": "error", "message": "教学会话已在运行"})

    try:
        data = request.json
        student_id = data.get('student_id')
        topic = data.get('topic')

        # 准备教学会话
        success = system.prepare_teaching_session(student_id, topic)
        if not success:
            return jsonify({"status": "error", "message": "准备教学会话失败"})

        # 启动教学会话
        welcome_message = system.start_teaching_session()

        # 更新状态
        system_status["running"] = True
        system_status["student_id"] = student_id
        system_status["topic"] = topic

        return jsonify({
            "status": "success",
            "message": "教学会话已启动",
            "welcome_message": welcome_message
        })

    except Exception as e:
        logger.error(f"启动教学会话失败: {e}")
        return jsonify({"status": "error", "message": f"启动教学会话失败: {str(e)}"})


@app.route('/api/stop_session', methods=['POST'])
def stop_session():
    """停止教学会话"""
    global system, system_status

    if not system_status["initialized"]:
        return jsonify({"status": "error", "message": "系统未初始化"})

    if not system_status["running"]:
        return jsonify({"status": "error", "message": "没有正在运行的教学会话"})

    try:
        # 停止教学会话
        success = system.stop_teaching_session()

        # 更新状态
        system_status["running"] = False

        return jsonify({
            "status": "success",
            "message": "教学会话已停止"
        })

    except Exception as e:
        logger.error(f"停止教学会话失败: {e}")
        return jsonify({"status": "error", "message": f"停止教学会话失败: {str(e)}"})


@app.route('/api/process_input', methods=['POST'])
def process_input():
    """处理用户输入"""
    global system, system_status

    if not system_status["initialized"]:
        return jsonify({"status": "error", "message": "系统未初始化"})

    if not system_status["running"]:
        return jsonify({"status": "error", "message": "没有正在运行的教学会话"})

    try:
        data = request.json
        input_text = data.get('text')

        if not input_text:
            return jsonify({"status": "error", "message": "输入文本为空"})

        # 处理文本输入
        response = system.process_text_input(input_text)

        return jsonify({
            "status": "success",
            "response": response
        })

    except Exception as e:
        logger.error(f"处理输入失败: {e}")
        return jsonify({"status": "error", "message": f"处理输入失败: {str(e)}"})


@app.route('/api/get_students')
def get_students():
    """获取学生列表"""
    global system, system_status

    if not system_status["initialized"]:
        return jsonify({"status": "error", "message": "系统未初始化"})

    try:
        students = []
        for student_id, profile in system.teaching.student_profiles.items():
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
        return jsonify({"status": "error", "message": f"获取学生列表失败: {str(e)}"})


@app.route('/api/clean_up', methods=['POST'])
def clean_up():
    """清理系统资源"""
    global system, system_status

    if not system_status["initialized"]:
        return jsonify({"status": "error", "message": "系统未初始化"})

    try:
        if system_status["running"]:
            system.stop_teaching_session()

        # 清理资源
        success = system.clean_up()

        # 重置状态
        system_status["initialized"] = False
        system_status["running"] = False
        system_status["student_id"] = None
        system_status["topic"] = None

        return jsonify({
            "status": "success",
            "message": "系统资源已清理"
        })

    except Exception as e:
        logger.error(f"清理系统资源失败: {e}")
        return jsonify({"status": "error", "message": f"清理系统资源失败: {str(e)}"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)