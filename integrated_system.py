"""
PEPPER机器人智能教学集成系统
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
from queue import Queue

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入多模态交互模块
from ai_service.multimodal.speech_recognition import SpeechRecognizer
from ai_service.multimodal.image_recognition import ImageRecognizer

# 导入知识图谱模块
from ai_service.knowledge_graph.knowledge_graph import KnowledgeGraph
from ai_service.knowledge_graph.education_knowledge_processor import EducationKnowledgeProcessor

# 导入大语言模型模块
from ai_service.llm_module.llm_interface import LLMService

# 导入个性化教学模块
from ai_service.teaching_module.personalized_teaching import PersonalizedTeaching

# 导入PEPPER机器人控制模块
from pepper_robot.robot_control.robot_controller import PepperRobot
from pepper_robot.motion_module.gestures import PepperGestures
from pepper_robot.sensor_module.sensor_handler import PepperSensors

# 导入通信桥接模块
from interface.bridge.websocket_client import WebSocketClient

from env_caller import call_spacy_env, call_langchain_env


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pepper_integrated_system.log')
    ]
)
logger = logging.getLogger("PEPPER_INTEGRATED")


class PepperIntegratedSystem:
    """PEPPER机器人智能教学集成系统"""

    def __init__(self, config=None):
        """初始化集成系统"""
        logger.info("正在初始化PEPPER机器人智能教学集成系统...")

        # 加载配置
        self.config = config if config else self._load_default_config()

        # 初始化标志
        self.is_initialized = False
        self.is_running = False

        # 消息队列（用于模块间通信）
        self.message_queue = Queue()

        # 初始化各个模块
        self._init_components()

        # 设置运行状态
        self.current_student_id = None
        self.current_topic = None
        self.conversation_history = []

        # 完成初始化
        self.is_initialized = True
        logger.info("PEPPER机器人智能教学集成系统初始化完成")

    def _load_default_config(self):
        """加载默认配置"""
        config = {
            "robot": {
                "ip": "127.0.0.1",
                "port": 9559,
                "simulation_mode": True
            },
            "knowledge_graph": {
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "password"
            },
            "llm": {
                "model_path": "models/deepseek-coder-1.3b-base",
                "use_8bit": True
            },
            "data_paths": {
                "student_profiles": "data/student_profiles",
                "course_materials": "data/course_materials"
            },
            "websocket": {
                "url": "ws://localhost:8765"
            }
        }
        return config

    def _init_components(self):
        """初始化各个组件"""
        try:
            # 初始化多模态交互模块
            logger.info("正在初始化多模态交互模块...")
            self.speech_recognizer = SpeechRecognizer()
            self.image_recognizer = ImageRecognizer()

            logger.info("正在初始化知识图谱模块...")
            kg_config = self.config["knowledge_graph"]
            self.knowledge_graph = KnowledgeGraph(
                kg_config["uri"], kg_config["user"], kg_config["password"]
            )
            self.knowledge_processor = EducationKnowledgeProcessor(
                kg_config["uri"], kg_config["user"], kg_config["password"]
            )

            # 初始化大语言模型模块
            logger.info("正在初始化大语言模型模块...")
            llm_config = self.config["llm"]
            self.llm_service = LLMService(llm_config["model_path"])

            # 初始化个性化教学模块
            logger.info("正在初始化个性化教学模块...")
            self.teaching = PersonalizedTeaching(
                kg_config["uri"], kg_config["user"], kg_config["password"]
            )

            # 加载课程教学知识
            self._load_education_knowledge()

            # 加载学生档案
            self._load_student_profiles()

            # 初始化PEPPER机器人控制（如果不是模拟模式）
            robot_config = self.config["robot"]
            if not robot_config["simulation_mode"]:
                logger.info("正在连接PEPPER机器人...")
                self.robot = PepperRobot(robot_config["ip"], robot_config["port"])
                self.gestures = PepperGestures(robot_config["ip"], robot_config["port"])
                self.sensors = PepperSensors(robot_config["ip"], robot_config["port"])
                self.robot_connected = True
            else:
                logger.info("PEPPER机器人运行在模拟模式")
                self.robot_connected = False

            # 初始化WebSocket客户端（如果需要）
            if "websocket" in self.config:
                logger.info("正在初始化WebSocket客户端...")
                ws_config = self.config["websocket"]
                self.ws_client = WebSocketClient(ws_config["url"])
                # 仅连接，不启动
                self.ws_connected = False
            else:
                self.ws_connected = False

        except Exception as e:
            logger.error(f"初始化组件失败: {e}")
            raise

    def _load_education_knowledge(self):
        """加载教育知识"""
        try:
            # 检查知识图谱是否已存在数据
            query = "MATCH (n) RETURN count(n) as count"
            result = self.knowledge_graph.query(query)

            if not result or result[0]["count"] < 10:
                logger.info("知识图谱为空，正在创建基础教育知识库...")
                self.knowledge_processor.create_educational_knowledge_base()
            else:
                logger.info(f"知识图谱已包含 {result[0]['count']} 个节点")
        except Exception as e:
            logger.error(f"加载教育知识失败: {e}")

    def _load_student_profiles(self):
        """加载学生档案"""
        try:
            # 确保目录存在
            profile_dir = self.config["data_paths"]["student_profiles"]
            if not os.path.exists(profile_dir):
                os.makedirs(profile_dir)

            # 加载已有学生档案
            self.teaching.load_student_profiles(profile_dir)

            # 如果没有学生档案，创建演示档案
            if not self.teaching.student_profiles:
                logger.info("没有找到学生档案，创建演示用学生档案...")
                self.teaching.create_demo_student_profiles()

                # 保存创建的档案
                self.teaching.save_student_profiles(profile_dir)

        except Exception as e:
            logger.error(f"加载学生档案失败: {e}")

    def connect_robot(self):
        """连接PEPPER机器人"""
        if self.config["robot"]["simulation_mode"]:
            logger.info("运行在模拟模式，不连接实际机器人")
            return True

        try:
            if not self.robot_connected:
                robot_config = self.config["robot"]
                self.robot = PepperRobot(robot_config["ip"], robot_config["port"])
                self.gestures = PepperGestures(robot_config["ip"], robot_config["port"])
                self.sensors = PepperSensors(robot_config["ip"], robot_config["port"])
                self.robot_connected = True
                logger.info("成功连接到PEPPER机器人")
            return True
        except Exception as e:
            logger.error(f"连接PEPPER机器人失败: {e}")
            return False

    def connect_websocket(self):
        """连接WebSocket服务"""
        if "websocket" not in self.config:
            logger.info("未配置WebSocket，跳过连接")
            return False

        try:
            if not self.ws_connected:
                ws_config = self.config["websocket"]
                self.ws_client = WebSocketClient(ws_config["url"])
                if self.ws_client.connect():
                    self.ws_connected = True
                    logger.info("成功连接到WebSocket服务")
                    return True
                else:
                    logger.error("连接WebSocket服务失败")
                    return False
            return True
        except Exception as e:
            logger.error(f"连接WebSocket服务失败: {e}")
            return False

    def prepare_teaching_session(self, student_id=None, topic=None):
        """准备教学会话"""
        # 如果未指定学生，使用第一个可用的学生档案
        if not student_id:
            if self.teaching.student_profiles:
                student_id = list(self.teaching.student_profiles.keys())[0]
            else:
                logger.error("没有可用的学生档案")
                return False

        # 检查学生档案是否存在
        student_profile = self.teaching.get_student_profile(student_id)
        if not student_profile:
            logger.error(f"找不到学生档案: {student_id}")
            return False

        # 设置当前学生
        self.current_student_id = student_id
        self.current_topic = topic

        # 清空会话历史
        self.conversation_history = []

        # 初始化语音识别词汇表（如果在非模拟模式）
        if self.robot_connected:
            # 设置基础词汇
            vocabulary = ["PEPPER", "你好", "问题", "帮助", "谢谢", "再见"]

            # 添加主题相关词汇
            if topic:
                # 从知识图谱获取主题相关的关键词
                try:
                    results = self.knowledge_graph.find_related_knowledge(topic)
                    for item in results:
                        if "start_node" in item and "name" in item["start_node"]:
                            vocabulary.append(item["start_node"]["name"])
                        if "end_node" in item and "name" in item["end_node"]:
                            vocabulary.append(item["end_node"]["name"])
                except Exception as e:
                    logger.error(f"获取主题词汇失败: {e}")

            # 添加学生常用词汇
            try:
                preferences = student_profile.get_top_preferences(5)
                for topic, _ in preferences:
                    vocabulary.append(topic)
            except Exception as e:
                logger.error(f"获取学生偏好词汇失败: {e}")

            # 设置语音识别词汇表
            try:
                self.sensors.setup_speech_recognition(vocabulary)
                logger.info(f"语音识别词汇表设置完成，包含{len(vocabulary)}个词")
            except Exception as e:
                logger.error(f"设置语音识别词汇表失败: {e}")

        logger.info(f"教学会话准备完成，学生: {student_id}, 主题: {topic or '未指定'}")
        return True

    def start_teaching_session(self):
        """启动教学会话"""
        if not self.is_initialized:
            logger.error("系统未完成初始化，无法启动教学会话")
            return False

        if not self.current_student_id:
            logger.error("未设置当前学生，请先准备教学会话")
            return False

        # 设置运行标志
        self.is_running = True

        # 获取学生信息
        student_profile = self.teaching.get_student_profile(self.current_student_id)
        student_name = student_profile.name or f"学生{self.current_student_id}"

        # 欢迎开场
        welcome_message = f"你好，{student_name}！我是PEPPER机器人。"

        if self.current_topic:
            welcome_message += f"今天我们将学习关于{self.current_topic}的知识。"
        else:
            welcome_message += "我可以回答你的问题，帮助你学习。"

        welcome_message += "你可以随时向我提问，我会尽力帮助你。"

        logger.info(f"开始教学会话: {welcome_message}")

        # 如果连接了机器人，使用机器人说话和做手势
        if self.robot_connected:
            try:
                self.robot.say(welcome_message)
                self.gestures.wave_hand()

                # 启动语音识别
                self.sensors.start_speech_recognition()
            except Exception as e:
                logger.error(f"机器人交互失败: {e}")

        return welcome_message

    def stop_teaching_session(self):
        """停止教学会话"""
        self.is_running = False

        # 如果连接了机器人，关闭语音识别
        if self.robot_connected:
            try:
                self.sensors.stop_speech_recognition()
                self.robot.say("教学会话已结束，谢谢参与！")
                self.gestures.bow()
            except Exception as e:
                logger.error(f"停止机器人交互失败: {e}")

        # 保存学生档案
        try:
            profile_dir = self.config["data_paths"]["student_profiles"]
            self.teaching.save_student_profiles(profile_dir)
            logger.info("学生档案已保存")
        except Exception as e:
            logger.error(f"保存学生档案失败: {e}")

        logger.info("教学会话已结束")
        return True

    def process_speech_input(self, max_wait_time=10):
        """处理语音输入"""
        if not self.robot_connected:
            logger.warning("机器人未连接，无法处理语音输入")
            return None

        logger.info("等待语音输入...")
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                # 获取识别到的词语
                recognized_words = self.sensors.get_recognized_words()

                if recognized_words:
                    logger.info(f"识别到语音输入: {recognized_words}")
                    return recognized_words

                # 短暂等待，避免CPU占用过高
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"处理语音输入失败: {e}")
                return None

        logger.info("语音输入等待超时")
        return None

    def process_text_input(self, text):
        """处理文本输入"""
        if not text:
            return None

        # 记录到会话历史
        self.conversation_history.append({
            "role": "user",
            "content": text,
            "timestamp": datetime.datetime.now().isoformat()
        })

        # 使用文本处理器分析输入
        try:
            stdout, stderr, returncode = call_spacy_env("spacy_functions.py", text)
            if returncode != 0:
                logger.error(f"文本处理失败: {stderr}")
                return None

            text_analysis = json.loads(stdout)
            keywords = text_analysis["keywords"]
            question_type = text_analysis["question_type"]

            logger.info(f"文本分析结果: 问题类型={question_type}, 关键词={keywords}")

            # 获取知识图谱相关信息
            knowledge_items = []
            for keyword in keywords:
                items = self.knowledge_graph.find_related_knowledge(keyword)
                knowledge_items.extend(items)

            # 生成个性化回答
            response = self.teaching.generate_personalized_answer(
                self.current_student_id, text
            )

            # 如果连接了机器人，使用机器人说话
            if self.robot_connected:
                try:
                    # 思考手势
                    self.gestures.thinking_gesture()

                    # 分段输出长回答
                    if len(response) > 200:
                        sentences = response.split('。')
                        for i in range(0, len(sentences), 3):
                            sentence_group = '。'.join(sentences[i:i + 3]) + '。'
                            self.robot.say(sentence_group)
                            time.sleep(0.5)
                    else:
                        self.robot.say(response)
                except Exception as e:
                    logger.error(f"机器人语音输出失败: {e}")

            # 记录到会话历史
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.datetime.now().isoformat()
            })

            # 记录学习交互
            self.teaching.add_learning_interaction(
                self.current_student_id,
                keywords[0] if keywords else "general",
                text
            )

            return response

        except Exception as e:
            logger.error(f"处理文本输入失败: {e}")
            error_response = "抱歉，我无法处理这个问题。请换一种方式提问。"

            if self.robot_connected:
                try:
                    self.robot.say(error_response)
                    self.gestures.shake_head()
                except:
                    pass

            return error_response

    def recommend_next_steps(self):
        """推荐下一步学习内容"""
        if not self.current_student_id:
            logger.error("未设置当前学生，无法推荐学习内容")
            return None

        try:
            # 获取当前主题或学生最近交互的主题
            topic = self.current_topic
            if not topic and self.conversation_history:
                # 尝试从最近的对话中提取主题
                for entry in reversed(self.conversation_history):
                    if entry["role"] == "user":
                        keywords = self.text_processor.extract_keywords(entry["content"])
                        if keywords:
                            topic = keywords[0]
                            break

            # 如果仍未确定主题，使用学生偏好
            if not topic:
                student_profile = self.teaching.get_student_profile(self.current_student_id)
                preferences = student_profile.get_top_preferences(1)
                if preferences:
                    topic = preferences[0][0]
                else:
                    topic = "Python编程"  # 默认主题

            # 生成学习路径
            learning_path = self.teaching.generate_learning_path(
                self.current_student_id, topic
            )

            # 推荐学习资源
            resources = self.teaching.recommend_learning_resources(
                self.current_student_id, topic, count=3
            )

            # 构建完整推荐
            recommendation = {
                "topic": topic,
                "learning_path": learning_path["learning_path"] if learning_path else "",
                "resources": resources
            }

            return recommendation

        except Exception as e:
            logger.error(f"生成学习推荐失败: {e}")
            return None

    def interactive_teaching_loop(self, max_interactions=10):
        """交互式教学循环"""
        if not self.is_running:
            if not self.start_teaching_session():
                return False

        interaction_count = 0

        logger.info(f"开始交互式教学循环，计划进行{max_interactions}次交互")

        while self.is_running and interaction_count < max_interactions:
            try:
                # 等待并处理输入
                if self.robot_connected:
                    # 语音输入
                    input_text = self.process_speech_input()
                    if not input_text:
                        # 如果没有收到语音输入，提供提示
                        if interaction_count > 0 and interaction_count % 3 == 0:
                            # 每隔3次交互，提供主动提示
                            prompt_message = "你有什么问题想问我吗？"
                            self.robot.say(prompt_message)

                        continue
                else:
                    # 模拟模式，使用预设问题
                    preset_questions = [
                        "Python中for循环和while循环有什么区别？",
                        "人工智能在教育中有哪些应用？",
                        "什么是多模态交互？",
                        "PEPPER机器人在课堂上的优势是什么？"
                    ]
                    input_text = preset_questions[interaction_count % len(preset_questions)]
                    logger.info(f"模拟用户输入: {input_text}")

                # 处理输入并生成回答
                response = self.process_text_input(input_text)

                # 如果是模拟模式，打印回答
                if not self.robot_connected:
                    logger.info(f"系统回答: {response[:100]}...")
                    # 模拟交互间隔
                    time.sleep(2)

                interaction_count += 1

                # 检查是否结束会话（如果用户说"再见"或类似词语）
                end_keywords = ["再见", "结束", "停止", "退出"]
                if any(keyword in input_text for keyword in end_keywords):
                    logger.info("用户请求结束会话")
                    self.stop_teaching_session()
                    break

            except Exception as e:
                logger.error(f"交互过程出错: {e}")
                if self.robot_connected:
                    try:
                        self.robot.say("抱歉，我遇到了一些问题。请稍后再试。")
                    except:
                        pass

        # 会话结束时，提供学习推荐
        if self.is_running:
            try:
                recommendation = self.recommend_next_steps()
                if recommendation:
                    next_steps_message = f"关于{recommendation['topic']}的学习，我建议你下一步可以："

                    # 从学习路径中提取1-2个关键建议
                    if recommendation['learning_path']:
                        path_lines = recommendation['learning_path'].split('\n')
                        key_suggestions = [line for line in path_lines if '- ' in line][:2]
                        if key_suggestions:
                            next_steps_message += ''.join(key_suggestions)

                    # 推荐资源
                    if recommendation['resources']:
                        next_steps_message += "我还为你准备了一些相关学习资源。"

                    logger.info(f"学习推荐: {next_steps_message}")

                    if self.robot_connected:
                        self.robot.say(next_steps_message)
            except Exception as e:
                logger.error(f"生成学习推荐失败: {e}")

            # 结束会话
            self.stop_teaching_session()

        return True

    def clean_up(self):
        """清理资源"""
        logger.info("正在清理系统资源...")

        # 停止教学会话
        if self.is_running:
            self.stop_teaching_session()

        # 关闭WebSocket连接
        if self.ws_connected:
            try:
                self.ws_client.close()
                logger.info("WebSocket连接已关闭")
            except Exception as e:
                logger.error(f"关闭WebSocket连接失败: {e}")

        # 关闭机器人连接
        if self.robot_connected:
            try:
                self.sensors.clean_up()
                self.robot.rest()
                logger.info("PEPPER机器人连接已关闭")
            except Exception as e:
                logger.error(f"关闭机器人连接失败: {e}")

        # 关闭知识图谱连接
        try:
            self.knowledge_graph.close()
            logger.info("知识图谱连接已关闭")
        except Exception as e:
            logger.error(f"关闭知识图谱连接失败: {e}")

        logger.info("系统资源清理完成")
        return True

    def run(self, student_id=None, topic=None, interaction_count=10):
        """运行完整教学流程"""
        try:
            # 准备教学会话
            if not self.prepare_teaching_session(student_id, topic):
                logger.error("教学会话准备失败")
                return False

            # 执行交互式教学
            self.interactive_teaching_loop(interaction_count)

            # 清理资源
            self.clean_up()

            return True

        except Exception as e:
            logger.error(f"运行教学流程失败: {e}")
            self.clean_up()
            return False

    def use_langchain_query(self, query):
        """使用langchain查询知识图谱"""
        kg_config = self.config["knowledge_graph"]

        stdout, stderr, returncode = call_langchain_env(
            "langchain_functions.py",
            query,
            kg_config["uri"],
            kg_config["user"],
            kg_config["password"]
        )

        if returncode != 0:
            logger.error(f"LangChain查询失败: {stderr}")
            return None

        try:
            result = json.loads(stdout)
            return result["response"]
        except Exception as e:
            logger.error(f"解析LangChain结果失败: {e}")
            return None

    def _initialize_teaching_actions(self):
        """初始化教学动作"""
        try:
            robot_config = self.config["robot"]
            from pepper_robot.motion_module.teaching_actions import TeachingActions

            self.teaching_actions = TeachingActions(
                ip=robot_config["ip"],
                port=robot_config["port"],
                simulation_mode=robot_config["simulation_mode"]
            )
            logger.info("教学动作模块初始化成功")
            return True
        except Exception as e:
            logger.error(f"初始化教学动作模块失败: {e}")
            return False

    def _initialize_kg_llm_integration(self):
        """初始化知识图谱和LLM集成"""
        try:
            kg_config = self.config["knowledge_graph"]
            from ai_service.llm_module.kg_llm_integration import KnowledgeLLMIntegration

            self.kg_llm = KnowledgeLLMIntegration(
                kg_uri=kg_config["uri"],
                kg_user=kg_config["user"],
                kg_password=kg_config["password"]
            )
            logger.info("知识图谱和LLM集成模块初始化成功")
            return True
        except Exception as e:
            logger.error(f"初始化知识图谱和LLM集成模块失败: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PEPPER机器人智能教学集成系统')
    parser.add_argument('--config', type=str, default='config.json',
                        help='配置文件路径')
    parser.add_argument('--student', type=str, default=None,
                        help='学生ID')
    parser.add_argument('--topic', type=str, default=None,
                        help='教学主题')
    parser.add_argument('--simulation', action='store_true',
                        help='使用模拟模式（不连接实际机器人）')
    parser.add_argument('--interactions', type=int, default=10,
                        help='交互次数')

    args = parser.parse_args()

    # 加载配置
    config = None
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 如果指定了模拟模式，覆盖配置
            if args.simulation:
                config["robot"]["simulation_mode"] = True

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            config = None

    # 创建并运行集成系统
    integrated_system = PepperIntegratedSystem(config)
    integrated_system.run(args.student, args.topic, args.interactions)


if __name__ == "__main__":
    main()
