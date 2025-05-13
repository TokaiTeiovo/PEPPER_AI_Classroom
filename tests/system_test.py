"""
PEPPER机器人智能教学系统测试模块

该模块提供了系统级功能测试，用于验证智能教学系统的各个组件正常运行
"""

import json
import logging
import os
import sys
import unittest

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入系统组件
from integrated_system import PepperIntegratedSystem
from ai_service.multimodal.text_processor import TextProcessor
from ai_service.knowledge_graph.knowledge_graph import KnowledgeGraph
from ai_service.llm_module.llm_interface import LLMService
from ai_service.teaching_module.personalized_teaching import PersonalizedTeaching

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tests/system_test.log')
    ]
)
logger = logging.getLogger("SYSTEM_TEST")


class TestIntegratedSystem(unittest.TestCase):
    """测试集成系统功能"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        # 加载测试配置
        config_path = "tests/test_config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cls.config = json.load(f)
            except Exception as e:
                logger.error(f"加载测试配置失败: {e}")
                cls.config = None
        else:
            # 使用默认测试配置
            cls.config = {
                "robot": {
                    "ip": "127.0.0.1",
                    "port": 9559,
                    "simulation_mode": True  # 测试时始终使用模拟模式
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
                    "student_profiles": "tests/test_data/student_profiles",
                    "course_materials": "tests/test_data/course_materials"
                }
            }

        # 确保测试目录存在
        os.makedirs(cls.config["data_paths"]["student_profiles"], exist_ok=True)
        os.makedirs(cls.config["data_paths"]["course_materials"], exist_ok=True)

    def setUp(self):
        """每个测试前执行"""
        # 创建系统实例
        self.system = PepperIntegratedSystem(self.config)

    def tearDown(self):
        """每个测试后执行"""
        # 清理资源
        if hasattr(self, 'system'):
            self.system.clean_up()

    def test_system_initialization(self):
        """测试系统初始化"""
        self.assertTrue(self.system.is_initialized)
        self.assertIsNotNone(self.system.text_processor)
        self.assertIsNotNone(self.system.knowledge_graph)
        self.assertIsNotNone(self.system.llm_service)
        self.assertIsNotNone(self.system.teaching)
        logger.info("系统初始化测试通过")

    def test_teaching_session_preparation(self):
        """测试教学会话准备"""
        # 测试会话准备
        result = self.system.prepare_teaching_session()
        self.assertTrue(result)
        self.assertIsNotNone(self.system.current_student_id)
        logger.info("教学会话准备测试通过")

    def test_text_processing(self):
        """测试文本处理功能"""
        # 首先准备会话
        self.system.prepare_teaching_session()

        # 测试处理用户输入
        test_inputs = [
            "Python中for循环和while循环有什么区别？",
            "什么是多模态交互？",
            "人工智能在教育中有什么应用？"
        ]

        for test_input in test_inputs:
            response = self.system.process_text_input(test_input)
            self.assertIsNotNone(response)
            self.assertGreater(len(response), 50)  # 确保回复有一定长度
            self.assertIn(self.system.current_student_id, self.system.teaching.student_profiles)

            # 验证会话历史记录
            found = False
            for entry in self.system.conversation_history:
                if entry["role"] == "user" and entry["content"] == test_input:
                    found = True
                    break
            self.assertTrue(found)

        logger.info("文本处理功能测试通过")

    def test_next_steps_recommendation(self):
        """测试学习推荐功能"""
        # 首先准备会话
        self.system.prepare_teaching_session()

        # 添加一些会话历史
        self.system.process_text_input("什么是Python循环？")

        # 测试学习推荐
        recommendation = self.system.recommend_next_steps()
        self.assertIsNotNone(recommendation)
        self.assertIn("topic", recommendation)
        self.assertIn("learning_path", recommendation)
        self.assertIn("resources", recommendation)
        self.assertGreater(len(recommendation["learning_path"]), 50)

        logger.info("学习推荐功能测试通过")


class TestComponentIntegration(unittest.TestCase):
    """测试关键组件集成"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        # 测试配置
        cls.kg_config = {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        }

    def test_text_processor(self):
        """测试文本处理器"""
        processor = TextProcessor()

        # 测试分词
        test_text = "PEPPER机器人可以帮助学生学习Python编程"
        words = processor.preprocess_text(test_text)
        self.assertIsNotNone(words)
        self.assertGreater(len(words), 0)

        # 测试关键词提取
        keywords = processor.extract_keywords(test_text)
        self.assertIsNotNone(keywords)
        self.assertGreater(len(keywords), 0)

        # 测试问题分类
        question = "什么是Python循环？"
        question_type = processor.classify_question(question)
        self.assertIsNotNone(question_type)

        logger.info("文本处理器测试通过")

    def test_knowledge_graph(self):
        """测试知识图谱"""
        try:
            kg = KnowledgeGraph(
                self.kg_config["uri"],
                self.kg_config["user"],
                self.kg_config["password"]
            )

            # 测试创建节点
            kg.create_node("TestConcept", {"name": "测试概念", "description": "这是一个测试概念"})

            # 测试查询
            results = kg.find_related_knowledge("测试概念")

            # 清理测试数据
            kg.query("MATCH (n:TestConcept {name: '测试概念'}) DELETE n")

            kg.close()
            logger.info("知识图谱测试通过")
        except Exception as e:
            logger.error(f"知识图谱测试失败: {e}")
            self.fail(f"知识图谱测试失败: {e}")

    def test_llm_service(self):
        """测试大语言模型服务"""
        try:
            llm = LLMService()

            # 测试生成回答
            response = llm.generate_response("Python是什么编程语言？", max_length=100)
            self.assertIsNotNone(response)
            self.assertGreater(len(response), 10)

            logger.info("大语言模型服务测试通过")
        except Exception as e:
            logger.error(f"大语言模型服务测试失败: {e}")
            self.fail(f"大语言模型服务测试失败: {e}")

    def test_personalized_teaching(self):
        """测试个性化教学模块"""
        try:
            teaching = PersonalizedTeaching(
                self.kg_config["uri"],
                self.kg_config["user"],
                self.kg_config["password"]
            )

            # 创建测试学生档案
            student_ids = teaching.create_demo_student_profiles()
            self.assertGreater(len(student_ids), 0)

            # 测试推荐学习资源
            resources = teaching.recommend_learning_resources(student_ids[0], "Python编程")
            self.assertIsNotNone(resources)
            self.assertGreater(len(resources), 0)

            # 测试生成个性化回答
            answer = teaching.generate_personalized_answer(
                student_ids[0], "什么是Python循环？"
            )
            self.assertIsNotNone(answer)
            self.assertGreater(len(answer), 50)

            logger.info("个性化教学模块测试通过")
        except Exception as e:
            logger.error(f"个性化教学模块测试失败: {e}")
            self.fail(f"个性化教学模块测试失败: {e}")


def run_tests():
    """运行所有测试"""
    logger.info("开始运行系统测试...")

    # 创建测试套件
    suite = unittest.TestSuite()

    # 添加集成系统测试
    suite.addTest(unittest.makeSuite(TestIntegratedSystem))

    # 添加组件集成测试
    suite.addTest(unittest.makeSuite(TestComponentIntegration))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 分析结果
    if result.wasSuccessful():
        logger.info("所有测试通过")
        return True
    else:
        logger.error(f"测试失败: 错误数={len(result.errors)}, 失败数={len(result.failures)}")
        return False


if __name__ == "__main__":
    # 运行测试
    success = run_tests()
    sys.exit(0 if success else 1)

    # 也可以单独运行特定测试
    # unittest.main(defaultTest="TestIntegratedSystem")
