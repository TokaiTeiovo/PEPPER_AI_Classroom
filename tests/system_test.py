"""
综合测试模块 - 验证PEPPER机器人教学系统的各项功能

该模块执行一系列测试来验证系统的核心功能，包括:
1. LoRA模型微调
2. 知识图谱构建与查询
3. 个性化教学推荐
4. 多模态交互
"""

import json
import logging
import os
import sys
import unittest

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入各个模块
from ai_service.llm_module.lora_fine_tuning import LoRAFineTuner
from ai_service.knowledge_graph.education_knowledge_processor import EducationKnowledgeProcessor
from ai_service.teaching_module.personalized_teaching import PersonalizedTeaching
from ai_service.multimodal.text_processor import TextProcessor
from ai_service.multimodal.speech_recognition import SpeechRecognizer
from ai_service.multimodal.image_recognition import ImageRecognizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('tests/system_test.log')]
)
logger = logging.getLogger("SYSTEM_TEST")


class TestLoRAFineTuning(unittest.TestCase):
    """测试LoRA微调功能"""

    def setUp(self):
        """初始化测试环境"""
        self.test_data_path = 'tests/test_data/lora_test_data.json'
        self.output_dir = 'tests/test_output/lora_fine_tuned'

        # 创建测试数据目录
        os.makedirs(os.path.dirname(self.test_data_path), exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # 创建测试数据
        self.create_test_data()

    def create_test_data(self):
        """创建测试用的微调数据"""
        test_data = [
            {
                "instruction": "解释Python中for循环和while循环的区别",
                "input": "",
                "output": "在Python中，for循环和while循环是两种不同的循环结构。for循环用于遍历一个已知的序列，它在每次迭代时会自动获取序列中的下一个元素，直到遍历完所有元素。而while循环则是在指定条件为真时重复执行代码块，需要手动更新循环变量，适合在不确定迭代次数的情况下使用。"
            },
            {
                "instruction": "什么是多模态交互",
                "input": "",
                "output": "多模态交互是指系统能够通过多种感知通道（如视觉、听觉、触觉等）与用户进行交流的能力。在PEPPER机器人的教学应用中，多模态交互使机器人能够通过摄像头识别学生的表情和动作，通过麦克风接收语音指令，并通过语音合成和肢体动作做出回应，提供更自然、更丰富的教学体验。"
            }
        ]

        with open(self.test_data_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        logger.info(f"已创建测试数据: {self.test_data_path}")

    @unittest.skip("模型微调测试耗时较长，默认跳过")
    def test_fine_tuning(self):
        """测试LoRA微调过程"""
        # 创建LoRA微调器实例
        fine_tuner = LoRAFineTuner(
            base_model_path="models/deepseek-coder-1.3b-base",
            output_dir=self.output_dir
        )

        # 加载基础模型（使用小型模型以加快测试）
        self.assertTrue(fine_tuner.load_base_model())

        # 准备LoRA配置
        peft_model = fine_tuner.prepare_lora_config()
        self.assertIsNotNone(peft_model)

        # 准备训练数据集
        dataset = fine_tuner.prepare_dataset(self.test_data_path)
        self.assertIsNotNone(dataset)

        # 训练模型（设置较少的轮次以加快测试）
        fine_tuner.train(dataset, epochs=1, batch_size=2)

        # 测试生成
        test_prompts = [
            "Python中for循环和while循环有什么区别？",
            "什么是多模态交互？"
        ]

        for prompt in test_prompts:
            response = fine_tuner.generate_response(prompt)
            self.assertIsNotNone(response)
            self.assertGreater(len(response), 50)  # 确保回答有足够长度
            logger.info(f"测试提示: {prompt}")
            logger.info(f"生成回答: {response[:100]}...")

        logger.info("LoRA微调测试完成")


class TestKnowledgeGraph(unittest.TestCase):
    """测试知识图谱构建与查询功能"""

    def setUp(self):
        """初始化测试环境"""
        self.edu_processor = EducationKnowledgeProcessor()
        self.test_output_dir = 'tests/test_output/knowledge_graph'
        os.makedirs(self.test_output_dir, exist_ok=True)

    def test_knowledge_extraction(self):
        """测试知识提取功能"""
        # 测试文本
        test_text = """
        Python是一种高级编程语言，它的设计强调代码的可读性。
        Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。
        在Python中，for循环用于遍历序列，而while循环基于条件表达式。
        """

        # 提取知识点
        knowledge_items = self.edu_processor.extract_knowledge_from_text(test_text, "Python")

        # 验证结果
        self.assertIsNotNone(knowledge_items)
        logger.info(f"从文本中提取了{len(knowledge_items)}个知识点")

        # 保存知识点到测试输出
        output_path = os.path.join(self.test_output_dir, 'knowledge_items.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_items, f, ensure_ascii=False, indent=2)

    def test_curriculum_knowledge(self):
        """测试课程知识生成功能"""
        # 测试几个不同学科
        subjects = ["Python", "数学", "人工智能"]

        for subject in subjects:
            # 生成学科知识点
            knowledge_items = self.edu_processor.generate_curriculum_knowledge(subject, "")

            # 验证结果
            self.assertIsNotNone(knowledge_items)
            self.assertGreater(len(knowledge_items), 0)
            logger.info(f"为学科'{subject}'生成了{len(knowledge_items)}个知识点")

            # 验证知识点结构
            for item in knowledge_items:
                self.assertIn("subject", item)
                self.assertIn("predicate", item)
                self.assertIn("object", item)

    def test_knowledge_graph_build(self):
        """测试知识图谱构建功能"""
        # 生成测试知识
        knowledge_items = self.edu_processor.generate_curriculum_knowledge("Python", "")

        # 构建知识图谱
        added_count = self.edu_processor.build_knowledge_graph(knowledge_items)

        # 验证结果
        self.assertGreater(added_count, 0)
        logger.info(f"向知识图谱添加了{added_count}个知识点")

        # 查询知识点
        results = self.edu_processor.knowledge_graph.find_related_knowledge("Python")

        # 验证查询结果
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        logger.info(f"查询到{len(results)}个与'Python'相关的知识点")


class TestPersonalizedTeaching(unittest.TestCase):
    """测试个性化教学功能"""

    def setUp(self):
        """初始化测试环境"""
        self.teaching = PersonalizedTeaching()
        self.test_output_dir = 'tests/test_output/personalized_teaching'
        os.makedirs(self.test_output_dir, exist_ok=True)

        # 创建测试学生档案
        self.student_ids = self.teaching.create_demo_student_profiles()

    def test_student_profile(self):
        """测试学生档案创建和更新"""
        # 验证学生档案创建成功
        self.assertGreater(len(self.student_ids), 0)

        # 获取学生档案
        student_id = self.student_ids[0]
        profile = self.teaching.get_student_profile(student_id)

        # 验证档案内容
        self.assertIsNotNone(profile)
        self.assertEqual(profile.student_id, student_id)
        self.assertGreater(len(profile.learning_history), 0)

        # 添加学习记录
        profile.add_learning_record("API接口", "practice", 85)

        # 验证记录已添加
        topics = [record["topic"] for record in profile.learning_history]
        self.assertIn("API接口", topics)

        # 导出档案到JSON
        output_path = os.path.join(self.test_output_dir, f'student_{student_id}.json')
        profile.export_to_json(output_path)
        self.assertTrue(os.path.exists(output_path))

    def test_learning_resource_recommendation(self):
        """测试学习资源推荐功能"""
        student_id = self.student_ids[0]

        # 测试默认推荐
        resources = self.teaching.recommend_learning_resources(student_id)
        self.assertIsNotNone(resources)
        self.assertGreater(len(resources), 0)
        logger.info(f"为学生{student_id}推荐了{len(resources)}个默认资源")

        # 测试特定主题推荐
        topic = "Python编程"
        resources = self.teaching.recommend_learning_resources(student_id, topic)
        self.assertIsNotNone(resources)
        self.assertGreater(len(resources), 0)
        logger.info(f"为学生{student_id}推荐了{len(resources)}个'{topic}'相关资源")

    def test_learning_path_generation(self):
        """测试学习路径生成功能"""
        student_id = self.student_ids[0]
        goal_topic = "Python函数"

        # 生成学习路径
        learning_path = self.teaching.generate_learning_path(student_id, goal_topic)

        # 验证结果
        self.assertIsNotNone(learning_path)
        self.assertIn("learning_path", learning_path)
        self.assertGreater(len(learning_path["learning_path"]), 100)

        # 保存学习路径
        output_path = os.path.join(self.test_output_dir, 'learning_path.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(learning_path, f, ensure_ascii=False, indent=2)

        logger.info(f"已生成学习路径，长度: {len(learning_path['learning_path'])}")

    def test_personalized_answer(self):
        """测试个性化回答生成功能"""
        student_id = self.student_ids[0]
        question = "Python中如何定义和使用函数？"

        # 生成个性化回答
        answer = self.teaching.generate_personalized_answer(student_id, question)

        # 验证结果
        self.assertIsNotNone(answer)
        self.assertGreater(len(answer), 100)

        # 记录学习交互
        self.teaching.add_learning_interaction(student_id, "函数", question)

        # 验证交互记录是否更新
        profile = self.teaching.get_student_profile(student_id)
        topics = [record["topic"] for record in profile.learning_history]
        self.assertIn("函数", topics)

        logger.info(f"已生成个性化回答，长度: {len(answer)}")
        logger.info(f"回答前100个字符: {answer[:100]}...")


class TestMultimodalInteraction(unittest.TestCase):
    """测试多模态交互功能"""

    def setUp(self):
        """初始化测试环境"""
        self.text_processor = TextProcessor()
        self.test_output_dir = 'tests/test_output/multimodal'
        os.makedirs(self.test_output_dir, exist_ok=True)

    def test_text_processing(self):
        """测试文本处理功能"""
        # 测试文本
        test_text = "PEPPER机器人能够理解自然语言并提供个性化的教学辅助。"

        # 测试分词
        words = self.text_processor.preprocess_text(test_text)
        self.assertIsNotNone(words)
        self.assertGreater(len(words), 0)
        logger.info(f"分词结果: {words}")

        # 测试关键词提取
        keywords = self.text_processor.extract_keywords(test_text)
        self.assertIsNotNone(keywords)
        self.assertGreater(len(keywords), 0)
        logger.info(f"关键词: {keywords}")

        # 测试问题分类
        question = "PEPPER机器人在教学中有什么应用？"
        question_type = self.text_processor.classify_question(question)
        self.assertIsNotNone(question_type)
        logger.info(f"问题类型: {question_type}")

    @unittest.skip("语音识别需要实际环境，默认跳过")
    def test_speech_recognition(self):
        """测试语音识别功能"""
        recognizer = SpeechRecognizer()

        # 测试文件识别（需要准备测试音频文件）
        test_audio = 'tests/test_data/test_audio.wav'
        if os.path.exists(test_audio):
            text = recognizer.recognize_from_file(test_audio)
            self.assertIsNotNone(text)
            logger.info(f"语音识别结果: {text}")

    @unittest.skip("图像识别需要实际环境，默认跳过")
    def test_image_recognition(self):
        """测试图像识别功能"""
        recognizer = ImageRecognizer()

        # 测试图像识别（需要准备测试图像文件）
        test_image = 'tests/test_data/test_image.jpg'
        if os.path.exists(test_image):
            result = recognizer.recognize_image(test_image)
            self.assertIsNotNone(result)
            logger.info(f"图像识别结果: {result}")


if __name__ == '__main__':
    # 创建测试套件
    suite = unittest.TestSuite()

    # 添加测试用例
    # suite.addTest(unittest.makeSuite(TestLoRAFineTuning))  # 默认跳过，耗时较长
    suite.addTest(unittest.makeSuite(TestKnowledgeGraph))
    suite.addTest(unittest.makeSuite(TestPersonalizedTeaching))
    suite.addTest(unittest.makeSuite(TestMultimodalInteraction))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
