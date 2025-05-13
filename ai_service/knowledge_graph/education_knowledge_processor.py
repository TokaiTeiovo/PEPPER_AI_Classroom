"""
教育知识数据处理模块 - 用于PEPPER机器人教学系统知识图谱的构建和更新

该模块实现了从教育资源中提取知识点并构建知识图谱的功能
支持从文本文件、结构化数据和在线资源获取教育领域知识
"""

import json
import logging
import re

import pandas as pd
import requests
import spacy
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

from ai_service.knowledge_graph.knowledge_graph import KnowledgeGraph

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EDU_KNOWLEDGE_PROCESSOR")

# 加载NLP模型
try:
    nlp = spacy.load("zh_core_web_sm")
    logger.info("中文NLP模型加载成功")
except OSError:
    logger.warning("中文NLP模型未找到，尝试下载...")
    try:
        from spacy.cli import download

        download("zh_core_web_sm")
        nlp = spacy.load("zh_core_web_sm")
        logger.info("中文NLP模型下载和加载成功")
    except Exception as e:
        logger.error(f"无法加载中文NLP模型: {e}")
        nlp = None


class EducationKnowledgeProcessor:
    def __init__(self, kg_uri="bolt://localhost:7687", kg_user="neo4j", kg_password="password"):
        """初始化教育知识处理器"""
        self.knowledge_graph = KnowledgeGraph(kg_uri, kg_user, kg_password)
        logger.info("教育知识处理器初始化完成")

    def extract_knowledge_from_text(self, text, subject="general"):
        """从文本中提取知识点"""
        if not nlp:
            logger.error("NLP模型未加载，无法进行知识提取")
            return []

        logger.info(f"从文本中提取知识点，主题: {subject}")
        knowledge_items = []

        # 将文本分割成句子
        sentences = sent_tokenize(text)

        for sentence in sentences:
            # 使用spaCy进行NLP分析
            doc = nlp(sentence)

            # 提取实体
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            # 提取主语-谓语-宾语关系
            for token in doc:
                if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                    subject_entity = token.text
                    verb = token.head.text

                    # 尝试找到直接宾语
                    object_entity = None
                    for child in token.head.children:
                        if child.dep_ in ("dobj", "pobj"):
                            object_entity = child.text
                            break

                    if object_entity:
                        knowledge_items.append({
                            "subject": subject_entity,
                            "predicate": verb,
                            "object": object_entity,
                            "source_sentence": sentence,
                            "domain": subject
                        })

        logger.info(f"提取了{len(knowledge_items)}个知识点")
        return knowledge_items

    def extract_from_file(self, file_path, subject="general"):
        """从文件中提取知识点"""
        try:
            logger.info(f"从文件提取知识: {file_path}")

            # 根据文件类型读取内容
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return self.extract_knowledge_from_text(text, subject)

            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    # 假设是知识点列表
                    return data
                elif isinstance(data, dict) and "text" in data:
                    # 假设是带有文本字段的对象
                    return self.extract_knowledge_from_text(data["text"], subject)
                else:
                    logger.error("不支持的JSON格式")
                    return []

            elif file_path.endswith(('.csv', '.xlsx')):
                df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

                # 检查是否已经是结构化的知识点
                if all(col in df.columns for col in ["subject", "predicate", "object"]):
                    return df.to_dict('records')
                else:
                    # 尝试从内容列提取知识
                    content_col = None
                    for col in ["content", "text", "description"]:
                        if col in df.columns:
                            content_col = col
                            break

                    if content_col:
                        all_knowledge = []
                        for _, row in df.iterrows():
                            knowledge = self.extract_knowledge_from_text(row[content_col], subject)
                            all_knowledge.extend(knowledge)
                        return all_knowledge
                    else:
                        logger.error("未找到内容列")
                        return []
            else:
                logger.error(f"不支持的文件类型: {file_path}")
                return []

        except Exception as e:
            logger.error(f"从文件提取知识失败: {e}")
            return []

    def extract_from_url(self, url, subject="general"):
        """从网页URL提取知识点"""
        try:
            logger.info(f"从URL提取知识: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # 解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # 提取正文内容
            # 移除脚本和样式元素
            for script in soup(["script", "style"]):
                script.extract()

            # 获取文本
            text = soup.get_text()

            # 去除多余空白
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return self.extract_knowledge_from_text(text, subject)

        except Exception as e:
            logger.error(f"从URL提取知识失败: {e}")
            return []

    def build_knowledge_graph(self, knowledge_items):
        """将知识点添加到知识图谱"""
        try:
            logger.info(f"向知识图谱添加{len(knowledge_items)}个知识点")

            added_count = 0
            for item in knowledge_items:
                try:
                    # 创建主语节点
                    subject_properties = {
                        "name": item["subject"],
                        "description": item.get("subject_description", ""),
                        "domain": item.get("domain", "general")
                    }
                    self.knowledge_graph.create_node("Concept", subject_properties)

                    # 创建宾语节点
                    object_properties = {
                        "name": item["object"],
                        "description": item.get("object_description", ""),
                        "domain": item.get("domain", "general")
                    }
                    self.knowledge_graph.create_node("Concept", object_properties)

                    # 创建关系
                    relationship_type = self._normalize_relationship(item["predicate"])
                    relationship_props = {
                        "source": item.get("source_sentence", ""),
                        "confidence": item.get("confidence", 1.0)
                    }

                    self.knowledge_graph.create_relationship(
                        "Concept", {"name": item["subject"]},
                        "Concept", {"name": item["object"]},
                        relationship_type, relationship_props
                    )

                    added_count += 1

                except Exception as e:
                    logger.warning(f"添加知识点时出错: {e}")
                    continue

            logger.info(f"成功添加{added_count}个知识点到知识图谱")
            return added_count

        except Exception as e:
            logger.error(f"构建知识图谱失败: {e}")
            return 0

    def _normalize_relationship(self, predicate):
        """将谓语标准化为图数据库关系类型"""
        # 移除特殊字符并转换为大写下划线格式
        normalized = re.sub(r'[^\w\s]', '', predicate)
        normalized = re.sub(r'\s+', '_', normalized.upper())

        # 确保关系类型有效
        if not normalized:
            normalized = "RELATED_TO"

        return normalized

    def generate_curriculum_knowledge(self, subject, grade_level):
        """生成特定学科和年级的课程知识点"""
        # 这里实现一个示例，为不同学科生成基础知识点
        knowledge_items = []

        if subject.lower() == "python":
            # Python编程基础知识点
            knowledge_items = [
                {
                    "subject": "变量",
                    "predicate": "是",
                    "object": "存储数据的容器",
                    "subject_description": "在程序中用于存储值的命名引用",
                    "object_description": "可以存储不同类型数据的内存位置",
                    "domain": "Python编程"
                },
                {
                    "subject": "for循环",
                    "predicate": "用于",
                    "object": "遍历序列",
                    "subject_description": "Python中用于迭代序列的循环结构",
                    "object_description": "按顺序访问序列中的每个元素",
                    "domain": "Python编程"
                },
                {
                    "subject": "while循环",
                    "predicate": "基于",
                    "object": "条件表达式",
                    "subject_description": "在条件为真时重复执行的循环结构",
                    "object_description": "结果为布尔值的表达式",
                    "domain": "Python编程"
                },
                {
                    "subject": "函数",
                    "predicate": "是",
                    "object": "可重用代码块",
                    "subject_description": "执行特定任务的代码块",
                    "object_description": "可以在程序中多次调用的代码段",
                    "domain": "Python编程"
                },
                {
                    "subject": "列表",
                    "predicate": "是",
                    "object": "有序集合",
                    "subject_description": "Python中的可变序列类型",
                    "object_description": "元素按顺序排列且可以修改的集合",
                    "domain": "Python编程"
                }
            ]

        elif subject.lower() == "数学":
            # 数学基础知识点
            knowledge_items = [
                {
                    "subject": "函数",
                    "predicate": "是",
                    "object": "输入输出对应关系",
                    "subject_description": "在数学中，函数是输入值与输出值之间的对应关系",
                    "object_description": "每个输入值对应唯一的输出值",
                    "domain": "数学"
                },
                {
                    "subject": "导数",
                    "predicate": "表示",
                    "object": "函数变化率",
                    "subject_description": "函数在某一点的瞬时变化率",
                    "object_description": "函数输出值相对于输入值的变化比率",
                    "domain": "数学"
                },
                {
                    "subject": "积分",
                    "predicate": "计算",
                    "object": "曲线下面积",
                    "subject_description": "求函数曲线与坐标轴围成区域的面积",
                    "object_description": "函数图像下方与坐标轴之间的区域面积",
                    "domain": "数学"
                }
            ]

        elif subject.lower() in ["人工智能", "ai"]:
            # 人工智能基础知识点
            knowledge_items = [
                {
                    "subject": "机器学习",
                    "predicate": "是",
                    "object": "人工智能子领域",
                    "subject_description": "通过数据学习和预测的计算机算法",
                    "object_description": "人工智能的一个主要研究方向",
                    "domain": "人工智能"
                },
                {
                    "subject": "深度学习",
                    "predicate": "基于",
                    "object": "神经网络",
                    "subject_description": "使用多层神经网络的机器学习方法",
                    "object_description": "模拟人脑结构的计算模型",
                    "domain": "人工智能"
                },
                {
                    "subject": "自然语言处理",
                    "predicate": "研究",
                    "object": "人机语言交互",
                    "subject_description": "研究计算机理解和生成人类语言的技术",
                    "object_description": "人类与计算机之间的语言沟通",
                    "domain": "人工智能"
                },
                {
                    "subject": "计算机视觉",
                    "predicate": "处理",
                    "object": "图像数据",
                    "subject_description": "使计算机能理解和处理视觉信息的技术",
                    "object_description": "从现实世界获取的图像和视频数据",
                    "domain": "人工智能"
                }
            ]

        return knowledge_items

    def create_educational_knowledge_base(self):
        """创建基础教育知识库"""
        # 为多个学科创建知识图谱
        all_knowledge = []

        # 添加Python编程知识
        python_knowledge = self.generate_curriculum_knowledge("python", "")
        all_knowledge.extend(python_knowledge)

        # 添加数学知识
        math_knowledge = self.generate_curriculum_knowledge("数学", "")
        all_knowledge.extend(math_knowledge)

        # 添加人工智能知识
        ai_knowledge = self.generate_curriculum_knowledge("人工智能", "")
        all_knowledge.extend(ai_knowledge)

        # 添加"人工智能+教育"领域特定知识
        ai_edu_knowledge = [
            {
                "subject": "PEPPER机器人",
                "predicate": "应用于",
                "object": "智能教学",
                "subject_description": "具有人机交互功能的智能机器人",
                "object_description": "利用AI技术提供个性化学习体验的教学方式",
                "domain": "人工智能+教育"
            },
            {
                "subject": "多模态交互",
                "predicate": "提升",
                "object": "学习体验",
                "subject_description": "通过视觉、听觉、触觉等多种方式进行交互",
                "object_description": "学生在学习过程中的感受和参与度",
                "domain": "人工智能+教育"
            },
            {
                "subject": "知识图谱",
                "predicate": "支持",
                "object": "个性化学习",
                "subject_description": "知识点及其关系的网络结构",
                "object_description": "根据学生特点定制的学习内容和路径",
                "domain": "人工智能+教育"
            }
        ]
        all_knowledge.extend(ai_edu_knowledge)

        # 构建知识图谱
        added_count = self.build_knowledge_graph(all_knowledge)

        logger.info(f"教育知识库创建完成，共添加{added_count}个知识点")
        return added_count

    def enrich_knowledge_graph(self, keyword, depth=2):
        """通过网络搜索丰富特定关键词相关的知识图谱"""
        logger.info(f"开始丰富关键词'{keyword}'相关的知识图谱")

        try:
            # 构建搜索查询
            search_query = f"{keyword} 教育 知识点"

            # 模拟搜索结果（实际应用中可以使用搜索API）
            urls = [
                f"https://example.com/education/{keyword}",
                f"https://example.org/learn/{keyword}"
            ]

            all_knowledge = []
            for url in urls:
                try:
                    # 从URL提取知识
                    knowledge = self.extract_from_url(url, subject=keyword)
                    all_knowledge.extend(knowledge)
                except Exception as e:
                    logger.warning(f"从URL提取知识失败: {e}")
                    continue

            # 添加到知识图谱
            if all_knowledge:
                added_count = self.build_knowledge_graph(all_knowledge)
                logger.info(f"已从网络搜索结果添加{added_count}个关于'{keyword}'的知识点")
                return added_count
            else:
                logger.warning(f"未找到关于'{keyword}'的额外知识")
                return 0

        except Exception as e:
            logger.error(f"丰富知识图谱失败: {e}")
            return 0

    def create_test_knowledge_file(self, output_path="test_knowledge.json"):
        """创建测试用的知识数据文件"""
        # 生成测试知识数据
        test_knowledge = []

        # 添加Python测试知识
        test_knowledge.extend(self.generate_curriculum_knowledge("python", ""))

        # 添加AI测试知识
        test_knowledge.extend(self.generate_curriculum_knowledge("人工智能", ""))

        # 写入JSON文件
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(test_knowledge, f, ensure_ascii=False, indent=2)
            logger.info(f"测试知识数据已保存到: {output_path}")
            return True
        except Exception as e:
            logger.error(f"创建测试知识文件失败: {e}")
            return False


# 当脚本直接运行时，执行示例过程
if __name__ == "__main__":
    # 创建教育知识处理器实例
    processor = EducationKnowledgeProcessor()

    # 创建测试知识文件
    processor.create_test_knowledge_file()

    # 可选：创建基础教育知识库
    # processor.create_educational_knowledge_base()

    print("教育知识数据处理模块测试完成")
