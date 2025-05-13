"""
知识图谱初始化脚本 - 独立版本
不依赖于其他可能有问题的模块
"""
import json
import logging
import os

from neo4j import GraphDatabase

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('kg_init.log')
    ]
)
logger = logging.getLogger("KG_INIT")


def load_config():
    """加载配置"""
    try:
        if os.path.exists("config.json"):
            with open("config.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning("未找到配置文件，使用默认配置")
            return {
                "knowledge_graph": {
                    "uri": "bolt://localhost:7687",
                    "user": "neo4j",
                    "password": "adminadmin"
                }
            }
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        return None


class KnowledgeGraph:
    """知识图谱操作类"""

    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        """初始化知识图谱连接"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("知识图谱连接已创建")

    def close(self):
        """关闭连接"""
        self.driver.close()

    def create_node(self, label, properties):
        """创建节点"""
        with self.driver.session() as session:
            result = session.write_transaction(self._create_node, label, properties)
            return result

    @staticmethod
    def _create_node(tx, label, properties):
        query = f"CREATE (n:{label} $properties) RETURN n"
        result = tx.run(query, properties=properties)
        return result.single()[0]

    def create_relationship(self, start_node_label, start_node_props,
                            end_node_label, end_node_props,
                            relationship_type, relationship_props=None):
        """创建关系"""
        if relationship_props is None:
            relationship_props = {}

        with self.driver.session() as session:
            # 确保起始节点存在
            session.write_transaction(
                self._merge_node, start_node_label, start_node_props
            )

            # 确保终止节点存在
            session.write_transaction(
                self._merge_node, end_node_label, end_node_props
            )

            # 创建关系
            result = session.write_transaction(
                self._create_relationship,
                start_node_label, start_node_props,
                end_node_label, end_node_props,
                relationship_type, relationship_props
            )
            return result

    @staticmethod
    def _merge_node(tx, label, properties):
        """确保节点存在（如果不存在则创建）"""
        name = properties.get("name", "")
        query = f"MERGE (n:{label} {{name: $name}}) "
        if properties:
            query += "SET n = $properties "
        query += "RETURN n"
        result = tx.run(query, name=name, properties=properties)
        return result.single()[0]

    @staticmethod
    def _create_relationship(tx, start_node_label, start_node_props,
                             end_node_label, end_node_props,
                             relationship_type, relationship_props):
        """创建关系"""
        query = (
            f"MATCH (a:{start_node_label}), (b:{end_node_label}) "
            f"WHERE a.name = $start_name AND b.name = $end_name "
            f"CREATE (a)-[r:{relationship_type} $relationship_props]->(b) "
            f"RETURN a, r, b"
        )
        result = tx.run(
            query,
            start_name=start_node_props.get("name"),
            end_name=end_node_props.get("name"),
            relationship_props=relationship_props
        )
        return result.single()

    def query(self, cypher_query, parameters=None):
        """执行Cypher查询"""
        if parameters is None:
            parameters = {}

        with self.driver.session() as session:
            result = session.run(cypher_query, parameters)
            return [record for record in result]


def generate_curriculum_knowledge(subject):
    """生成特定学科的课程知识点"""
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

    # 如果是其他主题，返回"人工智能+教育"知识点
    if not knowledge_items:
        return ai_edu_knowledge

    # 否则合并特定主题和"人工智能+教育"知识点
    knowledge_items.extend(ai_edu_knowledge)
    return knowledge_items


def build_knowledge_graph(kg, knowledge_items):
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

                # 创建宾语节点
                object_properties = {
                    "name": item["object"],
                    "description": item.get("object_description", ""),
                    "domain": item.get("domain", "general")
                }

                # 创建关系
                relationship_type = normalize_relationship(item["predicate"])
                relationship_props = {
                    "source": item.get("source_sentence", ""),
                    "confidence": item.get("confidence", 1.0)
                }

                kg.create_relationship(
                    "Concept", subject_properties,
                    "Concept", object_properties,
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


def normalize_relationship(predicate):
    """将谓语标准化为图数据库关系类型"""
    import re
    # 移除特殊字符并转换为大写下划线格式
    normalized = re.sub(r'[^\w\s]', '', predicate)
    normalized = re.sub(r'\s+', '_', normalized.upper())

    # 确保关系类型有效
    if not normalized:
        normalized = "RELATED_TO"

    return normalized


def create_educational_knowledge_base():
    """创建基础教育知识库"""
    # 为多个学科创建知识图谱
    all_knowledge = []

    # 添加Python编程知识
    python_knowledge = generate_curriculum_knowledge("python")
    all_knowledge.extend(python_knowledge)

    # 添加数学知识
    math_knowledge = generate_curriculum_knowledge("数学")
    all_knowledge.extend(math_knowledge)

    # 添加人工智能知识
    ai_knowledge = generate_curriculum_knowledge("人工智能")
    all_knowledge.extend(ai_knowledge)

    return all_knowledge


def init_knowledge_graph():
    """初始化知识图谱"""
    config = load_config()
    if not config:
        return False

    kg_config = config.get("knowledge_graph", {})

    try:
        # 创建知识图谱连接
        kg = KnowledgeGraph(
            kg_config.get("uri", "bolt://localhost:7687"),
            kg_config.get("user", "neo4j"),
            kg_config.get("password", "adminadmin")
        )

        # 生成教育知识
        knowledge_items = create_educational_knowledge_base()

        # 构建知识图谱
        added_count = build_knowledge_graph(kg, knowledge_items)

        # 关闭连接
        kg.close()

        return added_count > 0
    except Exception as e:
        logger.error(f"知识图谱初始化出错: {e}")
        return False


if __name__ == "__main__":
    logger.info("开始初始化知识图谱...")
    success = init_knowledge_graph()
    logger.info(f"知识图谱初始化{'成功' if success else '失败'}")
