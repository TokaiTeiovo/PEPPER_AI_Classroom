"""
个性化教学模块 - 用于PEPPER机器人教学系统提供个性化学习建议

该模块基于学生的历史学习数据，结合知识图谱和大语言模型，
为学生提供个性化学习建议，如学习资料推荐、学习路径规划等
"""

import json
import logging
import os
from datetime import datetime

from ai_service.knowledge_graph.knowledge_graph import KnowledgeGraph
from ai_service.llm_module.llm_interface import LLMService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PERSONALIZED_TEACHING")


class StudentProfile:
    """学生档案类，存储和管理学生的学习数据"""

    def __init__(self, student_id, name=""):
        self.student_id = student_id
        self.name = name
        self.learning_history = []  # 学习记录列表
        self.topic_preferences = {}  # 主题偏好
        self.strengths = {}  # 擅长的主题
        self.weaknesses = {}  # 薄弱的主题
        self.learning_style = "visual"  # 默认为视觉学习风格
        self.last_updated = datetime.now()
        logger.info(f"创建学生档案: {student_id}")

    def add_learning_record(self, topic, activity_type, performance, timestamp=None):
        """添加学习记录"""
        if timestamp is None:
            timestamp = datetime.now()

        record = {
            "timestamp": timestamp,
            "topic": topic,
            "activity_type": activity_type,  # quiz, reading, practice, etc.
            "performance": performance,  # 0-100 or descriptive
        }

        self.learning_history.append(record)
        self._update_topic_stats(topic, performance)
        self.last_updated = datetime.now()
        logger.info(f"为学生 {self.student_id} 添加学习记录: {topic}")

    def _update_topic_stats(self, topic, performance):
        """更新主题统计信息"""
        # 更新主题偏好 (基于互动频率)
        if topic in self.topic_preferences:
            self.topic_preferences[topic] += 1
        else:
            self.topic_preferences[topic] = 1

        # 更新强弱项 (基于表现)
        if isinstance(performance, (int, float)):
            if topic in self.strengths:
                self.strengths[topic] = (self.strengths[topic] * 0.7) + (performance * 0.3)
            else:
                self.strengths[topic] = performance

            # 设置阈值，低于70分为薄弱项
            if performance < 70:
                if topic in self.weaknesses:
                    self.weaknesses[topic] = (self.weaknesses[topic] * 0.7) + ((100 - performance) * 0.3)
                else:
                    self.weaknesses[topic] = 100 - performance
            elif topic in self.weaknesses:
                # 如果表现良好，减少薄弱度
                self.weaknesses[topic] = max(0, self.weaknesses[topic] * 0.8)
                if self.weaknesses[topic] < 10:  # 阈值以下则移除
                    del self.weaknesses[topic]

    def set_learning_style(self, style):
        """设置学习风格"""
        valid_styles = ["visual", "auditory", "reading", "kinesthetic"]
        if style.lower() in valid_styles:
            self.learning_style = style.lower()
            logger.info(f"更新学生 {self.student_id} 的学习风格: {style}")
            return True
        else:
            logger.warning(f"无效的学习风格: {style}")
            return False

    def get_top_preferences(self, limit=5):
        """获取学生最感兴趣的主题"""
        sorted_prefs = sorted(self.topic_preferences.items(), key=lambda x: x[1], reverse=True)
        return sorted_prefs[:limit]

    def get_top_strengths(self, limit=5):
        """获取学生最擅长的主题"""
        sorted_strengths = sorted(self.strengths.items(), key=lambda x: x[1], reverse=True)
        return sorted_strengths[:limit]

    def get_top_weaknesses(self, limit=5):
        """获取学生最薄弱的主题"""
        sorted_weaknesses = sorted(self.weaknesses.items(), key=lambda x: x[1], reverse=True)
        return sorted_weaknesses[:limit]

    def export_to_json(self, file_path=None):
        """导出学生档案到JSON文件"""
        profile_data = {
            "student_id": self.student_id,
            "name": self.name,
            "learning_history": self.learning_history,
            "topic_preferences": self.topic_preferences,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "learning_style": self.learning_style,
            "last_updated": self.last_updated.isoformat()
        }

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(profile_data, f, ensure_ascii=False, indent=2, default=str)
                logger.info(f"学生档案已导出到: {file_path}")
            except Exception as e:
                logger.error(f"导出学生档案失败: {e}")

        return profile_data

    @classmethod
    def load_from_json(cls, file_path):
        """从JSON文件加载学生档案"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            profile = cls(data["student_id"], data["name"])
            profile.learning_history = data["learning_history"]
            profile.topic_preferences = data["topic_preferences"]
            profile.strengths = data["strengths"]
            profile.weaknesses = data["weaknesses"]
            profile.learning_style = data["learning_style"]
            profile.last_updated = datetime.fromisoformat(data["last_updated"])

            logger.info(f"从文件加载学生档案: {file_path}")
            return profile

        except Exception as e:
            logger.error(f"加载学生档案失败: {e}")
            return None


class PersonalizedTeaching:
    """个性化教学模块，基于学生档案提供个性化学习建议"""

    def __init__(self, kg_uri="bolt://localhost:7687", kg_user="neo4j", kg_password="password"):
        """初始化个性化教学模块"""
        self.knowledge_graph = KnowledgeGraph(kg_uri, kg_user, kg_password)
        self.llm_service = LLMService()
        self.student_profiles = {}  # 学生档案字典，学生ID为键
        self.learning_resources = self._load_learning_resources()
        logger.info("个性化教学模块初始化完成")

    def _load_learning_resources(self):
        """加载学习资源"""
        # 这里可以从数据库或文件加载实际资源
        # 示例数据
        resources = {
            "Python编程": [
                {"type": "video", "title": "Python基础入门", "url": "https://example.com/python-basics",
                 "level": "beginner"},
                {"type": "tutorial", "title": "Python循环详解", "url": "https://example.com/python-loops",
                 "level": "intermediate"},
                {"type": "exercise", "title": "Python函数练习", "url": "https://example.com/python-functions",
                 "level": "intermediate"},
                {"type": "project", "title": "构建Python计算器", "url": "https://example.com/python-calculator",
                 "level": "advanced"},
            ],
            "人工智能": [
                {"type": "article", "title": "人工智能简介", "url": "https://example.com/ai-intro",
                 "level": "beginner"},
                {"type": "video", "title": "机器学习基础", "url": "https://example.com/machine-learning",
                 "level": "intermediate"},
                {"type": "tutorial", "title": "神经网络实践", "url": "https://example.com/neural-networks",
                 "level": "advanced"},
                {"type": "exercise", "title": "AI模型评估", "url": "https://example.com/ai-evaluation",
                 "level": "advanced"},
            ],
            "数学": [
                {"type": "article", "title": "微积分入门", "url": "https://example.com/calculus-intro",
                 "level": "beginner"},
                {"type": "video", "title": "函数与图像", "url": "https://example.com/functions-graphs",
                 "level": "intermediate"},
                {"type": "tutorial", "title": "微分方程解法", "url": "https://example.com/differential-equations",
                 "level": "advanced"},
                {"type": "exercise", "title": "数学建模练习", "url": "https://example.com/math-modeling",
                 "level": "advanced"},
            ]
        }
        return resources

    def add_student_profile(self, student_id, name=""):
        """添加学生档案"""
        if student_id not in self.student_profiles:
            self.student_profiles[student_id] = StudentProfile(student_id, name)
            logger.info(f"添加学生档案: {student_id}")
            return True
        else:
            logger.warning(f"学生档案已存在: {student_id}")
            return False

    def get_student_profile(self, student_id):
        """获取学生档案"""
        if student_id in self.student_profiles:
            return self.student_profiles[student_id]
        else:
            logger.warning(f"学生档案不存在: {student_id}")
            return None

    def load_student_profiles(self, directory):
        """从目录中加载所有学生档案"""
        if not os.path.isdir(directory):
            logger.error(f"目录不存在: {directory}")
            return False

        loaded_count = 0
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                profile = StudentProfile.load_from_json(file_path)
                if profile:
                    self.student_profiles[profile.student_id] = profile
                    loaded_count += 1

        logger.info(f"已加载{loaded_count}个学生档案")
        return loaded_count > 0

    def save_student_profiles(self, directory):
        """保存所有学生档案到目录"""
        if not os.path.exists(directory):
            os.makedirs(directory)

        saved_count = 0
        for student_id, profile in self.student_profiles.items():
            file_path = os.path.join(directory, f"{student_id}.json")
            profile.export_to_json(file_path)
            saved_count += 1

        logger.info(f"已保存{saved_count}个学生档案")
        return saved_count > 0

    def recommend_learning_resources(self, student_id, topic=None, count=3):
        """推荐学习资源"""
        profile = self.get_student_profile(student_id)
        if not profile:
            logger.error(f"无法找到学生档案: {student_id}")
            return []

        # 如果没有指定主题，则推荐学生薄弱项相关的资源
        if not topic:
            weaknesses = profile.get_top_weaknesses(3)
            if weaknesses:
                topic = weaknesses[0][0]  # 选取最弱的主题
            else:
                # 如果没有明显的弱项，则选择学生最常交互的主题
                preferences = profile.get_top_preferences(3)
                if preferences:
                    topic = preferences[0][0]
                else:
                    # 如果没有交互记录，默认使用一个通用主题
                    topic = "Python编程"

        # 根据学生的学习风格选择资源类型
        preferred_type = "video" if profile.learning_style == "visual" else \
            "article" if profile.learning_style == "reading" else \
                "tutorial"

        # 找出主题相关资源
        if topic in self.learning_resources:
            resources = self.learning_resources[topic]

            # 首先尝试匹配学习风格
            matching_resources = [r for r in resources if r["type"] == preferred_type]

            # 如果没有足够资源，添加其他类型
            if len(matching_resources) < count:
                other_resources = [r for r in resources if r["type"] != preferred_type]
                matching_resources.extend(other_resources)

            # 返回推荐数量的资源
            return matching_resources[:count]
        else:
            # 如果没有找到主题资源，尝试使用知识图谱寻找相关主题
            related_topics = self._find_related_topics(topic)
            all_resources = []

            for related_topic in related_topics:
                if related_topic in self.learning_resources:
                    all_resources.extend(self.learning_resources[related_topic])

            # 优先选择匹配学习风格的资源
            matching_resources = [r for r in all_resources if r["type"] == preferred_type]

            # 如果没有足够资源，添加其他类型
            if len(matching_resources) < count:
                other_resources = [r for r in all_resources if r["type"] != preferred_type]
                matching_resources.extend(other_resources)

            return matching_resources[:count]

    def _find_related_topics(self, topic):
        """通过知识图谱查找相关主题"""
        try:
            # 查询与主题相关的知识点
            results = self.knowledge_graph.find_related_knowledge(topic)

            # 提取相关主题
            related_topics = set()
            for item in results:
                if "start_node" in item and "name" in item["start_node"]:
                    related_topics.add(item["start_node"]["name"])
                if "end_node" in item and "name" in item["end_node"]:
                    related_topics.add(item["end_node"]["name"])

            # 去除原主题
            if topic in related_topics:
                related_topics.remove(topic)

            return list(related_topics)

        except Exception as e:
            logger.error(f"查找相关主题失败: {e}")
            return []

    def generate_learning_path(self, student_id, goal_topic):
        """生成个性化学习路径"""
        profile = self.get_student_profile(student_id)
        if not profile:
            logger.error(f"无法找到学生档案: {student_id}")
            return None

        # 尝试从知识图谱中查找相关知识点
        knowledge_items = self.knowledge_graph.find_related_knowledge(goal_topic)

        # 构建知识图谱上下文
        kg_context = self._build_kg_context(knowledge_items)

        # 构建学生上下文
        student_context = self._build_student_context(profile)

        # 使用LLM生成学习路径
        prompt = f"""作为一个教育AI助手，请为学生生成一个个性化的学习路径，帮助他们掌握"{goal_topic}"。

学生信息:
{student_context}

相关知识点:
{kg_context}

基于以上信息，请提供一个结构化的学习路径，包括:
1. 学习目标
2. 前置知识
3. 学习步骤（按顺序排列）
4. 每个步骤的推荐学习资源类型（考虑学生的学习风格）
5. 如何评估学习成果

请确保学习路径考虑学生的已有知识、强弱项和学习偏好。"""

        learning_path = self.llm_service.generate_response(prompt, max_length=1500)

        return {
            "student_id": student_id,
            "goal_topic": goal_topic,
            "learning_path": learning_path,
            "generated_at": datetime.now().isoformat()
        }

    def _build_kg_context(self, knowledge_items):
        """构建知识图谱上下文"""
        if not knowledge_items:
            return "没有找到相关知识点。"

        context = "相关知识概念和关系:\n"

        for idx, item in enumerate(knowledge_items, 1):
            if "start_node" in item and "end_node" in item:
                start_name = item["start_node"].get("name", "")
                relation = item.get("relationship", "")
                end_name = item["end_node"].get("name", "")

                start_desc = item["start_node"].get("description", "")
                end_desc = item["end_node"].get("description", "")

                context += f"{idx}. {start_name} ({start_desc}) -- {relation} --> {end_name} ({end_desc})\n"

        return context

    def _build_student_context(self, profile):
        """构建学生上下文"""
        context = f"学生ID: {profile.student_id}\n"
        context += f"学习风格: {profile.learning_style}\n"

        # 添加擅长项
        strengths = profile.get_top_strengths(3)
        if strengths:
            context += "擅长的主题:\n"
            for topic, score in strengths:
                context += f"- {topic}: {score:.1f}/100\n"

        # 添加弱项
        weaknesses = profile.get_top_weaknesses(3)
        if weaknesses:
            context += "需要加强的主题:\n"
            for topic, score in weaknesses:
                context += f"- {topic}: 弱项程度 {score:.1f}/100\n"

        # 添加偏好
        preferences = profile.get_top_preferences(3)
        if preferences:
            context += "感兴趣的主题:\n"
            for topic, count in preferences:
                context += f"- {topic}: 互动次数 {count}\n"

        return context

    def generate_personalized_answer(self, student_id, question):
        """生成针对学生的个性化回答"""
        profile = self.get_student_profile(student_id)
        if not profile:
            # 如果找不到学生档案，仍然可以回答，但不会个性化
            return self.llm_service.generate_response(question)

        # 查询知识图谱中与问题相关的知识
        # 提取问题中的关键词
        keywords = question.split()
        knowledge_items = []

        for keyword in keywords:
            if len(keyword) > 2:  # 忽略太短的词
                items = self.knowledge_graph.find_related_knowledge(keyword)
                knowledge_items.extend(items)

        # 去重
        unique_items = []
        unique_relations = set()
        for item in knowledge_items:
            if "start_node" in item and "end_node" in item and "relationship" in item:
                relation_key = f"{item['start_node'].get('name', '')}-{item['relationship']}-{item['end_node'].get('name', '')}"
                if relation_key not in unique_relations:
                    unique_relations.add(relation_key)
                    unique_items.append(item)

        # 使用知识图谱和学生信息生成个性化回答
        student_context = self._build_student_context(profile)
        kg_context = self._build_kg_context(unique_items)

        prompt = f"""作为一个教育AI助手，请针对学生的问题提供个性化的回答。

学生信息:
{student_context}

相关知识:
{kg_context}

学生问题: {question}

请提供一个针对这个学生的个性化回答，考虑他们的学习风格、强弱项和兴趣。解释应该清晰、准确，并适合学生的水平。如果回答涉及学生的弱项，请提供更详细的解释和例子。"""

        personalized_answer = self.llm_service.generate_response(prompt, max_length=1000)

        # 添加学习资源推荐
        # 尝试从问题中提取主题
        topic = None
        for keyword in keywords:
            if keyword in self.learning_resources:
                topic = keyword
                break

        # 如果找到相关主题，推荐资源
        if topic:
            resources = self.recommend_learning_resources(student_id, topic, count=2)
            if resources:
                resource_text = "\n\n还可以参考以下学习资源:\n"
                for resource in resources:
                    resource_text += f"- {resource['title']} ({resource['type']}): {resource['url']}\n"
                personalized_answer += resource_text

        return personalized_answer

    def add_learning_interaction(self, student_id, topic, question, answer_quality=None):
        """记录学习交互，更新学生档案"""
        profile = self.get_student_profile(student_id)
        if not profile:
            logger.warning(f"学生档案不存在，无法记录交互: {student_id}")
            return False

        # 添加交互记录
        activity_type = "question"

        # 表现是可选的，可以是学生对回答的评分或系统评估的问题难度
        performance = answer_quality if answer_quality else 100

        profile.add_learning_record(topic, activity_type, performance)

        # 如果问题涉及多个主题，尝试更新相关主题偏好
        keywords = question.split()
        for keyword in keywords:
            if len(keyword) > 2 and keyword != topic:  # 忽略太短的词和主题本身
                # 查询相关性
                items = self.knowledge_graph.find_related_knowledge(keyword)
                if items:
                    # 找到相关主题，也添加弱关联
                    profile.add_learning_record(keyword, "related_topic", 50)

        return True

    def create_demo_student_profiles(self):
        """创建演示用的学生档案"""
        # 学生1 - 编程爱好者
        student1 = StudentProfile("001", "张三")
        student1.set_learning_style("visual")

        # 添加Python学习记录
        student1.add_learning_record("Python编程", "quiz", 85)
        student1.add_learning_record("Python编程", "practice", 90)
        student1.add_learning_record("for循环", "exercise", 95)
        student1.add_learning_record("while循环", "exercise", 80)
        student1.add_learning_record("函数", "quiz", 70)

        # 添加人工智能学习记录
        student1.add_learning_record("人工智能", "reading", 60)
        student1.add_learning_record("机器学习", "video", 50)

        # 添加数学学习记录
        student1.add_learning_record("数学", "quiz", 75)

        # 学生2 - 数学爱好者
        student2 = StudentProfile("002", "李四")
        student2.set_learning_style("reading")

        # 添加数学学习记录
        student2.add_learning_record("数学", "quiz", 95)
        student2.add_learning_record("函数", "exercise", 90)
        student2.add_learning_record("导数", "quiz", 85)

        # 添加编程学习记录
        student2.add_learning_record("Python编程", "practice", 60)
        student2.add_learning_record("变量", "exercise", 70)

        # 添加到学生档案集合
        self.student_profiles[student1.student_id] = student1
        self.student_profiles[student2.student_id] = student2

        logger.info("已创建演示用学生档案")
        return [student1.student_id, student2.student_id]


# 当脚本直接运行时，执行示例过程
if __name__ == "__main__":
    # 创建个性化教学模块实例
    teaching = PersonalizedTeaching()

    # 创建演示学生档案
    student_ids = teaching.create_demo_student_profiles()

    # 测试学习资源推荐
    student_id = student_ids[0]
    resources = teaching.recommend_learning_resources(student_id, "Python编程")
    print(f"\n为学生 {student_id} 推荐的Python学习资源:")
    for res in resources:
        print(f"- {res['title']} ({res['type']}): {res['level']}")

    # 测试生成学习路径
    learning_path = teaching.generate_learning_path(student_id, "Python循环")
    print(f"\n为学生 {student_id} 生成的学习路径:")
    print(learning_path["learning_path"])

    # 测试个性化回答
    question = "Python中for循环和while循环有什么区别？"
    answer = teaching.generate_personalized_answer(student_id, question)
    print(f"\n学生问题: {question}")
    print(f"个性化回答:\n{answer}")

    # 保存学生档案
    teaching.save_student_profiles("student_profiles")
    print("\n演示完成，学生档案已保存")
