# 在ai_service/teaching_module/learning_analytics.py中创建新模块

import datetime
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np


class LearningAnalytics:
    """学习数据分析模块"""

    def __init__(self, data_dir="data/learning_analytics"):
        self.data_dir = data_dir
        self.logger = logging.getLogger("LEARNING_ANALYTICS")

        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)

    def record_interaction(self, student_id, interaction_data):
        """记录学习交互数据"""
        try:
            # 确保学生目录存在
            student_dir = os.path.join(self.data_dir, student_id)
            os.makedirs(student_dir, exist_ok=True)

            # 添加时间戳
            if "timestamp" not in interaction_data:
                interaction_data["timestamp"] = datetime.datetime.now().isoformat()

            # 保存交互数据
            filename = f"interaction_{interaction_data['timestamp'].replace(':', '-')}.json"
            filepath = os.path.join(student_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(interaction_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"已记录学生{student_id}的交互数据")
            return True

        except Exception as e:
            self.logger.error(f"记录交互数据失败: {e}")
            return False

    def analyze_student_progress(self, student_id):
        """分析学生学习进度"""
        try:
            student_dir = os.path.join(self.data_dir, student_id)
            if not os.path.exists(student_dir):
                self.logger.warning(f"找不到学生{student_id}的数据目录")
                return None

            # 读取所有交互数据
            interactions = []
            for filename in os.listdir(student_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(student_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        interactions.append(data)

            # 按时间排序
            interactions.sort(key=lambda x: x.get("timestamp", ""))

            # 提取主题和表现
            topics = {}
            for interaction in interactions:
                topic = interaction.get("topic", "未分类")
                performance = interaction.get("performance", 0)

                if topic not in topics:
                    topics[topic] = []

                topics[topic].append(performance)

            # 计算每个主题的平均表现和进步
            analysis = {}
            for topic, performances in topics.items():
                if len(performances) > 0:
                    analysis[topic] = {
                        "average": sum(performances) / len(performances),
                        "trend": performances[-1] - performances[0] if len(performances) > 1 else 0,
                        "samples": len(performances)
                    }

            return analysis

        except Exception as e:
            self.logger.error(f"分析学生进度失败: {e}")
            return None

    def generate_progress_chart(self, student_id, output_path=None):
        """生成学习进度图表"""
        try:
            analysis = self.analyze_student_progress(student_id)
            if not analysis:
                return None

            # 准备数据
            topics = list(analysis.keys())
            averages = [data["average"] for topic, data in analysis.items()]
            trends = [data["trend"] for topic, data in analysis.items()]

            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))

            # 绘制条形图
            x = np.arange(len(topics))
            width = 0.35

            ax.bar(x - width / 2, averages, width, label='平均表现')
            ax.bar(x + width / 2, trends, width, label='进步趋势')

            ax.set_title(f'学生{student_id}学习进度分析')
            ax.set_xticks(x)
            ax.set_xticklabels(topics)
            ax.legend()

            plt.tight_layout()

            # 保存或显示图表
            if output_path:
                plt.savefig(output_path)
                return output_path
            else:
                # 生成默认路径
                default_path = os.path.join(self.data_dir, f"{student_id}_progress.png")
                plt.savefig(default_path)
                return default_path

        except Exception as e:
            self.logger.error(f"生成进度图表失败: {e}")
            return None