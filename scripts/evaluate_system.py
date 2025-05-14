# 在scripts/evaluate_system.py创建

import argparse
import json
import logging
import os
import time

import matplotlib.pyplot as plt
import pandas as pd

from integrated_system import PepperIntegratedSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EVALUATE_SYSTEM")

# 评估问题集
evaluation_questions = [
    {
        "question": "Python中for循环和while循环有什么区别？",
        "topic": "Python编程",
        "difficulty": "basic"
    },
    {
        "question": "什么是人工智能在教育中的应用？",
        "topic": "人工智能",
        "difficulty": "intermediate"
    },
    {
        "question": "多模态交互是什么，它如何在PEPPER机器人中实现？",
        "topic": "多模态交互",
        "difficulty": "advanced"
    },
    {
        "question": "如何使用LoRA技术微调大语言模型？",
        "topic": "大语言模型",
        "difficulty": "advanced"
    },
    {
        "question": "知识图谱在个性化教学中有什么作用？",
        "topic": "知识图谱",
        "difficulty": "intermediate"
    }
]


def run_system_evaluation(config_path, student_id, output_path):
    """运行系统评估"""
    try:
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 初始化系统
        logger.info("初始化系统...")
        system = PepperIntegratedSystem(config)

        # 准备会话
        logger.info(f"准备会话，学生ID: {student_id}")
        system.prepare_teaching_session(student_id)

        # 启动会话
        system.start_teaching_session()

        # 评估结果
        results = []

        # 运行问题评估
        for i, question_data in enumerate(evaluation_questions):
            question = question_data["question"]
            topic = question_data["topic"]
            difficulty = question_data["difficulty"]

            logger.info(f"评估问题 {i + 1}/{len(evaluation_questions)}: {question}")

            # 记录开始时间
            start_time = time.time()

            # 处理问题
            response = system.process_text_input(question)

            # 计算响应时间
            response_time = time.time() - start_time

            # 记录结果
            results.append({
                "question": question,
                "topic": topic,
                "difficulty": difficulty,
                "response": response[:100] + "..." if response and len(response) > 100 else response,
                "response_time": response_time,
                "response_length": len(response) if response else 0
            })

            # 短暂等待
            time.sleep(1)

        # 停止会话
        system.stop_teaching_session()

        # 清理资源
        system.clean_up()

        # 保存结果
        results_df = pd.DataFrame(results)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存CSV
        results_df.to_csv(output_path, index=False)
        logger.info(f"评估结果已保存到: {output_path}")

        # 生成图表
        generate_evaluation_charts(results_df, os.path.splitext(output_path)[0] + "_charts.png")

        return results_df

    except Exception as e:
        logger.error(f"系统评估失败: {e}")
        return None


def generate_evaluation_charts(results_df, output_path):
    """生成评估图表"""
    try:
        # 创建图表
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        # 按难度级别统计响应时间
        response_time_by_difficulty = results_df.groupby('difficulty')['response_time'].mean()
        axs[0].bar(response_time_by_difficulty.index, response_time_by_difficulty.values)
        axs[0].set_title('平均响应时间（按难度）')
        axs[0].set_ylabel('响应时间（秒）')
        axs[0].set_xlabel('难度级别')

        # 按主题统计响应长度
        response_length_by_topic = results_df.groupby('topic')['response_length'].mean()
        axs[1].bar(response_length_by_topic.index, response_length_by_topic.values)
        axs[1].set_title('平均响应长度（按主题）')
        axs[1].set_ylabel('响应长度（字符）')
        axs[1].set_xlabel('主题')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"评估图表已保存到: {output_path}")

    except Exception as e:
        logger.error(f"生成评估图表失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='评估PEPPER机器人智能教学系统')
    parser.add_argument('--config', type=str, default='config.json',
                        help='配置文件路径')
    parser.add_argument('--student', type=str, default="001",
                        help='学生ID')
    parser.add_argument('--output', type=str, default="evaluation/results.csv",
                        help='输出文件路径')

    args = parser.parse_args()

    run_system_evaluation(args.config, args.student, args.output)


if __name__ == "__main__":
    main()