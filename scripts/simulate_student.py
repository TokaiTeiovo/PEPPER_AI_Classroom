# 在scripts/simulate_student.py创建

import argparse
import logging
import random
import time

import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SIMULATE_STUDENT")

# 预设问题列表
preset_questions = {
    "Python编程": [
        "Python中for循环和while循环有什么区别？",
        "如何在Python中定义一个函数？",
        "什么是Python的列表推导式？",
        "Python中如何处理异常？",
        "如何使用Python读写文件？"
    ],
    "人工智能": [
        "什么是人工智能？",
        "机器学习和深度学习有什么区别？",
        "什么是神经网络？",
        "自然语言处理的主要应用有哪些？",
        "计算机视觉技术在教育中如何应用？"
    ],
    "PEPPER机器人": [
        "PEPPER机器人有哪些传感器？",
        "PEPPER机器人如何实现多模态交互？",
        "PEPPER机器人在教育中的优势是什么？",
        "如何编程控制PEPPER机器人的动作？",
        "PEPPER机器人能够识别人脸吗？"
    ]
}

# 反馈词语
feedback_phrases = [
    "谢谢你的解答！",
    "这个回答很有帮助。",
    "我现在明白了。",
    "你能再详细解释一下吗？",
    "这个概念有点难理解。"
]


def simulate_interaction(base_url, student_id, topic, num_interactions):
    """模拟学生互动"""
    try:
        # 初始化系统
        resp = requests.post(f"{base_url}/api/initialize", json={"config_path": "config.json"})
        if resp.status_code != 200 or resp.json().get("status") != "success":
            logger.error("初始化系统失败")
            return False

        # 启动会话
        resp = requests.post(f"{base_url}/api/start_session", json={
            "student_id": student_id,
            "topic": topic
        })
        if resp.status_code != 200 or resp.json().get("status") != "success":
            logger.error("启动会话失败")
            return False

        # 获取问题列表
        questions = preset_questions.get(topic, [])
        if not questions:
            # 如果没有特定主题的问题，随机选择一个主题
            questions = random.choice(list(preset_questions.values()))

        # 随机打乱问题顺序
        random.shuffle(questions)

        # 根据互动次数选择问题
        selected_questions = questions[:num_interactions]
        if len(selected_questions) < num_interactions:
            # 如果问题不够，循环使用
            additional = random.sample(questions, num_interactions - len(selected_questions))
            selected_questions.extend(additional)

        # 开始互动
        for i, question in enumerate(selected_questions):
            logger.info(f"互动 {i + 1}/{num_interactions}: {question}")

            # 发送问题
            resp = requests.post(f"{base_url}/api/process_input", json={"text": question})
            if resp.status_code != 200 or resp.json().get("status") != "success":
                logger.error(f"处理问题失败: {question}")
                continue

            response = resp.json().get("response", "")
            logger.info(f"系统回答: {response[:100]}...")

            # 模拟思考时间
            time.sleep(random.uniform(1.0, 3.0))

            # 50%的概率发送反馈
            if random.random() < 0.5:
                feedback = random.choice(feedback_phrases)
                logger.info(f"发送反馈: {feedback}")

                resp = requests.post(f"{base_url}/api/process_input", json={"text": feedback})
                if resp.status_code != 200 or resp.json().get("status") != "success":
                    logger.error(f"处理反馈失败: {feedback}")

                # 模拟短暂等待
                time.sleep(random.uniform(0.5, 1.5))

        # 发送结束消息
        end_message = "谢谢你的帮助，再见！"
        logger.info(f"发送结束消息: {end_message}")

        resp = requests.post(f"{base_url}/api/process_input", json={"text": end_message})

        # 停止会话
        resp = requests.post(f"{base_url}/api/stop_session")
        if resp.status_code != 200 or resp.json().get("status") != "success":
            logger.error("停止会话失败")

        # 清理资源
        resp = requests.post(f"{base_url}/api/clean_up")

        logger.info("模拟互动完成")
        return True

    except Exception as e:
        logger.error(f"模拟互动失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='模拟学生与PEPPER机器人教学系统互动')
    parser.add_argument('--url', type=str, default="http://localhost:5000/api",
                        help='API地址')
    parser.add_argument('--student', type=str, default="001",
                        help='学生ID')
    parser.add_argument('--topic', type=str, default="Python编程",
                        help='教学主题')
    parser.add_argument('--interactions', type=int, default=5,
                        help='互动次数')

    args = parser.parse_args()

    simulate_interaction(args.url, args.student, args.topic, args.interactions)


if __name__ == "__main__":
    main()