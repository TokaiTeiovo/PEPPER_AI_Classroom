# 在pepper_robot/motion_module/teaching_actions.py中创建新模块

import logging
import time

from pepper_robot.motion_module.gestures import PepperGestures
from pepper_robot.robot_control.robot_controller import PepperRobot


class TeachingActions:
    """适用于教学场景的机器人动作组合"""

    def __init__(self, ip="127.0.0.1", port=9559, simulation_mode=False):
        self.simulation_mode = simulation_mode
        self.logger = logging.getLogger("TEACHING_ACTIONS")

        try:
            if not simulation_mode:
                # 实际机器人
                self.robot = PepperRobot(ip, port)
                self.gestures = PepperGestures(ip, port)
            else:
                # 导入模拟器模块
                from pepper_robot.simulation.pepper_simulator import PepperRobotSimulated, PepperGesturesSimulated
                self.robot = PepperRobotSimulated(ip, port)
                self.gestures = PepperGesturesSimulated(ip, port)

            self.logger.info("教学动作模块初始化成功")
        except Exception as e:
            self.logger.error(f"初始化教学动作模块失败: {e}")
            raise

    def greet_student(self, student_name=None):
        """向学生打招呼"""
        greeting = "你好"
        if student_name:
            greeting += f"，{student_name}"
        greeting += "！我是PEPPER机器人，很高兴能帮助你学习。"

        self.robot.say(greeting)
        self.gestures.wave_hand()
        return True

    def explain_concept(self, concept, explanation):
        """解释概念"""
        intro = f"让我来解释一下{concept}的概念。"
        self.robot.say(intro)

        # 思考手势
        self.gestures.thinking_gesture()
        time.sleep(0.5)

        # 分段解释长文本
        if len(explanation) > 200:
            sentences = explanation.split('。')
            for i in range(0, len(sentences), 3):
                sentence_group = '。'.join(sentences[i:i + 3]) + ('。' if i + 3 < len(sentences) else '')
                if sentence_group.strip():
                    self.robot.say(sentence_group)
                    time.sleep(0.3)
        else:
            self.robot.say(explanation)

        return True

    def answer_question(self, answer, show_thinking=True):
        """回答问题，可以展示思考过程"""
        if show_thinking:
            self.robot.say("让我思考一下...")
            self.gestures.thinking_gesture()
            time.sleep(1.5)

        # 分段输出长回答
        if len(answer) > 200:
            sentences = answer.split('。')
            for i in range(0, len(sentences), 3):
                sentence_group = '。'.join(sentences[i:i + 3]) + ('。' if i + 3 < len(sentences) else '')
                if sentence_group.strip():
                    self.robot.say(sentence_group)
                    time.sleep(0.3)
        else:
            self.robot.say(answer)

        return True

    def encourage_student(self):
        """鼓励学生"""
        encouragements = [
            "很好！你的思考很有深度。",
            "你的问题很有见地！",
            "继续保持好奇心，这是学习的关键。",
            "你正在进步，继续加油！"
        ]

        import random
        message = random.choice(encouragements)
        self.robot.say(message)
        self.gestures.applause()
        return True

    def end_session(self):
        """结束教学会话"""
        message = "今天的学习就到这里，谢谢你的参与！有任何问题随时可以问我。"
        self.robot.say(message)
        self.gestures.bow()
        return True