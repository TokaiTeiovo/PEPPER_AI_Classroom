import os
import sys
import time
import json
import argparse

# 添加当前目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入PEPPER机器人相关模块
from pepper_robot.robot_control.robot_controller import PepperRobot
from pepper_robot.motion_module.gestures import PepperGestures
from pepper_robot.sensor_module.sensor_handler import PepperSensors
from interface.bridge.websocket_client import WebSocketClient


class PepperController:
    def __init__(self, robot_ip="127.0.0.1", robot_port=9559, ws_url="ws://localhost:8765"):
        """初始化PEPPER控制器"""
        print("正在初始化PEPPER控制器...")

        # 连接PEPPER机器人
        self.robot = PepperRobot(robot_ip, robot_port)
        self.gestures = PepperGestures(robot_ip, robot_port)
        self.sensors = PepperSensors(robot_ip, robot_port)

        # 连接WebSocket桥接服务
        self.ws_client = WebSocketClient(ws_url)
        self.ws_client.connect()

        print("PEPPER控制器初始化完成")

    def handle_speech_to_text(self, audio_data):
        """处理语音识别请求"""
        response = self.ws_client.send_and_receive("speech_recognition", {"audio_data": audio_data})
        if response and response.get("status") == "success":
            return response.get("data", {}).get("text")
        return None

    def handle_text_to_speech(self, text):
        """处理文本转语音请求"""
        self.robot.say(text)

    def handle_llm_query(self, query):
        """处理大语言模型查询"""
        response = self.ws_client.send_and_receive("llm_query", {"query": query})
        if response and response.get("status") == "success":
            return response.get("data", {}).get("response")
        return None

    def handle_image_recognition(self, image_data):
        """处理图像识别请求"""
        response = self.ws_client.send_and_receive("image_recognition", {"image_data": image_data})
        if response and response.get("status") == "success":
            return response.get("data", {}).get("objects")
        return None

    def demo_interaction(self):
        """演示交互流程"""
        # 欢迎语
        self.robot.say("大家好，我是PEPPER机器人。我可以回答你们的问题，帮助你们学习。")
        self.gestures.wave_hand()

        # 设置语音识别
        vocabulary = ["你好", "PEPPER", "问题", "解释", "什么", "循环", "编程", "人工智能"]
        self.sensors.setup_speech_recognition(vocabulary)
        self.sensors.start_speech_recognition()

        try:
            # 简单的交互循环
            for _ in range(10):  # 演示10次交互
                print("等待用户提问...")

                # 模拟获取用户提问
                time.sleep(2)  # 实际应用中应该是事件驱动的
                word = self.sensors.get_recognized_words()

                if word:
                    print(f"识别到: {word}")

                    # 思考动作
                    self.gestures.thinking_gesture()

                    # 处理问题并回答
                    if "循环" in word:
                        response = self.handle_llm_query("Python中的循环有哪些类型？它们有什么区别？")
                        if response:
                            self.handle_text_to_speech(response)

                    elif "人工智能" in word:
                        response = self.handle_llm_query("什么是人工智能？它在教育中有哪些应用？")
                        if response:
                            self.handle_text_to_speech(response)

                    else:
                        self.handle_text_to_speech("对不起，我没有理解你的问题。请再说一次。")
                        self.gestures.shake_head()

                else:
                    print("未识别到语音输入")

        finally:
            # 清理资源
            self.sensors.stop_speech_recognition()
            self.sensors.clean_up()

    def run(self):
        """运行PEPPER控制器"""
        try:
            print("PEPPER控制器正在运行...")
            self.demo_interaction()
        except Exception as e:
            print(f"运行PEPPER控制器时出错: {e}")
        finally:
            # 关闭连接
            self.ws_client.close()
            self.robot.rest()
            print("PEPPER控制器已关闭")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PEPPER机器人控制器')
    parser.add_argument('--robot-ip', type=str, default='127.0.0.1',
                        help='PEPPER机器人IP地址')
    parser.add_argument('--robot-port', type=int, default=9559,
                        help='PEPPER机器人端口')
    parser.add_argument('--ws-url', type=str, default='ws://localhost:8765',
                        help='WebSocket服务器URL')
    args = parser.parse_args()

    controller = PepperController(args.robot_ip, args.robot_port, args.ws_url)
    controller.run()
