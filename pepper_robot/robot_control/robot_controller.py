import time
import sys
import argparse
from naoqi import ALProxy


class PepperRobot:
    def __init__(self, ip="127.0.0.1", port=9559):
        """初始化PEPPER机器人连接"""
        try:
            # 连接基本服务
            self.motion = ALProxy("ALMotion", ip, port)
            self.posture = ALProxy("ALRobotPosture", ip, port)
            self.tts = ALProxy("ALTextToSpeech", ip, port)
            self.animation = ALProxy("ALAnimationPlayer", ip, port)
            self.behavior = ALProxy("ALBehaviorManager", ip, port)
            self.audio = ALProxy("ALAudioDevice", ip, port)
            self.camera = ALProxy("ALVideoDevice", ip, port)
            self.memory = ALProxy("ALMemory", ip, port)

            # 初始化
            self.motion.wakeUp()  # 唤醒机器人
            print("成功连接到PEPPER机器人")
        except Exception as e:
            print(f"连接PEPPER机器人失败: {e}")
            sys.exit(1)

    def say(self, text, language="Chinese"):
        """机器人说话"""
        try:
            self.tts.setLanguage(language)
            self.tts.say(text)
            return True
        except Exception as e:
            print(f"语音输出失败: {e}")
            return False

    def move_to(self, x, y, theta):
        """控制机器人移动到指定位置"""
        try:
            self.motion.moveTo(x, y, theta)
            return True
        except Exception as e:
            print(f"移动失败: {e}")
            return False

    def set_posture(self, posture_name="Stand", speed=1.0):
        """设置机器人姿势"""
        try:
            self.posture.goToPosture(posture_name, speed)
            return True
        except Exception as e:
            print(f"设置姿势失败: {e}")
            return False

    def play_animation(self, animation_name):
        """播放预设动画"""
        try:
            self.animation.run(animation_name)
            return True
        except Exception as e:
            print(f"播放动画失败: {e}")
            return False

    def rest(self):
        """机器人休息"""
        try:
            self.motion.rest()
            return True
        except Exception as e:
            print(f"休息失败: {e}")
            return False

    def get_camera_image(self, camera_id=0):
        """获取摄像头图像"""
        try:
            resolution = 2  # VGA
            colorSpace = 11  # RGB
            fps = 5

            # 订阅摄像头
            videoClient = self.camera.subscribeCamera(
                "python_client", camera_id, resolution, colorSpace, fps)

            # 获取图像
            naoImage = self.camera.getImageRemote(videoClient)

            # 取消订阅
            self.camera.unsubscribe(videoClient)

            # 返回图像数据
            return naoImage
        except Exception as e:
            print(f"获取图像失败: {e}")
            return None

    def record_audio(self, duration=5):
        """录制音频"""
        try:
            # 设置音频参数
            self.audio.enableEnergyComputation()

            # 开始录制
            print(f"开始录制音频，持续{duration}秒...")
            time.sleep(duration)

            # 获取音频能量
            energy = self.audio.getFrontMicEnergy()
            print(f"音频能量: {energy}")

            # 实际上这只是一个示例，真正的音频录制需要更复杂的实现
            return True
        except Exception as e:
            print(f"录制音频失败: {e}")
            return False


# 测试PEPPER机器人控制
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PEPPER机器人控制')
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                        help='机器人IP地址')
    parser.add_argument('--port', type=int, default=9559,
                        help='机器人端口')
    args = parser.parse_args()

    # 创建机器人实例
    pepper = PepperRobot(args.ip, args.port)

    # 测试基本功能
    pepper.say("你好，我是PEPPER机器人，我可以帮助你进行学习。")
    pepper.set_posture("Stand")
    pepper.play_animation("animations/Stand/Gestures/Hey_1")

    # 测试结束，让机器人休息
    pepper.rest()
    print("测试完成")
