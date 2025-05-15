import time

from naoqi import ALProxy


class PepperGestures:
    def __init__(self, ip="127.0.0.1", port=9559):
        """初始化PEPPER手势处理"""
        try:
            self.animation = ALProxy("ALAnimationPlayer", ip, port)
            self.motion = ALProxy("ALMotion", ip, port)
            self.posture = ALProxy("ALRobotPosture", ip, port)

            print("成功初始化PEPPER手势处理")
        except Exception as e:
            print(f"初始化PEPPER手势处理失败: {e}")
            raise

    def wave_hand(self):
        """挥手手势"""
        try:
            self.animation.run("animations/Hello")
            return True
        except Exception as e:
            print(f"挥手手势失败: {e}")
            return False

    def thinking_gesture(self):
        """思考手势"""
        try:
            self.animation.run("animations/Thinking")
            return True
        except Exception as e:
            print(f"思考手势失败: {e}")
            return False

    def point_to_board(self):
        """指向黑板手势"""
        try:
            self.animation.run("animations/Pointing")
            return True
        except Exception as e:
            print(f"指向黑板手势失败: {e}")
            return False

    def shake_head(self):
        """摇头手势"""
        try:
            self.animation.run("animations/No")
            return True
        except Exception as e:
            print(f"摇头手势失败: {e}")
            return False

    def nod_head(self):
        """点头手势"""
        try:
            self.animation.run("animations/Yes")
            return True
        except Exception as e:
            print(f"点头手势失败: {e}")
            return False

    def applause(self):
        """鼓掌手势"""
        try:
            self.animation.run("animations/Applause")
            return True
        except Exception as e:
            print(f"鼓掌手势失败: {e}")
            return False

    def bow(self):
        """鞠躬手势"""
        try:
            self.animation.run("animations/Bow")
            return True
        except Exception as e:
            print(f"鞠躬手势失败: {e}")
            return False

# 测试PEPPER传感器
if __name__ == "__main__":
    import sys

    # 解析参数
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    else:
        ip = "127.0.0.1"

    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    else:
        port = 9559

    # 创建传感器处理实例
    sensors = PepperSensors(ip, port)

    # 测试声纳
    front_distance = sensors.get_sonar_distance("front")
    back_distance = sensors.get_sonar_distance("back")
    print(f"前方距离: {front_distance} 米")
    print(f"后方距离: {back_distance} 米")

    # 测试语音识别
    vocabulary = ["你好", "PEPPER", "开始", "停止", "问题"]
    if sensors.setup_speech_recognition(vocabulary):
        print("语音识别设置成功，开始识别...")
        sensors.start_speech_recognition()

        try:
            # 持续监听10秒
            for _ in range(10):
                word = sensors.get_recognized_words()
                if word:
                    print(f"识别到: {word}")
                time.sleep(1)
        finally:
            sensors.stop_speech_recognition()
            print("语音识别已停止")

    # 清理资源
    sensors.clean_up()
    print("测试完成")
