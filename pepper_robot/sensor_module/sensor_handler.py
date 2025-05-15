from naoqi import ALProxy


class PepperSensors:
    def __init__(self, ip="127.0.0.1", port=9559):
        """初始化PEPPER传感器处理"""
        try:
            self.memory = ALProxy("ALMemory", ip, port)
            self.sonar = ALProxy("ALSonar", ip, port)
            self.touch = ALProxy("ALTouch", ip, port)
            self.camera = ALProxy("ALVideoDevice", ip, port)
            self.speech = ALProxy("ALSpeechRecognition", ip, port)

            # 激活声纳传感器
            self.sonar.subscribe("PepperSensors")

            print("成功初始化PEPPER传感器处理")
        except Exception as e:
            print(f"初始化PEPPER传感器处理失败: {e}")
            raise

    def get_sonar_distance(self, side="front"):
        """获取声纳测量的距离"""
        try:
            if side.lower() == "front":
                return self.memory.getData("Device/SubDeviceList/Platform/Front/Sonar/Sensor/Value")
            elif side.lower() == "back":
                return self.memory.getData("Device/SubDeviceList/Platform/Back/Sonar/Sensor/Value")
            else:
                print(f"未知的声纳位置: {side}")
                return None
        except Exception as e:
            print(f"获取声纳距离失败: {e}")
            return None

    def is_touched(self, sensor_name):
        """检查指定的触摸传感器是否被触摸"""
        try:
            sensor_key = f"Device/SubDeviceList/{sensor_name}/Actuator/Value"
            return self.memory.getData(sensor_key) == 1.0
        except Exception as e:
            print(f"检查触摸传感器失败: {e}")
            return False

    def setup_speech_recognition(self, vocabulary, language="Chinese"):
        """设置语音识别"""
        try:
            self.speech.setLanguage(language)
            self.speech.setVocabulary(vocabulary, False)
            return True
        except Exception as e:
            print(f"设置语音识别失败: {e}")
            return False

    def start_speech_recognition(self, callback_module="PepperSensors"):
        """启动语音识别"""
        try:
            self.speech.subscribe(callback_module)
            return True
        except Exception as e:
            print(f"启动语音识别失败: {e}")
            return False

    def stop_speech_recognition(self, callback_module="PepperSensors"):
        """停止语音识别"""
        try:
            self.speech.unsubscribe(callback_module)
            return True
        except Exception as e:
            print(f"停止语音识别失败: {e}")
            return False

    def get_recognized_words(self):
        """获取识别到的词语"""
        try:
            words = self.memory.getData("WordRecognized")
            if words and len(words) >= 2 and words[1] > 0.5:  # 置信度阈值
                return words[0]
            return None
        except Exception as e:
            print(f"获取识别词语失败: {e}")
            return None

    def clean_up(self):
        """清理资源"""
        try:
            self.sonar.unsubscribe("PepperSensors")
            self.stop_speech_recognition()
            print("成功清理PEPPER传感器资源")
            return True
        except Exception as e:
            print(f"清理传感器资源失败: {e}")
            return False

