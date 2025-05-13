import speech_recognition as sr


class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def recognize_from_microphone(self, language='zh-CN'):
        """从麦克风接收音频并识别为文本"""
        try:
            with sr.Microphone() as source:
                print("请说话...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source)
                print("识别中...")
                text = self.recognizer.recognize_google(audio, language=language)
                return text
        except sr.UnknownValueError:
            return "无法识别语音"
        except sr.RequestError as e:
            return f"无法请求Google语音识别服务; {e}"

    def recognize_from_file(self, audio_file_path, language='zh-CN'):
        """从文件识别语音为文本"""
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio, language=language)
                return text
        except sr.UnknownValueError:
            return "无法识别语音"
        except sr.RequestError as e:
            return f"无法请求Google语音识别服务; {e}"


# 测试语音识别
if __name__ == "__main__":
    recognizer = SpeechRecognizer()
    text = recognizer.recognize_from_microphone()
    print(f"识别结果: {text}")
