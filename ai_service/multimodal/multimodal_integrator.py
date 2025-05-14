# 在ai_service/multimodal/multimodal_integrator.py中创建新模块

import logging
import os

from ai_service.multimodal.image_recognition import ImageRecognizer
from ai_service.multimodal.speech_recognition import SpeechRecognizer
from ai_service.multimodal.text_processor import TextProcessor


class MultimodalIntegrator:
    """多模态集成器，用于协调多种模态的输入输出"""

    def __init__(self):
        self.speech_recognizer = SpeechRecognizer()
        self.image_recognizer = ImageRecognizer()
        self.text_processor = TextProcessor()
        self.logger = logging.getLogger("MULTIMODAL_INTEGRATOR")

    def process_input(self, input_type, input_data):
        """处理不同类型的输入并返回标准化结果"""
        if input_type == "speech":
            # 语音输入处理
            if os.path.exists(input_data):  # 如果是文件路径
                text = self.speech_recognizer.recognize_from_file(input_data)
            else:  # 否则尝试直接从麦克风识别
                text = self.speech_recognizer.recognize_from_microphone()

            # 文本处理
            if text and text != "无法识别语音":
                keywords = self.text_processor.extract_keywords(text)
                question_type = self.text_processor.classify_question(text)
                return {
                    "text": text,
                    "keywords": keywords,
                    "question_type": question_type
                }
            return None

        elif input_type == "image":
            # 图像处理
            if os.path.exists(input_data):
                result = self.image_recognizer.recognize_image(input_data)
                return result
            return None

        elif input_type == "text":
            # 直接文本处理
            keywords = self.text_processor.extract_keywords(input_data)
            question_type = self.text_processor.classify_question(input_data)
            return {
                "text": input_data,
                "keywords": keywords,
                "question_type": question_type
            }

        return None