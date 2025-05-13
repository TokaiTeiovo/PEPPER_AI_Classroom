import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


class ImageRecognizer:
    def __init__(self, model_name="google/vit-base-patch16-224"):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)

    def preprocess_image(self, image_path):
        """预处理图像为模型输入格式"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs

    def recognize_image(self, image_path):
        """识别图像内容"""
        inputs = self.preprocess_image(image_path)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

        return {
            "class_id": predicted_class_idx,
            "class_name": self.model.config.id2label[predicted_class_idx]
        }


# 测试图像识别
if __name__ == "__main__":
    image_recognizer = ImageRecognizer()
    result = image_recognizer.recognize_image("test_image.jpg")
    print(f"识别结果: {result}")
