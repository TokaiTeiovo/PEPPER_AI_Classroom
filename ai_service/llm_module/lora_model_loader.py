
"""
LoRA模型加载器 - 专门用于加载已训练的LoRA模型进行推理
"""

import json
import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("LORA_MODEL_LOADER")


class LoRAModelLoader:
    """专门用于加载LoRA微调后的模型进行推理"""

    def __init__(self, lora_model_path):
        self.lora_model_path = lora_model_path
        self.base_model_path = None
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 读取训练信息
        self._load_training_info()

    def _load_training_info(self):
        """从训练信息中获取基础模型路径"""
        training_info_path = os.path.join(self.lora_model_path, 'training_info.json')

        if os.path.exists(training_info_path):
            try:
                with open(training_info_path, 'r', encoding='utf-8') as f:
                    training_info = json.load(f)
                    self.base_model_path = training_info.get('base_model')
                    logger.info(f"从训练信息获取基础模型: {self.base_model_path}")
            except Exception as e:
                logger.warning(f"读取训练信息失败: {e}")

        # 如果没有找到，使用默认路径
        if not self.base_model_path:
            self.base_model_path = "models/deepseek-coder-1.3b-base"
            logger.info(f"使用默认基础模型路径: {self.base_model_path}")

    def load_model(self, use_4bit=True, use_8bit=False):
        """加载LoRA模型"""
        try:
            # 验证路径
            if not os.path.exists(self.lora_model_path):
                raise FileNotFoundError(f"LoRA模型路径不存在: {self.lora_model_path}")

            if not os.path.exists(self.base_model_path):
                raise FileNotFoundError(f"基础模型路径不存在: {self.base_model_path}")

            # 检查必要的LoRA文件
            adapter_files = [
                'adapter_config.json',
                'adapter_model.bin',
                'adapter_model.safetensors'
            ]

            has_adapter = any(
                os.path.exists(os.path.join(self.lora_model_path, f))
                for f in adapter_files
            )

            if not has_adapter:
                raise FileNotFoundError("未找到LoRA适配器文件")

            # 加载分词器
            logger.info("加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 配置量化
            quantization_config = None
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            if self.device == "cuda" and (use_4bit or use_8bit):
                try:
                    if use_4bit:
                        logger.info("使用4bit量化")
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                        )
                    elif use_8bit:
                        logger.info("使用8bit量化")
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                        )
                except Exception as e:
                    logger.warning(f"量化配置失败: {e}")
                    quantization_config = None

            # 加载基础模型
            logger.info(f"加载基础模型: {self.base_model_path}")
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": dtype,
            }

            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = "auto"
            elif self.device == "cuda":
                load_kwargs["device_map"] = "auto"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                **load_kwargs
            )

            # 加载LoRA适配器
            logger.info(f"加载LoRA适配器: {self.lora_model_path}")
            from peft import PeftModel
            self.peft_model = PeftModel.from_pretrained(self.model, self.lora_model_path)

            logger.info("LoRA模型加载完成")
            return True

        except Exception as e:
            logger.error(f"LoRA模型加载失败: {e}")
            return False

    def generate_response(self, prompt, max_length=512):
        """生成回答"""
        if not self.peft_model or not self.tokenizer:
            return "模型未加载"

        try:
            # 简化提示格式
            if not prompt.startswith("Human:"):
                formatted_prompt = f"Human: {prompt}\n\nAssistant:"
            else:
                formatted_prompt = prompt

            # 编码输入
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.peft_model.device)

            # 优化生成参数
            gen_kwargs = {
                "max_new_tokens": min(120, max_length),  # 限制长度
                "temperature": 0.4,  # 降低随机性
                "top_p": 0.85,  # 更保守的采样
                "top_k": 25,  # 减少候选词
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.4,  # 强化重复惩罚
                "early_stopping": True,
                "no_repeat_ngram_size": 3,
            }

            # 生成回答
            with torch.no_grad():
                output_ids = self.peft_model.generate(**inputs, **gen_kwargs)

            # 只解码新生成的部分
            new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # 后处理
            response = self._clean_response(response)

            return response

        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return "抱歉，我现在无法处理您的问题。"

    def _clean_response(self, response):
        """清理回复内容"""
        if not response:
            return "请提供更具体的问题。"

        # 基本清理
        response = response.strip()

        # 移除可能的提示词残留
        if "Human:" in response:
            response = response.split("Human:")[0].strip()
        if "Assistant:" in response:
            response = response.replace("Assistant:", "").strip()

        # 确保回复完整性
        if len(response) < 5:
            return "请重新表述您的问题。"

        # 简单的句子完整性检查
        if not any(response.endswith(punct) for punct in ['。', '！', '？', '.', '!', '?']):
            # 如果没有合适的结尾标点，尝试在合理位置截断
            if '，' in response:
                response = response.rsplit('，', 1)[0] + '。'
            elif len(response) > 10:
                response += '。'

        return response


# 在 enhanced_api_server.py 中使用这个加载器
def load_lora_model_properly(lora_model_path, use_4bit=True, use_8bit=False):
    """正确加载LoRA模型的函数"""
    try:
        from ai_service.llm_module.lora_model_loader import LoRAModelLoader

        loader = LoRAModelLoader(lora_model_path)
        success = loader.load_model(use_4bit=use_4bit, use_8bit=use_8bit)

        if success:
            return loader
        else:
            return None

    except Exception as e:
        logger.error(f"LoRA模型加载失败: {e}")
        return None