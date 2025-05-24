"""
LoRA微调模块 - 用于PEPPER机器人教学系统的大语言模型微调
"""

import logging
import os
from datetime import datetime

import torch
from datasets import Dataset
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("LORA_FINE_TUNING")


class LoRAFineTuner:
    def __init__(
            self,
            base_model_path="models/deepseek-coder-1.3b-base",
            output_dir=None,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05
    ):
        """初始化LoRA微调器"""
        self.base_model_path = base_model_path

        # 如果没有指定输出目录，使用时间戳生成
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"models/deepseek-{timestamp}"
        else:
            self.output_dir = output_dir

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"模型将保存到: {self.output_dir}")

    def load_base_model(self, use_4bit=True, use_8bit=False):
        """加载基础模型"""
        logger.info(f"正在加载基础模型: {self.base_model_path}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
        )

        # 设置分词器的填充和特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 确定设备和数据类型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info(f"使用设备: {device}, 数据类型: {dtype}")

        # 配置量化参数
        quantization_config = None

        if device == "cuda" and (use_4bit or use_8bit):
            try:
                if use_4bit:
                    logger.info("使用4bit量化配置")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",  # 使用NF4量化类型
                        bnb_4bit_compute_dtype=torch.float16,  # 计算时使用float16
                        bnb_4bit_use_double_quant=True,  # 双重量化，进一步节省内存
                    )
                elif use_8bit:
                    logger.info("使用8bit量化配置")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
            except Exception as e:
                logger.warning(f"量化配置失败，将使用标准加载: {e}")
                quantization_config = None

        # 加载模型
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }

        # 添加量化配置
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        elif device == "cuda":
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32

            # 加载模型
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    **load_kwargs
                )

                # 如果使用CPU且没有量化，手动移动到设备
                if device == "cpu" and quantization_config is None:
                    self.model = self.model.to(device)

                logger.info("基础模型加载完成")

                # 打印模型信息
                if hasattr(self.model, 'get_memory_footprint'):
                    memory_mb = self.model.get_memory_footprint() / 1024 / 1024
                    logger.info(f"模型显存占用: {memory_mb:.2f} MB")

                return True

            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                return False

    def prepare_lora_config(self):
        """准备LoRA配置"""
        logger.info("配置LoRA参数")

        # 为DeepSeek模型获取正确的目标模块
        if hasattr(self.model, "config") and hasattr(self.model.config,"model_type"):
            model_type = self.model.config.model_type
            logger.info(f"检测到模型类型: {model_type}")
            if model_type == "deepseek":
                # DeepSeek模型的特定模块
                target_modules = ["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif model_type in ["llama", "mistral"]:
                # LLaMA/Mistral类型模型
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            else:
                # 通用目标模块
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
        # 默认目标模块
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        logger.info(f"目标模块: {target_modules}")

        # 定义LoRA配置
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias="none",  # 不训练bias
            modules_to_save=None,  # 不保存额外模块
        )

        # 为量化训练准备模型
        if hasattr(self.model, "is_loaded_in_8bit") and self.model.is_loaded_in_8bit:
            logger.info("为8bit量化训练准备模型")
            self.model = prepare_model_for_kbit_training(self.model)
        elif hasattr(self.model, "is_loaded_in_4bit") and self.model.is_loaded_in_4bit:
            logger.info("为4bit量化训练准备模型")
            self.model = prepare_model_for_kbit_training(self.model)

        # 应用LoRA配置
        self.peft_model = get_peft_model(self.model, peft_config)

        # 打印可训练参数信息
        trainable_params = 0
        all_param = 0
        for _, param in self.peft_model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        logger.info(f"可训练参数: {trainable_params:,} / 总参数: {all_param:,} "
                    f"({100 * trainable_params / all_param:.2f}%)")

        return self.peft_model

    def prepare_dataset(self, data_path=None, instruction_data=None):
        """
        准备训练数据集
        支持从 JSON 或 CSV 读取 [{instruction, input, output}] 格式
        """
        if data_path and os.path.exists(data_path):
            if data_path.endswith('.json'):
                import json
                with open(data_path, 'r', encoding='utf-8') as f:
                    instruction_data = json.load(f)
            elif data_path.endswith('.csv'):
                import pandas as pd
                instruction_data = pd.read_csv(data_path).to_dict('records')

        if not instruction_data:
            # 示例数据
            instruction_data = [
                {
                    "instruction": "解释Python中for循环和while循环的区别",
                    "input": "",
                    "output": "for循环用于遍历序列，while用于条件循环。"
                },
                {
                    "instruction": "PEPPER机器人在课堂上有哪些优势",
                    "input": "",
                    "output": "互动性强，具备视觉语音交互，提升教学效率。"
                }
            ]

        logger.info(f"准备训练数据集，共{len(instruction_data)}条记录")

        # 转换为 prompt + response 格式
        def format_instruction(example):
            instruction = example["instruction"]
            input_text = example.get("input", "")
            output = example["output"]
            prompt = f"Human: {instruction}\n{input_text}\n\nAssistant: " if input_text else f"Human: {instruction}\n\nAssistant: "
            full_text = prompt + output
            return {"text": full_text}

        formatted_data = [format_instruction(item) for item in instruction_data]
        raw_dataset = Dataset.from_list(formatted_data)

        # 分词 + 添加 labels，并删除原始text字段
        def tokenize_function(examples):
            tokens = self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=1024
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        tokenized_dataset = raw_dataset.map(tokenize_function, batched=False)
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])  # 删除text字段，避免转换tensor时报错

        return tokenized_dataset

    def train(self, dataset, epochs=3, batch_size=4, learning_rate=2e-4):
        """训练模型"""
        logger.info("开始训练过程")

        # 根据是否使用量化调整训练参数
        if hasattr(self.model, "is_loaded_in_4bit") and self.model.is_loaded_in_4bit:
            logger.info("使用4bit量化训练配置")
            # 4bit训练建议使用更小的batch size和更大的gradient accumulation
            if batch_size > 2:
                batch_size = 2
                logger.info("4bit训练建议batch_size设置为2")
            gradient_accumulation_steps = 8  # 增加梯度累积步数
        elif hasattr(self.model, "is_loaded_in_8bit") and self.model.is_loaded_in_8bit:
            logger.info("使用8bit量化训练配置")
            gradient_accumulation_steps = 4
        else:
            gradient_accumulation_steps = 4

        # 定义训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,  # 增加保存间隔，减少IO
            save_total_limit=2,  # 减少保存的检查点数量
            fp16=(self.device == "cuda"),  # 如果使用GPU则启用fp16
            bf16=False,  # 对于量化训练，通常不使用bf16
            report_to="none",  # 禁用报告
            remove_unused_columns=False,  # 保留所有列
            dataloader_pin_memory=False,  # 量化训练时关闭pin_memory
            gradient_checkpointing=True,  # 启用梯度检查点节省内存
            optim="adamw_torch",  # 使用标准AdamW优化器
        )

        # 创建数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # 不使用掩码语言模型训练
        )

        # 创建训练器
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        # 开始训练
        logger.info("开始LoRA微调训练")
        try:
            trainer.train()
        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            return False

        # 保存训练信息
        training_info = {
            "base_model": self.base_model_path,
            "training_time": datetime.now().isoformat(),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "output_dir": self.output_dir,
            "quantization": "4bit" if hasattr(self.model, "is_loaded_in_4bit") and self.model.is_loaded_in_4bit else
            "8bit" if hasattr(self.model, "is_loaded_in_8bit") and self.model.is_loaded_in_8bit else "none",
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "total_trainable_params": sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        }

        info_file = os.path.join(self.output_dir, "training_info.json")
        import json
        try:
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(training_info, f, ensure_ascii=False, indent=2)
            logger.info(f"训练信息已保存到: {info_file}")
        except Exception as e:
            logger.warning(f"保存训练信息失败: {e}")

        return True

    def generate_response(self, prompt, max_new_tokens=512):
        """使用微调后的模型生成回答"""
        if not self.peft_model:
            logger.error("模型未加载")
            return "模型未加载，无法生成回答"
        try:
            # 为DeepSeek模型调整提示格式
            if not prompt.startswith("Human:"):
                prompt = f"Human: {prompt}\n\nAssistant:"

            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # 设置生成参数
            gen_kwargs = {
                "max_new_tokens": min(max_new_tokens, 512),
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1,  # 避免重复
            }

            # 生成回答
            with torch.no_grad():
                output_ids = self.peft_model.generate(**inputs, **gen_kwargs)

            # 解码输出
            full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # 提取回答部分（去除提示部分）
            if "Assistant:" in full_output:
                response = full_output.split("Assistant:", 1)[1].strip()
            else:
                response = full_output[len(prompt):].strip()

            return response

        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return f"生成回答时出错: {str(e)}"


# 当脚本直接运行时，执行示例训练过程
if __name__ == "__main__":
    # 创建LoRA微调器实例
    fine_tuner = LoRAFineTuner()

    # 加载基础模型
    if fine_tuner.load_base_model():
        # 准备LoRA配置
        fine_tuner.prepare_lora_config()

        # 准备训练数据集（使用默认示例数据）
        dataset = fine_tuner.prepare_dataset()

        # 训练模型
        fine_tuner.train(dataset, epochs=3)

        # 测试微调后的模型
        test_prompts = [
            "Python中for循环和while循环有什么区别？",
            "人工智能在教育中有哪些应用？",
            "什么是多模态交互？",
            "PEPPER机器人在课堂上的优势是什么？"
        ]

        for prompt in test_prompts:
            response = fine_tuner.generate_response(prompt)
            print(f"问题: {prompt}")
            print(f"回答: {response}")
            print("-" * 80)
