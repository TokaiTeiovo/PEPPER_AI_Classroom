"""
LoRA微调模块 - 用于PEPPER机器人教学系统的大语言模型微调

该模块实现了基于教育领域知识的大语言模型微调功能
使用LoRA (Low-Rank Adaptation) 技术，以低资源方式微调现有大模型
"""

import logging
import os

import pandas as pd
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
    DataCollatorForLanguageModeling
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
            output_dir="models/deepseek-finetuned",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05
    ):
        """初始化LoRA微调器"""
        self.base_model_path = base_model_path
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

    def load_base_model(self, use_8bit=True):
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

        # 加载模型
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
        }

        # 根据设备和配置决定模型加载方式
        if self.device == "cuda":
            if use_8bit:
                load_kwargs["load_in_8bit"] = True
            else:
                load_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            **load_kwargs
        )

        logger.info("基础模型加载完成")

        return True

    def prepare_lora_config(self):
        """准备LoRA配置"""
        logger.info("配置LoRA参数")

        # 为DeepSeek模型获取正确的目标模块
        if hasattr(self.model, "config") and hasattr(self.model.config,
                                                     "model_type") and self.model.config.model_type == "deepseek":
            # DeepSeek模型的特定模块
            target_modules = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
        else:
            # 查找模型中包含的层
            import re
            target_modules = []
            for name, _ in self.model.named_modules():
                if any(re.search(pattern, name) for pattern in ["attention", "mlp", "dense", "linear"]):
                    parts = name.split('.')
                    if len(parts) > 1 and parts[-1] in ["query", "key", "value", "out_proj", "fc1", "fc2", "up_proj",
                                                        "down_proj", "gate_proj"]:
                        target_modules.append(parts[-1])

            # 如果没有找到任何目标模块，使用默认值
            if not target_modules:
                target_modules = ["query", "key", "value", "out_proj", "fc1", "fc2"]

            # 去重
            target_modules = list(set(target_modules))
            logger.info(f"自动检测到的目标模块: {target_modules}")

        # 定义LoRA配置
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules
        )

        # 为量化训练准备模型（如果使用8bit）
        if hasattr(self.model, "is_loaded_in_8bit") and self.model.is_loaded_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # 应用LoRA配置
        self.peft_model = get_peft_model(self.model, peft_config)

        # 打印模型训练信息
        logger.info(f"可训练参数: {self.peft_model.print_trainable_parameters()}")

        return self.peft_model

    def prepare_dataset(self, data_path=None, instruction_data=None):
        """
        准备训练数据集

        可以从文件加载数据，或直接传入指令数据
        数据格式应为[{"instruction": "...", "input": "...", "output": "..."}]
        """
        if data_path and os.path.exists(data_path):
            # 从文件加载数据
            if data_path.endswith('.json'):
                import json
                with open(data_path, 'r', encoding='utf-8') as f:
                    instruction_data = json.load(f)
            elif data_path.endswith('.csv'):
                instruction_data = pd.read_csv(data_path).to_dict('records')

        if not instruction_data:
            # 如果没有提供数据，创建一个简单的教育领域示例数据集
            instruction_data = [
                {
                    "instruction": "解释Python中for循环和while循环的区别",
                    "input": "",
                    "output": "在Python中，for循环和while循环是两种不同的循环结构。for循环用于遍历一个已知的序列（如列表、元组、字符串等），它在每次迭代时会自动获取序列中的下一个元素，直到遍历完所有元素。而while循环则是在指定条件为真时重复执行代码块，需要手动更新循环变量，适合在不确定迭代次数的情况下使用。"
                },
                {
                    "instruction": "什么是人工智能在教育中的应用",
                    "input": "",
                    "output": "人工智能在教育中的应用多种多样，包括：个性化学习系统，根据学生的学习进度和能力提供定制化的学习内容；智能辅导系统，能够回答学生的问题并提供即时反馈；自动评分系统，对作业和考试进行自动评分；教育数据分析，帮助教师了解学生的学习行为和模式；智能机器人助教，如PEPPER机器人，可以在课堂上辅助教学，提供互动体验。这些应用共同目标是提高教学效率和学习效果，为教师和学生提供更好的教育体验。"
                },
                {
                    "instruction": "解释什么是多模态交互",
                    "input": "",
                    "output": "多模态交互是指人机交互系统能够通过多种感知通道（或称模态）与用户进行交流的能力。这些模态包括视觉（如图像识别、手势识别）、听觉（如语音识别、声音识别）、触觉等。在PEPPER机器人的教学应用中，多模态交互指的是机器人能够通过摄像头识别学生的表情和动作，通过麦克风接收语音指令，并通过语音合成和肢体动作做出回应。这种多模态交互使机器人能够更自然地与学生互动，提供更丰富的教学体验。"
                },
                {
                    "instruction": "PEPPER机器人在课堂上有哪些优势",
                    "input": "",
                    "output": "PEPPER机器人在'人工智能+'课堂上具有多方面优势：首先，它拥有物理实体，与纯软件AI相比能提供更直观、生动的互动体验；其次，多模态交互能力（语音、视觉、触摸）使其能适应多样化的教学需求；第三，它能收集和分析学生的学习数据，提供个性化的学习建议；第四，拟人化的外形和动作有助于增强学生的学习兴趣和参与度；最后，它能承担部分教学任务，如知识点讲解、答疑等，为教师减轻负担。这些优势使PEPPER机器人成为'人工智能+'课堂的理想辅助工具。"
                }
            ]

        logger.info(f"准备训练数据集，共{len(instruction_data)}条记录")

        # 处理数据格式，转换为模型训练所需的格式
        def format_instruction(example):
            """将指令数据格式化为训练格式"""
            instruction = example["instruction"]
            input_text = example.get("input", "")
            output = example["output"]

            # 格式化为适合DeepSeek模型的指令格式
            if input_text:
                prompt = f"Human: {instruction}\n{input_text}\n\nAssistant: "
            else:
                prompt = f"Human: {instruction}\n\nAssistant: "

            text = prompt + output
            return {"text": text}

        # 格式化数据
        formatted_data = [format_instruction(item) for item in instruction_data]

        # 创建Dataset对象
        dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))

        # 数据集预处理：分词
        def tokenize_function(examples):
            """对文本进行分词"""
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=1024
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        return tokenized_dataset

    def train(self, dataset, epochs=3, batch_size=4, learning_rate=2e-4):
        """训练模型"""
        logger.info("开始训练过程")

        # 定义训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=10,
            save_steps=200,
            save_total_limit=3,
            fp16=(self.device == "cuda"),  # 如果使用GPU则启用fp16
            report_to="none",  # 禁用报告
            remove_unused_columns=False  # 保留所有列
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
        trainer.train()

        # 保存模型
        logger.info(f"保存微调后的模型到: {self.output_dir}")
        self.peft_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        return True

    def generate_response(self, prompt, max_new_tokens=512):
        """使用微调后的模型生成回答"""
        # 确保提示符格式正确
        if not prompt.startswith("Human:"):
            prompt = f"Human: {prompt}\n\nAssistant:"

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # 生成回答
        with torch.no_grad():
            outputs = self.peft_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取回答部分
        if "Assistant:" in response:
            response = response.split("Assistant:", 1)[1].strip()
        else:
            response = response[len(prompt):].strip()

        return response


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
