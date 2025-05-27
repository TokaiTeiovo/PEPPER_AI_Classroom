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

        # 只有明确指定output_dir时才创建目录
        if output_dir is not None:
            self.output_dir = output_dir
            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"模型将保存到: {self.output_dir}")
        else:
            # 如果没有指定output_dir，不创建任何目录
            self.output_dir = None
            logger.info("推理模式，不创建输出目录")

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def load_base_model(self, use_4bit=True, use_8bit=False):
        """加载基础模型"""
        try:
            logger.info(f"正在加载基础模型: {self.base_model_path}")

            # 检查模型路径是否存在
            if not os.path.exists(self.base_model_path):
                logger.error(f"模型路径不存在: {self.base_model_path}")
                return False

            # 加载分词器
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_path,
                    trust_remote_code=True,
                )
                logger.info("分词器加载成功")
            except Exception as e:
                logger.error(f"分词器加载失败: {e}")
                return False

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
                logger.info("开始加载模型...")
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
                    try:
                        memory_mb = self.model.get_memory_footprint() / 1024 / 1024
                        logger.info(f"模型显存占用: {memory_mb:.2f} MB")
                    except:
                        pass

                return True

            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                return False

        except Exception as e:
            logger.error(f"load_base_model 执行失败: {e}")
            return False

    def prepare_lora_config(self):
        """准备LoRA配置"""
        try:
            logger.info("配置LoRA参数")

            # 为DeepSeek模型获取正确的目标模块
            if hasattr(self.model, "config") and hasattr(self.model.config, "model_type"):
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

        except Exception as e:
            logger.error(f"LoRA配置失败: {e}")
            return None

    def prepare_dataset(self, data_path=None, instruction_data=None):
        """
        准备训练数据集 - 优化中文对话格式
        支持从 JSON 或 CSV 读取 [{instruction, input, output}] 格式
        """
        try:
            if data_path and os.path.exists(data_path):
                if data_path.endswith('.json'):
                    import json
                    with open(data_path, 'r', encoding='utf-8') as f:
                        instruction_data = json.load(f)
                elif data_path.endswith('.csv'):
                    import pandas as pd
                    instruction_data = pd.read_csv(data_path).to_dict('records')

            if not instruction_data:
                # 中文示例数据
                instruction_data = [
                    {
                        "instruction": "解释Python中for循环和while循环的区别",
                        "input": "",
                        "output": "您好！for循环用于遍历已知序列，while循环基于条件判断。for循环次数确定，while循环次数不确定。for循环语法简洁，while循环需要手动管理循环变量。"
                    },
                    {
                        "instruction": "你好",
                        "input": "",
                        "output": "您好！我是PEPPER智能教学助手，很高兴为您服务！有什么可以帮助您的吗？"
                    },
                    {
                        "instruction": "介绍一下自己",
                        "input": "",
                        "output": "我是PEPPER智能教学助手，基于DeepSeek模型。我可以帮助您学习编程、人工智能、数学等知识，提供个性化的学习建议。"
                    }
                ]

            logger.info(f"准备训练数据集，共{len(instruction_data)}条记录")

            # 转换为中文对话格式
            def format_instruction(example):
                instruction = example["instruction"]
                input_text = example.get("input", "")
                output = example["output"]

                # 使用中文格式的对话模板
                if input_text.strip():
                    prompt = f"用户: {instruction}\n补充信息: {input_text}\n\n助手: "
                else:
                    prompt = f"用户: {instruction}\n\n助手: "

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
            tokenized_dataset = tokenized_dataset.remove_columns(["text"])

            return tokenized_dataset

        except Exception as e:
            logger.error(f"数据集准备失败: {e}")
            return None

        def generate_response(self, prompt, max_new_tokens=512):
            """使用微调后的模型生成回答 - 强化中文输出"""
            if not self.peft_model:
                logger.error("模型未加载")
                return "模型未加载，无法生成回答"

            try:
                # 为DeepSeek模型调整中文提示格式
                if not prompt.startswith("用户:"):
                    system_prompt = "你是PEPPER智能教学助手，请用中文回答所有问题。"
                    prompt = f"{system_prompt}\n用户: {prompt}\n\n助手: "

                # 编码输入
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                # 设置生成参数，优化中文生成
                gen_kwargs = {
                    "max_new_tokens": min(max_new_tokens, 200),  # 限制长度
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "top_k": 40,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "repetition_penalty": 1.2,
                    "no_repeat_ngram_size": 3,
                }

                # 生成回答
                with torch.no_grad():
                    output_ids = self.peft_model.generate(**inputs, **gen_kwargs)

                # 解码输出
                full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

                # 提取回答部分
                if "助手:" in full_output:
                    response = full_output.split("助手:")[-1].strip()
                else:
                    response = full_output[len(prompt):].strip()

                # 限制回答长度，避免过长输出
                if len(response) > 300:
                    response = response[:300] + "..."

                return response

            except Exception as e:
                logger.error(f"生成回答失败: {e}")
                return "抱歉，我现在无法回答这个问题。请稍后再试。"

    def train(self, dataset, epochs=3, batch_size=4, learning_rate=2e-4, progress_callback=None):
        """训练模型"""
        try:
            logger.info("开始训练过程")

            # 检查是否收到了进度回调
            if progress_callback:
                logger.info("已接收到进度回调函数")
            else:
                logger.warning("未接收到进度回调函数")

            if not self.peft_model:
                logger.error("PEFT模型未初始化")
                return False

            if not self.peft_model:
                logger.error("PEFT模型未初始化")
                return False

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
                logging_steps=1,
                save_steps=500,
                save_total_limit=2,
                fp16=(self.device == "cuda"),
                bf16=False,
                report_to="none",
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                gradient_checkpointing=True,
                optim="adamw_torch",
            )

            # 创建数据收集器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # 不使用掩码语言模型训练
            )

            # 自定义进度回调类
            from transformers import TrainerCallback

            class ProgressCallback(TrainerCallback):
                def __init__(self, callback_func):
                    self.callback_func = callback_func
                    self.last_reported_progress = 0
                    logger.info("ProgressCallback 初始化完成")

                def on_train_begin(self, args, state, control, **kwargs):
                    logger.info(f"训练开始 - 总步数: {state.max_steps}")
                    if self.callback_func:
                        self.callback_func(0)  # 开始时设置为0%

                def on_step_end(self, args, state, control, **kwargs):
                    if self.callback_func and state.max_steps > 0:
                        # 计算真实进度百分比
                        progress_percent = (state.global_step / state.max_steps) * 100

                        # 只有进度变化超过1%才报告，避免过于频繁
                        if progress_percent - self.last_reported_progress >= 1.0 or state.global_step == state.max_steps:
                            logger.info(
                                f"训练步骤: {state.global_step}/{state.max_steps}, 进度: {progress_percent:.1f}%")
                            self.callback_func(progress_percent)
                            self.last_reported_progress = progress_percent

                def on_epoch_end(self, args, state, control, **kwargs):
                    if self.callback_func and state.max_steps > 0:
                        progress_percent = (state.global_step / state.max_steps) * 100
                        self.callback_func(progress_percent)

                def on_train_end(self, args, state, control, **kwargs):
                    logger.info("训练结束")
                    if self.callback_func:
                        self.callback_func(100)  # 训练结束时设置为100%

            # 创建训练器
            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )

            # 如果有进度回调，添加回调
            if progress_callback:
                logger.info("添加进度回调到训练器")
                progress_cb = ProgressCallback(progress_callback)
                trainer.add_callback(progress_cb)
            else:
                logger.warning("没有进度回调函数，将无法更新训练进度")

            # 开始训练
            logger.info("开始LoRA微调训练")
            trainer.train()

            # 保存模型
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)

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

            logger.info("LoRA微调训练完成")
            return True

        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            return False

    def generate_response(self, prompt, max_new_tokens=512):
        """使用微调后的模型生成回答"""
        if not self.peft_model:
            logger.error("模型未加载")
            return "模型未加载，无法生成回答"

        try:
            # 为DeepSeek模型调整提示格式 - 简化格式
            if not prompt.startswith("Human:"):
                formatted_prompt = f"Human: {prompt}\n\nAssistant:"
            else:
                formatted_prompt = prompt

            # 编码输入
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

            # 优化生成参数 - 专为教学问答优化
            gen_kwargs = {
                "max_new_tokens": min(150, max_new_tokens),  # 限制长度，避免冗长回复
                "temperature": 0.3,  # 降低随机性，提高一致性
                "top_p": 0.8,  # 更保守的nucleus sampling
                "top_k": 30,  # 减少候选词数量
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.3,  # 增强重复惩罚
                "length_penalty": 1.0,  # 长度惩罚
                "early_stopping": True,  # 启用早停
                "no_repeat_ngram_size": 3,  # 避免3-gram重复
            }

            # 生成回答
            with torch.no_grad():
                output_ids = self.peft_model.generate(**inputs, **gen_kwargs)

            # 解码输出 - 只取新生成的部分
            new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # 后处理：清理和优化回复
            response = self._post_process_response(response)

            return response

        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return "抱歉，我现在无法回答这个问题。"

    def _post_process_response(self, response):
        """后处理生成的回复"""
        if not response:
            return "抱歉，我没有理解您的问题。"

        # 移除多余的空白
        response = response.strip()

        # 如果响应为空或过短
        if len(response) < 3:
            return "抱歉，我需要更多信息来回答您的问题。"

        # 移除可能的重复句子
        sentences = response.split('。')
        unique_sentences = []
        seen = set()

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen and len(sentence) > 5:
                unique_sentences.append(sentence)
                seen.add(sentence)

        # 重新组合句子
        if unique_sentences:
            cleaned_response = '。'.join(unique_sentences)
            if not cleaned_response.endswith('。') and not cleaned_response.endswith(
                    '！') and not cleaned_response.endswith('？'):
                cleaned_response += '。'
            return cleaned_response

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