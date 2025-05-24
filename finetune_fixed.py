#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用于微调的脚本 - 修复版
专门处理Neo4j转换的数据
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DefaultDataCollator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('finetune.log')
    ]
)
logger = logging.getLogger("FINETUNE_SCRIPT")


def load_data(data_path):
    """加载微调数据"""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"从 {data_path} 加载了 {len(data)} 条数据")

        # 验证数据格式
        for i, item in enumerate(data[:5]):
            if not all(k in item for k in ["instruction", "input", "output"]):
                logger.error(f"数据项 {i} 格式不正确: {item}")
                raise ValueError(f"数据格式不正确")

        return data
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise


def preprocess_data(tokenizer, data, max_length=512):
    """预处理数据"""

    def format_prompt(item):
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item["output"]

        if input_text:
            prompt = f"Human: {instruction}\n{input_text}\n\nAssistant: {output}"
        else:
            prompt = f"Human: {instruction}\n\nAssistant: {output}"

        return prompt

    # 创建训练对
    train_samples = []
    for item in data:
        formatted_prompt = format_prompt(item)
        train_samples.append({"text": formatted_prompt})

    # 创建数据集
    df = pd.DataFrame(train_samples)
    dataset = Dataset.from_pandas(df)

    # 定义标记化函数
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None  # 重要: 修复错误
        )

        # 创建标签
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    # 标记化数据集
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description='微调脚本')
    parser.add_argument('--model_path', type=str, default="models/deepseek-coder-1.3b-base",
                        help='基础模型路径')
    parser.add_argument('--output_dir', type=str, default="models/deepseek-finetuned",
                        help='微调后的模型输出路径')
    parser.add_argument('--data_path', type=str, default="neo4j_data/fine_tuning_data.json",
                        help='训练数据路径')
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # 加载数据
        data = load_data(args.data_path)

        # 加载模型和分词器
        logger.info(f"加载模型: {args.model_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True
        )

        # 确保分词器有padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # 预处理数据
        tokenized_dataset = preprocess_data(tokenizer, data, max_length=args.max_length)

        # 设置LoRA配置
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=target_modules
        )

        # 应用LoRA
        model = get_peft_model(model, peft_config)
        logger.info(f"可训练参数: {model.print_trainable_parameters()}")

        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=20,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            report_to="none",
            remove_unused_columns=False
        )

        # 创建数据收集器
        data_collator = DefaultDataCollator()

        # 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # 开始训练
        logger.info("开始训练")
        trainer.train()

        # 保存模型
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"模型已保存至: {args.output_dir}")

        return 0

    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())