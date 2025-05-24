# 在scripts/finetune_model.py中创建微调脚本

import argparse
import logging
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ai_service.llm_module.lora_fine_tuning import LoRAFineTuner

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FINETUNE_MODEL")


def main():
    parser = argparse.ArgumentParser(description='使用LoRA技术微调大语言模型')
    parser.add_argument('--model_path', type=str, default="models/deepseek-coder-1.3b-base",
                        help='基础模型路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='微调后模型输出路径（不指定则自动生成时间戳）')
    parser.add_argument('--data_path', type=str, default=None,
                        help='训练数据路径（JSON或CSV格式）')
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')

    args = parser.parse_args()

    # 如果没有指定输出目录，自动生成带时间戳的目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"models/deepseek-{timestamp}"

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建LoRA微调器实例
    fine_tuner = LoRAFineTuner(
        base_model_path=args.model_path,
        output_dir=args.output_dir
    )

    # 加载基础模型
    if fine_tuner.load_base_model():
        # 准备LoRA配置
        fine_tuner.prepare_lora_config()

        # 准备训练数据集
        if args.data_path and os.path.exists(args.data_path):
            logger.info(f"使用数据文件: {args.data_path}")
            dataset = fine_tuner.prepare_dataset(data_path=args.data_path)
        else:
            logger.info("使用默认示例数据")
            dataset = fine_tuner.prepare_dataset()

        # 训练模型
        logger.info(f"开始训练，epochs={args.epochs}, batch_size={args.batch_size}")
        fine_tuner.train(dataset, epochs=args.epochs, batch_size=args.batch_size)

        logger.info(f"模型微调完成，已保存到: {args.output_dir}")

        # 测试微调后的模型
        test_prompts = [
            "Python中for循环和while循环有什么区别？",
            "人工智能在教育中有哪些应用？",
            "什么是多模态交互？",
            "PEPPER机器人在课堂上的优势是什么？"
        ]

        for prompt in test_prompts:
            response = fine_tuner.generate_response(prompt)
            logger.info(f"测试问题: {prompt}")
            logger.info(f"模型回答: {response[:100]}...")
            logger.info("-" * 50)
    else:
        logger.error("加载基础模型失败")


if __name__ == "__main__":
    main()