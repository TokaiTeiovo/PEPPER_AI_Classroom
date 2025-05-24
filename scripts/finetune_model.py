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
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批次大小（4bit训练推荐使用2）')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                        help='使用4bit量化训练（默认启用）')
    parser.add_argument('--use_8bit', action='store_true', default=False,
                        help='使用8bit量化训练')
    parser.add_argument('--no_quantization', action='store_true', default=False,
                        help='不使用量化训练')

    args = parser.parse_args()

    # 确定量化设置
    use_4bit = args.use_4bit and not args.no_quantization
    use_8bit = args.use_8bit and not args.no_quantization and not use_4bit

    # 如果没有指定输出目录，自动生成带时间戳的目录
    if args.no_quantization:
        logger.info("使用标准精度训练（无量化）")
    elif use_4bit:
        logger.info("使用4bit量化训练")
    elif use_8bit:
        logger.info("使用8bit量化训练")

    # 如果没有指定输出目录，自动生成带时间戳的目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quantization_suffix = "_4bit" if use_4bit else "_8bit" if use_8bit else ""
        args.output_dir = f"models/deepseek-{timestamp}{quantization_suffix}"

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建LoRA微调器实例
    fine_tuner = LoRAFineTuner(
        base_model_path=args.model_path,
        output_dir=args.output_dir
    )

    # 如果没有指定输出目录，自动生成带时间戳的目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quantization_suffix = "_4bit" if use_4bit else "_8bit" if use_8bit else ""
        args.output_dir = f"models/deepseek-{timestamp}{quantization_suffix}"


if __name__ == "__main__":
    main()