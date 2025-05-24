#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PEPPER机器人智能教学系统启动程序 - 优化版
该脚本会检测GPU状态并提供相应的优化选项
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pepper_system_optimized.log')
    ]
)
logger = logging.getLogger("PEPPER_LAUNCH_OPTIMIZED")

# 添加当前目录到路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def check_gpu_status():
    """详细检查GPU状态并尝试解决问题"""
    logger.info("正在检查GPU状态...")

    try:
        import torch

        # 检查PyTorch是否能检测到CUDA
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"PyTorch检测到 {gpu_count} 个GPU设备")
            logger.info(f"主GPU: {gpu_name}")

            # 打印CUDA版本
            cuda_version = torch.version.cuda
            logger.info(f"CUDA版本: {cuda_version}")

            # 打印每个GPU的信息
            for i in range(gpu_count):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

            # 尝试在GPU上执行一个简单操作以确认其正常工作
            try:
                test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
                test_result = test_tensor * 2
                logger.info(f"GPU测试成功: {test_result.cpu().numpy()}")

                # 设置环境变量，优先使用GPU
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                return True, "GPU可用且工作正常"
            except Exception as e:
                logger.error(f"GPU测试失败: {e}")
                return False, f"检测到GPU但无法使用: {e}"
        else:
            # 尝试获取系统GPU信息
            logger.warning("PyTorch未检测到GPU，尝试获取系统GPU信息...")

            # 检查操作系统
            system = platform.system()
            if system == "Windows":
                try:
                    # 使用Windows系统命令获取GPU信息
                    result = subprocess.run(["wmic", "path", "win32_VideoController", "get", "name"],
                                            capture_output=True, text=True, check=True)
                    gpu_info = result.stdout.strip()
                    logger.info(f"系统GPU信息:\n{gpu_info}")

                    # 检查是否需要安装CUDA
                    logger.warning("可能需要安装或更新NVIDIA驱动和CUDA")
                    logger.warning("请访问 https://developer.nvidia.com/cuda-downloads 下载安装CUDA")
                except Exception as e:
                    logger.error(f"获取系统GPU信息失败: {e}")
            elif system == "Linux":
                try:
                    # 使用Linux系统命令获取GPU信息
                    result = subprocess.run(["lspci", "|", "grep", "-i", "nvidia"],
                                            shell=True, capture_output=True, text=True)
                    gpu_info = result.stdout.strip()
                    logger.info(f"系统GPU信息:\n{gpu_info}")

                    # 尝试检查NVIDIA驱动
                    try:
                        nvidia_result = subprocess.run(["nvidia-smi"],
                                                       capture_output=True, text=True)
                        logger.info(f"NVIDIA-SMI输出:\n{nvidia_result.stdout}")
                    except:
                        logger.warning("无法执行nvidia-smi，可能未安装NVIDIA驱动")
                except Exception as e:
                    logger.error(f"获取系统GPU信息失败: {e}")

            return False, "未检测到可用的GPU"
    except ImportError:
        logger.error("未安装PyTorch，无法检测GPU状态")
        return False, "未安装PyTorch"
    except Exception as e:
        logger.error(f"检查GPU状态时发生错误: {e}")
        return False, f"检查出错: {e}"


def optimize_for_cpu():
    """优化CPU模式下的运行配置"""
    logger.info("正在优化CPU运行配置...")

    try:
        import torch

        # 设置PyTorch使用最大CPU线程
        torch.set_num_threads(os.cpu_count())
        logger.info(f"PyTorch线程数设置为: {os.cpu_count()}")

        # 禁用CUDA以确保使用CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # 如果是Intel CPU，尝试启用MKL优化
        try:
            # 设置环境变量启用MKL
            os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
            os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
            logger.info("已启用MKL优化")
        except:
            pass

        logger.info("CPU优化配置完成")
        return True
    except Exception as e:
        logger.error(f"优化CPU配置失败: {e}")
        return False


def check_system_requirements():
    """检查系统要求"""
    logger.info("检查系统要求...")

    # 检查文件夹
    required_dirs = ["data", "models", "logs"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            logger.info(f"创建目录: {dir_name}")

    # 检查配置文件
    if not os.path.exists("config.json"):
        logger.warning("未找到配置文件，将使用默认配置")
        return False

    # 检查模型文件
    model_path = "models/deepseek-coder-1.3b-base"
    if not os.path.exists(model_path):
        logger.warning(f"未找到模型文件: {model_path}")
        logger.warning("请确保已下载模型文件")

    logger.info("系统要求检查完成")
    return True


def update_config_for_system(config, use_gpu=False):
    """根据系统状态更新配置"""
    if config:
        # 更新LLM配置
        if "llm" in config:
            if use_gpu:
                config["llm"]["use_gpu"] = True
                config["llm"]["use_8bit"] = True
                config["llm"]["device"] = "cuda"
                logger.info("配置已更新为GPU模式")
            else:
                config["llm"]["use_gpu"] = False
                config["llm"]["device"] = "cpu"
                logger.info("配置已更新为CPU模式")

        # 更新其他配置项
        if not use_gpu and "llm" in config:
            # 如果在CPU上，减少批处理大小等参数以降低内存使用
            if "batch_size" in config["llm"]:
                config["llm"]["batch_size"] = min(config["llm"].get("batch_size", 4), 2)

            # 可能需要使用较小的模型
            if config["llm"].get("model_path", "").endswith("1.3b-base"):
                # 已经是小模型，不需要更改
                pass
            elif "model_path" in config["llm"] and not config["llm"]["model_path"].endswith("1.3b-base"):
                logger.warning(f"当前模型可能较大，考虑使用较小的模型以在CPU上获得更好性能")

    return config


def load_config(config_path="config.json"):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return None


def connect_to_robot(config):
    """连接到PEPPER机器人"""
    logger.info("正在连接PEPPER机器人...")

    if config["robot"]["simulation_mode"]:
        logger.info("使用模拟模式，不连接实际机器人")
        return True

    try:
        # 导入机器人控制模块
        from pepper_robot.robot_control.robot_controller import PepperRobot

        # 尝试连接机器人
        robot = PepperRobot(config["robot"]["ip"], config["robot"]["port"])
        robot.say("机器人连接成功")
        logger.info("成功连接到PEPPER机器人")
        return True
    except Exception as e:
        logger.error(f"连接机器人失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PEPPER机器人智能教学系统 - 优化版')
    parser.add_argument('--config', type=str, default='config.json',
                        help='配置文件路径')
    parser.add_argument('--simulation', action='store_true',
                        help='使用模拟模式（不连接实际机器人）')
    parser.add_argument('--student', type=str, default=None,
                        help='学生ID')
    parser.add_argument('--topic', type=str, default="Python编程",
                        help='教学主题')
    parser.add_argument('--interaction-count', type=int, default=10,
                        help='交互次数')
    parser.add_argument('--check-only', action='store_true',
                        help='仅检查系统，不启动')
    parser.add_argument('--force-cpu', action='store_true',
                        help='强制使用CPU模式')
    parser.add_argument('--optimize', action='store_true',
                        help='使用优化选项')

    args = parser.parse_args()

    # 显示系统信息
    logger.info(f"操作系统: {platform.system()} {platform.version()}")
    logger.info(f"Python版本: {platform.python_version()}")
    try:
        import torch
        logger.info(f"PyTorch版本: {torch.__version__}")
    except ImportError:
        logger.warning("未安装PyTorch")

    # 检查GPU状态
    use_gpu = False
    if not args.force_cpu:
        use_gpu, gpu_message = check_gpu_status()
        logger.info(f"GPU状态: {gpu_message}")
    else:
        logger.info("已强制指定使用CPU模式")

    # 如果使用CPU，进行CPU优化
    if not use_gpu:
        optimize_for_cpu()

    # 检查系统要求
    check_system_requirements()

    # 加载配置
    config = load_config(args.config)
    if not config:
        logger.error("无法加载配置，使用默认配置")
        config = None

    # 更新配置
    config = update_config_for_system(config, use_gpu)

    # 如果使用模拟模式，更新配置
    if args.simulation and config:
        config["robot"]["simulation_mode"] = True
        logger.info("已启用模拟模式")

    # 如果只检查系统，则到此为止
    if args.check_only:
        logger.info("系统检查完成")
        return 0

    # 尝试连接机器人
    if config and not connect_to_robot(config):
        logger.warning("无法连接PEPPER机器人，将使用模拟模式")
        if config:
            config["robot"]["simulation_mode"] = True

    try:
        # 导入集成系统之前设置优化参数
        if args.optimize:
            logger.info("启用系统优化选项")
            # 设置缓存目录
            os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.path.dirname(__file__), "cache")
            # 减少内存使用
            os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

        # 导入集成系统
        logger.info("正在导入集成系统模块...")
        from integrated_system import PepperIntegratedSystem

        # 创建集成系统
        logger.info("初始化集成系统...")
        integrated_system = PepperIntegratedSystem(config)

        # 准备教学会话
        logger.info(f"准备教学会话，学生ID: {args.student}, 主题: {args.topic}")
        if not integrated_system.prepare_teaching_session(args.student, args.topic):
            logger.error("教学会话准备失败")
            return 1

        # 启动教学会话
        welcome_message = integrated_system.start_teaching_session()
        logger.info(f"教学会话已启动: {welcome_message}")

        # 进入交互循环
        logger.info(f"开始教学交互，计划交互次数: {args.interaction_count}")
        integrated_system.interactive_teaching_loop(args.interaction_count)

        # 结束教学会话
        logger.info("教学会话结束")
        integrated_system.stop_teaching_session()

        # 清理资源
        integrated_system.clean_up()
        logger.info("系统已正常退出")

        return 0

    except KeyboardInterrupt:
        logger.info("收到终止信号，系统退出")
        return 0
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())