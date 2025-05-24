#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PEPPER机器人智能教学系统启动程序
"""

import argparse
import json
import logging
import os
import sys

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pepper_system.log')
    ]
)
logger = logging.getLogger("PEPPER_LAUNCH")

# 添加当前目录到路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入集成系统
from integrated_system import PepperIntegratedSystem


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

    logger.info("系统要求检查完成")
    return True


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
    parser = argparse.ArgumentParser(description='PEPPER机器人智能教学系统')
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

    args = parser.parse_args()

    # 检查系统要求
    check_system_requirements()

    # 加载配置
    config = load_config(args.config)
    if not config:
        logger.error("无法加载配置，使用默认配置")
        config = None

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
        return 1


if __name__ == "__main__":
    sys.exit(main())