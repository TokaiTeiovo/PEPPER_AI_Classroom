#!/usr/bin/env python
"""
PEPPER机器人智能教学系统启动脚本

用法：
    python run_pepper_system.py [参数]

参数：
    --config CONFIG       配置文件路径
    --student STUDENT     学生ID
    --topic TOPIC         教学主题
    --simulation          使用模拟模式（不连接实际机器人）
    --interactions NUM    交互次数
    --test                运行系统测试
    --init-kg             初始化知识图谱
    --monitor-gpu         监控GPU使用

例如：
    python run_pepper_system.py --simulation --topic "Python编程"
    python run_pepper_system.py --student "001" --interactions 15
    python run_pepper_system.py --test
"""

import argparse
import json
import logging
import os
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入集成系统
from integrated_system import PepperIntegratedSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pepper_system.log')
    ]
)
logger = logging.getLogger("PEPPER_SYSTEM")


def run_system(args):
    """运行智能教学系统"""
    logger.info("正在启动PEPPER机器人智能教学系统...")

    # 加载配置
    config = None
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 如果指定了模拟模式，覆盖配置
            if args.simulation:
                config["robot"]["simulation_mode"] = True

            logger.info(f"已加载配置文件: {args.config}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            config = None
    else:
        logger.warning(f"配置文件不存在: {args.config}，将使用默认配置")

    try:
        # 创建并运行系统
        system = PepperIntegratedSystem(config)
        success = system.run(args.student, args.topic, args.interactions)

        if success:
            logger.info("PEPPER机器人智能教学系统运行完成")
            return 0
        else:
            logger.error("PEPPER机器人智能教学系统运行失败")
            return 1
    except Exception as e:
        logger.error(f"PEPPER机器人智能教学系统运行出错: {e}")
        return 1


def run_system_test():
    """运行系统测试"""
    logger.info("正在运行PEPPER机器人智能教学系统测试...")

    try:
        from tests.system_test import run_tests
        success = run_tests()

        if success:
            logger.info("系统测试通过")
            return 0
        else:
            logger.error("系统测试失败")
            return 1
    except ImportError:
        logger.error("未找到系统测试模块")
        return 1
    except Exception as e:
        logger.error(f"系统测试出错: {e}")
        return 1


def init_knowledge_graph(config_path="config.json"):
    """初始化知识图谱"""
    logger.info("正在初始化知识图谱...")

    try:
        # 加载配置
        config = None
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
                return 1
        else:
            logger.warning(f"配置文件不存在: {config_path}，将使用默认配置")

        # 初始化知识图谱
        from ai_service.knowledge_graph.education_knowledge_processor import EducationKnowledgeProcessor

        # 获取知识图谱配置
        kg_config = config["knowledge_graph"] if config else {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        }

        # 创建知识处理器
        processor = EducationKnowledgeProcessor(
            kg_config["uri"], kg_config["user"], kg_config["password"]
        )

        # 创建教育知识库
        count = processor.create_educational_knowledge_base()

        if count > 0:
            logger.info(f"知识图谱初始化完成，添加了{count}个知识点")
            return 0
        else:
            logger.error("知识图谱初始化失败")
            return 1
    except Exception as e:
        logger.error(f"知识图谱初始化出错: {e}")
        return 1


def monitor_gpu():
    """监控GPU使用情况"""
    logger.info("正在启动GPU监控...")

    try:
        from monitor_gpu import monitor_gpu
        monitor_gpu()
        return 0
    except ImportError:
        logger.error("未找到GPU监控模块")
        return 1
    except Exception as e:
        logger.error(f"GPU监控出错: {e}")
        return 1


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PEPPER机器人智能教学系统')
    parser.add_argument('--config', type=str, default='config.json',
                        help='配置文件路径')
    parser.add_argument('--student', type=str, default=None,
                        help='学生ID')
    parser.add_argument('--topic', type=str, default=None,
                        help='教学主题')
    parser.add_argument('--simulation', action='store_true',
                        help='使用模拟模式（不连接实际机器人）')
    parser.add_argument('--interactions', type=int, default=10,
                        help='交互次数')
    parser.add_argument('--test', action='store_true',
                        help='运行系统测试')
    parser.add_argument('--init-kg', action='store_true',
                        help='初始化知识图谱')
    parser.add_argument('--monitor-gpu', action='store_true',
                        help='监控GPU使用')

    args = parser.parse_args()

    # 根据参数执行不同操作
    if args.test:
        return run_system_test()
    elif args.init_kg:
        return init_knowledge_graph(args.config)
    elif args.monitor_gpu:
        return monitor_gpu()
    else:
        return run_system(args)


if __name__ == "__main__":
    sys.exit(main())
