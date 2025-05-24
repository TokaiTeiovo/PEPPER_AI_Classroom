#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PEPPER智能教学系统 - 主启动脚本
集成大语言模型、知识图谱、多模态交互、智能教学四大功能模块
"""

import argparse
import logging
import os
import sys
import threading
import time
import webbrowser

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 直接使用API服务器的Flask应用
from interface.api.enhanced_api_server import app as main_app, initialize_services

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pepper_system.log')
    ]
)
logger = logging.getLogger("PEPPER_MAIN")


def check_dependencies():
    """检查系统依赖"""
    logger.info("检查系统依赖...")

    required_packages = [
        'flask', 'flask_cors', 'transformers', 'torch', 'jieba', 'spacy'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.warning(f"缺少以下依赖包: {missing_packages}")
        logger.info("请运行以下命令安装:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False

    logger.info("依赖检查完成")
    return True


def setup_directories():
    """创建必要的目录"""
    directories = [
        'data/student_profiles',
        'data/course_materials',
        'data/learning_analytics',
        'uploads',
        'reports',
        'logs',
        'models'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    logger.info("目录结构创建完成")


def check_model_availability():
    """检查模型可用性"""
    model_path = "models/deepseek-coder-1.3b-base"

    if os.path.exists(model_path):
        logger.info(f"找到模型: {model_path}")
        return True
    else:
        logger.warning(f"未找到模型: {model_path}")
        logger.info("请确保已下载DeepSeek模型，或修改config.json中的模型路径")
        return False


def check_neo4j_availability():
    """检查Neo4j可用性"""
    try:
        from neo4j import GraphDatabase

        # 尝试连接Neo4j（使用默认配置）
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "admin123"))
        with driver.session() as session:
            result = session.run("RETURN 1")
            result.single()
        driver.close()

        logger.info("Neo4j数据库连接正常")
        return True
    except Exception as e:
        logger.warning(f"Neo4j连接失败: {e}")
        logger.info("请确保Neo4j数据库已启动，或在系统中手动连接")
        return False


def open_browser_delayed(url, delay=3):
    """延迟打开浏览器"""

    def open_browser():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            logger.info(f"已在浏览器中打开: {url}")
        except Exception as e:
            logger.warning(f"自动打开浏览器失败: {e}")

    thread = threading.Thread(target=open_browser)
    thread.daemon = True
    thread.start()


def print_system_info():
    """打印系统信息"""
    logger.info("=" * 80)
    logger.info("PEPPER智能教学系统")
    logger.info("=" * 80)
    logger.info("功能模块:")
    logger.info("  🧠 大语言模型集成 - DeepSeek模型接口、LoRA微调")
    logger.info("  🗂️ 知识图谱系统 - Neo4j数据库、教育知识处理")
    logger.info("  🎯 多模态交互 - 语音识别、图像识别、文本处理")
    logger.info("  📚 智能教学功能 - 个性化学习路径、资源推荐、报告生成")
    logger.info("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PEPPER智能教学系统')
    parser.add_argument('--host', type=str, default='localhost',
                        help='服务器主机地址')
    parser.add_argument('--port', type=int, default=5000,
                        help='服务器端口')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    parser.add_argument('--no-browser', action='store_true',
                        help='不自动打开浏览器')
    parser.add_argument('--check-only', action='store_true',
                        help='仅检查系统环境')

    args = parser.parse_args()

    # 打印系统信息
    print_system_info()

    # 检查依赖
    if not check_dependencies():
        logger.error("依赖检查失败，请安装缺少的包后重试")
        return 1

    # 创建目录结构
    setup_directories()

    # 检查模型和数据库
    model_available = check_model_availability()
    neo4j_available = check_neo4j_availability()

    if args.check_only:
        logger.info("系统环境检查完成")
        logger.info(f"  模型可用: {'✓' if model_available else '✗'}")
        logger.info(f"  Neo4j可用: {'✓' if neo4j_available else '✗'}")
        return 0

    try:
        # 初始化服务
        logger.info("正在初始化系统服务...")
        initialize_services()

        # 准备服务器URL
        server_url = f"http://{args.host}:{args.port}"

        # 自动打开浏览器
        if not args.no_browser:
            open_browser_delayed(server_url)

        logger.info("=" * 80)
        logger.info("系统启动成功！")
        logger.info(f"访问地址: {server_url}")
        logger.info("功能说明:")
        logger.info("  • 大语言模型: 加载DeepSeek模型，进行LoRA微调")
        logger.info("  • 知识图谱: 连接Neo4j，导入教育知识")
        logger.info("  • 多模态交互: 语音、图像、文本处理")
        logger.info("  • 智能教学: 个性化学习路径和资源推荐")
        logger.info("=" * 80)
        logger.info("按 Ctrl+C 停止服务器")

        # 启动Flask应用
        main_app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=False  # 避免重复初始化
        )

    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭系统...")
        return 0
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())