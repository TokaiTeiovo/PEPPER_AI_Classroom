#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PEPPER机器人智能教学系统 - Web应用启动器
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
import webbrowser

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pepper_web_app.log')
    ]
)
logger = logging.getLogger("PEPPER_WEB_APP")


def check_dependencies():
    """检查依赖项"""
    logger.info("检查依赖项...")

    required_packages = [
        'flask', 'flask_cors', 'transformers', 'torch', 'neo4j'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('_', '-'))
        except ImportError:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

    if missing_packages:
        logger.warning(f"缺少依赖包: {missing_packages}")
        logger.info("请运行: pip install " + " ".join(missing_packages))
        return False

    logger.info("依赖项检查完成")
    return True


def setup_directories():
    """创建必要的目录"""
    directories = [
        "data/student_profiles",
        "data/course_materials",
        "data/learning_analytics",
        "logs",
        "models"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    logger.info("目录结构创建完成")


def load_config(config_path="config.json"):
    """加载配置文件"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        else:
            logger.warning(f"配置文件不存在: {config_path}")
            return None
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return None


def start_web_console(config):
    """启动Web控制台"""
    try:
        # 导入Web控制台应用
        from interface.web_console.app import app

        # 从配置获取参数
        web_config = config.get("web_console", {})
        host = web_config.get("host", "0.0.0.0")
        port = web_config.get("port", 5000)
        debug = web_config.get("debug", True)

        logger.info(f"启动Web控制台: http://{host}:{port}")

        # 启动Flask应用
        app.run(host=host, port=port, debug=debug, use_reloader=False)

    except Exception as e:
        logger.error(f"启动Web控制台失败: {e}")
        raise


def open_browser(url, delay=2):
    """延迟打开浏览器"""

    def open_browser_delayed():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            logger.info(f"已在浏览器中打开: {url}")
        except Exception as e:
            logger.warning(f"自动打开浏览器失败: {e}")

    thread = threading.Thread(target=open_browser_delayed)
    thread.daemon = True
    thread.start()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PEPPER机器人智能教学系统 - Web版')
    parser.add_argument('--config', type=str, default='config.json',
                        help='配置文件路径')
    parser.add_argument('--host', type=str, default=None,
                        help='Web服务器主机地址')
    parser.add_argument('--port', type=int, default=None,
                        help='Web服务器端口')
    parser.add_argument('--no-browser', action='store_true',
                        help='不自动打开浏览器')
    parser.add_argument('--check-only', action='store_true',
                        help='仅检查环境，不启动服务')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PEPPER机器人智能教学系统 - Web版启动中...")
    logger.info("=" * 60)

    # 检查依赖项
    if not check_dependencies():
        logger.error("依赖项检查失败，请安装缺少的包")
        return 1

    # 创建目录结构
    setup_directories()

    # 加载配置
    config = load_config(args.config)
    if not config:
        # 使用默认配置
        config = {
            "robot": {"simulation_mode": True, "web_control": True},
            "web_console": {"host": "0.0.0.0", "port": 5000, "debug": True}
        }
        logger.info("使用默认配置")

    # 命令行参数覆盖配置
    if args.host:
        config.setdefault("web_console", {})["host"] = args.host
    if args.port:
        config.setdefault("web_console", {})["port"] = args.port

    # 确保使用模拟模式
    config["robot"]["simulation_mode"] = True
    config["robot"]["web_control"] = True

    if args.check_only:
        logger.info("环境检查完成，系统就绪")
        return 0

    # 准备Web服务器URL
    web_config = config.get("web_console", {})
    host = web_config.get("host", "0.0.0.0")
    port = web_config.get("port", 5000)

    # 为本地访问准备URL
    if host == "0.0.0.0":
        browser_url = f"http://localhost:{port}"
    else:
        browser_url = f"http://{host}:{port}"

    # 自动打开浏览器
    if not args.no_browser:
        open_browser(browser_url)

    logger.info(f"Web控制台将在以下地址启动: {browser_url}")
    logger.info("按 Ctrl+C 停止服务器")

    try:
        # 启动Web控制台
        start_web_console(config)
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭服务器...")
        return 0
    except Exception as e:
        logger.error(f"服务器运行出错: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())