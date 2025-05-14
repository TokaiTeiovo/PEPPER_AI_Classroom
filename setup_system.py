# 在setup_system.py中创建设置脚本

import argparse
import json
import logging
import os
import subprocess
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SETUP_SYSTEM")


def check_dependencies():
    """检查系统依赖"""
    try:
        # 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            logger.warning(f"Python版本过低: {python_version.major}.{python_version.minor}, 建议使用Python 3.8+")

        # 检查必要的包
        required_packages = [
            "numpy", "transformers", "torch", "langchain", "neo4j",
            "flask", "websockets", "spacy", "matplotlib"
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            logger.warning(f"缺少必要的包: {', '.join(missing_packages)}")
            logger.info("请安装这些包: pip install " + " ".join(missing_packages))
            return False

        # 检查CUDA可用性（可选）
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA可用，使用设备: {device_name}")
            else:
                logger.warning("CUDA不可用，将使用CPU模式（可能会较慢）")
        except:
            logger.warning("无法检查CUDA可用性")

        return True

    except Exception as e:
        logger.error(f"检查依赖失败: {e}")
        return False


def create_default_config(config_path="config.json"):
    """创建默认配置文件"""
    default_config = {
        "robot": {
            "ip": "127.0.0.1",
            "port": 9559,
            "simulation_mode": True
        },
        "knowledge_graph": {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        },
        "llm": {
            "model_path": "models/deepseek-coder-1.3b-base",
            "use_8bit": True
        },
        "data_paths": {
            "student_profiles": "data/student_profiles",
            "course_materials": "data/course_materials",
            "learning_analytics": "data/learning_analytics"
        },
        "websocket": {
            "url": "ws://localhost:8765"
        }
    }

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        logger.info(f"默认配置文件已创建: {config_path}")
        return True
    except Exception as e:
        # 继续在setup_system.py中创建设置脚本

        logger.error(f"创建配置文件失败: {e}")
        return False

    def setup_directories():
        """创建必要的目录结构"""
        directories = [
            "data/student_profiles",
            "data/course_materials",
            "data/learning_analytics",
            "models",
            "logs"
        ]

        try:
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
            logger.info("必要的目录结构已创建")
            return True
        except Exception as e:
            logger.error(f"创建目录结构失败: {e}")
            return False

    def check_neo4j():
        """检查Neo4j数据库"""
        try:
            # 简单检查Neo4j是否安装
            result = subprocess.run(["neo4j", "version"],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)

            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"找到Neo4j: {version}")
                return True
            else:
                logger.warning("未找到Neo4j，请确保Neo4j数据库已安装并启动")
                logger.info("可从https://neo4j.com/download/下载Neo4j")
                return False
        except Exception as e:
            logger.warning(f"检查Neo4j失败: {e}")
            logger.info("请确保Neo4j数据库已安装并启动")
            return False

    def download_model(model_name="deepseek-coder-1.3b-base"):
        """下载预训练模型"""
        try:
            from huggingface_hub import snapshot_download

            logger.info(f"正在下载模型: {model_name}")
            model_path = os.path.join("models", model_name)

            if os.path.exists(model_path):
                logger.info(f"模型已存在: {model_path}")
                return True

            snapshot_download(repo_id=model_name, local_dir=model_path)
            logger.info(f"模型下载完成: {model_path}")
            return True
        except Exception as e:
            logger.error(f"下载模型失败: {e}")
            logger.info("请手动下载模型并放置在models目录下")
            return False

    def main():
        parser = argparse.ArgumentParser(description='PEPPER机器人智能教学系统设置')
        parser.add_argument('--config', type=str, default='config.json',
                            help='配置文件路径')
        parser.add_argument('--download-model', action='store_true',
                            help='下载预训练模型')
        parser.add_argument('--check-only', action='store_true',
                            help='仅检查依赖，不执行设置')

        args = parser.parse_args()

        # 检查依赖
        dependencies_ok = check_dependencies()

        if args.check_only:
            sys.exit(0 if dependencies_ok else 1)

        # 创建目录结构
        setup_directories()

        # 创建配置文件（如果不存在）
        if not os.path.exists(args.config):
            create_default_config(args.config)

        # 检查Neo4j
        check_neo4j()

        # 下载模型（如果需要）
        if args.download_model:
            download_model()

        logger.info("系统设置完成")

    if __name__ == "__main__":
        main()