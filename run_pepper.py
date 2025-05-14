# run_pepper.py
import subprocess
import sys

# 确保当前环境中有基本依赖
required_packages = ["neo4j", "numpy", "pandas"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"缺少必要依赖: {package}")
        sys.exit(1)

# 运行主程序
subprocess.run([sys.executable, "run_pepper_system.py"] + sys.argv[1:])
