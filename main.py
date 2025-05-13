import os
import subprocess
import sys
import threading
import time

# 添加当前目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入各个模块
from interface.bridge.websocket_bridge import WebSocketBridge
from interface.api.api_server import app as api_app


def start_websocket_bridge():
    """启动WebSocket桥接服务"""
    print("正在启动WebSocket桥接服务...")
    bridge = WebSocketBridge()
    bridge.run()


def start_api_server():
    """启动API服务"""
    print("正在启动API服务...")
    api_app.run(host='0.0.0.0', port=5000)


def start_pepper_controller():
    """启动PEPPER机器人控制器（在单独的进程中运行Python 2.7代码）"""
    print("正在启动PEPPER机器人控制器...")
    pepper_script_path = os.path.join(os.path.dirname(__file__), 'pepper_controller.py')

    # 使用Python 2.7执行PEPPER控制脚本
    pepper_python_path = os.path.join(os.path.dirname(__file__), 'venv_pepper', 'Scripts', 'python.exe')
    subprocess.Popen([pepper_python_path, pepper_script_path])


def main():
    """主函数，启动所有服务"""
    print("正在启动PEPPER机器人智能教学系统...")

    # 创建并启动WebSocket桥接服务线程
    bridge_thread = threading.Thread(target=start_websocket_bridge)
    bridge_thread.daemon = True
    bridge_thread.start()

    # 创建并启动API服务线程
    api_thread = threading.Thread(target=start_api_server)
    api_thread.daemon = True
    api_thread.start()

    # 启动PEPPER机器人控制器
    start_pepper_controller()

    print("所有服务已启动，系统正在运行...")

    try:
        # 保持主线程运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("接收到中断信号，正在关闭系统...")
        sys.exit(0)


if __name__ == "__main__":
    main()
