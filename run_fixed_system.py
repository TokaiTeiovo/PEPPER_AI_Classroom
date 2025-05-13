"""
启动修复版PEPPER AI系统
"""
import subprocess
import sys
import time
import os


def main():
    print("=" * 60)
    print("修复版PEPPER机器人智能教学系统")
    print("解决GPU使用和回答质量问题")
    print("=" * 60)

    try:
        # 先检查GPU
        print("\n[1/3] 检查GPU状态...")
        subprocess.run([sys.executable, "-c",
                        "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); "
                        "print(f'GPU设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"])

        # 启动修复版API服务器
        print("\n[2/3] 正在启动修复版API服务器...")
        api_process = subprocess.Popen([sys.executable, "fix_gpu_response.py"])

        # 等待API服务器启动
        print("等待API服务器启动 (30秒)...")
        for i in range(30, 0, -1):
            print(f"剩余时间: {i}秒", end="\r")
            time.sleep(1)
        print(" " * 30, end="\r")  # 清除倒计时

        # 启动现有的机器人模拟器
        print("\n[3/3] 正在启动机器人模拟器...")
        simulator_process = subprocess.Popen([sys.executable, "robot_simulator.py"])

        print("\n系统已启动! 按Ctrl+C退出...")

        # 等待任一进程结束
        while True:
            api_status = api_process.poll()
            sim_status = simulator_process.poll()

            if api_status is not None:
                print(f"\nAPI服务器已退出，状态码: {api_status}")
                break

            if sim_status is not None:
                print(f"\n机器人模拟器已退出，状态码: {sim_status}")
                break

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n接收到中断信号，正在关闭系统...")

    finally:
        # 清理进程
        print("\n关闭所有进程...")

        try:
            if 'api_process' in locals() and api_process.poll() is None:
                api_process.terminate()
                print("API服务器已关闭")

            if 'simulator_process' in locals() and simulator_process.poll() is None:
                simulator_process.terminate()
                print("机器人模拟器已关闭")
        except Exception as e:
            print(f"关闭进程时出错: {e}")

        print("\n系统已关闭")


if __name__ == "__main__":
    main()
