# start_pepper.py
import argparse
import os
import subprocess
import sys


def main():
    """启动PEPPER机器人智能教学系统"""
    parser = argparse.ArgumentParser(description='启动PEPPER机器人智能教学系统')
    parser.add_argument('--simulation', action='store_true', help='使用模拟模式')
    parser.add_argument('--topic', type=str, default="Python编程", help='教学主题')
    parser.add_argument('--student', type=str, help='学生ID')
    parser.add_argument('--interactions', type=int, default=5, help='交互次数')
    args = parser.parse_args()

    # 构建命令行参数
    cmd_args = []
    if args.simulation:
        cmd_args.append("--simulation")
    if args.topic:
        cmd_args.extend(["--topic", args.topic])
    if args.student:
        cmd_args.extend(["--student", args.student])
    if args.interactions:
        cmd_args.extend(["--interactions", str(args.interactions)])

    # 调用主环境运行
    python_exe = os.path.abspath(os.path.join("venv_ai", "Scripts", "python.exe"))
    if not os.path.exists(python_exe):
        python_exe = sys.executable  # 使用当前环境

    run_script = os.path.abspath("run_pepper_system.py")
    cmd = [python_exe, run_script] + cmd_args

    print(f"执行命令: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
