# check_all_files.py
import os


def check_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            if b'\x00' in content:
                print(f"文件包含 null 字节: {file_path}")
                return False
        return True
    except Exception as e:
        print(f"检查文件 {file_path} 时出错: {e}")
        return False


def scan_directory(directory):
    clean_files = 0
    problem_files = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if check_file(file_path):
                    clean_files += 1
                else:
                    problem_files += 1

    print(f"扫描完成: 正常文件 {clean_files} 个, 有问题文件 {problem_files} 个")


# 扫描当前目录
scan_directory('.')
