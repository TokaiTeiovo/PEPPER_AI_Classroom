# fix_init_files.py
import os


def fix_init_file(file_path):
    """修复 __init__.py 文件中的 null 字节"""
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            # 如果文件不存在，创建一个新的空文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# 自动生成的 __init__.py 文件\n")
            print(f"创建了新的文件: {file_path}")
            return True

        # 读取文件内容
        with open(file_path, 'rb') as f:
            content = f.read()

        # 检查是否包含 null 字节
        if b'\x00' in content:
            # 如果文件只包含 null 字节，创建一个新的文件
            if set(content) == {0}:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("# 自动生成的 __init__.py 文件\n")
                print(f"替换了空文件: {file_path}")
            else:
                # 如果文件包含其他内容，移除 null 字节
                cleaned_content = content.replace(b'\x00', b'')
                with open(file_path, 'wb') as f:
                    f.write(cleaned_content)
                print(f"修复了文件: {file_path}")
            return True
        else:
            return False
    except Exception as e:
        print(f"处理文件失败: {file_path}, 错误: {e}")
        return False


def fix_all_init_files():
    """修复所有 __init__.py 文件"""
    problem_files = [
        "./ai_service/__init__.py",
        "./ai_service/knowledge_graph/__init__.py",
        "./ai_service/llm_module/__init__.py",
        "./ai_service/multimodal/__init__.py",
        "./interface/__init__.py",
        "./interface/bridge/__init__.py",
        "./pepper_robot/__init__.py",
        "./pepper_robot/motion_module/__init__.py",
        "./pepper_robot/robot_control/__init__.py",
        "./pepper_robot/sensor_module/__init__.py"
    ]

    fixed_count = 0
    for file_path in problem_files:
        if fix_init_file(file_path):
            fixed_count += 1

    print(f"修复了 {fixed_count} 个文件")

    # 另外检查项目中所有的 __init__.py 文件
    print("\n检查项目中所有的 __init__.py 文件...")
    additional_fixed = 0

    for root, dirs, files in os.walk("."):
        if "__pycache__" in root:
            continue

        for file in files:
            if file == "__init__.py":
                file_path = os.path.join(root, file)
                if file_path not in problem_files:
                    if fix_init_file(file_path):
                        additional_fixed += 1

    print(f"额外修复了 {additional_fixed} 个文件")
    return fixed_count + additional_fixed


if __name__ == "__main__":
    print("开始修复 __init__.py 文件...")
    total_fixed = fix_all_init_files()
    print(f"总共修复了 {total_fixed} 个文件")
