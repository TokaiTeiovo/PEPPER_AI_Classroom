"""
项目修复脚本 - 检查并修复项目文件
"""
import os
import shutil


def check_file(file_path):
    """检查文件是否可以正常打开"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return True, None
    except Exception as e:
        return False, str(e)


def backup_file(file_path):
    """备份文件"""
    backup_path = file_path + ".bak"
    try:
        shutil.copy2(file_path, backup_path)
        print(f"已备份文件: {file_path} -> {backup_path}")
        return True
    except Exception as e:
        print(f"备份文件失败: {file_path}, 错误: {e}")
        return False


def main():
    """主函数"""
    print("项目修复工具 - 检查并修复关键文件")
    print("=" * 50)

    # 检查关键文件
    key_files = [
        "integrated_system.py",
        "run_pepper_system.py",
        "run_demo.py",
        "ai_service/multimodal/speech_recognition.py",
        "ai_service/multimodal/text_processor.py",
        "ai_service/multimodal/image_recognition.py"
    ]

    print("\n正在检查关键文件...")
    problem_files = []

    for file_path in key_files:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            problem_files.append((file_path, "不存在"))
            continue

        ok, error = check_file(file_path)
        if not ok:
            print(f"文件有问题: {file_path}, 错误: {error}")
            # 备份有问题的文件
            if backup_file(file_path):
                problem_files.append((file_path, error))
        else:
            print(f"文件正常: {file_path}")

    # 检查目录结构
    print("\n正在检查项目目录结构...")
    required_dirs = [
        "ai_service",
        "ai_service/knowledge_graph",
        "ai_service/llm_module",
        "ai_service/multimodal",
        "ai_service/teaching_module",
        "pepper_robot",
        "pepper_robot/motion_module",
        "pepper_robot/robot_control",
        "pepper_robot/sensor_module",
        "pepper_robot/simulation",
        "interface",
        "interface/bridge",
        "interface/api",
        "data",
        "data/student_profiles",
        "data/course_materials",
        "models",
        "tests"
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"目录不存在: {dir_path}")
            missing_dirs.append(dir_path)
        else:
            # 确保__init__.py文件存在
            init_file = os.path.join(dir_path, "__init__.py")
            if not os.path.exists(init_file):
                print(f"缺少__init__.py文件: {dir_path}")
                with open(init_file, "w", encoding="utf-8") as f:
                    f.write("# 自动生成的__init__.py文件\n")
                print(f"已创建: {init_file}")

    # 创建缺失的目录
    for dir_path in missing_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"已创建目录: {dir_path}")

            # 创建__init__.py文件
            init_file = os.path.join(dir_path, "__init__.py")
            with open(init_file, "w", encoding="utf-8") as f:
                f.write("# 自动生成的__init__.py文件\n")
            print(f"已创建: {init_file}")
        except Exception as e:
            print(f"创建目录失败: {dir_path}, 错误: {e}")

    print("\n项目修复报告:")
    print(f"- 检查了 {len(key_files)} 个关键文件")
    print(f"- 发现 {len(problem_files)} 个有问题的文件")
    print(f"- 检查了 {len(required_dirs)} 个必要目录")
    print(f"- 发现并创建了 {len(missing_dirs)} 个缺失的目录")

    if problem_files:
        print("\n需要修复的文件:")
        for file_path, error in problem_files:
            print(f"- {file_path}: {error}")
        print("\n请重新创建这些文件或从备份中恢复")
    else:
        print("\n没有发现问题文件，项目结构完整")


if __name__ == "__main__":
    main()
