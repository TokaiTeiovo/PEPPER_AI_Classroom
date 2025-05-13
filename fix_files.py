"""
修复项目文件中的null字节问题
"""
import os
import sys


def fix_file(file_path):
    """删除文件中的null字节"""
    try:
        # 以二进制模式读取文件
        with open(file_path, 'rb') as f:
            content = f.read()

        # 检查是否包含null字节
        if b'\x00' in content:
            print(f"文件包含null字节: {file_path}")
            # 移除null字节
            cleaned_content = content.replace(b'\x00', b'')
            # 写回文件
            with open(file_path, 'wb') as f:
                f.write(cleaned_content)
            print(f"已修复文件: {file_path}")
            return True
        else:
            print(f"文件正常: {file_path}")
            return False
    except Exception as e:
        print(f"处理文件时出错: {file_path}, 错误: {e}")
        return False


def main():
    """主函数"""
    # 需要修复的关键文件
    key_files = [
        "integrated_system.py",
        "run_pepper_system.py"
    ]

    fixed_count = 0
    for file_path in key_files:
        if os.path.exists(file_path):
            if fix_file(file_path):
                fixed_count += 1
        else:
            print(f"文件不存在: {file_path}")

    print(f"修复了 {fixed_count} 个文件")

    return 0


if __name__ == "__main__":
    sys.exit(main())
