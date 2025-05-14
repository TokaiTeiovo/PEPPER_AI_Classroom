# 创建 fix_all_files.py
import os


def fix_file(file_path):
    """修复文件中的空字节"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()

        # 检查是否包含空字节
        if b'\x00' in content:
            # 移除空字节
            content = content.replace(b'\x00', b'')

            # 重新保存文件
            with open(file_path, 'wb') as f:
                f.write(content)

            print(f"已修复文件: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False


def scan_directory(directory):
    """扫描目录中的所有Python文件"""
    fixed_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_file(file_path):
                    fixed_count += 1

    return fixed_count


# 修复项目中的所有Python文件
fixed_count = scan_directory('.')
print(f"共修复了 {fixed_count} 个文件")