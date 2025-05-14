with open('integrated_system.py', 'rb') as f:
    content = f.read()
    if b'\x00' in content:
        print("文件包含 null 字节")
    else:
        print("文件正常")
