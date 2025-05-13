import os

print("检查DeepSeek模型文件...")
model_path = "models/deepseek-coder-1.3b-base"
files = os.listdir(model_path)
print(f"发现{len(files)}个文件:")
for file in files:
    print(f"- {file}")
print("模型文件检查完成")
