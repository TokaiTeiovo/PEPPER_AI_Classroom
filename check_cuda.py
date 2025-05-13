import torch
import sys

print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    print(f"设备数量: {torch.cuda.device_count()}")
    print(f"设备属性: {torch.cuda.get_device_properties(0)}")
    print("尝试在GPU上运行简单的操作...")
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    y = x + 1
    print(f"结果: {y}")
    print("GPU测试成功！")
else:
    print("CUDA不可用，检查以下可能的原因:")
    print("1. 没有兼容的NVIDIA GPU")
    print("2. NVIDIA驱动没有正确安装")
    print("3. CUDA工具包没有正确安装")
    print("4. PyTorch没有安装CUDA版本")
