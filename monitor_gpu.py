import time
import torch
import psutil
import os
import datetime


def format_bytes(bytes):
    """以人类可读的格式显示字节数"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def monitor_gpu():
    """监控GPU使用情况"""
    print("\033c", end="")  # 清屏
    print("=" * 60)
    print(f"GPU监控工具 - 实时统计 (按Ctrl+C退出)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("错误: CUDA不可用，无法监控GPU")
        return

    try:
        # 获取初始信息
        device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory

        print(f"设备: {device_name}")
        print(f"总内存: {format_bytes(total_memory)}")
        print("-" * 60)

        # 表头
        print(f"{'时间':^12} | {'GPU利用率':^10} | {'已分配':^12} | {'缓存':^12} | {'可用':^12} | {'CPU利用率':^10}")
        print("-" * 60)

        start_time = time.time()
        while True:
            # 获取当前时间
            current_time = datetime.datetime.now().strftime("%H:%M:%S")

            # 获取GPU内存信息
            allocated_memory = torch.cuda.memory_allocated()
            reserved_memory = torch.cuda.memory_reserved()
            free_memory = total_memory - allocated_memory

            # GPU利用率（实际上PyTorch不直接提供这个，这里仅显示内存利用率）
            gpu_util = allocated_memory / total_memory * 100

            # CPU利用率
            cpu_util = psutil.cpu_percent()

            # 打印当前状态
            print(
                f"{current_time:^12} | {gpu_util:^10.2f}% | {format_bytes(allocated_memory):^12} | {format_bytes(reserved_memory):^12} | {format_bytes(free_memory):^12} | {cpu_util:^10.2f}%",
                end="\r")

            # 睡眠一秒
            time.sleep(1)

    except KeyboardInterrupt:
        # 计算运行时间
        total_runtime = time.time() - start_time
        minutes, seconds = divmod(total_runtime, 60)
        hours, minutes = divmod(minutes, 60)

        print("\n" + "=" * 60)
        print(f"监控结束，总运行时间: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
        print("=" * 60)

    except Exception as e:
        print(f"\n监控出错: {e}")


if __name__ == "__main__":
    monitor_gpu()
