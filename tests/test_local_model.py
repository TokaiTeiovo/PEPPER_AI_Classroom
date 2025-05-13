import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os


def test_local_deepseek_model(model_path="models/deepseek-coder-1.3b-base"):
    """测试本地DeepSeek模型"""
    print("=" * 50)
    print(f"开始测试本地DeepSeek模型: {model_path}")
    print("=" * 50)

    try:
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            print(f"错误: 模型路径不存在 - {model_path}")
            return False

        # 检查必要的模型文件
        key_files = ["config.json", "tokenizer_config.json"]
        missing_files = [f for f in key_files if not os.path.exists(os.path.join(model_path, f))]

        if missing_files:
            print(f"错误: 缺少关键模型文件 - {missing_files}")
            return False

        # 记录开始时间
        start_time = time.time()

        print("正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        tokenizer_time = time.time()
        print(f"分词器加载完成，耗时: {tokenizer_time - start_time:.2f}秒")

        # 确定设备和数据类型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"使用设备: {device}, 数据类型: {dtype}")

        print("正在加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # 如果使用CPU，确保模型在正确设备上
        if not torch.cuda.is_available():
            model = model.to(device)

        model_time = time.time()
        print(f"模型加载完成，耗时: {model_time - tokenizer_time:.2f}秒")
        print(f"总加载时间: {model_time - start_time:.2f}秒")

        # 测试生成
        print("\n正在测试文本生成...")
        test_prompt = "Human: 什么是人工智能？\n\nAssistant:"

        print(f"测试提示: {test_prompt}")

        gen_start_time = time.time()

        # 编码输入
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        # 生成参数
        gen_kwargs = {
            "max_new_tokens": 50,  # 生成较短的回答以便快速测试
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id
        }

        # 生成回答
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        # 解码输出
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 提取回答部分
        response = output[len(test_prompt):].strip()

        gen_time = time.time() - gen_start_time

        print(f"生成回答(耗时: {gen_time:.2f}秒): {response}")

        print("\n=" * 50)
        print("本地DeepSeek模型测试完成，模型工作正常！")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 测试本地DeepSeek模型
    test_local_deepseek_model()
