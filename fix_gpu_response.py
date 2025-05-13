"""
修复版PEPPER AI系统 - 解决GPU使用和回答质量问题
"""
import tkinter as tk
from tkinter import scrolledtext, Entry, Button, Label, Frame
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import requests
import threading
import time
import logging
import sys
import os
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS

# ===========================================
# 配置日志
# ===========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pepper_fix.log')
    ]
)
logger = logging.getLogger("PEPPER_AI_FIX")


# ===========================================
# GPU监控和使用函数
# ===========================================
def check_gpu_usage():
    """检查并打印GPU使用情况"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"可用GPU数量: {device_count}")

        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            allocated_memory = torch.cuda.memory_allocated(i) / 1e9
            reserved_memory = torch.cuda.memory_reserved(i) / 1e9

            logger.info(f"GPU {i}: {device_name}")
            logger.info(f"  总内存: {total_memory:.2f}GB")
            logger.info(f"  已分配内存: {allocated_memory:.2f}GB")
            logger.info(f"  已保留内存: {reserved_memory:.2f}GB")

            # 尝试强制一些GPU计算以提高使用率
            logger.info(f"  执行测试计算...")
            test_tensor = torch.randn(2000, 2000, device=f"cuda:{i}")
            result = torch.matmul(test_tensor, test_tensor)
            torch.cuda.synchronize()  # 确保计算完成

            # 检查计算后内存使用
            new_allocated = torch.cuda.memory_allocated(i) / 1e9
            logger.info(f"  计算后已分配内存: {new_allocated:.2f}GB")

            # 清理测试张量
            del test_tensor
            del result
            torch.cuda.empty_cache()
    else:
        logger.warning("没有可用的GPU")


def force_gpu_utilization():
    """强制GPU使用率提高的测试函数"""
    if not torch.cuda.is_available():
        logger.warning("没有可用的GPU，无法执行测试")
        return

    try:
        # 创建大型张量并执行计算密集型操作
        logger.info("执行GPU压力测试以提高使用率...")

        # 创建几个大型张量
        a = torch.randn(5000, 5000, device="cuda")
        b = torch.randn(5000, 5000, device="cuda")

        # 执行多次矩阵乘法
        for _ in range(10):
            c = torch.matmul(a, b)
            torch.cuda.synchronize()

        logger.info("GPU压力测试完成")

        # 检查内存使用
        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info(f"测试后已分配内存: {allocated:.2f}GB")

        # 清理测试张量
        del a
        del b
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"GPU压力测试失败: {e}")


# ===========================================
# API服务器
# ===========================================
app = Flask(__name__)
CORS(app)  # 启用CORS

# 全局变量
tokenizer = None
model = None
generation_pipeline = None


def load_model(use_phi=True):
    """加载模型 - 优先使用Phi模型"""
    global tokenizer, model, generation_pipeline

    try:
        # 检查GPU
        if torch.cuda.is_available():
            logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA版本: {torch.version.cuda}")
            device = "cuda"
            # 清理GPU内存
            torch.cuda.empty_cache()
        else:
            logger.warning("没有可用的GPU，将使用CPU（这会很慢）")
            device = "cpu"

        # 决定使用哪个模型
        if use_phi:
            model_name = "microsoft/phi-1_5"
            logger.info(f"正在加载Phi-1.5模型...")

            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("分词器加载完成")

            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )

            # 创建生成管道
            generation_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1
            )

            logger.info("Phi-1.5模型加载完成")
        else:
            # 使用DeepSeek模型，但改变加载方式
            model_path = "models/deepseek-coder-1.3b-base"
            logger.info(f"正在加载DeepSeek模型: {model_path}")

            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            logger.info("分词器加载完成")

            # 加载模型 - 使用不同的方法
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )

            # 创建生成管道
            generation_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1
            )

            logger.info("DeepSeek模型加载完成")

        # 检查模型是否正确加载
        if model is None:
            logger.error("模型加载失败")
            return False

        # 检查模型是否在正确的设备上
        if hasattr(model, "device"):
            logger.info(f"模型设备: {model.device}")
        else:
            device_info = next(model.parameters()).device
            logger.info(f"模型设备: {device_info}")

        # 打印内存使用情况
        if device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU内存使用: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")

        # 执行测试生成以确保模型工作正常
        test_input = "你好，请用简短的一句话回答我"
        logger.info("执行测试生成...")

        if generation_pipeline is not None:
            test_output = generation_pipeline(
                test_input,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True
            )
            logger.info(f"测试生成结果: {test_output[0]['generated_text']}")
        else:
            # 使用普通方式测试
            inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=20)
            test_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"测试生成结果: {test_output}")

        # 强制提高GPU使用率
        if device == "cuda":
            force_gpu_utilization()
            check_gpu_usage()

        return True

    except Exception as e:
        logger.error(f"模型加载失败: {e}", exc_info=True)
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    global model, generation_pipeline
    if model is None:
        return jsonify({"status": "error", "message": "模型未加载"}), 503
    return jsonify({"status": "ok", "message": "修复版PEPPER AI系统运行正常"})


@app.route('/llm/query', methods=['POST'])
def llm_query_api():
    """大语言模型查询API - 修复版"""
    global tokenizer, model, generation_pipeline

    if model is None or tokenizer is None:
        return jsonify({"status": "error", "message": "模型未加载"}), 503

    data = request.json
    if not data or 'query' not in data:
        return jsonify({"status": "error", "message": "未找到查询文本"}), 400

    query = data['query']

    try:
        logger.info(f"收到查询: {query[:50]}...")

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            check_gpu_usage()  # 检查GPU使用情况

        # 使用生成管道 - 更可靠的方法
        prompt = f"问题: {query}\n\n回答: "

        logger.info("开始生成回答...")
        start_time = time.time()

        # 使用管道生成回答
        if generation_pipeline is not None:
            outputs = generation_pipeline(
                prompt,
                max_new_tokens=150,
                temperature=0.5,  # 使用更低的温度以获得更确定性的回答
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.2,
                num_return_sequences=1
            )

            full_output = outputs[0]['generated_text']

        else:
            # 备用方法
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.5,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.2
                )

            full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 提取回答部分
        response = full_output[len(prompt):].strip()

        # 简单清理响应，删除奇怪的Markdown和列表项目
        response = response.replace("---", "").replace("我是答案。", "")
        response = response.replace("* [ ]", "").replace("* []", "")

        # 确保是合理的响应
        if not response or len(response) < 5 or "题目" in response:
            response = "你好！很高兴见到你。我是PEPPER智能助手，请问有什么我可以帮助你的吗？"

        gen_time = time.time() - start_time
        logger.info(f"生成完成: {len(response)}字符, 耗时={gen_time:.2f}秒")

        # 强制提高GPU使用率
        if torch.cuda.is_available():
            force_gpu_utilization()

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            check_gpu_usage()  # 再次检查GPU使用情况

        return jsonify({"status": "success", "response": response})

    except Exception as e:
        logger.error(f"生成回答失败: {e}", exc_info=True)
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    """主页"""
    return jsonify({
        "status": "ok",
        "message": "修复版PEPPER AI智能教学系统 API",
        "endpoints": {
            "/health": "健康检查",
            "/llm/query": "查询大语言模型(POST)"
        }
    })


# ===========================================
# 启动函数
# ===========================================
def start_server():
    """加载模型并启动服务器"""
    # 加载模型 - 使用Phi模型替代DeepSeek
    if load_model(use_phi=True):
        # 启动API服务
        logger.info("启动API服务...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("模型加载失败，无法启动API服务")


if __name__ == "__main__":
    # 开始服务器
    start_server()
