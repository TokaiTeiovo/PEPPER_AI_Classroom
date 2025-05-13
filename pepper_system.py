"""
PEPPER机器人智能教学系统 - 统一启动文件
整合了GPU优化、API服务器和机器人模拟器
"""
import tkinter as tk
from tkinter import scrolledtext, Entry, Button, Label, Frame, messagebox
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import threading
import time
import logging
import sys
import os
import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess

# ===========================================
# 配置日志
# ===========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pepper_system.log')
    ]
)
logger = logging.getLogger("PEPPER_AI")


# ===========================================
# GPU优化函数
# ===========================================
def optimize_gpu():
    """优化GPU设置以提高使用率"""
    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，无法优化GPU设置")
        return False

    try:
        # 清理GPU缓存
        torch.cuda.empty_cache()

        # 启用cudnn benchmark以提高性能
        torch.backends.cudnn.benchmark = True
        logger.info("已启用cudnn benchmark")

        # 设置更高效的CUDA分配器
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        # 预热GPU以提高初始性能
        logger.info("正在预热GPU...")
        x = torch.randn(1024, 1024, device='cuda')
        y = torch.randn(1024, 1024, device='cuda')
        for _ in range(5):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()  # 确保完成
        logger.info("GPU预热完成")

        # 打印GPU信息
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU内存: 总计={torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
        logger.info(f"CUDA版本: {torch.version.cuda}")

        return True
    except Exception as e:
        logger.error(f"GPU优化失败: {e}")
        return False


# ===========================================
# API服务器
# ===========================================
app = Flask(__name__)
CORS(app)  # 启用CORS

# 全局变量
tokenizer = None
model = None


def load_model(model_path="models/deepseek-coder-1.3b-base", use_8bit=True):
    """加载模型"""
    global tokenizer, model
    try:
        logger.info(f"正在加载模型: {model_path}")

        # 优化GPU(如果可用)
        if torch.cuda.is_available():
            optimize_gpu()
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        # 加载分词器
        logger.info("正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info("分词器加载完成")

        # 加载模型
        logger.info("正在加载模型...")

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }

        if device == "cuda":
            model_kwargs["device_map"] = "auto"

            # 使用8位量化（如果指定）
            if use_8bit:
                try:
                    import bitsandbytes as bnb
                    model_kwargs["load_in_8bit"] = True
                    logger.info("使用8位量化加载模型以节省GPU内存")
                except ImportError:
                    logger.warning("未找到bitsandbytes库，无法使用8位量化")

        # 使用低内存加载
        try:
            import accelerate
            model_kwargs["low_cpu_mem_usage"] = True
        except ImportError:
            pass

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        # 如果使用CPU，确保模型在正确设备上
        if device == "cpu" and not hasattr(model, "hf_device_map"):
            model = model.to(device)

        logger.info("模型加载完成")

        # 打印内存使用情况
        if device == "cuda":
            memory = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU内存使用: {memory:.2f}GB")

        return True
    except Exception as e:
        logger.error(f"模型加载失败: {e}", exc_info=True)
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    global model
    if model is None:
        return jsonify({"status": "error", "message": "模型未加载"}), 503
    return jsonify({"status": "ok", "message": "PEPPER AI系统运行正常"})


@app.route('/llm/query', methods=['POST'])
def llm_query_api():
    """大语言模型查询API"""
    global tokenizer, model

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

        # 构建提示 - 尝试更简单的格式
        prompt = f"问题: {query}\n回答:"

        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 生成设置 - 使用更保守的参数
        generation_kwargs = {
            "max_new_tokens": 200,  # 较保守的生成长度
            "temperature": 0.6,  # 更保守的温度
            "top_p": 0.92,  # 适中的top_p
            "do_sample": True,
            "repetition_penalty": 1.1,  # 轻微惩罚重复
            "no_repeat_ngram_size": 3  # 防止重复n-gram
        }

        # 确保有pad_token_id
        if tokenizer.pad_token_id is None:
            generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

        logger.info("开始生成回答...")
        start_time = time.time()

        # 生成回答
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        gen_time = time.time() - start_time

        # 解码输出
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 提取回答部分
        response = output[len(prompt):].strip()

        logger.info(f"生成完成: {len(response)}字符, 耗时={gen_time:.2f}秒")

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        "message": "PEPPER AI 智能教学系统 API",
        "endpoints": {
            "/health": "健康检查",
            "/llm/query": "查询大语言模型(POST)"
        }
    })


# ===========================================
# 机器人模拟器 GUI
# ===========================================
class RobotSimulator:
    def __init__(self, root, api_url="http://localhost:5000"):
        """初始化机器人模拟器"""
        self.root = root
        self.api_url = api_url
        self.setup_ui()

    def setup_ui(self):
        """设置用户界面"""
        self.root.title("PEPPER机器人模拟器")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # 标题
        title_frame = Frame(self.root, bg="#4a86e8")
        title_frame.pack(fill=tk.X, padx=10, pady=10)

        title_label = Label(
            title_frame,
            text="PEPPER机器人在'人工智能+'课堂上的智能化应用",
            font=("Arial", 16, "bold"),
            fg="white",
            bg="#4a86e8",
            padx=10,
            pady=10
        )
        title_label.pack()

        # 机器人状态框
        robot_frame = Frame(self.root, bg="#e6e6e6", relief=tk.RAISED, borderwidth=1)
        robot_frame.pack(fill=tk.X, padx=10, pady=5)

        self.robot_status = Label(
            robot_frame,
            text="机器人状态: 待机中",
            font=("Arial", 10),
            fg="#333333",
            bg="#e6e6e6",
            anchor="w",
            padx=10,
            pady=5
        )
        self.robot_status.pack(fill=tk.X)

        # 对话显示区域
        chat_frame = Frame(self.root)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Arial", 12),
            bg="white",
            fg="#333333"
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)

        # 输入区域
        input_frame = Frame(self.root, bg="#f0f0f0")
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        self.user_input = Entry(
            input_frame,
            font=("Arial", 12),
            width=50
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", self.send_message)

        self.send_button = Button(
            input_frame,
            text="发送",
            font=("Arial", 12),
            command=self.send_message,
            bg="#4a86e8",
            fg="white",
            padx=10
        )
        self.send_button.pack(side=tk.RIGHT)

        # 预设问题按钮
        preset_frame = Frame(self.root, bg="#f0f0f0")
        preset_frame.pack(fill=tk.X, padx=10, pady=5)

        preset_label = Label(
            preset_frame,
            text="预设问题:",
            font=("Arial", 10),
            bg="#f0f0f0"
        )
        preset_label.pack(side=tk.LEFT, padx=(0, 5))

        preset_questions = [
            "Python中for循环和while循环有什么区别？",
            "人工智能在教育中有哪些应用？",
            "什么是多模态交互？",
            "PEPPER机器人在课堂上的优势是什么？"
        ]

        for question in preset_questions:
            btn = Button(
                preset_frame,
                text=question[:15] + "..." if len(question) > 15 else question,
                font=("Arial", 9),
                command=lambda q=question: self.use_preset(q),
                bg="#e6e6e6",
                fg="#333333",
                padx=5
            )
            btn.pack(side=tk.LEFT, padx=5)

        # 测试API连接
        self.test_api_connection()

        # 初始欢迎消息
        self.add_message("PEPPER",
                         "你好！我是PEPPER机器人，基于DeepSeek本地模型。我可以回答关于Python编程、人工智能、教育应用等方面的问题。请在下方输入你的问题。")

    def test_api_connection(self):
        """测试API连接"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                self.robot_status.config(text="机器人状态: 在线 - 已连接到DeepSeek模型")
            else:
                self.robot_status.config(text=f"机器人状态: 错误 - API响应状态码 {response.status_code}")
        except Exception as e:
            self.robot_status.config(text=f"机器人状态: 离线 - 无法连接到API ({str(e)})")
            self.add_message("系统", "警告: 无法连接到API服务。请确保服务已启动。")

    def add_message(self, sender, message):
        """添加消息到对话显示区域"""
        self.chat_display.config(state=tk.NORMAL)

        # 添加时间戳
        time_str = time.strftime("%H:%M:%S")

        if sender == "PEPPER":
            self.chat_display.insert(tk.END, f"{time_str} PEPPER: ", "robot_tag")
            self.chat_display.insert(tk.END, f"{message}\n\n", "robot_msg")
            self.chat_display.tag_config("robot_tag", foreground="#4a86e8", font=("Arial", 12, "bold"))
            self.chat_display.tag_config("robot_msg", foreground="#333333", font=("Arial", 12))
        elif sender == "系统":
            self.chat_display.insert(tk.END, f"{time_str} 系统: ", "system_tag")
            self.chat_display.insert(tk.END, f"{message}\n\n", "system_msg")
            self.chat_display.tag_config("system_tag", foreground="#ff9800", font=("Arial", 12, "bold"))
            self.chat_display.tag_config("system_msg", foreground="#ff5722", font=("Arial", 12))
        else:
            self.chat_display.insert(tk.END, f"{time_str} 用户: ", "user_tag")
            self.chat_display.insert(tk.END, f"{message}\n\n", "user_msg")
            self.chat_display.tag_config("user_tag", foreground="#009688", font=("Arial", 12, "bold"))
            self.chat_display.tag_config("user_msg", foreground="#333333", font=("Arial", 12))

        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.yview(tk.END)

    def send_message(self, event=None):
        """发送用户消息"""
        message = self.user_input.get().strip()
        if not message:
            return

        # 清空输入框
        self.user_input.delete(0, tk.END)

        # 显示用户消息
        self.add_message("用户", message)

        # 更新机器人状态
        self.robot_status.config(text="机器人状态: 思考中...")

        # 在单独线程中处理API请求，避免界面卡顿
        threading.Thread(target=self.process_message, args=(message,)).start()

    def process_message(self, message):
        """处理消息并获取回答"""
        try:
            # 发送API请求
            self.root.after(0, lambda: self.add_message("系统", "正在生成回答，请稍等..."))

            response = requests.post(
                f"{self.api_url}/llm/query",
                json={"query": message},
                timeout=120  # 设置较长的超时时间
            )

            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "抱歉，我无法生成回答。")

                # 更新UI（必须在主线程中进行）
                self.root.after(0, lambda: self.add_message("PEPPER", answer))
                self.root.after(0, lambda: self.robot_status.config(text="机器人状态: 待机中"))
            else:
                error_msg = f"API请求失败: {response.status_code}"
                self.root.after(0, lambda: self.add_message("系统", error_msg))
                self.root.after(0, lambda: self.robot_status.config(text="机器人状态: 错误"))

        except Exception as e:
            error_msg = f"发生错误: {str(e)}"
            self.root.after(0, lambda: self.add_message("系统", error_msg))
            self.root.after(0, lambda: self.robot_status.config(text="机器人状态: 错误"))

    def use_preset(self, question):
        """使用预设问题"""
        self.user_input.delete(0, tk.END)
        self.user_input.insert(0, question)
        self.send_message()


# ===========================================
# 启动函数
# ===========================================
def start_api_server():
    """启动API服务器"""
    print("加载模型并启动API服务器...")
    if load_model():
        print("API服务器启动成功，模型加载完成")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("模型加载失败，无法启动API服务器")


def start_robot_simulator():
    """启动机器人模拟器"""
    root = tk.Tk()
    app = RobotSimulator(root)
    root.mainloop()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PEPPER机器人智能教学系统')
    parser.add_argument('--mode', type=str, choices=['all', 'api', 'gui'], default='all',
                        help='运行模式: all(完整系统), api(仅API服务器), gui(仅GUI)')
    parser.add_argument('--model', type=str, default='models/deepseek-coder-1.3b-base',
                        help='模型路径')
    args = parser.parse_args()

    print("=" * 60)
    print("PEPPER机器人智能教学系统")
    print("=" * 60)

    if args.mode == 'all':
        # 启动API服务器进程
        api_process = subprocess.Popen([sys.executable, __file__, '--mode', 'api', '--model', args.model])

        # 等待API服务器启动
        print("等待API服务器启动 (20秒)...")
        for i in range(20, 0, -1):
            print(f"剩余时间: {i}秒", end="\r")
            time.sleep(1)
        print(" " * 30, end="\r")  # 清除倒计时

        # 启动GUI
        print("正在启动机器人模拟器...")
        gui_process = subprocess.Popen([sys.executable, __file__, '--mode', 'gui'])

        try:
            # 等待任一进程结束
            while True:
                api_status = api_process.poll()
                gui_status = gui_process.poll()

                if api_status is not None:
                    print(f"API服务器已退出，状态码: {api_status}")
                    break

                if gui_status is not None:
                    print(f"机器人模拟器已退出，状态码: {gui_status}")
                    break

                time.sleep(1)

        except KeyboardInterrupt:
            print("\n接收到中断信号，正在关闭系统...")

        finally:
            # 清理进程
            if api_process.poll() is None:
                api_process.terminate()
                print("API服务器已关闭")

            if gui_process.poll() is None:
                gui_process.terminate()
                print("机器人模拟器已关闭")

    elif args.mode == 'api':
        # 只启动API服务器
        start_api_server()

    elif args.mode == 'gui':
        # 只启动GUI
        start_robot_simulator()


if __name__ == "__main__":
    main()
