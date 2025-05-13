"""
PEPPER机器人模拟器 - 图形界面
用于与DeepSeek模型API服务器交互
"""
import tkinter as tk
from tkinter import scrolledtext, Entry, Button, Label, Frame
import requests
import threading
import time


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
                timeout=60  # 增加超时时间，因为模型生成可能需要较长时间
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


if __name__ == "__main__":
    root = tk.Tk()
    app = RobotSimulator(root)
    root.mainloop()
