"""
PEPPER机器人智能教学系统运行示例

这是一个简单的演示脚本，展示如何使用集成系统进行教学应用。
"""
import argparse
import json
import logging
import os
import sys
import time

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入集成系统和模拟器
from integrated_system import PepperIntegratedSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pepper_demo.log')
    ]
)
logger = logging.getLogger("PEPPER_DEMO")


class PepperDemoApp:
    """PEPPER机器人教学演示应用"""

    def __init__(self, config_path="config.json"):
        """初始化演示应用"""
        self.config_path = config_path
        self.integrated_system = None
        self.simulator = None
        self.simulator_root = None
        self.simulator_thread = None

        # 加载配置
        self.config = self._load_config()

    def _load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"已加载配置文件: {self.config_path}")
                return config
            else:
                logger.warning(f"配置文件不存在: {self.config_path}，将使用默认配置")
                return None
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return None

    def start_simulator(self):
        """启动PEPPER机器人模拟器"""
        logger.info("正在启动PEPPER机器人模拟器...")

        # 创建一个不使用Tkinter的模拟器实例
        from pepper_robot.simulation.pepper_simulator import PepperSimulator
        self.simulator = PepperSimulator()  # 不传入Tkinter根窗口

        # 检查模拟器类中是否有start方法，如果没有，我们不调用它
        if hasattr(self.simulator, 'is_running'):
            self.simulator.is_running = True

        logger.info("PEPPER机器人模拟器已启动")

    def _run_simulator(self):
        """运行模拟器线程"""
        # 运行Tkinter主循环
        self.simulator_root.mainloop()

    def start_integrated_system(self):
        """启动集成系统"""
        logger.info("正在启动PEPPER机器人智能教学集成系统...")

        # 确保配置使用模拟模式
        if self.config:
            self.config["robot"]["simulation_mode"] = True

        # 创建集成系统实例
        self.integrated_system = PepperIntegratedSystem(self.config)

        logger.info("PEPPER机器人智能教学集成系统已启动")

    def run_demo_session(self, student_id=None, topic=None):
        """运行演示教学会话"""
        if not self.integrated_system:
            logger.error("集成系统未启动，无法运行教学会话")
            return False

        logger.info("正在准备演示教学会话...")

        # 准备教学会话
        success = self.integrated_system.prepare_teaching_session(student_id, topic)
        if not success:
            logger.error("教学会话准备失败")
            return False

        # 启动教学会话
        welcome_message = self.integrated_system.start_teaching_session()
        logger.info(f"教学会话已启动: {welcome_message}")

        # 如果有模拟器，使用模拟器说话
        if self.simulator:
            self.simulator.say(welcome_message)
            time.sleep(3)  # 等待模拟器说完

        # 模拟一些教学交互
        self._simulate_teaching_interactions()

        # 结束教学会话
        self.integrated_system.stop_teaching_session()
        logger.info("演示教学会话已结束")

        return True

    def _simulate_teaching_interactions(self):
        """模拟一些教学交互"""
        # 模拟学生提问
        questions = [
            "什么是Python循环？",
            "for循环和while循环有什么区别？",
            "如何在Python中实现条件判断？",
            "Python函数怎么定义？",
            "谢谢你的解答"
        ]

        for i, question in enumerate(questions):
            logger.info(f"模拟学生提问 {i + 1}: {question}")

            # 如果有模拟器，模拟学生提问
            if self.simulator:
                self.simulator.simulate_speech_input(question)
                time.sleep(2)  # 给模拟器一些时间来显示

            # 处理问题并获取回答
            response = self.integrated_system.process_text_input(question)
            logger.info(f"系统回答: {response[:100]}...")

            # 通过模拟器输出回答
            if self.simulator:
                self.simulator.say(response[:200] + ("..." if len(response) > 200 else ""))
                # 等待模拟器"说话"完成
                words = len(response[:200].split())
                speak_time = max(3, words * 0.3)  # 估计说话时间
                time.sleep(speak_time)
            else:
                # 没有模拟器，只暂停一下
                time.sleep(2)

    def clean_up(self):
        """清理资源"""
        logger.info("正在清理资源...")

        # 清理集成系统
        if self.integrated_system:
            self.integrated_system.clean_up()

        # 停止模拟器
        if self.simulator:
            self.simulator.stop()

        logger.info("资源清理完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PEPPER机器人智能教学系统演示')
    parser.add_argument('--config', type=str, default='config.json',
                        help='配置文件路径')
    parser.add_argument('--student', type=str, default=None,
                        help='学生ID')
    parser.add_argument('--topic', type=str, default="Python编程",
                        help='教学主题')
    parser.add_argument('--no-simulator', action='store_true',
                        help='不启动图形界面模拟器')

    args = parser.parse_args()

    try:
        # 创建演示应用
        demo_app = PepperDemoApp(args.config)

        # 启动模拟器（如果需要）
        if not args.no_simulator:
            demo_app.start_simulator()
            # 给模拟器界面一些时间初始化
            time.sleep(1)

        # 启动集成系统
        demo_app.start_integrated_system()

        # 运行演示教学会话
        demo_app.run_demo_session(args.student, args.topic)

        # 如果启动了模拟器，保持程序运行直到模拟器关闭
        if not args.no_simulator and demo_app.simulator_thread:
            logger.info("演示程序运行中，请关闭模拟器窗口退出...")
            demo_app.simulator_thread.join()

        # 清理资源
        demo_app.clean_up()

        logger.info("演示程序已结束")
        return 0

    except KeyboardInterrupt:
        logger.info("接收到中断信号，程序退出")
        return 0
    except Exception as e:
        logger.error(f"演示程序出错: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
