import json
import threading
import time
import ssl
from websocket import create_connection, WebSocketApp


class WebSocketClient:
    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        self.ws = None
        self.connected = False
        self.callback = None

    def connect(self):
        """连接到WebSocket服务器"""
        try:
            self.ws = create_connection(self.server_url)
            self.connected = True
            print(f"已连接到WebSocket服务器: {self.server_url}")
            return True
        except Exception as e:
            print(f"连接WebSocket服务器失败: {e}")
            return False

    def connect_async(self, on_message=None, on_error=None, on_close=None, on_open=None):
        """异步连接到WebSocket服务器"""
        try:
            websocket_app = WebSocketApp(
                self.server_url,
                on_message=on_message if on_message else self._on_message,
                on_error=on_error if on_error else self._on_error,
                on_close=on_close if on_close else self._on_close,
                on_open=on_open if on_open else self._on_open
            )

            # 启动WebSocket连接线程
            wst = threading.Thread(target=websocket_app.run_forever)
            wst.daemon = True
            wst.start()

            # 等待连接建立
            time.sleep(1)

            self.ws = websocket_app
            return True
        except Exception as e:
            print(f"异步连接WebSocket服务器失败: {e}")
            return False

    def _on_message(self, ws, message):
        """默认的消息处理函数"""
        try:
            data = json.loads(message)
            print(f"收到消息: {data}")
            if self.callback:
                self.callback(data)
        except json.JSONDecodeError:
            print(f"接收到无效的JSON消息: {message}")

    def _on_error(self, ws, error):
        """默认的错误处理函数"""
        print(f"发生错误: {error}")
        self.connected = False

    def _on_close(self, ws):
        """默认的关闭处理函数"""
        print("连接关闭")
        self.connected = False

    def _on_open(self, ws):
        """默认的连接建立处理函数"""
        print(f"已连接到WebSocket服务器: {self.server_url}")
        self.connected = True

    def set_callback(self, callback):
        """设置消息回调函数"""
        self.callback = callback

    def send(self, message_type, content):
        """发送消息"""
        if not self.connected:
            print("未连接到WebSocket服务器")
            return False

        try:
            message = {"type": message_type, "content": content}
            self.ws.send(json.dumps(message))
            return True
        except Exception as e:
            print(f"发送消息失败: {e}")
            return False

    def send_and_receive(self, message_type, content):
        """发送消息并等待接收响应"""
        if not self.connected or not hasattr(self.ws, "send"):
            print("未连接到WebSocket服务器")
            return None

        try:
            message = {"type": message_type, "content": content}
            self.ws.send(json.dumps(message))
            response = self.ws.recv()
            return json.loads(response)
        except Exception as e:
            print(f"发送接收消息失败: {e}")
            return None

    def close(self):
        """关闭WebSocket连接"""
        if self.connected and self.ws:
            try:
                self.ws.close()
                print("WebSocket连接已关闭")
                self.connected = False
                return True
            except Exception as e:
                print(f"关闭WebSocket连接失败: {e}")
                return False


# 示例用法
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    else:
        server_url = "ws://localhost:8765"

    client = WebSocketClient(server_url)

    # 测试同步连接和消息交换
    if client.connect():
        # 测试语音识别请求
        response = client.send_and_receive("speech_recognition", {"audio_data": "test_audio_data"})
        print(f"语音识别响应: {response}")

        # 测试大语言模型查询
        response = client.send_and_receive("llm_query", {"query": "什么是人工智能？"})
        print(f"LLM查询响应: {response}")

        # 测试图像识别请求
        response = client.send_and_receive("image_recognition", {"image_data": "test_image_data"})
        print(f"图像识别响应: {response}")

        # 关闭连接
        client.close()
