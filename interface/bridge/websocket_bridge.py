import asyncio
import json
import websockets


class WebSocketBridge:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.connected_clients = set()
        self.server = None

    async def handler(self, websocket, path):
        """处理WebSocket连接"""
        # 注册客户端
        self.connected_clients.add(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.process_message(data)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    print(f"接收到无效的JSON消息: {message}")
                    await websocket.send(json.dumps({"status": "error", "message": "无效的JSON格式"}))
        finally:
            # 注销客户端
            self.connected_clients.remove(websocket)

    async def process_message(self, data):
        """处理接收到的消息"""
        try:
            message_type = data.get("type")
            content = data.get("content", {})

            if message_type == "speech_recognition":
                # 处理语音识别请求
                text = content.get("audio_data")
                # 这里应该调用AI服务的语音识别模块
                # 由于是示例，我们直接返回一个模拟结果
                return {"status": "success", "data": {"text": "这是识别结果"}}

            elif message_type == "llm_query":
                # 处理大语言模型查询
                query = content.get("query")
                # 这里应该调用AI服务的大语言模型模块
                # 由于是示例，我们直接返回一个模拟结果
                return {"status": "success", "data": {"response": f"回答: {query}"}}

            elif message_type == "image_recognition":
                # 处理图像识别请求
                image_data = content.get("image_data")
                # 这里应该调用AI服务的图像识别模块
                # 由于是示例，我们直接返回一个模拟结果
                return {"status": "success", "data": {"objects": ["物体1", "物体2"]}}

            else:
                return {"status": "error", "message": f"未知的消息类型: {message_type}"}
        except Exception as e:
            print(f"处理消息出错: {e}")
            return {"status": "error", "message": str(e)}

    async def start_server(self):
        """启动WebSocket服务器"""
        self.server = await websockets.serve(self.handler, self.host, self.port)
        print(f"WebSocket服务器已启动: ws://{self.host}:{self.port}")
        await self.server.wait_closed()

    def run(self):
        """运行WebSocket服务器"""
        asyncio.run(self.start_server())

    async def stop_server(self):
        """停止WebSocket服务器"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("WebSocket服务器已停止")


# 示例用法
if __name__ == "__main__":
    bridge = WebSocketBridge()
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("接收到中断信号，正在关闭服务器...")
        asyncio.run(bridge.stop_server())
