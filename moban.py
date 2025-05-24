#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建必要的模板文件和目录结构
"""

import os


def create_template_structure():
    """创建模板目录结构"""

    # 创建必要的目录
    directories = [
        'interface',
        'interface/web_console',
        'interface/web_console/templates',
        'interface/api',
        'uploads',
        'reports',
        'logs',
        'data/student_profiles'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

    # 创建简化的HTML模板
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PEPPER智能教学系统</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background-color: #4a86e8;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }
        .nav-tabs {
            display: flex;
            background: white;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .nav-tab {
            flex: 1;
            padding: 15px 20px;
            background: white;
            border: none;
            cursor: pointer;
            font-size: 16px;
            border-bottom: 3px solid transparent;
        }
        .nav-tab.active {
            background: #f8f9ff;
            border-bottom-color: #4a86e8;
            color: #4a86e8;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .panel {
            background: white;
            border-radius: 5px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input, select, textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background-color: #4a86e8;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
            margin-bottom: 10px;
        }
        button:hover {
            background-color: #3b78e7;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-on {
            background-color: #28a745;
        }
        .status-off {
            background-color: #dc3545;
        }
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) {
            .grid-2 {
                grid-template-columns: 1fr;
            }
            .nav-tabs {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>PEPPER智能教学系统</h1>
            <p>基于DeepSeek模型的个性化AI教学助手</p>
        </div>

        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('llm')">🧠 大语言模型</button>
            <button class="nav-tab" onclick="showTab('knowledge')">🗂️ 知识图谱</button>
            <button class="nav-tab" onclick="showTab('multimodal')">🎯 多模态交互</button>
            <button class="nav-tab" onclick="showTab('teaching')">📚 智能教学</button>
        </div>

        <!-- 大语言模型标签页 -->
        <div id="llm" class="tab-content active">
            <div class="panel">
                <h2>DeepSeek模型管理</h2>
                <div class="form-group">
                    <label>模型状态</label>
                    <div>
                        <span class="status-indicator status-off" id="model-status"></span>
                        <span id="model-status-text">模型未加载</span>
                    </div>
                </div>
                <div class="form-group">
                    <label>模型路径</label>
                    <input type="text" id="model-path" value="models/deepseek-coder-1.3b-base">
                </div>
                <div class="form-group">
                    <button onclick="loadModel()">加载模型</button>
                    <button onclick="testModel()">测试模型</button>
                </div>
                <div class="form-group">
                    <label>测试问题</label>
                    <textarea id="test-question" rows="3" placeholder="输入测试问题..."></textarea>
                </div>
                <div class="form-group">
                    <label>模型回答</label>
                    <div id="model-response" style="background: #f8f9ff; padding: 15px; border-radius: 4px; min-height: 100px;">
                        等待测试...
                    </div>
                </div>
            </div>
        </div>

        <!-- 知识图谱标签页 -->
        <div id="knowledge" class="tab-content">
            <div class="panel">
                <h2>Neo4j数据库连接</h2>
                <div class="form-group">
                    <label>连接状态</label>
                    <div>
                        <span class="status-indicator status-off" id="neo4j-status"></span>
                        <span id="neo4j-status-text">数据库未连接</span>
                    </div>
                </div>
                <div class="form-group">
                    <label>服务器地址</label>
                    <input type="text" id="neo4j-uri" value="bolt://localhost:7687">
                </div>
                <div class="form-group">
                    <button onclick="connectNeo4j()">连接数据库</button>
                    <button onclick="generateSampleKnowledge()">生成示例知识</button>
                </div>
            </div>
        </div>

        <!-- 多模态交互标签页 -->
        <div id="multimodal" class="tab-content">
            <div class="panel">
                <h2>文本处理</h2>
                <div class="form-group">
                    <label>输入文本</label>
                    <textarea id="text-input" rows="4" placeholder="输入要处理的文本..."></textarea>
                </div>
                <div class="form-group">
                    <button onclick="processText()">处理文本</button>
                </div>
                <div class="form-group">
                    <label>处理结果</label>
                    <div id="text-result" style="background: #f8f9ff; padding: 15px; border-radius: 4px;">
                        等待处理...
                    </div>
                </div>
            </div>
        </div>

        <!-- 智能教学标签页 -->
        <div id="teaching" class="tab-content">
            <div class="panel">
                <h2>智能对话</h2>
                <div class="form-group">
                    <div id="chat-container" style="border: 1px solid #ddd; height: 300px; overflow-y: auto; padding: 10px; background: #fafafa;">
                        <div style="background: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <strong>PEPPER:</strong> 你好！我是你的AI教学助手，有什么问题吗？
                        </div>
                    </div>
                </div>
                <div class="form-group" style="display: flex; gap: 10px;">
                    <textarea id="chat-input" rows="2" placeholder="输入你的问题..." style="flex: 1;"></textarea>
                    <button onclick="sendMessage()" style="width: 80px;">发送</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 标签页切换
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.nav-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }

        // API调用函数
        async function callAPI(endpoint, method = 'GET', data = null) {
            const options = {
                method: method,
                headers: { 'Content-Type': 'application/json' }
            };
            if (data) options.body = JSON.stringify(data);

            const response = await fetch(`/api/${endpoint}`, options);
            return await response.json();
        }

        // 加载模型
        async function loadModel() {
            const modelPath = document.getElementById('model-path').value;
            const result = await callAPI('load_model', 'POST', { model_path: modelPath });

            if (result.status === 'success') {
                document.getElementById('model-status').className = 'status-indicator status-on';
                document.getElementById('model-status-text').textContent = '模型已加载';
                alert('模型加载成功');
            } else {
                alert('模型加载失败: ' + result.message);
            }
        }

        // 测试模型
        async function testModel() {
            const question = document.getElementById('test-question').value;
            if (!question.trim()) {
                alert('请输入测试问题');
                return;
            }

            const result = await callAPI('test_model', 'POST', { question: question });

            if (result.status === 'success') {
                document.getElementById('model-response').textContent = result.response;
            } else {
                document.getElementById('model-response').textContent = '错误: ' + result.message;
            }
        }

        // 连接Neo4j
        async function connectNeo4j() {
            const uri = document.getElementById('neo4j-uri').value;
            const result = await callAPI('connect_neo4j', 'POST', {
                uri: uri,
                user: 'neo4j',
                password: 'admin123'
            });

            if (result.status === 'success') {
                document.getElementById('neo4j-status').className = 'status-indicator status-on';
                document.getElementById('neo4j-status-text').textContent = '数据库已连接';
                alert('Neo4j连接成功');
            } else {
                alert('Neo4j连接失败: ' + result.message);
            }
        }

        // 生成示例知识
        async function generateSampleKnowledge() {
            const result = await callAPI('generate_sample_knowledge', 'POST');

            if (result.status === 'success') {
                alert('示例知识生成成功');
            } else {
                alert('示例知识生成失败: ' + result.message);
            }
        }

        // 处理文本
        async function processText() {
            const text = document.getElementById('text-input').value;
            if (!text.trim()) {
                alert('请输入要处理的文本');
                return;
            }

            const result = await callAPI('process_text', 'POST', { text: text });

            if (result.status === 'success') {
                document.getElementById('text-result').innerHTML = `
                    <strong>分词结果:</strong> ${result.tokens.join(', ')}<br>
                    <strong>关键词:</strong> ${result.keywords.join(', ')}<br>
                    <strong>问题类型:</strong> ${result.question_type}
                `;
            } else {
                document.getElementById('text-result').textContent = '错误: ' + result.message;
            }
        }

        // 发送消息
        async function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();

            if (!message) return;

            // 添加用户消息
            const chatContainer = document.getElementById('chat-container');
            const userMsg = document.createElement('div');
            userMsg.innerHTML = `<strong>你:</strong> ${message}`;
            userMsg.style.cssText = 'background: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 10px; text-align: right;';
            chatContainer.appendChild(userMsg);

            input.value = '';

            // 发送到AI助手
            const result = await callAPI('chat', 'POST', { message: message });

            if (result.status === 'success') {
                const assistantMsg = document.createElement('div');
                assistantMsg.innerHTML = `<strong>PEPPER:</strong> ${result.response}`;
                assistantMsg.style.cssText = 'background: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;';
                chatContainer.appendChild(assistantMsg);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }

        // 回车发送消息
        document.getElementById('chat-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>'''

    # 保存HTML模板
    template_path = os.path.join('interface', 'web_console', 'templates', 'index.html')
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"创建模板文件: {template_path}")

    print("模板结构创建完成！")


if __name__ == "__main__":
    create_template_structure()