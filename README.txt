PEPPER_AI_Classroom/
├── ai_service/                             # AI服务模块
│   ├── knowledge_graph/                    # 知识图谱模块
│   │   ├── __init__.py
│   │   ├── knowledge_graph.py              # 知识图谱操作类
│   │   └── education_knowledge_processor.py # 教育知识处理器
│   ├── llm_module/                         # 大语言模型模块
│   │   ├── __init__.py
│   │   ├── llm_interface.py                # 大模型接口
│   │   ├── langchain_integration.py        # LangChain集成
│   │   └── lora_fine_tuning.py             # LoRA微调模块
│   ├── multimodal/                         # 多模态交互模块
│   │   ├── __init__.py
│   │   ├── speech_recognition.py           # 语音识别
│   │   ├── image_recognition.py            # 图像识别
│   │   └── text_processor.py               # 文本处理
│   ├── teaching_module/                    # 个性化教学模块
│   │   ├── __init__.py
│   │   └── personalized_teaching.py        # 个性化教学实现
│   └── __init__.py
├── interface/                              # 接口模块
│   ├── bridge/                             # 桥接通信模块
│   │   ├── __init__.py
│   │   ├── websocket_bridge.py             # WebSocket服务桥接
│   │   └── websocket_client.py             # WebSocket客户端
│   ├── api/                                # API接口
│   │   ├── __init__.py
│   │   └── api_server.py                   # API服务器
│   └── __init__.py
├── pepper_robot/                           # PEPPER机器人控制模块
│   ├── motion_module/                      # 动作模块
│   │   ├── __init__.py
│   │   └── gestures.py                     # 手势和动作
│   ├── robot_control/                      # 机器人控制
│   │   ├── __init__.py
│   │   └── robot_controller.py             # 机器人控制器
│   ├── sensor_module/                      # 传感器模块
│   │   ├── __init__.py
│   │   └── sensor_handler.py               # 传感器处理
│   ├── simulation/                         # 模拟器模块
│   │   ├── __init__.py
│   │   └── pepper_simulator.py             # PEPPER机器人模拟器
│   └── __init__.py
├── tests/                                  # 测试模块
│   ├── __init__.py
│   ├── system_test.py                      # 系统测试
│   └── test_local_model.py                 # 本地模型测试
├── data/                                   # 数据目录
│   ├── student_profiles/                   # 学生档案
│   └── course_materials/                   # 课程材料
├── models/                                 # 模型目录
│   └── deepseek-coder-1.3b-base/           # 示例模型目录
├── scripts/                                # 脚本工具
│   └── download_model.py                   # 模型下载脚本
├── assets/                                 # 资源文件
│   └── pepper_robot.png                    # 机器人图像
├── config.json                             # 配置文件
├── integrated_system.py                    # 集成系统
├── run_pepper_system.py                    # 系统启动脚本
├── run_demo.py                             # 演示应用
├── check_cuda.py                           # CUDA检查脚本
├── monitor_gpu.py                          # GPU监控脚本
├── requirements.txt                        # 基本依赖项
├── requirements_gpu.txt                    # GPU加速依赖项
└── README.md                               # 项目说明