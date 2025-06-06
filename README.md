# 智能教学系统

基于DeepSeek模型的个性化AI教学助手，集成大语言模型、知识图谱、多模态交互、智能教学四大功能模块。

## 🌟 主要功能

### 🧠 大语言模型集成
- **DeepSeek模型接口**: 支持本地模型加载与推理
- **LoRA微调**: 基于教育数据的轻量级模型微调
- **参数调节**: 实时调整Temperature、Max Tokens等参数
- **模型测试**: 内置测试工具验证模型性能

### 🗂️ 知识图谱系统
- **Neo4j数据库**: 图数据库存储教育知识
- **知识导入**: 支持JSON/CSV/TXT格式知识文件导入
- **Cypher查询**: 可视化查询界面，支持复杂图查询
- **教育知识处理**: 自动生成示例教育知识图谱

### 🎯 多模态交互
- **语音识别**: 支持音频文件上传和实时录音识别
- **图像识别**: 基于ViT模型的图像内容识别
- **文本处理**: 中文分词、关键词提取、问题分类

### 📚 智能教学功能
- **学生管理**: 创建和管理学生档案，记录学习风格
- **个性化学习路径**: 根据学生特点生成定制化学习计划
- **资源推荐**: 智能推荐适合的学习资源和材料
- **报告生成**: 自动生成学习进度和表现报告

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <项目地址>
cd PEPPER_AI_Classroom

# 安装Python依赖
pip install flask flask-cors transformers torch jieba spacy neo4j pillow numpy pandas

# 下载中文分词模型
python -m spacy download zh_core_web_sm
```

### 2. 配置Neo4j（可选）

```bash
# 下载并启动Neo4j
# 访问 http://localhost:7474
# 设置用户名: neo4j, 密码: adminadmin
```

### 3. 下载语言模型（可选）

```bash
# 创建模型目录
mkdir -p models
cd models

# 下载DeepSeek模型（示例）
# 或将已有模型放置到 models/deepseek-coder-1.3b-base/ 目录
```

### 4. 启动系统

```bash
# 启动完整系统
python run_pepper_web_system.py

# 指定端口启动
python run_pepper_web_system.py --port 8080

# 调试模式启动
python run_pepper_web_system.py --debug

# 仅检查环境
python run_pepper_web_system.py --check-only
```

### 5. 访问系统

打开浏览器访问 `http://localhost:5000`

## 📖 使用指南

### 大语言模型模块

1. **加载模型**
   - 在"大语言模型"标签页中
   - 输入模型路径（默认: models/deepseek-coder-1.3b-base）
   - 点击"加载模型"按钮

2. **测试模型**
   - 在测试区域输入问题
   - 调整Temperature和Max Tokens参数
   - 点击"测试模型"查看回答

3. **LoRA微调**
   - 上传训练数据文件（JSON/CSV格式）
   - 设置训练轮数和学习率
   - 点击"开始微调"启动训练

### 知识图谱模块

1. **连接Neo4j**
   - 输入数据库连接信息
   - 默认: bolt://localhost:7687, neo4j/adminadmin
   - 点击"连接数据库"

2. **导入知识**
   - 上传知识文件或点击"生成示例知识"
   - 支持JSON、CSV、TXT格式
   - 查看导入统计信息

3. **查询知识**
   - 使用Cypher查询语言
   - 例如: `MATCH (n) RETURN n LIMIT 10`
   - 查看查询结果和图谱统计

### 多模态交互模块

1. **语音识别**
   - 点击"开始录音"进行实时录音
   - 或上传音频文件（WAV/MP3/M4A）
   - 查看识别结果

2. **图像识别**
   - 上传图像文件（JPG/PNG/GIF）
   - 系统自动识别图像内容
   - 显示识别结果和置信度

3. **文本处理**
   - 输入要处理的文本
   - 查看分词、关键词、问题类型分析结果

### 智能教学模块

1. **学生管理**
   - 添加新学生档案
   - 设置学习风格（视觉型/听觉型/阅读型/动手型）
   - 加载和查看学生信息

2. **智能对话**
   - 选择学生后进行个性化对话
   - 系统会根据学生特点调整回答
   - 自动记录学习交互

3. **学习路径生成**
   - 输入学习目标（如"Python循环语句"）
   - 系统生成个性化学习路径
   - 包含学习步骤和建议

4. **资源推荐**
   - 选择主题获取相关学习资源
   - 根据学生学习风格个性化推荐
   - 包含视频、文章、练习等多种类型

5. **报告生成**
   - 选择报告类型和时间范围
   - 生成学习进度、表现分析报告
   - 支持HTML格式导出

## ⚙️ 配置文件

系统配置文件 `config.json` 包含以下主要配置项：

- `llm`: 语言模型相关配置
- `knowledge_graph`: Neo4j数据库配置
- `multimodal`: 多模态交互配置
- `teaching`: 智能教学配置
- `data_paths`: 数据存储路径配置

## 🔧 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 确保有足够的内存空间
   - 考虑使用CPU模式

2. **Neo4j连接失败**
   - 确保Neo4j服务已启动
   - 检查连接信息（地址、用户名、密码）
   - 系统可在无Neo4j的情况下运行其他功能

3. **语音识别不工作**
   - 检查麦克风权限
   - 确保音频文件格式支持
   - 网络连接是否正常（使用在线识别服务）

4. **依赖包安装问题**
   - 使用 `pip install -r requirements.txt` 安装所有依赖
   - 如有GPU，可安装 `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### 日志查看

系统日志保存在以下位置：
- 主日志: `pepper_system.log`
- 错误日志: 控制台输出
- 详细日志: 各模块单独日志文件

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

本项目采用MIT许可证。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 项目Issues
- 邮箱联系

---

**注意**: 本系统为教育研究项目，请在合规范围内使用。