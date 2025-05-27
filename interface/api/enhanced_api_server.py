#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能教学系统 - 增强API服务器
支持大语言模型、知识图谱、多模态交互、智能教学四大功能模块
"""
import gc
import json
import logging
import os
import sys
from datetime import datetime

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入各个模块
from ai_service.llm_module.llm_interface import LLMService
from ai_service.llm_module.lora_fine_tuning import LoRAFineTuner
from ai_service.knowledge_graph.knowledge_graph import KnowledgeGraph
from ai_service.knowledge_graph.education_knowledge_processor import EducationKnowledgeProcessor
from ai_service.multimodal.speech_recognition import SpeechRecognizer
from ai_service.multimodal.image_recognition import ImageRecognizer
from ai_service.multimodal.text_processor import TextProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PEPPER_API")

app = Flask(__name__)
CORS(app)

# 全局变量存储服务实例
services = {
    'llm': None,
    'fine_tuner': None,
    'knowledge_graph': None,
    'knowledge_processor': None,
    'speech_recognizer': None,
    'image_recognizer': None,
    'text_processor': None,
    'teaching': None
}

# 系统状态
system_status = {
    'model_loaded': False,
    'model_path': None,
    'model_quantization': None,
    'neo4j_connected': False,
    'training_progress': 0,
    'training_active': False,
    'training_start_time': None,
    'training_end_time': None,
    'training_output_dir': None,
    'training_config': None,
    'training_error': None,
    'auto_cleanup_enabled': True  # 新增：自动清理开关
}

# 配置
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


def force_cleanup_model(model_service=None):
    """强制清理模型和释放内存"""
    try:
        if model_service:
            # 清理指定模型
            if hasattr(model_service, 'model') and model_service.model:
                del model_service.model
            if hasattr(model_service, 'peft_model') and model_service.peft_model:
                del model_service.peft_model
            if hasattr(model_service, 'tokenizer') and model_service.tokenizer:
                del model_service.tokenizer

        # 强制垃圾回收
        gc.collect()

        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("模型内存清理完成")
        return True

    except Exception as e:
        logger.error(f"强制清理模型失败: {e}")
        return False

def initialize_services():
    """初始化服务"""
    try:
        # 初始化文本处理器
        services['text_processor'] = TextProcessor()

        # 初始化语音识别器
        services['speech_recognizer'] = SpeechRecognizer()

        # 初始化图像识别器
        services['image_recognizer'] = ImageRecognizer()

        logger.info("基础服务初始化完成")
    except Exception as e:
        logger.error(f"服务初始化失败: {e}")

# ======================== 大语言模型API ========================

@app.route('/api/load_model', methods=['POST'])
def load_model():
    """加载大语言模型 - 简化修复版本"""
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "请求数据为空"})

        model_path = data.get('model_path', '')
        use_4bit = data.get('use_4bit', True)
        use_8bit = data.get('use_8bit', False)

        if not model_path or model_path.strip() == '':
            return jsonify({"status": "error", "message": "模型路径不能为空"})

        logger.info(f"正在加载推理模型: {model_path}")

        if not os.path.exists(model_path):
            return jsonify({"status": "error", "message": f"模型路径不存在: {model_path}"})

        # 清理现有模型
        if services['llm']:
            logger.info("清理现有推理模型")
            force_cleanup_model(services['llm'])
            services['llm'] = None

        if services['fine_tuner']:
            logger.info("清理训练模型")
            force_cleanup_model(services['fine_tuner'])
            services['fine_tuner'] = None

        # 检测是否是LoRA模型
        adapter_files = ['adapter_config.json', 'adapter_model.bin', 'adapter_model.safetensors']
        is_lora_model = any(os.path.exists(os.path.join(model_path, f)) for f in adapter_files)

        try:
            if is_lora_model:
                logger.info("检测到LoRA模型，使用LoRA加载方式")

                # 确定基础模型路径
                base_model_path = "models/deepseek-coder-1.3b-base"

                # 尝试从training_info.json读取基础模型路径
                training_info_path = os.path.join(model_path, 'training_info.json')
                if os.path.exists(training_info_path):
                    try:
                        with open(training_info_path, 'r', encoding='utf-8') as f:
                            training_info = json.load(f)
                            saved_base_path = training_info.get('base_model')
                            if saved_base_path and os.path.exists(saved_base_path):
                                base_model_path = saved_base_path
                                logger.info(f"使用保存的基础模型路径: {base_model_path}")
                    except:
                        pass

                if not os.path.exists(base_model_path):
                    return jsonify({
                        "status": "error",
                        "message": f"基础模型不存在: {base_model_path}"
                    })

                # 加载基础模型和分词器
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                logger.info("加载分词器...")
                tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # 配置量化
                quantization_config = None
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32

                if device == "cuda" and (use_4bit or use_8bit):
                    if use_4bit:
                        logger.info("使用4bit量化")
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                        )
                    elif use_8bit:
                        logger.info("使用8bit量化")
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

                # 加载基础模型
                logger.info(f"加载基础模型: {base_model_path}")
                load_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}

                if quantization_config:
                    load_kwargs["quantization_config"] = quantization_config
                    load_kwargs["device_map"] = "auto"
                elif device == "cuda":
                    load_kwargs["device_map"] = "auto"

                base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)

                # 加载LoRA适配器
                logger.info(f"加载LoRA适配器: {model_path}")
                from peft import PeftModel
                peft_model = PeftModel.from_pretrained(base_model, model_path)

                # 创建简单的包装类
                class SimpleLoRAWrapper:
                    def __init__(self, peft_model, tokenizer):
                        self.peft_model = peft_model
                        self.model = peft_model
                        self.tokenizer = tokenizer

                    def generate_response(self, prompt, max_length=512):
                        try:
                            # 简化输入格式
                            if not prompt.startswith("Human:"):
                                formatted_prompt = f"Human: {prompt}\n\nAssistant:"
                            else:
                                formatted_prompt = prompt

                            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.peft_model.device)

                            with torch.no_grad():
                                outputs = self.peft_model.generate(
                                    **inputs,
                                    max_new_tokens=80,  # 进一步限制长度
                                    temperature=0.2,  # 更低的温度
                                    top_p=0.9,
                                    top_k=20,
                                    do_sample=True,
                                    pad_token_id=self.tokenizer.eos_token_id,
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    repetition_penalty=1.5,  # 更强的重复惩罚
                                    early_stopping=True,
                                    no_repeat_ngram_size=2,
                                )

                            # 只取新生成的部分
                            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

                            # 简单清理
                            response = response.strip()

                            # 移除明显的重复或无意义内容
                            if len(response) < 3:
                                return "请具体说明您的问题。"

                            # 确保回复完整
                            if not response.endswith(('。', '！', '？', '.', '!', '?')):
                                if len(response) > 10:
                                    response += '。'

                            return response[:200]  # 硬性限制最大长度

                        except Exception as e:
                            logger.error(f"生成回答失败: {e}")
                            return "抱歉，我暂时无法回答这个问题。"

                services['llm'] = SimpleLoRAWrapper(peft_model, tokenizer)
                logger.info("LoRA模型加载完成")

            else:
                # 原始模型
                logger.info("检测到原始模型，使用LLMService")
                services['llm'] = LLMService(model_path)

            # 更新状态
            system_status['model_loaded'] = True
            system_status['model_path'] = model_path
            system_status['model_quantization'] = '4bit' if use_4bit else '8bit' if use_8bit else 'full'

            # 测试模型
            try:
                test_response = services['llm'].generate_response("你好", 20)
                logger.info(f"模型测试成功: {test_response[:30]}...")
            except Exception as test_error:
                logger.warning(f"模型测试失败: {test_error}")

            return jsonify({
                "status": "success",
                "message": "模型加载成功",
                "model_path": model_path,
                "quantization": system_status['model_quantization'],
                "model_type": "LoRA" if is_lora_model else "Base"
            })

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return jsonify({"status": "error", "message": f"模型加载失败: {str(e)}"})

    except Exception as e:
        logger.error(f"load_model函数执行失败: {e}")
        return jsonify({"status": "error", "message": f"加载失败: {str(e)}"})

@app.route('/api/unload_model', methods=['POST'])
def unload_model():
    """卸载大语言模型"""
    try:
        logger.info("开始卸载模型...")

        # 清理推理模型
        if services['llm']:
            force_cleanup_model(services['llm'])
            services['llm'] = None

        # 清理训练模型
        if services['fine_tuner']:
            force_cleanup_model(services['fine_tuner'])
            services['fine_tuner'] = None

        system_status['model_loaded'] = False
        system_status['model_path'] = None
        system_status['model_quantization'] = None

        logger.info("所有模型已卸载")
        return jsonify({
            "status": "success",
            "message": "模型已卸载"
        })

    except Exception as e:
        logger.error(f"模型卸载失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/start_finetuning', methods=['POST'])
def start_finetuning():
    """开始LoRA微调"""
    try:
        if 'training_data' not in request.files:
            return jsonify({
                "status": "error",
                "message": "未找到训练数据文件"
            })

        file = request.files['training_data']
        model_path = request.form.get('model_path', '')
        epochs = int(request.form.get('epochs', 3))
        learning_rate = float(request.form.get('learning_rate', 0.0002))
        batch_size = int(request.form.get('batch_size', 2))  # 4bit训练默认batch_size=2
        use_4bit = request.form.get('use_4bit', 'true').lower() == 'true'
        use_8bit = request.form.get('use_8bit', 'false').lower() == 'true'

        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "未选择文件"
            })

        if not model_path or model_path.strip() == '':
            return jsonify({
                "status": "error",
                "message": "未指定模型路径"
            })

        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            return jsonify({
                "status": "error",
                "message": f"模型路径不存在: {model_path}"
            })

        # 清理现有的推理模型，为训练让出内存
        if services['llm']:
            logger.info("清理推理模型，为训练让出内存")
            force_cleanup_model(services['llm'])
            services['llm'] = None
            system_status['model_loaded'] = False

        # 保存文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 生成带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quantization_suffix = "_4bit" if use_4bit else "_8bit" if use_8bit else ""
        output_dir = f"models/deepseek-{timestamp}{quantization_suffix}"

        # 创建微调器，使用指定的模型路径
        services['fine_tuner'] = LoRAFineTuner(
            base_model_path=model_path,  # 使用前端传递的模型路径
            output_dir=output_dir
        )

        # 记录训练信息到系统状态
        system_status['training_output_dir'] = output_dir
        system_status['training_start_time'] = datetime.now().isoformat()
        system_status['training_active'] = True
        system_status['training_progress'] = 0
        system_status['training_config'] = {
            'model_path': model_path,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'use_4bit': use_4bit,
            'use_8bit': use_8bit,
            'quantization': '4bit' if use_4bit else '8bit' if use_8bit else 'none'
        }

        # 启动异步训练任务
        import threading
        def run_training():
            try:
                logger.info(f"开始{system_status['training_config']['quantization']}量化训练")

                # 加载基础模型(0-10%)
                system_status['training_progress'] = 5
                success = services['fine_tuner'].load_base_model(use_4bit=use_4bit, use_8bit=use_8bit)
                if not success:
                    system_status['training_active'] = False
                    system_status['training_error'] = "模型加载失败"
                    logger.error("训练模型加载失败")
                    return

                system_status['training_progress'] = 10
                logger.info("训练模型加载成功，准备LoRA配置")

                # 准备LoRA配置 (10-15%)
                services['fine_tuner'].prepare_lora_config()
                system_status['training_progress'] = 12

                logger.info("准备数据集")
                # 准备数据集
                dataset = services['fine_tuner'].prepare_dataset(data_path=filepath)
                if dataset is None:
                    system_status['training_active'] = False
                    system_status['training_error'] = "数据集准备失败"
                    logger.error("数据集准备失败")
                    return

                system_status['training_progress'] = 15
                logger.info("开始LoRA微调训练")

                # 定义进度回调函数 - 这个函数会被训练器调用
                def training_progress_callback(progress):
                    # progress 是 0-100 的真实训练进度
                    # 映射到 15-99 的区间
                    mapped_progress = 15 + (progress * 0.84)  # 84% = 99-15
                    system_status['training_progress'] = int(min(99, mapped_progress))
                    logger.info(f"训练进度: {system_status['training_progress']}%")

                # 开始训练
                success = services['fine_tuner'].train(
                    dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    progress_callback=training_progress_callback
                )

                if success:
                    system_status['training_progress'] = 100
                    system_status['training_end_time'] = datetime.now().isoformat()
                    logger.info(f"训练完成，模型已保存到: {output_dir}")

                    # 训练完成后的自动清理
                    logger.info("训练完成，开始自动清理训练模型...")

                    # 等待确保模型保存完成
                    import time
                    time.sleep(3)  # 增加等待时间到3秒

                    # 执行清理
                    try:
                        if services['fine_tuner']:
                            force_cleanup_model(services['fine_tuner'])
                            services['fine_tuner'] = None
                            logger.info("✅ 训练模型已自动清理，内存已释放")
                        else:
                            logger.info("训练模型已经为空，无需清理")
                    except Exception as cleanup_error:
                        logger.error(f"自动清理失败: {cleanup_error}")

                else:
                    system_status['training_error'] = "训练过程失败"
                    logger.error("训练过程失败")

            except Exception as e:
                logger.error(f"训练过程出错: {e}")
                system_status['training_error'] = str(e)
            finally:
                system_status['training_active'] = False
                # 清理临时文件
                try:
                    os.remove(filepath)
                    logger.info("训练数据临时文件已清理")
                except:
                    pass

                # 最终清理检查
                try:
                    if services['fine_tuner'] is not None:
                        logger.info("执行最终保险清理...")
                        force_cleanup_model(services['fine_tuner'])
                        services['fine_tuner'] = None
                        logger.info("最终清理完成")
                except Exception as final_cleanup_error:
                    logger.error(f"最终清理失败: {final_cleanup_error}")

        # 启动训练线程
        thread = threading.Thread(target=run_training)
        thread.daemon = True
        thread.start()


        logger.info(f"LoRA微调已开始，输出目录: {output_dir}")
        return jsonify({
            "status": "success",
            "message": "微调已开始",
            "output_dir": output_dir,
            "timestamp": timestamp,
            "config": system_status['training_config']
        })

    except Exception as e:
        logger.error(f"微调启动失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/training_progress', methods=['GET'])
def get_training_progress():
    """获取训练进度"""
    progress_info = {
        "status": "success",
        "progress": system_status['training_progress'],
        "active": system_status['training_active'],
        "output_dir": system_status.get('training_output_dir', ''),
        "start_time": system_status.get('training_start_time', ''),
        "config": system_status.get('training_config', {}),
        "auto_cleanup_enabled": system_status.get('auto_cleanup_enabled', True)
    }

    # 添加结束时间（如果有）
    if 'training_end_time' in system_status:
        progress_info["end_time"] = system_status['training_end_time']

    # 添加错误信息（如果有）
    if 'training_error' in system_status:
        progress_info["error"] = system_status['training_error']

    # 计算预估剩余时间
    if system_status['training_active'] and 'training_start_time' in system_status:
        try:
            start_time = datetime.fromisoformat(system_status['training_start_time'])
            elapsed_time = (datetime.now() - start_time).total_seconds()
            progress = system_status['training_progress']

            if progress > 5:  # 避免除零错误
                estimated_total_time = elapsed_time * 100 / progress
                remaining_time = estimated_total_time - elapsed_time
                progress_info["estimated_remaining_seconds"] = max(0, int(remaining_time))
        except:
            pass

    # 添加内存使用信息
    try:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            gpu_allocated = torch.cuda.memory_allocated() / 1024 ** 3
            gpu_cached = torch.cuda.memory_reserved() / 1024 ** 3

            progress_info["memory_info"] = {
                "gpu_total_gb": round(gpu_memory, 2),
                "gpu_allocated_gb": round(gpu_allocated, 2),
                "gpu_cached_gb": round(gpu_cached, 2),
                "gpu_free_gb": round(gpu_memory - gpu_cached, 2)
            }
    except:
        pass

    return jsonify(progress_info)

@app.route('/api/toggle_auto_cleanup', methods=['POST'])
def toggle_auto_cleanup():
    """切换训练完成后的自动清理功能"""
    try:
        data = request.json
        enabled = data.get('enabled', True)

        system_status['auto_cleanup_enabled'] = enabled

        logger.info(f"自动清理功能已{'启用' if enabled else '禁用'}")

        return jsonify({
            "status": "success",
            "message": f"自动清理功能已{'启用' if enabled else '禁用'}",
            "auto_cleanup_enabled": enabled
        })

    except Exception as e:
        logger.error(f"切换自动清理功能失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/manual_cleanup', methods=['POST'])
def manual_cleanup():
    """手动清理模型内存"""
    try:
        cleanup_results = []

        # 清理推理模型
        if services['llm']:
            force_cleanup_model(services['llm'])
            services['llm'] = None
            system_status['model_loaded'] = False
            cleanup_results.append("推理模型已清理")

        # 清理训练模型
        if services['fine_tuner']:
            force_cleanup_model(services['fine_tuner'])
            services['fine_tuner'] = None
            cleanup_results.append("训练模型已清理")

        if not cleanup_results:
            cleanup_results.append("没有需要清理的模型")

        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            cleanup_results.append("GPU缓存已清理")

        logger.info(f"手动清理完成: {', '.join(cleanup_results)}")

        return jsonify({
            "status": "success",
            "message": "手动清理完成",
            "details": cleanup_results
        })

    except Exception as e:
        logger.error(f"手动清理失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/system_memory_info', methods=['GET'])
def get_system_memory_info():
    """获取系统内存使用信息"""
    try:
        import psutil

        memory_info = {
            "cpu_memory": {
                "total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
                "available_gb": round(psutil.virtual_memory().available / 1024 ** 3, 2),
                "used_gb": round(psutil.virtual_memory().used / 1024 ** 3, 2),
                "percent": psutil.virtual_memory().percent
            }
        }

        # GPU内存信息
        if torch.cuda.is_available():
            memory_info["gpu_memory"] = {
                "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2),
                "allocated_gb": round(torch.cuda.memory_allocated() / 1024 ** 3, 2),
                "cached_gb": round(torch.cuda.memory_reserved() / 1024 ** 3, 2),
                "free_gb": round(
                    (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024 ** 3, 2)
            }
        else:
            memory_info["gpu_memory"] = {"available": False}

        # 模型状态
        memory_info["model_status"] = {
            "inference_model_loaded": services['llm'] is not None,
            "training_model_loaded": services['fine_tuner'] is not None,
            "training_active": system_status['training_active']
        }

        return jsonify({
            "status": "success",
            "memory_info": memory_info
        })

    except Exception as e:
        logger.error(f"获取内存信息失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# ======================== 其他API保持不变 ========================

@app.route('/api/test_model', methods=['POST'])
def test_model():
    """测试大语言模型"""
    try:
        if not services['llm']:
            return jsonify({
                "status": "error",
                "message": "模型未加载"
            })

        data = request.json
        question = data.get('question', '')
        max_tokens = data.get('max_tokens', 512)
        temperature = data.get('temperature', 0.7)

        if not question.strip():
            return jsonify({
                "status": "error",
                "message": "问题不能为空"
            })

        # 使用模型生成回答
        try:
            if hasattr(services['llm'], 'generate_response'):
                response = services['llm'].generate_response(question, max_tokens)
            else:
                logger.error(f"模型对象没有 generate_response 方法: {type(services['llm'])}")
                return jsonify({
                    "status": "error",
                    "message": "模型接口不兼容"
                })

            return jsonify({
                "status": "success",
                "result": response,  # 注意：这里用 result，前端期望这个字段
                "model_quantization": system_status.get('model_quantization', 'unknown'),
                "model_type": type(services['llm']).__name__
            })

        except Exception as model_error:
            logger.error(f"模型推理失败: {model_error}")
            return jsonify({
                "status": "error",
                "message": f"模型推理失败: {str(model_error)}"
            })

    except Exception as e:
        logger.error(f"模型测试失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/debug_model_status', methods=['GET'])
def debug_model_status():
    """调试模型状态"""
    try:
        debug_info = {
            "services_llm_exists": services['llm'] is not None,
            "services_llm_type": type(services['llm']).__name__ if services['llm'] else None,
            "system_status_model_loaded": system_status['model_loaded'],
            "system_status_model_path": system_status.get('model_path'),
            "system_status_quantization": system_status.get('model_quantization'),
        }

        # 检查模型方法
        if services['llm']:
            debug_info["model_methods"] = [method for method in dir(services['llm']) if not method.startswith('_')]
            debug_info["has_generate_response"] = hasattr(services['llm'], 'generate_response')

        # 检查知识图谱状态
        debug_info["neo4j_connected"] = system_status['neo4j_connected']
        debug_info["knowledge_graph_exists"] = services['knowledge_graph'] is not None

        return jsonify({
            "status": "success",
            "debug_info": debug_info
        })

    except Exception as e:
        logger.error(f"调试状态获取失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# ======================== 知识图谱API ========================

@app.route('/api/connect_neo4j', methods=['POST'])
def connect_neo4j():
    """连接Neo4j数据库"""
    try:
        data = request.json
        uri = data.get('uri', 'bolt://localhost:7687')
        user = data.get('user', 'neo4j')
        password = data.get('password', 'password')

        services['knowledge_graph'] = KnowledgeGraph(uri, user, password)
        services['knowledge_processor'] = EducationKnowledgeProcessor(uri, user, password)

        # 测试连接
        test_result = services['knowledge_graph'].query("RETURN 1 as test")
        if test_result:
            system_status['neo4j_connected'] = True
            logger.info("Neo4j连接成功")
            return jsonify({
                "status": "success",
                "message": "数据库连接成功"
            })
        else:
            raise Exception("连接测试失败")

    except Exception as e:
        logger.error(f"Neo4j连接失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/generate_sample_knowledge', methods=['POST'])
def generate_sample_knowledge():
    """生成示例知识"""
    try:
        if not services['knowledge_processor']:
            return jsonify({
                "status": "error",
                "message": "未连接知识图谱数据库"
            })

        count = services['knowledge_processor'].create_educational_knowledge_base()

        logger.info("示例知识生成成功")
        return jsonify({
            "status": "success",
            "message": "示例知识生成成功",
            "count": count
        })

    except Exception as e:
        logger.error(f"示例知识生成失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/chat', methods=['POST'])
def chat():
    """智能对话"""
    try:
        data = request.json
        message = data.get('message', '')
        use_knowledge_graph = data.get('use_knowledge_graph', False)

        if not message.strip():
            return jsonify({
                "status": "error",
                "message": "消息不能为空"
            })

        # 检查模型是否加载
        if not services['llm']:
            return jsonify({
                "status": "error",
                "message": "模型未加载，请先在左侧面板加载模型"
            })

            # 检查模型类型并调用相应的生成方法
        try:
            if hasattr(services['llm'], 'generate_response'):
                # 标准 LLMService 或 LoRAFineTuner 都有这个方法
                if use_knowledge_graph and services['knowledge_graph'] and system_status['neo4j_connected']:
                    # 使用知识图谱增强回答
                    response = chat_with_knowledge_graph(message)
                else:
                    # 直接使用模型回答
                    response = services['llm'].generate_response(message)
            else:
                logger.error(f"模型对象类型错误: {type(services['llm'])}")
                return jsonify({
                    "status": "error",
                    "message": f"模型对象类型不支持: {type(services['llm'])}"
                })

            return jsonify({
                "status": "success",
                "response": response,
                "model_type": type(services['llm']).__name__,
                "use_knowledge_graph": use_knowledge_graph and system_status['neo4j_connected']
            })

        except Exception as model_error:
            logger.error(f"模型推理失败: {model_error}")
            return jsonify({
                "status": "error",
                "message": f"模型推理失败: {str(model_error)}"
            })

    except Exception as e:
        logger.error(f"智能对话失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

def chat_with_knowledge_graph(message):
    """结合知识图谱生成回答"""
    try:
        # 提取关键词 (简单版本)
        keywords = []
        if services['text_processor']:
            keywords = services['text_processor'].extract_keywords(message)
        else:
            # 简单的关键词提取
            import jieba
            words = jieba.cut(message)
            keywords = [word for word in words if len(word) > 1][:5]

        # 从知识图谱查询相关信息
        knowledge_items = []
        for keyword in keywords:
            try:
                items = services['knowledge_graph'].find_related_knowledge(keyword)
                knowledge_items.extend(items)
            except Exception as e:
                logger.warning(f"查询知识图谱失败 (关键词: {keyword}): {e}")

        # 构建知识上下文
        knowledge_context = ""
        if knowledge_items:
            knowledge_context = "\n相关知识:\n"
            for item in knowledge_items[:5]:  # 限制知识条目数量
                if isinstance(item, dict) and 'start_node' in item and 'end_node' in item:
                    start_name = item['start_node'].get('name', '')
                    relationship = item.get('relationship', '')
                    end_name = item['end_node'].get('name', '')
                    knowledge_context += f"- {start_name} {relationship} {end_name}\n"

        # 构建增强的提示
        enhanced_prompt = f"""基于以下知识用中文回答问题:
{knowledge_context}

用户问题: {message}

请提供准确、有帮助的回答:"""

        # 使用模型生成回答
        response = services['llm'].generate_response(enhanced_prompt)

        return response

    except Exception as e:
        logger.error(f"知识图谱增强对话失败: {e}")
        # 降级到普通对话
        return services['llm'].generate_response(message)

# ======================== 模型发现API ========================

@app.route('/api/discover_models', methods=['GET'])
def discover_models():
    """发现可用的模型"""
    try:
        models_dir = 'models'
        available_models = []

        # 检查models目录是否存在
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
            logger.info("创建了models目录")

        # 遍历models目录
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)

            # 只处理文件夹
            if os.path.isdir(item_path):
                model_info = {
                    "name": item,
                    "path": item_path,
                    "display_name": item,
                    "valid": False,
                    "size": 0,
                    "files": []
                }

                # 检查模型文件
                try:
                    model_files = []
                    total_size = 0

                    for root, dirs, files in os.walk(item_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            file_size = os.path.getsize(file_path)
                            total_size += file_size

                            # 检查关键模型文件
                            if file.endswith(('.bin', '.safetensors', '.pt', '.pth', '.ckpt')):
                                model_files.append({
                                    "name": file,
                                    "size": file_size,
                                    "type": "model"
                                })
                            elif file in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
                                model_files.append({
                                    "name": file,
                                    "size": file_size,
                                    "type": "config"
                                })

                    # 判断是否为有效模型
                    has_model_file = any(f['type'] == 'model' for f in model_files)
                    has_config = any(f['type'] == 'config' for f in model_files)

                    if has_model_file or has_config:
                        model_info["valid"] = True

                    model_info["size"] = total_size
                    model_info["files"] = model_files

                    # 生成显示名称
                    if "deepseek" in item.lower():
                        model_info["display_name"] = f"🧠 DeepSeek - {item}"
                    else:
                        model_info["display_name"] = f"🤖 {item}"

                    available_models.append(model_info)

                except Exception as e:
                    logger.warning(f"扫描模型目录 {item_path} 时出错: {e}")
                    continue

        available_models.sort(key=lambda x: (not x["valid"], x["name"]))

        logger.info(f"发现 {len(available_models)} 个模型目录")

        return jsonify({
            "status": "success",
            "models": available_models,
            "count": len(available_models)
        })

    except Exception as e:
        logger.error(f"模型发现失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/get_model_info', methods=['POST'])
def get_model_info():
    """获取特定模型的详细信息"""
    try:
        data = request.json
        model_path = data.get('model_path', '')

        if not model_path or not os.path.exists(model_path):
            return jsonify({
                "status": "error",
                "message": "模型路径不存在"
            })

        model_info = {
            "path": model_path,
            "name": os.path.basename(model_path),
            "files": [],
            "total_size": 0,
            "model_files_count": 0,
            "config_files": [],
            "tokenizer_files": []
        }

        # 扫描模型文件
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path)
                    model_info["total_size"] += file_size

                    relative_path = os.path.relpath(file_path, model_path)

                    file_info = {
                        "name": file,
                        "path": relative_path,
                        "size": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2)
                    }

                    if file.endswith(('.bin', '.safetensors', '.pt', '.pth', '.ckpt')):
                        file_info["type"] = "model"
                        model_info["model_files_count"] += 1
                    elif file == 'config.json':
                        file_info["type"] = "config"
                        model_info["config_files"].append(file_info)

                        # 尝试读取配置信息
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                                model_info["model_type"] = config.get("model_type", "unknown")
                                model_info["architectures"] = config.get("architectures", [])
                                model_info["vocab_size"] = config.get("vocab_size", 0)
                        except:
                            pass
                    elif file in ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'merges.txt']:
                        file_info["type"] = "tokenizer"
                        model_info["tokenizer_files"].append(file_info)
                    else:
                        file_info["type"] = "other"

                    model_info["files"].append(file_info)

                except Exception as e:
                    logger.warning(f"无法获取文件信息 {file_path}: {e}")
                    continue

        # 转换总大小为可读格式
        total_size_mb = model_info["total_size"] / (1024 * 1024)
        if total_size_mb > 1024:
            model_info["total_size_display"] = f"{total_size_mb / 1024:.1f} GB"
        else:
            model_info["total_size_display"] = f"{total_size_mb:.1f} MB"

        return jsonify({
            "status": "success",
            "model_info": model_info
        })

    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/check_quantization_support', methods=['GET'])
def check_quantization_support():
    """检查量化训练支持情况"""
    try:
        support_info = {
            "bitsandbytes_available": False,
            "accelerate_available": False,
            "peft_available": False,
            "torch_version": "",
            "cuda_available": False,
            "gpu_name": "",  # 新增：GPU名称
            "recommendations": []
        }

        # 检查必要的库
        try:
            import bitsandbytes
            support_info["bitsandbytes_available"] = True
            support_info["bitsandbytes_version"] = bitsandbytes.__version__
        except ImportError:
            support_info["recommendations"].append("需要安装bitsandbytes: pip install bitsandbytes")

        try:
            import accelerate
            support_info["accelerate_available"] = True
            support_info["accelerate_version"] = accelerate.__version__
        except ImportError:
            support_info["recommendations"].append("需要安装accelerate: pip install accelerate")

        try:
            import peft
            support_info["peft_available"] = True
            support_info["peft_version"] = peft.__version__
        except ImportError:
            support_info["recommendations"].append("需要安装peft: pip install peft")

        # 检查PyTorch版本
        try:
            import torch
            support_info["torch_version"] = torch.__version__
            support_info["cuda_available"] = torch.cuda.is_available()

            if torch.cuda.is_available():
                support_info["cuda_version"] = torch.version.cuda
                # 获取GPU名称
                support_info["gpu_name"] = torch.cuda.get_device_name(0)
                logger.info(f"检测到GPU: {support_info['gpu_name']}")
            else:
                support_info["gpu_name"] = "无GPU"

        except ImportError:
            support_info["recommendations"].append("需要安装PyTorch")

        # 判断是否支持量化训练
        quantization_ready = (
                support_info["bitsandbytes_available"] and
                support_info["accelerate_available"] and
                support_info["peft_available"]
        )

        support_info["quantization_ready"] = quantization_ready

        if not quantization_ready:
            support_info["recommendations"].append("安装命令: pip install bitsandbytes accelerate peft")

        return jsonify({
            "status": "success",
            "support_info": support_info
        })

    except Exception as e:
        logger.error(f"检查量化支持失败: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# ======================== 系统API ========================

@app.route('/api/system_status', methods=['GET'])
def get_system_status():
    """获取系统状态"""
    return jsonify({
        "status": "success",
        "system_status": system_status,
        "services": {
            "llm": services['llm'] is not None,
            "fine_tuner": services['fine_tuner'] is not None,
            "knowledge_graph": services['knowledge_graph'] is not None,
            "teaching": services['teaching'] is not None,
            "speech_recognizer": services['speech_recognizer'] is not None,
            "image_recognizer": services['image_recognizer'] is not None,
            "text_processor": services['text_processor'] is not None
        }
    })

@app.route('/')
def index():
    """主页"""
    try:
        template_path = os.path.join('interface', 'web_console', 'templates', 'index.html')
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return html_content
        else:
            return """
            <h1>智能教学系统</h1>
            <p>API服务器正在运行</p>
            <p>请访问 <a href="/api/system_status">/api/system_status</a> 查看系统状态</p>
            """
    except Exception as e:
        logger.error(f"主页路由出错: {e}")
        return f"<h1>系统错误</h1><p>{str(e)}</p>"

# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "API接口不存在"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "message": "服务器内部错误"
    }), 500

if __name__ == '__main__':
    # 初始化服务
    initialize_services()

    logger.info("智能教学系统API服务器启动中...")
    logger.info("访问地址: http://localhost:5000")

    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=True)