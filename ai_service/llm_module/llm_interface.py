import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMService:
    def __init__(self, model_path="models/deepseek-coder-1.3b-base"):
        print(f"正在加载本地DeepSeek模型: {model_path}...")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 确定设备和数据类型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"使用设备: {device}, 数据类型: {dtype}")

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # 如果使用CPU，确保模型在正确设备上
        if not torch.cuda.is_available():
            self.model = self.model.to(device)

        print("DeepSeek模型加载完成")

    def generate_response(self, prompt, max_length=1024):
        """生成回答 - 强化中文输出"""
        # 为DeepSeek模型调整提示格式，强制使用中文
        if not prompt.startswith("用户:"):
            # 添加中文系统提示，强制模型使用中文回答
            system_prompt = """你是PEPPER智能教学助手，一个专业的中文AI教学助手。请注意：
1. 必须用中文回答所有问题
2. 回答要专业、详细、易懂
3. 针对教育场景提供帮助
4. 保持友好和耐心的语气

"""
            prompt = f"{system_prompt}用户: {prompt}\n\n助手: "

        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 设置生成参数，优化中文生成
        gen_kwargs = {
            "max_new_tokens": min(256, max_length),  # 减少长度避免重复
            "temperature": 0.8,  # 稍微提高创造性
            "top_p": 0.9,
            "top_k": 40,  # 减少top_k提高质量
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.2,  # 增加重复惩罚
            "no_repeat_ngram_size": 3,  # 避免3-gram重复
        }

        # 生成回答
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # 解码输出
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 提取回答部分（去除提示部分）
        if "助手:" in output:
            response = output.split("助手:")[-1].strip()
        else:
            response = output[len(prompt):].strip()

        # 后处理：如果回答仍然是英文，添加中文提示重新生成
        if self._is_mainly_english(response):
            print("检测到英文回答，重新生成中文回答...")
            chinese_prompt = f"""你必须用中文回答。用户问题：{prompt.split('用户:')[-1].split('助手:')[0].strip()}

请用中文详细回答："""
            return self._force_chinese_response(chinese_prompt)

        return response

    def _is_mainly_english(self, text):
        """检测文本是否主要是英文"""
        if not text:
            return False

        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = english_chars + chinese_chars

        if total_chars == 0:
            return False

        return english_chars > chinese_chars

    def _force_chinese_response(self, prompt):
        """强制生成中文回答"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 30,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.3,
        }

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output[len(prompt):].strip()

        # 如果还是英文，返回默认中文回答
        if self._is_mainly_english(response):
            return "抱歉，我是PEPPER智能教学助手。我会用中文为您解答问题。请告诉我您需要什么帮助？"

        return response

    def answer_with_knowledge(self, question, knowledge):
        """结合知识图谱信息回答问题 - 强化中文输出"""
        # 将知识图谱的信息格式化为提示的一部分
        knowledge_str = ""
        for item in knowledge:
            if isinstance(item, dict) and 'start_node' in item and 'relationship' in item and 'end_node' in item:
                start_node = item['start_node'].get('name', '')
                relationship = item['relationship']
                end_node = item['end_node'].get('name', '')

                if 'description' in item['start_node']:
                    start_desc = item['start_node']['description']
                    knowledge_str += f"{start_node}: {start_desc}\n"

                if 'description' in item['end_node']:
                    end_desc = item['end_node']['description']
                    knowledge_str += f"{end_node}: {end_desc}\n"

                knowledge_str += f"{start_node} -- {relationship} --> {end_node}\n"
            else:
                # 处理其他格式的知识项
                knowledge_str += f"{str(item)}\n"

        # 构建中文提示
        prompt = f"""你是PEPPER智能教学助手。请基于以下知识用中文回答问题：

相关知识：
{knowledge_str}

用户问题：{question}

请用中文详细、准确地回答，语言要专业但易懂："""

        return self.generate_response(prompt)