from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


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
        """生成回答"""
        # 为DeepSeek模型调整提示格式
        if not prompt.startswith("Human:"):
            prompt = f"Human: {prompt}\n\nAssistant:"

        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 设置生成参数
        gen_kwargs = {
            "max_new_tokens": min(512, max_length),
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        # 生成回答
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # 解码输出
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 提取回答部分（去除提示部分）
        response = output[len(prompt):].strip()

        return response

    def answer_with_knowledge(self, question, knowledge):
        """结合知识图谱信息回答问题"""
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

        # 构建提示
        prompt = f"""Human: 我需要基于以下知识回答一个问题:

知识:
{knowledge_str}

问题: {question}
        
"""
        return self.generate_response(prompt)
