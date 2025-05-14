from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.llms import HuggingFacePipeline
from langchain_neo4j import Neo4jGraph
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class LangChainIntegration:
    def __init__(self,
                 uri="bolt://localhost:7687",
                 username="neo4j",
                 password="password",
                 model_name="gpt2"):  # 使用较小的模型进行测试
        # 连接Neo4j图数据库
        try:
            self.graph = Neo4jGraph(
                url=uri,
                username=username,
                password=password
            )

            # 加载模型
            print(f"正在加载模型: {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            # 创建HuggingFace pipeline
            text_generation = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

            # 创建LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=text_generation)

            # 创建GraphCypherQAChain
            self.chain = GraphCypherQAChain.from_llm(
                llm=self.llm,
                graph=self.graph,
                verbose=True
            )

            print("LangChain集成初始化完成")
            self.initialized = True

        except Exception as e:
            print(f"LangChain集成初始化失败: {e}")
            self.initialized = False

    def query(self, question):
        """使用GraphCypherQAChain查询知识图谱并回答问题"""
        if not self.initialized:
            return "系统初始化失败，无法处理查询。"

        try:
            result = self.chain.run(question)
            return result
        except Exception as e:
            print(f"查询处理失败: {e}")
            return f"处理查询时出错: {str(e)}"
