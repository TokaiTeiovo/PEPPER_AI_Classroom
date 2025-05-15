import subprocess
import sys

def call_langchain_env(script_name, query, uri, user, password):
    """调用 LangChain 环境中的脚本处理查询"""
    try:
        # 创建临时脚本文件
        with open("langchain_functions.py", "w") as f:
            f.write("""
import json
import sys
from langchain.llms import HuggingFacePipeline
from langchain.chains import GraphQAChain
from langchain.graphs import Neo4jGraph
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def query_knowledge_graph(query, uri, user, password):
    try:
        # 连接到Neo4j图数据库
        graph = Neo4jGraph(
            url=uri,
            username=user,
            password=password
        )

        # 简单的查询处理
        results = []
        try:
            # 尝试使用Cypher直接查询
            cypher_query = f"MATCH (n)-[r]-(m) WHERE n.name CONTAINS '{query}' OR m.name CONTAINS '{query}' RETURN n, r, m LIMIT 10"
            results = graph.query(cypher_query)
        except:
            pass

        # 如果没有结果，返回一个简单的回复
        if not results:
            return {
                "response": f"我无法在知识图谱中找到与'{query}'相关的信息。可以尝试其他问题或使用不同的关键词。"
            }

        # 简单处理结果
        knowledge = []
        for record in results:
            if 'n' in record and 'r' in record and 'm' in record:
                n = record['n']
                r = record['r']
                m = record['m']

                item = {
                    "start_node": n,
                    "relationship": r,
                    "end_node": m,
                }
                knowledge.append(item)

        # 构建回复
        response = f"关于'{query}'，我在知识图谱中找到了以下信息:\\n"
        for i, item in enumerate(knowledge):
            start_name = item.get('start_node', {}).get('name', '')
            rel = item.get('relationship', '')
            end_name = item.get('end_node', {}).get('name', '')

            if start_name and rel and end_name:
                response += f"{i+1}. {start_name} -- {rel} --> {end_name}\\n"

        return {
            "response": response,
            "knowledge": knowledge
        }
    except Exception as e:
        return {
            "response": f"查询知识图谱时出错: {str(e)}",
            "error": str(e)
        }

# 从命令行参数获取查询和Neo4j信息
if len(sys.argv) > 4:
    query = sys.argv[1]
    uri = sys.argv[2]
    user = sys.argv[3]
    password = sys.argv[4]
else:
    query = ""
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"

# 处理查询并输出结果
result = query_knowledge_graph(query, uri, user, password)
print(json.dumps(result, ensure_ascii=False))
""")

        # 调用脚本
        result = subprocess.run([sys.executable, "langchain_functions.py", query, uri, user, password],
                                capture_output=True, text=True)

        return result.stdout, result.stderr, result.returncode

    except Exception as e:
        return "", str(e), 1