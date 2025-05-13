from neo4j import GraphDatabase


class KnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_node(self, label, properties):
        """创建节点"""
        with self.driver.session() as session:
            result = session.write_transaction(self._create_node, label, properties)
            return result

    @staticmethod
    def _create_node(tx, label, properties):
        query = f"CREATE (n:{label} $properties) RETURN n"
        result = tx.run(query, properties=properties)
        return result.single()[0]

    def create_relationship(self, start_node_label, start_node_props,
                            end_node_label, end_node_props,
                            relationship_type, relationship_props=None):
        """创建关系"""
        if relationship_props is None:
            relationship_props = {}

        with self.driver.session() as session:
            result = session.write_transaction(
                self._create_relationship,
                start_node_label, start_node_props,
                end_node_label, end_node_props,
                relationship_type, relationship_props
            )
            return result

    @staticmethod
    def _create_relationship(tx, start_node_label, start_node_props,
                             end_node_label, end_node_props,
                             relationship_type, relationship_props):
        query = (
            f"MATCH (a:{start_node_label}), (b:{end_node_label}) "
            f"WHERE a.name = $start_name AND b.name = $end_name "
            f"CREATE (a)-[r:{relationship_type} $relationship_props]->(b) "
            f"RETURN a, r, b"
        )
        result = tx.run(
            query,
            start_name=start_node_props.get("name"),
            end_name=end_node_props.get("name"),
            relationship_props=relationship_props
        )
        return result.single()

    def query(self, cypher_query, parameters=None):
        """执行Cypher查询"""
        if parameters is None:
            parameters = {}

        with self.driver.session() as session:
            result = session.run(cypher_query, parameters)
            return [record for record in result]

    def find_related_knowledge(self, keyword):
        """查找与关键词相关的知识"""
        query = """
        MATCH (n)-[r]-(m)
        WHERE n.name CONTAINS $keyword OR m.name CONTAINS $keyword
        RETURN n, r, m
        LIMIT 10
        """
        results = self.query(query, {"keyword": keyword})

        # 处理结果为更友好的格式
        knowledge_items = []
        for record in results:
            n = record["n"]
            r = record["r"]
            m = record["m"]

            knowledge_items.append({
                "start_node": dict(n),
                "relationship": type(r).__name__,
                "end_node": dict(m)
            })

        return knowledge_items


# 测试知识图谱
if __name__ == "__main__":
    # 注意：需要先启动Neo4j数据库
    kg = KnowledgeGraph()

    # 创建教学知识节点
    kg.create_node("Concept", {"name": "循环语句", "description": "用于重复执行代码块的程序结构"})
    kg.create_node("Concept", {"name": "for循环", "description": "明确循环次数的循环结构"})
    kg.create_node("Concept", {"name": "while循环", "description": "基于条件的循环结构"})

    # 创建关系
    kg.create_relationship(
        "Concept", {"name": "循环语句"},
        "Concept", {"name": "for循环"},
        "HAS_TYPE"
    )
    kg.create_relationship(
        "Concept", {"name": "循环语句"},
        "Concept", {"name": "while循环"},
        "HAS_TYPE"
    )

    # 查询知识
    results = kg.find_related_knowledge("循环")
    for item in results:
        print(f"{item['start_node']['name']} --[{item['relationship']}]--> {item['end_node']['name']}")

    kg.close()
