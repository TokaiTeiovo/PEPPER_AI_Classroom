#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neo4j完整数据导出工具 - 将所有Neo4j数据转换为微调JSON格式
"""

import argparse
import json
import logging
import os
import sys
import time

from neo4j import GraphDatabase
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('neo4j_export.log')
    ]
)
logger = logging.getLogger("NEO4J_EXPORT")


def connect_to_neo4j(uri, user, password):
    """连接到Neo4j数据库"""
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        # 测试连接
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            test_value = result.single()["test"]
            if test_value != 1:
                raise Exception("连接测试失败")

        logger.info("成功连接到Neo4j数据库")
        return driver
    except Exception as e:
        logger.error(f"连接Neo4j失败: {e}")
        return None


def get_database_statistics(driver):
    """获取数据库统计信息"""
    stats = {}

    with driver.session() as session:
        # 获取节点总数
        result = session.run("MATCH (n) RETURN count(n) as node_count")
        stats["total_nodes"] = result.single()["node_count"]

        # 获取关系总数
        result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
        stats["total_relationships"] = result.single()["rel_count"]

        # 获取节点标签
        result = session.run("CALL db.labels()")
        stats["node_labels"] = [record["label"] for record in result]

        # 获取关系类型
        result = session.run("CALL db.relationshipTypes()")
        stats["relationship_types"] = [record["relationshipType"] for record in result]

        # 获取各标签节点数量
        node_counts = {}
        for label in stats["node_labels"]:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
            node_counts[label] = result.single()["count"]
        stats["label_counts"] = node_counts

        # 获取各关系类型数量
        rel_counts = {}
        for rel_type in stats["relationship_types"]:
            result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
            rel_counts[rel_type] = result.single()["count"]
        stats["relationship_counts"] = rel_counts

    return stats


def export_all_nodes(driver, output_dir):
    """导出所有节点"""
    nodes_file = os.path.join(output_dir, "all_nodes.json")

    with driver.session() as session:
        # 获取节点总数
        result = session.run("MATCH (n) RETURN count(n) as count")
        total_nodes = result.single()["count"]
        logger.info(f"开始导出 {total_nodes} 个节点...")

        # 分批导出节点
        batch_size = 5000
        num_batches = (total_nodes + batch_size - 1) // batch_size

        all_nodes = []

        for i in tqdm(range(num_batches), desc="导出节点"):
            skip = i * batch_size

            query = """
            MATCH (n)
            RETURN n, labels(n) as labels
            SKIP $skip LIMIT $limit
            """

            result = session.run(query, skip=skip, limit=batch_size)

            batch_nodes = []
            for record in result:
                try:
                    node = dict(record["n"])
                    node_labels = record["labels"]

                    node_data = {
                        "id": node.get("id", None),
                        "name": node.get("name", ""),
                        "description": node.get("description", ""),
                        "labels": node_labels,
                        "properties": {k: v for k, v in node.items() if k not in ["id", "name", "description"]}
                    }

                    batch_nodes.append(node_data)
                except Exception as e:
                    logger.warning(f"处理节点时出错: {e}")
                    continue

            all_nodes.extend(batch_nodes)

            # 定期保存以减少内存使用
            if (i + 1) % 10 == 0 or i == num_batches - 1:
                with open(nodes_file, 'w', encoding='utf-8') as f:
                    json.dump(all_nodes, f, ensure_ascii=False, indent=2)
                logger.info(f"已保存 {len(all_nodes)} 个节点到 {nodes_file}")

    logger.info(f"节点导出完成，共 {len(all_nodes)} 个节点")
    return nodes_file


def export_all_relationships(driver, output_dir):
    """导出所有关系"""
    relationships_file = os.path.join(output_dir, "all_relationships.json")

    with driver.session() as session:
        # 获取关系总数
        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        total_relationships = result.single()["count"]
        logger.info(f"开始导出 {total_relationships} 个关系...")

        # 分批导出关系
        batch_size = 5000
        num_batches = (total_relationships + batch_size - 1) // batch_size

        all_relationships = []

        for i in tqdm(range(num_batches), desc="导出关系"):
            skip = i * batch_size

            query = """
            MATCH (source)-[r]->(target)
            RETURN 
                source, 
                labels(source) as source_labels, 
                type(r) as relationship_type, 
                r as relationship_properties,
                target, 
                labels(target) as target_labels
            SKIP $skip LIMIT $limit
            """

            result = session.run(query, skip=skip, limit=batch_size)

            batch_relationships = []
            for record in result:
                try:
                    source = dict(record["source"])
                    source_labels = record["source_labels"]
                    relationship_type = record["relationship_type"]
                    relationship_properties = dict(record["relationship_properties"])
                    target = dict(record["target"])
                    target_labels = record["target_labels"]

                    relationship_data = {
                        "source": {
                            "id": source.get("id", None),
                            "name": source.get("name", ""),
                            "description": source.get("description", ""),
                            "labels": source_labels
                        },
                        "relationship": {
                            "type": relationship_type,
                            "properties": relationship_properties
                        },
                        "target": {
                            "id": target.get("id", None),
                            "name": target.get("name", ""),
                            "description": target.get("description", ""),
                            "labels": target_labels
                        }
                    }

                    batch_relationships.append(relationship_data)
                except Exception as e:
                    logger.warning(f"处理关系时出错: {e}")
                    continue

            all_relationships.extend(batch_relationships)

            # 定期保存以减少内存使用
            if (i + 1) % 10 == 0 or i == num_batches - 1:
                with open(relationships_file, 'w', encoding='utf-8') as f:
                    json.dump(all_relationships, f, ensure_ascii=False, indent=2)
                logger.info(f"已保存 {len(all_relationships)} 个关系到 {relationships_file}")

    logger.info(f"关系导出完成，共 {len(all_relationships)} 个关系")
    return relationships_file


def convert_to_fine_tuning_format(relationships_data, output_file, min_length=20, max_samples=None):
    """将关系数据转换为微调格式"""
    fine_tuning_data = []

    # 加载关系数据
    with open(relationships_data, 'r', encoding='utf-8') as f:
        relationships = json.load(f)

    logger.info(f"开始转换 {len(relationships)} 个关系为微调格式...")

    for item in tqdm(relationships, desc="转换为微调格式"):
        try:
            source = item["source"]
            relationship = item["relationship"]
            target = item["target"]

            source_name = source.get("name", "")
            source_desc = source.get("description", "")
            rel_type = relationship.get("type", "")
            target_name = target.get("name", "")
            target_desc = target.get("description", "")

            # 跳过名称为空的数据
            if not source_name or not target_name:
                continue

            # 1. 源概念解释
            if source_desc and len(source_desc) >= min_length:
                fine_tuning_data.append({
                    "instruction": f"解释什么是{source_name}",
                    "input": "",
                    "output": source_desc
                })

            # 2. 目标概念解释
            if target_desc and len(target_desc) >= min_length:
                fine_tuning_data.append({
                    "instruction": f"解释什么是{target_name}",
                    "input": "",
                    "output": target_desc
                })

            # 3. 关系解释
            relationship_output = f"{source_name}与{target_name}的关系是{rel_type}。"
            if source_desc:
                relationship_output += f" {source_name}是指{source_desc}。"
            if target_desc:
                relationship_output += f" {target_name}是指{target_desc}。"

            if len(relationship_output) >= min_length:
                fine_tuning_data.append({
                    "instruction": f"{source_name}与{target_name}的关系是什么？",
                    "input": "",
                    "output": relationship_output
                })

                # 另一种表达方式
                fine_tuning_data.append({
                    "instruction": f"{source_name}和{target_name}之间有什么联系？",
                    "input": "",
                    "output": relationship_output
                })

            # 如果达到最大样本数，停止处理
            if max_samples and len(fine_tuning_data) >= max_samples:
                logger.info(f"已达到最大样本数 {max_samples}，停止转换")
                fine_tuning_data = fine_tuning_data[:max_samples]
                break

        except Exception as e:
            logger.warning(f"转换数据时出错: {e}")
            continue

    # 保存微调数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fine_tuning_data, f, ensure_ascii=False, indent=2)

    logger.info(f"转换完成，生成了 {len(fine_tuning_data)} 条微调数据")

    # 打印几个示例
    for i, example in enumerate(fine_tuning_data[:3]):
        logger.info(f"示例 {i + 1}:")
        logger.info(f"  Instruction: {example['instruction']}")
        logger.info(f"  Input: {example['input']}")
        output_preview = example['output'][:100] + "..." if len(example['output']) > 100 else example['output']
        logger.info(f"  Output: {output_preview}")

    return len(fine_tuning_data)


def main():
    parser = argparse.ArgumentParser(description='Neo4j完整数据导出工具')
    parser.add_argument('--uri', type=str, default="bolt://localhost:7687",
                        help='Neo4j URI')
    parser.add_argument('--user', type=str, default="neo4j",
                        help='Neo4j用户名')
    parser.add_argument('--password', type=str, required=True,
                        help='Neo4j密码')
    parser.add_argument('--output-dir', type=str, default="neo4j_data",
                        help='输出目录')
    parser.add_argument('--min-length', type=int, default=20,
                        help='最小输出长度')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='最大样本数量（不设置则导出全部）')
    parser.add_argument('--stats-only', action='store_true',
                        help='仅显示统计信息，不导出数据')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 连接Neo4j
    driver = connect_to_neo4j(args.uri, args.user, args.password)
    if not driver:
        logger.error("无法连接到Neo4j，程序退出")
        return 1

    try:
        # 获取数据库统计信息
        stats = get_database_statistics(driver)
        logger.info("数据库统计信息:")
        logger.info(f"  节点总数: {stats['total_nodes']}")
        logger.info(f"  关系总数: {stats['total_relationships']}")
        logger.info(f"  节点标签: {', '.join(stats['node_labels'])}")
        logger.info(f"  关系类型: {', '.join(stats['relationship_types'])}")

        # 保存统计信息
        stats_file = os.path.join(args.output_dir, "database_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"统计信息已保存到 {stats_file}")

        if args.stats_only:
            logger.info("仅统计模式，程序结束")
            return 0

        # 导出节点
        nodes_file = export_all_nodes(driver, args.output_dir)

        # 导出关系
        relationships_file = export_all_relationships(driver, args.output_dir)

        # 转换为微调格式
        fine_tuning_file = os.path.join(args.output_dir, "fine_tuning_data.json")
        convert_to_fine_tuning_format(
            relationships_file,
            fine_tuning_file,
            min_length=args.min_length,
            max_samples=args.max_samples
        )

        logger.info("数据导出和转换完成")
        logger.info(f"原始节点数据: {nodes_file}")
        logger.info(f"原始关系数据: {relationships_file}")
        logger.info(f"微调格式数据: {fine_tuning_file}")

        return 0

    except Exception as e:
        logger.error(f"导出过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        driver.close()


if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    end_time = time.time()
    logger.info(f"程序运行时间: {end_time - start_time:.2f} 秒")
    sys.exit(exit_code)