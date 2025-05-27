#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速导入知识图谱数据脚本
"""

import os
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ai_service.knowledge_graph.knowledge_graph import KnowledgeGraph


def quick_import_knowledge(uri="bolt://localhost:7687", user="neo4j", password="admin123"):
    """快速导入预定义的知识图谱数据"""

    # 连接知识图谱
    kg = KnowledgeGraph(uri, user, password)

    # 清空现有数据（可选）
    print("是否清空现有数据？(y/N): ", end="")
    clear_data = input().strip().lower()
    if clear_data == 'y':
        try:
            kg.query("MATCH (n) DETACH DELETE n")
            print("✅ 已清空现有数据")
        except Exception as e:
            print(f"❌ 清空数据失败: {e}")

    # 批量创建教学相关概念
    education_concepts = [
        # PEPPER机器人相关
        {"name": "PEPPER机器人", "description": "人形智能教学机器人", "category": "教育机器人", "type": "硬件"},
        {"name": "人机交互", "description": "人与机器之间的交流方式", "category": "交互技术", "type": "技术"},
        {"name": "情感识别", "description": "识别人类情绪状态的技术", "category": "AI技术", "type": "技术"},
        {"name": "语音合成", "description": "将文本转换为语音", "category": "语音技术", "type": "技术"},
        {"name": "手势识别", "description": "识别和理解手势动作", "category": "视觉技术", "type": "技术"},

        # 编程教学概念
        {"name": "编程思维", "description": "解决问题的逻辑思维方式", "category": "思维能力", "type": "能力"},
        {"name": "代码调试", "description": "查找和修复程序错误", "category": "编程技能", "type": "技能"},
        {"name": "算法思维", "description": "设计高效解决方案的思维", "category": "思维能力", "type": "能力"},
        {"name": "项目管理", "description": "规划和管理软件项目", "category": "管理技能", "type": "技能"},
        {"name": "团队协作", "description": "与他人合作完成任务", "category": "软技能", "type": "技能"},

        # AI教学概念
        {"name": "模型训练", "description": "使用数据训练AI模型", "category": "AI实践", "type": "实践"},
        {"name": "数据预处理", "description": "清理和准备训练数据", "category": "数据处理", "type": "技术"},
        {"name": "特征提取", "description": "从数据中提取有用特征", "category": "数据处理", "type": "技术"},
        {"name": "模型评估", "description": "评价模型性能指标", "category": "AI实践", "type": "实践"},
        {"name": "超参数调优", "description": "优化模型参数设置", "category": "AI实践", "type": "实践"},

        # 教学方法
        {"name": "启发式教学", "description": "引导学生自主思考的教学法", "category": "教学方法", "type": "方法"},
        {"name": "案例教学", "description": "通过实际案例进行教学", "category": "教学方法", "type": "方法"},
        {"name": "互动教学", "description": "师生互动参与的教学方式", "category": "教学方法", "type": "方法"},
        {"name": "实践教学", "description": "通过实际操作学习知识", "category": "教学方法", "type": "方法"},
        {"name": "游戏化学习", "description": "通过游戏元素促进学习", "category": "教学方法", "type": "方法"},

        # 学习理论
        {"name": "建构主义学习", "description": "学习者主动构建知识的理论", "category": "学习理论", "type": "理论"},
        {"name": "社会学习理论", "description": "通过观察和模仿学习", "category": "学习理论", "type": "理论"},
        {"name": "认知负荷理论", "description": "关于大脑处理信息能力的理论", "category": "学习理论", "type": "理论"},
        {"name": "多元智能理论", "description": "人具有多种智能类型", "category": "学习理论", "type": "理论"},

        # 评估方法
        {"name": "形成性评估", "description": "学习过程中的持续评估", "category": "评估方法", "type": "方法"},
        {"name": "总结性评估", "description": "学习结束后的综合评估", "category": "评估方法", "type": "方法"},
        {"name": "同伴评估", "description": "学生之间相互评价", "category": "评估方法", "type": "方法"},
        {"name": "自我评估", "description": "学习者对自己的评价", "category": "评估方法", "type": "方法"},

        # 技术工具
        {"name": "学习管理系统", "description": "管理在线学习的平台", "category": "教育技术", "type": "工具"},
        {"name": "智能辅导系统", "description": "提供个性化辅导的AI系统", "category": "教育技术", "type": "工具"},
        {"name": "虚拟实验室", "description": "在线模拟实验环境", "category": "教育技术", "type": "工具"},
        {"name": "协作学习平台", "description": "支持团队学习的平台", "category": "教育技术", "type": "工具"},
    ]

    print("📚 正在创建教育概念节点...")
    created_nodes = 0
    for concept in education_concepts:
        try:
            kg.create_node("Concept", concept)
            created_nodes += 1
        except Exception as e:
            print(f"⚠️ 创建节点失败: {concept['name']} - {e}")

    print(f"✅ 创建了 {created_nodes} 个概念节点")

    # 批量创建关系
    relationships = [
        # PEPPER机器人相关关系
        ("PEPPER机器人", "支持", "人机交互", "PEPPER机器人支持多种人机交互方式"),
        ("PEPPER机器人", "具备", "情感识别", "PEPPER机器人具备情感识别能力"),
        ("PEPPER机器人", "使用", "语音合成", "PEPPER机器人使用语音合成技术"),
        ("PEPPER机器人", "支持", "手势识别", "PEPPER机器人支持手势识别功能"),
        ("PEPPER机器人", "应用", "多模态交互", "PEPPER机器人应用多模态交互技术"),

        # 编程教学关系
        ("Python", "培养", "编程思维", "Python学习培养编程思维"),
        ("编程", "需要", "代码调试", "编程过程需要代码调试技能"),
        ("算法", "体现", "算法思维", "算法设计体现算法思维"),
        ("软件开发", "需要", "项目管理", "软件开发需要项目管理"),
        ("编程学习", "强调", "团队协作", "编程学习强调团队协作能力"),

        # AI教学关系
        ("机器学习", "包含", "模型训练", "机器学习包含模型训练过程"),
        ("模型训练", "需要", "数据预处理", "模型训练需要数据预处理"),
        ("数据预处理", "包含", "特征提取", "数据预处理包含特征提取"),
        ("模型训练", "需要", "模型评估", "模型训练需要模型评估"),
        ("模型优化", "包含", "超参数调优", "模型优化包含超参数调优"),

        # 教学方法关系
        ("个性化学习", "采用", "启发式教学", "个性化学习采用启发式教学"),
        ("实践教学", "使用", "案例教学", "实践教学使用案例教学方法"),
        ("PEPPER机器人", "支持", "互动教学", "PEPPER机器人支持互动教学"),
        ("编程教学", "强调", "实践教学", "编程教学强调实践教学"),
        ("智能教育", "应用", "游戏化学习", "智能教育应用游戏化学习"),

        # 学习理论关系
        ("个性化学习", "基于", "建构主义学习", "个性化学习基于建构主义学习理论"),
        ("协作学习", "基于", "社会学习理论", "协作学习基于社会学习理论"),
        ("教学设计", "考虑", "认知负荷理论", "教学设计考虑认知负荷理论"),
        ("个性化教学", "应用", "多元智能理论", "个性化教学应用多元智能理论"),

        # 评估方法关系
        ("智能教育", "采用", "形成性评估", "智能教育采用形成性评估"),
        ("传统教育", "依赖", "总结性评估", "传统教育依赖总结性评估"),
        ("协作学习", "包含", "同伴评估", "协作学习包含同伴评估"),
        ("个性化学习", "鼓励", "自我评估", "个性化学习鼓励自我评估"),

        # 技术工具关系
        ("在线教育", "使用", "学习管理系统", "在线教育使用学习管理系统"),
        ("个性化学习", "依赖", "智能辅导系统", "个性化学习依赖智能辅导系统"),
        ("实践教学", "使用", "虚拟实验室", "实践教学使用虚拟实验室"),
        ("团队学习", "使用", "协作学习平台", "团队学习使用协作学习平台"),

        # 跨领域关系
        ("人工智能", "赋能", "智能教育", "人工智能赋能智能教育"),
        ("知识图谱", "支持", "个性化推荐", "知识图谱支持个性化推荐"),
        ("大数据", "支持", "学习分析", "大数据支持学习分析"),
        ("云计算", "支持", "在线教育", "云计算支持在线教育平台"),

        # 编程语言之间关系
        ("JavaScript", "用于", "前端开发", "JavaScript主要用于前端开发"),
        ("Python", "适用", "人工智能", "Python适用于人工智能开发"),
        ("Java", "适用", "企业开发", "Java适用于企业级开发"),
        ("C++", "适用", "系统编程", "C++适用于系统级编程"),

        # 数据科学关系
        ("数据科学", "结合", "统计学", "数据科学结合统计学方法"),
        ("数据科学", "结合", "计算机科学", "数据科学结合计算机科学"),
        ("数据挖掘", "应用", "机器学习", "数据挖掘应用机器学习算法"),
        ("数据可视化", "帮助", "数据理解", "数据可视化帮助数据理解"),
    ]

    print("🔗 正在创建知识关系...")
    created_relationships = 0
    for start, relation, end, desc in relationships:
        try:
            kg.create_relationship(
                "Concept", {"name": start},
                "Concept", {"name": end},
                relation.upper().replace(" ", "_"),
                {"description": desc}
            )
            created_relationships += 1
        except Exception as e:
            print(f"⚠️ 创建关系失败: {start} -> {end} - {e}")

    print(f"✅ 创建了 {created_relationships} 个知识关系")

    # 创建学习路径
    learning_paths = [
        {
            "name": "Python编程入门路径",
            "description": "零基础学习Python编程的完整路径",
            "steps": ["变量", "数据类型", "控制结构", "函数", "面向对象", "项目实践"],
            "difficulty": "初级",
            "duration": "3个月"
        },
        {
            "name": "人工智能基础路径",
            "description": "人工智能基础知识学习路径",
            "steps": ["数学基础", "机器学习", "深度学习", "自然语言处理", "项目实战"],
            "difficulty": "中级",
            "duration": "6个月"
        },
        {
            "name": "Web开发全栈路径",
            "description": "全栈Web开发学习路径",
            "steps": ["HTML/CSS", "JavaScript", "前端框架", "后端开发", "数据库", "部署"],
            "difficulty": "中级",
            "duration": "4个月"
        }
    ]

    print("🛤️ 正在创建学习路径...")
    for path in learning_paths:
        try:
            kg.create_node("LearningPath", path)
            # 创建路径与步骤的关系
            for i, step in enumerate(path["steps"]):
                kg.create_relationship(
                    "LearningPath", {"name": path["name"]},
                    "Concept", {"name": step},
                    "INCLUDES_STEP",
                    {"order": i + 1, "description": f"学习路径第{i + 1}步"}
                )
        except Exception as e:
            print(f"⚠️ 创建学习路径失败: {path['name']} - {e}")

    # 创建学生类型
    student_types = [
        {"name": "视觉学习者", "description": "通过图像和图表学习效果最好", "learning_style": "visual"},
        {"name": "听觉学习者", "description": "通过听讲和讨论学习效果最好", "learning_style": "auditory"},
        {"name": "动手学习者", "description": "通过实践操作学习效果最好", "learning_style": "kinesthetic"},
        {"name": "阅读学习者", "description": "通过阅读文本学习效果最好", "learning_style": "reading"}
    ]

    print("👥 正在创建学生类型...")
    for student_type in student_types:
        try:
            kg.create_node("StudentType", student_type)
        except Exception as e:
            print(f"⚠️ 创建学生类型失败: {student_type['name']} - {e}")

    # 创建教学资源类型
    resource_types = [
        {"name": "视频教程", "description": "视频形式的教学内容", "media_type": "video", "suitable_for": "visual"},
        {"name": "音频讲座", "description": "音频形式的教学内容", "media_type": "audio", "suitable_for": "auditory"},
        {"name": "互动练习", "description": "可操作的练习题", "media_type": "interactive",
         "suitable_for": "kinesthetic"},
        {"name": "文档资料", "description": "文本形式的学习材料", "media_type": "text", "suitable_for": "reading"},
        {"name": "虚拟实验", "description": "模拟实验环境", "media_type": "simulation", "suitable_for": "kinesthetic"},
        {"name": "思维导图", "description": "可视化知识结构", "media_type": "diagram", "suitable_for": "visual"}
    ]

    print("📚 正在创建教学资源类型...")
    for resource in resource_types:
        try:
            kg.create_node("ResourceType", resource)
            # 创建资源类型与学习风格的适配关系
            kg.create_relationship(
                "ResourceType", {"name": resource["name"]},
                "StudentType", {"name": f"{resource['suitable_for']}学习者"},
                "SUITABLE_FOR",
                {"description": f"{resource['name']}适合{resource['suitable_for']}学习者"}
            )
        except Exception as e:
            print(f"⚠️ 创建资源类型失败: {resource['name']} - {e}")

    # 验证数据导入
    print("\n📊 验证数据导入结果...")
    try:
        # 统计节点数量
        concept_count = kg.query("MATCH (n:Concept) RETURN count(n) as count")[0]["count"]
        path_count = kg.query("MATCH (n:LearningPath) RETURN count(n) as count")[0]["count"]
        student_type_count = kg.query("MATCH (n:StudentType) RETURN count(n) as count")[0]["count"]
        resource_count = kg.query("MATCH (n:ResourceType) RETURN count(n) as count")[0]["count"]

        # 统计关系数量
        relationship_count = kg.query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]

        print(f"✅ 概念节点: {concept_count}")
        print(f"✅ 学习路径: {path_count}")
        print(f"✅ 学生类型: {student_type_count}")
        print(f"✅ 资源类型: {resource_count}")
        print(f"✅ 总关系数: {relationship_count}")

        # 显示一些示例查询结果
        print("\n🔍 示例查询结果:")

        # 查询PEPPER机器人相关知识
        pepper_relations = kg.query("""
            MATCH (pepper:Concept {name: 'PEPPER机器人'})-[r]->(related)
            RETURN related.name as related_concept, type(r) as relationship
            LIMIT 5
        """)
        print("PEPPER机器人相关概念:")
        for item in pepper_relations:
            print(f"  - {item['related_concept']} ({item['relationship']})")

        # 查询编程相关知识
        programming_concepts = kg.query("""
            MATCH (concept:Concept)
            WHERE concept.category CONTAINS '编程' OR concept.name IN ['Python', 'JavaScript', 'Java']
            RETURN concept.name as name, concept.description as desc
            LIMIT 5
        """)
        print("\n编程相关概念:")
        for item in programming_concepts:
            print(f"  - {item['name']}: {item['desc']}")

    except Exception as e:
        print(f"❌ 验证查询失败: {e}")

    print("\n🎉 知识图谱数据导入完成!")
    print("💡 你现在可以:")
    print("   1. 启动PEPPER系统进行智能问答")
    print("   2. 通过Neo4j浏览器查看知识图谱: http://localhost:7474")
    print("   3. 使用个性化教学功能")

    kg.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='快速导入教育知识图谱数据')
    parser.add_argument('--uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--user', default='neo4j', help='Neo4j用户名')
    parser.add_argument('--password', default='admin123', help='Neo4j密码')

    args = parser.parse_args()

    print("🚀 开始导入教育知识图谱数据...")
    print(f"📡 连接到: {args.uri}")

    try:
        quick_import_knowledge(args.uri, args.user, args.password)
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        print("请确保:")
        print("  1. Neo4j数据库已启动")
        print("  2. 连接参数正确")
        print("  3. 用户有写入权限")


if __name__ == "__main__":
    main()