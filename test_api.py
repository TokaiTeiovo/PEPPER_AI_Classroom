import requests
import json
import time


def test_api(api_url="http://localhost:5000"):
    """测试API服务"""
    print("=" * 50)
    print(f"开始测试API服务: {api_url}")
    print("=" * 50)

    try:
        # 检查健康状态
        print("\n测试健康检查API...")
        health_start = time.time()
        health_response = requests.get(f"{api_url}/health")
        health_time = time.time() - health_start

        if health_response.status_code == 200:
            print(f"健康检查成功 (耗时: {health_time:.2f}秒)")
            print(f"响应: {health_response.json()}")
        else:
            print(f"健康检查失败: {health_response.status_code}")
            print(f"响应: {health_response.text}")
            return False

        # 测试查询API
        test_queries = [
            "什么是人工智能？",
            "Python中for循环和while循环有什么区别？",
            "PEPPER机器人在'人工智能+'课堂上有哪些应用？"
        ]

        for query in test_queries:
            print(f"\n测试查询API: {query}")

            query_start = time.time()
            query_response = requests.post(
                f"{api_url}/llm/query",
                json={"query": query}
            )
            query_time = time.time() - query_start

            if query_response.status_code == 200:
                print(f"查询成功 (耗时: {query_time:.2f}秒)")
                result = query_response.json()
                print(f"回答: {result['response'][:100]}...")  # 只显示前100个字符
            else:
                print(f"查询失败: {query_response.status_code}")
                print(f"响应: {query_response.text}")

        print("\n=" * 50)
        print("API服务测试完成，服务正常运行！")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_api()
