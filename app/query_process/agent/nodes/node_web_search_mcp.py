import asyncio
import json
import sys

from agents.mcp import MCPServerStreamableHttp

from app.conf.bailian_mcp_config import mcp_config
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger

async def mcp_call_streamable(query):
    """
    调用百炼的网络搜索工具
    :param query:
    :return:
    """
    # logger.info(f"开始调用百炼网络搜索工具,base_url:{mcp_config.mcp_base_url},token:{mcp_config.api_key}")
    # 1. 创建MCPServerStreamableHttp对象
    search_mcp = MCPServerStreamableHttp(
        name = "search_mcp",
        params={
            # 核心参数
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
            "timeout": 30, #连接超时时间
        },
        max_retry_attempts=3
    )
    # 2. 连接 - 调用 - 关闭
    try:
        # 连接
        await search_mcp.connect()

        # 获取工具
        tools = await search_mcp.list_tools()
        print(f"工具列表：{tools}")
        # 调用
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": query,
                "count": 5,
            }
        )
        return result
    finally:
        await search_mcp.cleanup()

def node_web_search_mcp(state:QueryGraphState):
    func_name = sys._getframe().f_code.co_name
    add_running_task(state['session_id'], func_name, is_stream=state['is_stream'])

    rewritten_query = state.get("rewritten_query")
    result=asyncio.run(mcp_call_streamable(rewritten_query))
    """
      {
       "isError": false,
       "content": [
         {
           "text": "{\"pages\":[{\"snippet\":\"和讯首页|手机和讯 登录注册 股票客户端 Android 股票客户端 iPhone\",
                                 \"hostname\":\"和讯网\",
                                 \"hostlogo\":\"https://img.alicdn.com/imgextra/i3/O1CN01VcUfI91cc0kCH3Gt2_!!6000000003620-73-tps-32-32.ico\",
                                 \"title\":\"行情中心-和讯网 国内全面的即时行情数据服务中心\",
                                 \"url\":\"https://quote.hexun.com/\"},
                                 
                                {\"snippet\":\"数据中心\",
                                 \"hostname\":\"东方财富网\",
                                 \"hostlogo\":\"https://img.alicdn.com/imgextra/i1/O1CN01iL4mYC1cF6vgiem0A_!!6000000003570-55-tps-32-32.svg\",
                                 \"title\":\"股票\",
                                  \"url\":\"https://stock.eastmoney.com/\"},
                                 
                                 {\"snippet\":\"意见反馈\",
                                 \"hostname\":\"东方财富网\",
                                 \"hostlogo\":\"https://quote.eastmoney.com/favicon.ico\",
                                 \"title\":\"行情中心:国内快捷全面的股票、基金、期货、美股、港股、外汇、黄金、债券行情系统_东方财富网\",
                                 \"url\":\"https://quote.eastmoney.com/center/qqzs.html#!/stealingyourhistory\"}],
                                 
                    \"request_id\":\"faa40120-ee17-4401-a6c5-9970da077c05\",\"tools\":[],\"status\":0}",
           "type": "text"
         }
       ]
     }
    """
    # 结果处理
    search_result=json.loads(result.content[0].text).get("pages", [])
    # search_result = result["content"][0]["text"].get("pages", [])
    add_done_task(state["session_id"], func_name, is_stream=state['is_stream'])
    logger.info(f"节点{func_name}执行完毕，状态数据：{state}")
    return {"web_search_docs":search_result}   # search_result: dict  # 网络搜索的结果

from dotenv import load_dotenv

if __name__ == '__main__':
    # load_dotenv()
    test_state = {
        "session_id":"mcp_01",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream":True
    }

    # 调用 websearch_node 函数
    result_state = node_web_search_mcp(test_state)

    # 验证结果
    print("测试结果:")
    print(f"查询内容: {test_state.get('rewritten_query')}")

    # 输出搜索结果
    search_results = result_state.get('web_search_docs', [])
    print(f"搜索结果数量: {len(search_results)}")
    print("search_results", search_results)