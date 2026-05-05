import json
import sys

from langchain_classic.agents.chat.prompt import HUMAN_MESSAGE
from langchain_core.messages import SystemMessage, HumanMessage

from app.clients.mongo_history_utils import get_recent_messages, save_chat_message
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.query_process.api.query_server import query
from app.utils.task_utils import add_running_task, add_done_task


def step_3_extract_info(original_query, history_messages):
    history_text = ""
    for message in history_messages:
        history_text += f"角色{message['role']}：{message['text']},重写的问题：{message['rewritten_query']},主体名称：{message.get('item_names',[])},时间：{message['ts']}\n"
    # 调用大模型，根据历史对话和用户提问，确定主体，并返回结果
    llm = get_llm_client()
    prompt = load_prompt("rewritten_query_and_itemnames.prompt", history_text=history_text,
                         query=original_query)

    messages = [
        SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
        HumanMessage(content=prompt)
    ]

    try:
        response = llm.invoke(messages)
        content = response.content
        # 处理LLM可能返回的代码块格式（如```json ... ```），去除包裹符
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
        # 将处理后的文本转为JSON字典，解析LLM返回结果
        result = json.loads(content)
        logger.info(f"Step 3: 解析 LLM 结果: {result}")
        # 健壮性处理：确保返回结果包含item_names字段，无则设为空列表
        if "item_names" not in result:
            result["item_names"] = []
        # 健壮性处理：确保返回结果包含rewritten_query字段，无则复用原始查询
        if "rewritten_query" not in result:
            result["rewritten_query"] = query
        # 返回解析后的提取结果
        return result
    except Exception as e:
        # 捕获所有异常（如LLM调用失败、JSON解析失败等），记录错误日志
        logger.error(f"Step 3 LLM 提取失败: {e}")
        # 异常时返回默认结果：空商品名列表+原始查询
        return {"item_names": [], "rewritten_query": query}


# def save_chat_message(session_id, role, original_query, rewritten_query, item_names):
#
#     message_id=save_chat_message(session_id=session_id,
#                       role=role,
#                       text=original_query,
#                       rewritten_query=rewritten_query,
#                       item_names=item_names)
#     return  message_id

def node_item_name_confirm(state:QueryGraphState):
    """
    确定用户提问的主体，并且重写问题，重写用户提问 可以去除一些不明确的代词（他能xxxx）,合并历史上下文
    """
    #从对堆栈中获取到方法名
    func_name=sys._getframe().f_back.f_code.co_name
    add_running_task(state['session_id'], func_name, is_stream=state['is_stream'])
    
    session_id=state.get("session_id")
    original_query=state.get("original_query")
    # 1.获取历史对话
    history_messages=get_recent_messages(session_id=state["session_id"], limit=5)
    logger.info(f"Node: 获取到 {len(history_messages)} 条历史消息")
    # 2.保存用户当前消息
    message_id=save_chat_message(session_id, "user", original_query, "", state.get("item_names", []))
    logger.info(f"保存用户当前消息成功，message_id: {message_id}")
    # 3.提取消息 {"item_names":[大模型从用户提问以及结合历史对话提取的主体名称列表],"rewritten_query":"重写的用户提问"}
    extract_res=step_3_extract_info(state["original_query"], history_messages)
    # 更新 State 中的 rewrite_query
    rewritten_query = extract_res.get("rewritten_query", original_query)
    state["rewritten_query"] = rewritten_query
    # 4.根据
    add_done_task(state["session_id"], func_name, is_stream=state['is_stream'])
    logger.info(f"节点{func_name}执行完毕，状态数据：{state}")
    return state