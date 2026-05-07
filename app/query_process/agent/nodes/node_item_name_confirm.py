import json
import sys

from langchain_classic.agents.chat.prompt import HUMAN_MESSAGE
from langchain_core.messages import SystemMessage, HumanMessage

from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.lm.embedding_utils import generate_embeddings
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

def step_4_query_milvus_item_names(item_names):
    # 对查询的item_names进行向量化
    embeddings=generate_embeddings(item_names)
    milvus_client=get_milvus_client()
    final_result=[]
    # 构建查询参数reqs
    for index,item_name in enumerate(item_names):
        dense_vector=embeddings["dense"][index]
        sparse_vector=embeddings["sparse"][index]

        reqs=create_hybrid_search_requests(dense_vector=dense_vector,sparse_vector= sparse_vector)

        # 进行混合检索
        # 返回的数据结构为：
        """
        [[  
            { id:xx,distance:0.xx,entity:{item_name:xxx} },
            { id:xx,distance:0.xx,entity:{item_name:xxx} }
        ]]
        """
        response=hybrid_search(
                      client=milvus_client,
                      collection_name=milvus_config.item_name_collection,
                      reqs=reqs,
                      ranker_weights=(0.5, 0.5),  # 混合向量权重
                      norm_score=True             # 归一化分数  0-1
                      )
        # 对返回结果进行解析
        matches=[]
        if response and len(response)>0:
            for hit in response[0]:
                hit_name=hit.get("entity",{}).get("item_name","")
                score=hit.get("distance",0)
                if hit_name:
                    matches.append({"item_name":hit_name,"score":score})

        final_result.append({
                            "extracted":item_name,       # 大模型提取结果
                             "matches":matches           # 向量数据库匹配结果
        })
    logger.info(f"Step 4: 匹配结果: {final_result}")
    return  final_result


def step_5_confirm_and_option_item_names(query_milvus_result):
    confirm_item_names = []
    option_item_names = []
    # 循环遍历query_milvus_result
    for item_name_info in query_milvus_result:
        extracted_name = item_name_info.get("extracted", "")
        matches=item_name_info.get("matches", [])

        # 对匹配的item_name按照匹配分数进行降序排序
        matches.sort(key=lambda x: x["score"], reverse=True)
        # 过滤高分
        high_matches = [match for match in matches if match.get("score",0) >= 0.85]
        middle_matches = [match for match in matches if 0.6 < match.get("score",0) >= 0.6]
        # 只有一个高分匹配
        if len(high_matches)==1 :
            confirm_item_names.append(high_matches[0].get("item_name"))
            continue
        # 有多个高分匹配
        if len(high_matches)>1:
            # 优先考虑名称相同的
            same_item_name=""
            for match in high_matches:
                if match.get("item_name")==extracted_name:
                    same_item_name=match
                    break
            if not same_item_name:
                same_item_name=high_matches[0] # 没有与数据库中主体名称相同的，把最高分的匹配项作为最终结果
            confirm_item_names.append(same_item_name)
            continue
        if len(middle_matches)>0:
            # 保留前两个
            for item in middle_matches[:2]:
                option_item_names.append(item.get("item_name"))
            continue
        logger.info(f"没有匹配到主体，保留原始结果: {extracted_name}")
        # state[]
    result={
        "confirm_item_names":list(set(confirm_item_names)),
        "option_item_names":list(set(option_item_names))
    }
    return  result


def step_6_deal_list(state,item_result, history_messages,rewritten_query):
    confirm_item_names=item_result.get("confirm_item_names", [])
    option_item_names=item_result.get("option_item_names", [])
    if len(confirm_item_names)>0:
        # TODO 更新聊天记录 item_names->confirm_item_name
        # 确定主体，并返回结果
        state["item_names"]=confirm_item_names
        state["rewritten_query"]=rewritten_query
        state["history"]=history_messages
        return  state
    if len(option_item_names)>0:
        # 可选主体，并返回结果
        option_names=",".join(option_item_names)
        answer=f"你是想咨询以下哪个商品：{option_names},请提问的时候明确主体名称"
        state["answer"]=answer
        return  state
    answer=f"没有找到对应的商品，请重新提问"
    state["answer"]=answer
    return  state

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


    # 3.提取消息 {"item_names":[大模型从用户提问以及结合历史对话提取的主体名称列表],"rewritten_query":"重写的用户提问"}
    extract_res=step_3_extract_info(state["original_query"], history_messages)
    # 更新 State 中的 rewrite_query
    rewritten_query = extract_res.get("rewritten_query", original_query)
    state["rewritten_query"] = rewritten_query
    item_names=extract_res.get("item_names", [])
    item_results={}
    if len(item_names)>0:
        # 4.将大模型生成的主体名称到向量数据库中作相似度查询获取到精确的名称
        query_milvus_result=step_4_query_milvus_item_names(item_names)
        # 5.处理查询结果，返回结果{确定的item_name:[],可选的item_name:[] }
        item_results=step_5_confirm_and_option_item_names(query_milvus_result)

    # 6.处理列表
    state=step_6_deal_list(state,item_results,history_messages,rewritten_query)
    # 保存历史对话
    # if state.get("answer"):
    #     save_chat_message(session_id=session_id, role="assistant", text=state.get("answer", ""),item_names=[],image_urls=[],rewritten_query= rewritten_query)

    # 2.保存用户当前消息
    message_id=save_chat_message(session_id=session_id, role="user", rewritten_query=rewritten_query, item_names=state.get("item_names", []),image_urls=state.get("image_urls", []))
    logger.info(f"保存用户当前消息成功，message_id: {message_id}")
    add_done_task(state["session_id"], func_name, is_stream=state['is_stream'])
    logger.info(f"节点{func_name}执行完毕，状态数据：{state}")
    return state