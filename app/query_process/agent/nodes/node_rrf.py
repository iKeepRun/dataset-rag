import sys

from sympy.multipledispatch.dispatcher import source

from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger

def node_rrf(state:QueryGraphState):
    func_name = sys._getframe().f_code.co_name
    add_running_task(state['session_id'], func_name, is_stream=state['is_stream'])
    # 保留top_k个结果
    top_k=5
    # 获取节点node_search_embedding_hyde和node_search_embedding的输出
    hyde_embedding_chunks = state.get("hyde_embedding_chunks")
    embedding_chunks = state.get("embedding_chunks")
    # 给结果加权
    source_with_weight=[(embedding_chunks,1.0),(hyde_embedding_chunks,1.0) ]
    # 存储id和rrf评分的字典
    score_dict={}
    # 存储id 和 chunk的字典
    chunk_dict={}
    # 遍历两个结果，计算rrf评分
    for source, weight in source_with_weight:
        for rank, chunk in enumerate(source,start=1):
            chunk_id=chunk.get("id") or chunk.get("entity").get("chunk_id")
            chunk_score=(1/ (60+ rank))*weight
            score_dict[chunk_id]=score_dict.get(chunk_id,0)+chunk_score
            chunk_dict[chunk_id]=chunk
    score_dict_sorted=dict(sorted(score_dict.items(),key=lambda x:x[1],reverse=True))

    result=[chunk_dict[chunk_id] for chunk_id in score_dict_sorted][:top_k]

    state["rrf_chunks"]= result
    add_done_task(state["session_id"], func_name, is_stream=state['is_stream'])
    logger.info(f"节点{func_name}执行完毕，状态数据：{state}")
    return state