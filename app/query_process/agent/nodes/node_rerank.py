import sys

from modelscope.pipelines.multi_modal.disco_guided_diffusion_pipeline.disco_guided_diffusion import normalize

from app.lm.reranker_utils import get_reranker_model
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def step_1_merge_result(state):
    rrf_chunks = state["rrf_chunks"]
    web_docs = state["web_search_docs"]

    merge_result = []
    for chunk in rrf_chunks:
        entity = chunk["entity"]
        merge_result.append(
            {
                "chunk_id": entity["chunk_id"],
                "text": entity["content"],
                "title": entity["title"],
                "url": "",
                "source": "local"
            })
    for doc in web_docs:
        merge_result.append(
            {
                "chunk_id": "",
                "text": doc["snippet"],
                "title": doc["title"],
                "url": doc["url"],
                "source": "web"
            })
    return merge_result


def step_2_rerank(merge_result, state):
    # 构建查询参数 [[问题：答案],[问题：答案]]
    qustion=state.get("rewritten_query") or state.get("original_query")
    query_pair=[]
    for chunk in merge_result:
        query_pair.append([qustion,chunk["text"]])

    rerank_model=get_reranker_model()

    scores=rerank_model.compute_score(sentence_pairs=query_pair,normalize= True)  # 分数归一化
    for merge,score in zip(merge_result,scores):
        merge["score"]=score
    return merge_result.sort(key=lambda x:x["score"],reverse=True)

def node_rerank(state: QueryGraphState):
    func_name = sys._getframe().f_code.co_name
    add_running_task(state['session_id'], func_name, is_stream=state['is_stream'])

    # 1. 合并node_rrf 和 node_web_search_mcp 的结果
    merge_result = step_1_merge_result(state)
    # 2. 调用模型进行rerank
    rerank_result = step_2_rerank(merge_result,state)
    """
    rerank_result=[
                     {
                       "chunk_id":"",
                       "text":"",
                       "title":"",
                       "url":"",
                       "source":""
                     },
                     {
                        ...
                     }
                   ]
    """
    # 3. 返回结果
    add_done_task(state["session_id"], func_name, is_stream=state['is_stream'])
    logger.info(f"节点{func_name}执行完毕，状态数据：{state}")
    return state
