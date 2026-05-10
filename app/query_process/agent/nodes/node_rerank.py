import sys

from modelscope.pipelines.multi_modal.disco_guided_diffusion_pipeline.disco_guided_diffusion import normalize

from app.lm.reranker_utils import get_reranker_model
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


RERANK_MAX_TOPK:int= 10
RERABK_MIN_TOPK:int= 1
# 断崖阀值（相对）
RERANK_GAP_RATIO:float= 0.25
# 断崖阀值 （绝对）
RERANK_GAP_ABS:float=0.5  # 最大间隔分数


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


def step_3_topk(rerank_result, state):
    # 最多获取的元素数量
    max_topk=RERANK_MAX_TOPK
    # 最少获取的元素数量
    min_topk=RERABK_MIN_TOPK
    # 绝对分
    gap_abs=RERANK_GAP_ABS
    # 相对分
    gap_ratio=RERANK_GAP_RATIO

    actual_topk=min(max_topk,len(rerank_result))
    for i in range(actual_topk-1):
        front=rerank_result[i]
        back=rerank_result[i+1]

        score_diff=back["score"]-front["score"]
        if (i+1)>=min_topk:
            if score_diff>gap_abs or score_diff>gap_ratio*front["score"]:
                return  i+1
    return actual_topk

def strp_3_topk_1(rerank_result, state):
    # 最多获取的元素数量
    max_topk = RERANK_MAX_TOPK
    # 最少获取的元素数量
    min_topk = RERABK_MIN_TOPK
    # 绝对分
    gap_abs = RERANK_GAP_ABS
    # 相对分
    gap_ratio = RERANK_GAP_RATIO
    topk=min(max_topk,len(rerank_result))
    if topk>min_topk:
        for index in range(min_topk-1,topk-1):
            score_1=rerank_result[index]["score"]
            score_2=rerank_result[index+1]["score"]
            gap=score_2-score_1
            rel=gap/(abs(score_1)+1e-6)
            if gap>=gap_abs or rel>=gap_ratio:
                return index+1
    return topk

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
    # 3. 处理结果，取topk个
    actual_topk=step_3_topk(rerank_result,state)
    state["reranked_docs"] = rerank_result[:actual_topk]

    add_done_task(state["session_id"], func_name, is_stream=state['is_stream'])
    logger.info(f"节点{func_name}执行完毕，状态数据：{state}")
    return state
