import sys

from langchain_core.messages import HumanMessage

from app.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def step_1_llm_output(rewritten_query):
    llm=get_llm_client()
    prompt=load_prompt("hyde_prompt.prompt", rewritten_query=rewritten_query)
    messages=[
        HumanMessage(content=prompt)
    ]

    response=llm.invoke( messages)
    result=response.content
    return result


def step_2_hyde_embedding(llm_answer, rewritten_query, item_names):
    query=rewritten_query+llm_answer
    embeddings=generate_embeddings([query])

    reqs=create_hybrid_search_requests(
                            dense_vector=embeddings["dense"][0],
                            sparse_vector= embeddings["sparse"][0],
                            expr=f"item_name in [{','.join(repr(name) for name in item_names)}]")

    response=hybrid_search(client=get_milvus_client(),
                           collection_name=milvus_config.chunks_collection,
                           reqs=reqs,
                           ranker_weights=(0.9, 0.1),  # 混合向量权重
                           norm_score=True,
                           output_fields=["chunk_id","content","file_title","title","parent_title","item_name"])
    logger.info(f"混合检索结果：{response}")
    return response[0] if response else []
def node_search_embedding_hyde(state:QueryGraphState):
    func_name = sys._getframe().f_code.co_name
    add_running_task(state['session_id'], func_name, is_stream=state['is_stream'])

    # 1.获取上一个节点数据
    rewritten_query = state.get("rewritten_query")  # 重写的问题答案
    item_names = state.get("item_names")
    # 2.让模型生成参考答案
    llm_answer=step_1_llm_output(rewritten_query)
    # 3.将问题+生成的答案向量化进行混合检索
    result=step_2_hyde_embedding(llm_answer, rewritten_query, item_names)

    add_done_task(state["session_id"], func_name, is_stream=state['is_stream'])
    logger.info(f"节点{func_name}执行完毕，状态数据：{state}")
    return {"hyde_embedding_chunks":result}
