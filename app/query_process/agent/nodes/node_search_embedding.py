import sys

from app.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from app.conf.milvus_config import milvus_config
from app.lm.embedding_utils import generate_embeddings
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger

def node_search_embedding(state:QueryGraphState):
    func_name = sys._getframe().f_code.co_name
    add_running_task(state['session_id'], func_name, is_stream=state['is_stream'])


    # 1.获取上一个节点数据
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names")
    # 2.将重写问题进行向量化
    embeddings=generate_embeddings([rewritten_query])
    # 3.到向量数据库进行混合检索
    item_names_no_space = [name.replace(" ", "") for name in item_names]
    expr=f"item_name in [{','.join(repr(name) for name in item_names_no_space)}]"
      # 3.1 构建查询参数reqs
    reqs=create_hybrid_search_requests(
                            dense_vector=embeddings["dense"][0],
                            sparse_vector= embeddings["sparse"][0],
                            expr=expr
        )
      # 3.2 进行混合检索
    response=hybrid_search(client=get_milvus_client(),
                           collection_name=milvus_config.chunks_collection,
                           reqs=reqs,
                           ranker_weights=(0.9, 0.1),  # 混合向量权重
                           norm_score=True,
                           output_fields=["chunk_id","content","file_title","title","parent_title","item_name"])
    """
      ["
        [
            'id: 1, distance: 0.006047376897186041, entity: {"chunk_id","content","file_title","title","parent_title","item_name"}', 
            'id: 2, distance: 0.006422005593776703, entity: {"chunk_id","content","file_title","title","parent_title","item_name"}'
        ]
      "]
    """
    # 4. 处理数据
    embedding_chunks=response[0]
    add_done_task(state["session_id"], func_name, is_stream=state['is_stream'])
    logger.info(f"节点{func_name}执行完毕，状态数据：{state}")
    return  {"embedding_chunks":embedding_chunks}


if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 烫金机使用说明",  # 模拟改写后的查询
        "item_names": ["HAK 180 烫金机"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")

        if chunks:
            print("\n>>> Top 1 结果详情:")
            top1 = chunks[0]
            # 打印关键字段（注意：entity字段可能包含具体业务数据）
            print(f"ID: {top1.get('id')}")
            print(f"Distance: {top1.get('distance')}")
            entity = top1.get('entity', {})
            print(f"Item Name: {entity.get('item_name')}")
            print(f"Content Preview: {entity.get('content', '')[:100]}...")
        else:
            print("\n>>> 警告：未检索到任何结果，请检查 Milvus 数据或 item_names 是否匹配")

    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)