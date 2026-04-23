import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_done_task, add_running_task


def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    未来要实现:
    1. 加载 BGE-M3 模型。
    2. 对每个 Chunk 的文本进行 Dense (稠密) 和 Sparse (稀疏) 向量化。
    3. 准备好写入 Milvus 的数据格式。
    """
    # 1. 记录开始任务的日志和任务状态的配置
    func_name = sys._getframe().f_code.co_name
    logger.info(f"节点{func_name}执行,参数状态:{state}")
    add_running_task(state['task_id'], func_name)

    try:
        chunks=state.get("chunks")

        final_chunks=[]
        # 每次向量化五个块，因为bgem模型一次只能8192序列长度（token）
        # 每个content 拆分的时候最大长度限制了 2000，实际长度会比2000小
        batch_size=5
        for i in range(0,len(chunks),batch_size):
            batch_chunks=chunks[i:i+batch_size]

            current_chunk_contents=[]
            for chunk in batch_chunks:
                chunk_content=f"商品：{chunk['item_name']},内容：{chunk["content"]}"
                current_chunk_contents.append(chunk_content)
            result=generate_embeddings(current_chunk_contents)

            for i,chunk in enumerate(batch_chunks):
                chunk_copy=chunk.copy()
                chunk_copy["dense_vector"]=result['dense'][i]
                chunk_copy["sparse_vector"]=result['sparse'][i]
                final_chunks.append(chunk_copy)
        state["chunks"]=final_chunks
        return  state

    except Exception as e:
        logger.error(f"节点{func_name}执行失败:{e}")
        raise e
    finally:
        logger.info(f"节点{func_name}完成,数据状态{state}")
        add_done_task(state['task_id'], func_name)
