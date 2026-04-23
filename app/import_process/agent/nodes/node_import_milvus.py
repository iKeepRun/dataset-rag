import sys

from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task


def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    # 1. 记录开始任务的日志和任务状态的配置
    func_name = sys._getframe().f_code.co_name
    logger.info(f"节点{func_name}执行,参数状态:{state}")
    add_running_task(state['task_id'], func_name)

    # 从配置类中获取
    milvus_uri = milvus_config.milvus_url
    collection_name = milvus_config.chunks_collection

    # 配置缺失校验：任一配置为空则跳过Milvus存储，记录警告
    if not all([milvus_uri, collection_name]):
        logger.warning("Milvus配置缺失（MILVUS_URL/ITEM_NAME_COLLECTION），跳过数据保存")
        return state

    chunks = state.get("chunks")
    try:
        # 获取Milvus单例客户端，连接失败则直接返回
        client = get_milvus_client()
        if not client:
            logger.error("无法获取Milvus客户端（连接失败），跳过数据保存")
            raise Exception("无法获取Milvus客户端")

        # 集合初始化：不存在则创建（定义Schema+索引），存在则直接使用
        if not client.has_collection(collection_name=collection_name):
            logger.info(f"Milvus集合[{collection_name}]不存在，开始创建Schema和索引")
            # 创建集合Schema：自增主键+动态字段，适配灵活的数据存储
            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
            # 添加自增主键字段：INT64类型，唯一标识每条数据
            schema.add_field(
                field_name="chunk_id",
                datatype=DataType.INT64,
                is_primary=True,
                auto_id=True
            )
            # 添加文件标题字段：VARCHAR类型，最大长度65535，适配长标题
            schema.add_field(
                field_name="title",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            # 添加文件父标题字段：VARCHAR类型，最大长度65535，适配长标题
            schema.add_field(
                field_name="parent_title",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            schema.add_field(
                field_name="part",
                datatype=DataType.INT8,
            )
            # 添加文件标题字段：VARCHAR类型，最大长度65535，适配长标题
            schema.add_field(
                field_name="file_title",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            # 添加商品名字段：VARCHAR类型，最大长度65535，去重依据
            schema.add_field(
                field_name="item_name",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            # 添加文件内容字段：VARCHAR类型，最大长度65535
            schema.add_field(
                field_name="content",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            # 添加稠密向量字段：FLOAT_VECTOR，1024维（BGE-M3固定维度）
            schema.add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=1024
            )
            # 添加稀疏向量字段：SPARSE_FLOAT_VECTOR，变长
            schema.add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR
            )

            # 构建索引参数：为向量字段创建索引，提升检索性能
            index_params = client.prepare_index_params()
            # 优化版稠密向量索引：HNSW + COSINE (恢复最佳性能配置)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                # HNSW (Hierarchical Navigable Small World) 是目前性能最好、最常用的基于图的索引，检索速度极快，精度极高。
                index_type="HNSW",
                # 使用 COSINE 作为稠密向量相似度计算方式
                metric_type="COSINE",
                # M: 图中每个节点的最大连接数(常用16-64)
                # efConstruction: 构建索引时的搜索范围(越大建索引越慢，但精度越高，常用100-200)
                # 不同数据体量的推荐建议(万级)：
                # 10000 条数据：M=16, efConstruction=200
                # 50000 条数据：M=32, efConstruction=300
                # 100000 条数据：M=64, efConstruction=400
                params={"M": 32, "efConstruction": 300}
            )

            # 稀疏向量索引：专用SPARSE_INVERTED_INDEX+IP，关闭量化保证精度
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector_index",
                # 稀疏倒排索引 专门为稀疏向量（比如文本的 TF-IDF 向量、关键词权重向量，特点是大部分元素为 0，只有少数维度有值）设计的倒排索引，是稀疏向量检索的标配索引类型。
                index_type="SPARSE_INVERTED_INDEX",
                # IP（内积，Inner Product）如果向量是 “文本语义向量 + 关键词权重”，长度代表文本与主题的关联强度，此时用 IP 能同时体现 “语义匹配度” 和 “关联强度”。
                metric_type="IP",
                # DAAT_MAXSCORE：稀疏向量检索时，只计算可能得高分的维度，跳过大量0值，速度更快。
                # quantization="none"：稀疏向量里的权重是小数，不做压缩，保证精度不丢。
                params={"inverted_index_algo": "DAAT_MAXSCORE", "quantization": "none"}
            )

            # 创建集合：Schema + 索引参数
            client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
            logger.info(f"Milvus集合[{collection_name}]创建成功，包含Schema和向量索引")

        # 幂等性处理：删除同名商品数据，避免重复存储（核心：先加载集合才能删除）
        for chunk in chunks:
            item_name=chunk["item_name"]
            clean_item_name = (item_name or "").strip()
            if clean_item_name:
                client.load_collection(collection_name=collection_name)
                # 商品名称转义，防止特殊字符导致过滤表达式解析失败
                safe_item_name = escape_milvus_string(clean_item_name)
                filter_expr = f'item_name=="{safe_item_name}"'
                # 执行删除操作
                client.delete(collection_name=collection_name, filter=filter_expr)

                client.load_collection(collection_name=collection_name)
                logger.info(f"Milvus幂等性处理完成，已删除集合中[{clean_item_name}]的历史数据")

        # 构造插入Milvus的数据：基础字段+非空向量字段
        # data = {
        #     "file_title": file_title,
        #     "item_name": item_name
        # }
        # # 稠密向量非空才添加，避免空值入库报错
        # if dense_vector is not None:
        #     data["dense_vector"] = dense_vector
        # # 稀疏向量非空则归一化后添加，保证检索准确性
        # if sparse_vector is not None:
        #     data["sparse_vector"] = sparse_vector

        # data_list=[]
        # for chunk in chunks:
        #     data = {
        #         "file_title": chunk["file_title"],
        #         "item_name": chunk["item_name"],
        #         "content": chunk["content"],
        #         "dense_vector": chunk["dense_vector"],
        #         "sparse_vector": chunk["sparse_vector"]
        #     }
        #     data_list.append(data)
        # 插入数据：列表格式支持批量插入，单条数据保持格式统一
        client.insert(collection_name=collection_name, data=chunks)
        # 插入后强制加载集合，确保数据立即可查、Attu可视化界面可见
        client.load_collection(collection_name=collection_name)

        # 最终同步商品名称到全局状态
        # state["item_name"] = item_name
        # logger.info(f"步骤6：商品名称[{item_name}]成功存入Milvus集合[{collection_name}]，数据：{list(data.keys())}")
        # 捕获所有Milvus操作异常：连接中断、入库失败、索引错误等，不中断主流程
    except Exception as e:
        logger.error(f"步骤6：数据存入Milvus失败，原因：{str(e)}", exc_info=True)

    return state



if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")