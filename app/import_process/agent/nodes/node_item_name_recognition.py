import os
import sys
from typing import Tuple, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task, add_done_task


def step_1_get_chunks(state: ImportGraphState)->Tuple[list[Dict],str]:
    """
    参数校验
    :param state:
    :return:
    """
    chunks = state.get("chunks") or []
    file_title = state.get("file_title", "default_title")

    if not chunks:
        logger.error(f"参数校验失败: chunks 不能为空")
        raise ValueError(f"参数校验失败: chunks 不能为空")
    return chunks, file_title

def step_2_get_item_name(chunks:list[Dict], file_title:str):
    """
    获取item_name,后续根据用户的提问 确定用哪个item_name进行回答（确定使用哪个文档）
    :param chunks:
    :param file_title:
    :return: item_name
    """
    if not chunks:
        logger.error(f"参数校验失败: chunks 不能为空")
        raise ValueError(f"参数校验失败: chunks 不能为空")
    # 截取前5个chunks
    chunks = chunks[:5]
    # 拼接提示词
    human_prompt=load_prompt(name="item_name_recognition", file_title=file_title,context=chunks)
    system_prompt=load_prompt(name="product_recognition_system")

    llm_client = get_llm_client()

    response=llm_client.invoke([
        SystemMessage(content= system_prompt),
        HumanMessage(content= human_prompt)
       ])

    result=response.content.strip().replace(" ", "")
    if not result:
        logger.warning("大模型返回空内容，使用文件标题作为商品名称兜底")
        result=file_title
    return result

def step_3_update_chunks(state: ImportGraphState,chunks:list[Dict], item_name:str):
    """
    更新切片数据
    :param state:
    :param chunks:
    :param item_name:
    :return: 更新item_name之后的切片集合
    """
    for chunk in chunks:
        chunk["item_name"] = item_name

    state['item_name']=item_name
    state['chunks']=chunks



def step_4_get_item_name_vector(item_name:str)->Tuple[list[float],list[float]]:
    """
    获取item_name的稠密/稀疏向量数据
    :param item_name:
    :return:dense_vector,sparse_vector
    """
    embedding_result=generate_embeddings([item_name])
    dense_vector=embedding_result["dense"][0]
    sparse_vector=embedding_result["sparse"][0]
    return dense_vector,sparse_vector


def step_5_store_item_name_vector(state: ImportGraphState,file_title,item_name,dense_vector:list[float], sparse_vector:list[float]):
    """
        步骤 6: 将商品名称、文件标题、双向量持久化到Milvus向量数据库
        核心逻辑：
            1. 配置校验：检查Milvus连接地址和集合名配置，缺失则跳过
            2. 客户端获取：获取单例Milvus客户端，连接失败则跳过
            3. 集合初始化：无集合则创建（定义Schema+索引），有集合则直接使用（保留原有配置）
            4. 幂等性处理：删除同名商品数据，避免重复存储
            5. 数据插入：构造符合Schema的数据，非空向量才添加
            6. 集合加载：插入后强制加载集合，确保数据立即可查/Attu可见
        参数：
            state: 流程状态对象，用于最终状态同步
            file_title: 处理后的文件标题
            item_name: 识别后的商品名称（主键去重依据）
            dense_vector: 步骤5生成的稠密向量（1024维列表）
            sparse_vector: 步骤5生成的稀疏向量（字典格式）
        """
    # 从配置类中获取
    milvus_uri = milvus_config.milvus_url
    collection_name = milvus_config.item_name_collection

    # 配置缺失校验：任一配置为空则跳过Milvus存储，记录警告
    if not all([milvus_uri, collection_name]):
        logger.warning("Milvus配置缺失（MILVUS_URL/ITEM_NAME_COLLECTION），跳过数据保存")
        return

    logger.info(f"开始执行步骤6：将商品名称[{item_name}]保存到Milvus集合[{collection_name}]")

    try:
        # 获取Milvus单例客户端，连接失败则直接返回
        client = get_milvus_client()
        if not client:
            logger.error("无法获取Milvus客户端（连接失败），跳过数据保存")
            return

        # 集合初始化：不存在则创建（定义Schema+索引），存在则直接使用
        if not client.has_collection(collection_name=collection_name):
            logger.info(f"Milvus集合[{collection_name}]不存在，开始创建Schema和索引")
            # 创建集合Schema：自增主键+动态字段，适配灵活的数据存储
            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
            # 添加自增主键字段：INT64类型，唯一标识每条数据
            schema.add_field(
                field_name="pk",
                datatype=DataType.INT64,
                is_primary=True,
                auto_id=True
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
                params={"M": 16, "efConstruction": 200}
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
        clean_item_name = (item_name or "").strip()
        if clean_item_name:
            client.load_collection(collection_name=collection_name)
            # 商品名称转义，防止特殊字符导致过滤表达式解析失败
            safe_item_name = escape_milvus_string(clean_item_name)
            filter_expr = f'item_name=="{safe_item_name}"'
            # 执行删除操作
            client.delete(collection_name=collection_name, filter=filter_expr)
            logger.info(f"Milvus幂等性处理完成，已删除集合中[{clean_item_name}]的历史数据")

        # 构造插入Milvus的数据：基础字段+非空向量字段
        data = {
            "file_title": file_title,
            "item_name": item_name
        }
        # 稠密向量非空才添加，避免空值入库报错
        if dense_vector is not None:
            data["dense_vector"] = dense_vector
        # 稀疏向量非空则归一化后添加，保证检索准确性
        if sparse_vector is not None:
            data["sparse_vector"] = sparse_vector

        # 插入数据：列表格式支持批量插入，单条数据保持格式统一
        client.insert(collection_name=collection_name, data=[data])
        # 插入后强制加载集合，确保数据立即可查、Attu可视化界面可见
        client.load_collection(collection_name=collection_name)

        # 最终同步商品名称到全局状态
        state["item_name"] = item_name
        logger.info(f"步骤6：商品名称[{item_name}]成功存入Milvus集合[{collection_name}]，数据：{list(data.keys())}")

    # 捕获所有Milvus操作异常：连接中断、入库失败、索引错误等，不中断主流程
    except Exception as e:
        logger.error(f"步骤6：数据存入Milvus失败，原因：{str(e)}", exc_info=True)




def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。

    步骤一：获取参数 进行校验 chunks file_title(提取不到item_name做兜底值)
    步骤二：截取前5个chunks,拼接提示词，向让文本大模型提取item_name
    步骤三：更新chunks数据，添加item_name
    步骤四：调用本地向量大模型生成向量数据item_name
    步骤五：存储向量数据到向量数据库中 kb_item_name (id/file_title/item_name /稠密和稀疏 dense/sparse)
    """
    # 1. 记录开始任务的日志和任务状态的配置
    func_name = sys._getframe().f_code.co_name
    logger.info(f"节点{func_name}执行,参数状态:{state}")
    add_running_task(state['task_id'], func_name)


    try:
       # 步骤一：获取参数 进行校验 chunks file_name(提取不到item_name做兜底值)
       chunks,file_title= step_1_get_chunks(state)
       # 步骤二：截取前5个chunks,拼接提示词，向让文本大模型提取item_name
       item_name=step_2_get_item_name(chunks,file_title)
       # 步骤三：更新chunks数据，添加item_name
       step_3_update_chunks(state,chunks,item_name)
       # 步骤四：调用本地向量大模型生成向量数据item_name
       dense_vector,sparse_vector=step_4_get_item_name_vector(item_name)
       # 步骤五：存储向量数据到向量数据库中 kb_item_name (id/file_title/item_name /稠密和稀疏 dense/sparse)
       step_5_store_item_name_vector(state,file_title,item_name,dense_vector,sparse_vector)
    except Exception as e:
        logger.error(f"节点{func_name}发生异常：{e}")
        raise
    finally:
        logger.info(f"节点{func_name}完成,数据状态{state}")
        add_done_task(state['task_id'], func_name)
    return state








# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()