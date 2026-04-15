import copy
from typing import TypedDict
from app.core.logger import logger
from app.utils.format_utils import format_state


class ImportGraphState(TypedDict):

    # 节点开始之前的准备数据
    task_id: str  # 任务id, 用于追踪日志

    local_dir: str        #   输出文件的目录
    local_file_path: str  #   原始输入的文件路径

    # 文件识别节点
    is_pdf_read_enabled: bool
    is_md_read_enabled: bool

    pdf_path: str         #   存储pdf文件目录
    md_path: str          #   存储md文件目录
    file_title: str       #   提取文件名，后续添加到向量数据库（如果大模型没有提取出item_name，用file_title做兜底）

    # 多模态图片理解节点（识别pdf和md文件中的图片）
    md_content: str       #   记录md文本内容

    #智能文档切割节点
    chunks: list          #   存储切分后的文本块
    max_content_length: int   #   切片最大值

    # 主体识别和标签提取节点
    item_name: str        #   存储设备的名称

    # 混合向量化节点
    embeddings_content: list  #    切片转向量化之后的数据，包含向量数据的列表，准备写入 Milvus


graph_default_state: ImportGraphState = {
    "task_id":"",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "is_normal_split_enabled": True,
    "is_silicon_flow_api_enabled": True,
    "is_advanced_split_enabled": False,
    "is_vllm_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "split_path": "",
    "embeddings_path": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": []
}

def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖
    :param overrides: 要覆盖的字段（关键字参数解包）
    :return:   新的状态实例
                Examples:
                     state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """

    # 默认状态
    state = copy.deepcopy(graph_default_state)
    # 用 overrides 覆盖默认值
    state.update(overrides)
    # 返回创建好的状态字典实例
    return state

def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染
    """
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    """
    测试
    """
    # 创建默认状态
    state = create_default_state(local_file_path="万用表RS-12的使用.pdf")
    logger.info(format_state(state))