from typing import TypedDict, List


class QueryGraphState(TypedDict):
    session_id:str
    original_query: str  # 用户提问



    # rewrite_result: dict  # 用户问题的识别与改写，item_name确认结果

    web_search_docs:str   # search_result: dict  # 网络搜索的结果
    embedding_chunks:list # vector_search_result: dict  # 普通向量搜索的结果
    hyde_embedding_chunks:list# hyde_search_result: dict  # HyDE检索的切片

    # fusion_result: dict  # 向量和HyDE检索融合与粗排的结果

    rrf_chunks:list  # 融合排序之后的切片     # re_rank_result: dict  # function_result和 网络搜索重排序的结果
    reranked_docs:list # 重排序之后的最终Top-k文档

    prompt:str  # 组装好的提示词
    answer: str  # 最终答案

    item_names:List[str]  # 识别出的主体名称列表
    rewritten_query: str  # 改写后的用户问题
    history:str # 交互历史
    is_stream:bool # 是否流式返回
