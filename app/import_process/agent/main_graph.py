from app.import_process.agent.state import ImportGraphState
from langgraph.graph import StateGraph, END

from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md

# 初始化图对象
workflow = StateGraph(ImportGraphState)

workflow.add_node("node_entry", node_entry)  # 流程入口：参数初始化、输入校验
workflow.add_node("node_pdf_to_md", node_pdf_to_md)  # PDF转MD：非MD格式文件的前置处理
workflow.add_node("node_md_img", node_md_img)  # MD图片处理：保证文档中图片的可访问性
workflow.add_node("node_document_split", node_document_split)  # 文档分块：解决大文本无法向量化/推理的问题
workflow.add_node("node_item_name_recognition", node_item_name_recognition)  # 项目名识别：业务定制化步骤，提取核心业务标识
workflow.add_node("node_bge_embedding", node_bge_embedding)  # BGE向量化：文本→向量，为Milvus存储做准备
workflow.add_node("node_import_milvus", node_import_milvus)  # 向量入库：将向量数据持久化到Milvus

# ===================== 3. 设置工作流入口节点 =====================
# 语法：set_entry_point("节点标识") → 推荐写法，直接指定流程起始节点
# 等效写法：workflow.add_edge(START, "node_entry")（START是LangGraph内置起始常量）
# 作用：指定工作流执行的第一个节点，替代手动添加START到目标节点的边，代码更简洁
workflow.set_entry_point("node_entry")


# 定义条件路由函数
def route_after_entry(state: ImportGraphState) -> str:
    if state["is_pdf_read_enabled"]:
        return "node_pdf_to_md"
    elif state["is_md_read_enabled"]:
        return "node_md_img"
    else:
        return END


workflow.add_conditional_edges(
           "node_entry",
                  route_after_entry,
                  {
                      "node_pdf_to_md": "node_pdf_to_md",
                      "node_md_img": "node_md_img",
                      END: END
                  })

workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

kb_import_graph = workflow.compile()

