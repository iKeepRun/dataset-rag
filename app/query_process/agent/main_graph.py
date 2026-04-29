import sys

from langgraph.graph import StateGraph,END

from app.import_process.agent.main_graph import kb_import_graph
from app.import_process.agent.state import ImportGraphState
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.state import QueryGraphState

# 创建主图节点对象
work_flow=StateGraph(QueryGraphState)

work_flow.add_node("node_item_name_confirm",node_item_name_confirm)
work_flow.add_node("node_web_search_mcp",node_web_search_mcp)
work_flow.add_node("node_search_embedding",node_search_embedding)
work_flow.add_node("node_search_embedding_hyde",node_search_embedding_hyde)
work_flow.add_node("node_rrf",node_rrf)
work_flow.add_node("node_rerank",node_rerank)
work_flow.add_node("node_answer_output",node_answer_output)

work_flow.set_entry_point("node_item_name_confirm")

def route_after_node_item_name_confirm(state:QueryGraphState):

    # 添加条件边
    if state['answer']:
        return "node_answer_output"
    return "node_web_search_mcp","node_search_embedding","node_search_embedding_hyde"

# 添加条件边,参数 为：起始节点，目标节点，映射（函数返回参数和节点名称的映射关系）
work_flow.add_conditional_edges("node_item_name_confirm",route_after_node_item_name_confirm,{
    "node_web_search_mcp":"node_web_search_mcp",
    "node_search_embedding":"node_search_embedding",
    "node_search_embedding_hyde":"node_search_embedding_hyde",
    "node_answer_output":"node_answer_output"
})
# work_flow.add_edge("node_item_name_confirm","node_web_search_mcp")
# work_flow.add_edge("node_item_name_confirm","node_search_embedding")
# work_flow.add_edge("node_item_name_confirm","node_search_embedding_hyde")

work_flow.add_edge("node_search_embedding","node_rrf")
work_flow.add_edge("node_search_embedding_hyde","node_rrf")
work_flow.add_edge("node_web_search_mcp","node_rrf")
work_flow.add_edge("node_rrf","node_rerank")
work_flow.add_edge("node_rerank","node_answer_output")
work_flow.add_edge("node_answer_output",END)


query_agent=work_flow.compile()

print(query_agent.get_graph().print_ascii())


def test_pdf_flow():
    print("\n==测试PDF文件处理流程==")
    # 模拟初始化状态
    initial_state = ImportGraphState(
        task_id="test_task_001",
        local_file_path="test.pdf",
        local_dir="./output",
        # 确保相关开关被正确初始化（根据您的 state 定义，有些可能是默认值）
        is_pdf_read_enabled=True
    )

    # 运行图
    print("开始运行....")
    try:
        # 修正点：使用 .invoke() 方法
        result = kb_import_graph.invoke(initial_state)
        print("运行结束，最终的状态 keys:", result.keys())
    except Exception as e:
        print(f"运行报错：{e}")
        # 打印详细堆栈以便调试
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("----", sys.path)
    test_pdf_flow()