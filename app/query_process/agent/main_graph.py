from langgraph.graph import StateGraph,END

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


agent=work_flow.compile()

print(agent.get_graph().print_ascii())