import sys

from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task


def node_item_name_confirm(state:QueryGraphState):
    #从对堆栈中获取到方法名
    func_name=sys._getframe().f_back.f_code.co_name
    add_running_task(state['session_id'], func_name, is_stream=state['is_stream'])

    add_done_task(state["session_id"], func_name, is_stream=state['is_stream'])
    logger.info(f"节点{func_name}执行完毕，状态数据：{state}")
    return state