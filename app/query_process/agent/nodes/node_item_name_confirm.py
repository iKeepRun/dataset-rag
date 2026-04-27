import sys

from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger

def node_item_name_confirm(state:QueryGraphState):
    #从对堆栈中获取到方法名
    func_name=sys._getframe().f_back.f_code.co_name


    logger.info(f"当前节点：{func_name}")
    return state