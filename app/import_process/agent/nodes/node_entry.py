import sys

from app.import_process.agent.state import ImportGraphState, create_default_state
from app.core.logger import logger


def node_entry(state:ImportGraphState)->ImportGraphState:


    logger.info(f">>> [Stub] 执行节点: {sys._getframe().f_code.co_name}")
    logger.info("入口节点执行了")
    return state