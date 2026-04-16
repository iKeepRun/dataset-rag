import os
import sys

from app.import_process.agent.state import ImportGraphState, create_default_state
from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state:ImportGraphState)->ImportGraphState:

    func_name=sys._getframe().f_code.co_name
    logger.info(f"节点{func_name}开始执行,参数状态:{state}")

    # 向前端推送
    add_running_task(state['task_id'], func_name)
    # 验参
    local_file_path=state['local_file_path']
    if not local_file_path:
        logger.error(f"{func_name}核心参数 local_file_path缺失，请先上传文件")
        return state
    # 提取文件名
    file_name=os.path.basename(local_file_path).split( ".")[0]

    if local_file_path.endswith(".pdf"):
        state['is_pdf_read_enabled']= True
        state['pdf_path']=local_file_path
    elif local_file_path.endswith(".md"):
        state['is_md_read_enabled']= True
        state['md_path']=local_file_path
    else:
        logger.warning("不支持的文件格式，仅支持 pdf/md 文件")
        return state

    state['file_title']=file_name


    add_done_task(state['task_id'], func_name)

    logger.info(f"节点{func_name}执行完毕,参数状态:{state}")
    return state