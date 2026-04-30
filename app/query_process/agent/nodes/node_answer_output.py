import sys
import time

from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.sse_utils import push_to_session, SSEEvent
from app.utils.task_utils import add_running_task, add_done_task, set_task_result


def node_answer_output(state:QueryGraphState):
    func_name=sys._getframe().f_code.co_name
    add_running_task(state['session_id'],func_name,is_stream=state['is_stream'])

    session_id=state['session_id']
    is_stream=state.get("is_stream",True)
    answer=state['answer'] or f"答案：{state.get("original_query","问题"),"这是测试流式返回的测试数据，通过sse技术将答案一点一点推送到前端。。。。。。。"}"

    pending_answer=""


    if is_stream:
        for i in answer:
            pending_answer+=i
            push_to_session(session_id, SSEEvent.DELTA,{"data":i} )
            time.sleep(0.03)
        # image_urls = ["https://apps.truenas.com/catalog/openclaw",
        #               "https://www.alibabacloud.com/blog/openclaw-launches-on-alibaba-cloud-simple-application-server_602845"]
        image_urls = ["http://www.baidu.com/img/bd_logo.png", "https://example.com/demo-2.png"]
        push_to_session(session_id, SSEEvent.FINAL, {"answer": pending_answer,
                                                         "status":"completed",
                                                         "image_urls":image_urls
                                                         })
        logger.info(f"流式输出完毕，状态数据总长度：{len(pending_answer)}")
    else:
        pending_answer=answer
        set_task_result(task_id=session_id,key="answer",value=pending_answer)
    add_done_task(state["session_id"],func_name,is_stream=state['is_stream'])
    logger.info(f"节点{func_name}执行完毕，状态数据：{state}")
    return state