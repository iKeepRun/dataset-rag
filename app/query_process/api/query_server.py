import uuid

import uvicorn
from aiohttp.web_response import StreamResponse
from fastapi import FastAPI, HTTPException, BackgroundTasks,Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse,FileResponse

from app.core.logger import logger
from app.import_process.agent.state import create_default_state
from app.query_process.agent.main_graph import query_agent
from app.query_process.agent.state import create_query_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.sse_utils import create_sse_queue, sse_generator, push_to_session, SSEEvent
from app.utils.task_utils import get_task_result, update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED

app=FastAPI ()

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 健康检查接口
@app.get("/health")
async  def health():
    logger.info("health check: ok!!!")
    return {"status": "ok"}



# 返回页面
@app.get("/chat")
def get_chat_page():
    file_path=PROJECT_ROOT / "app" / "query_process" / "page" / "chat.html"
    if not file_path.exists():
        logger.error(f"页面不存在：{file_path}")
        raise HTTPException(status_code=404, detail=f"页面不存在：{file_path}")
    # 返回静态页面
    return FileResponse(file_path)


class QueryRequest(BaseModel):
    query:str=Field(...,title="用户提问")
    is_stream:bool=Field(False,title="是否流式返回")
    session_id:str=Field(None,title="会话ID")

# 用户提问
def run_query_graph(session_id:str,query:str,is_stream:bool):
    state=create_query_default_state(query=query,session_id=session_id,is_stream=is_stream)

    update_task_status(task_id=session_id,status_name=TASK_STATUS_PROCESSING,push_queue=is_stream)
    try:
      query_agent.invoke(state)
    except Exception as e:
      logger.exception(f"{session_id}任务执行异常：{str(e)}")
      update_task_status(task_id=session_id,status_name=TASK_STATUS_FAILED,push_queue=is_stream)
      # 向队列推送报错相关数据
      push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
    update_task_status(task_id=session_id,status_name=TASK_STATUS_COMPLETED)


@app.post("/query")
async def query(query_request:QueryRequest,background_tasks: BackgroundTasks):
    session_id=query_request.session_id or str(uuid.uuid4())
    query=query_request.query
    is_stream=query_request.is_stream

    if is_stream:
        # 创建 SSE 队列
        create_sse_queue(session_id=session_id)
        # 异步执行
        background_tasks.add_task(run_query_graph,session_id,query,is_stream)
        logger.info(f"{session_id}开始异步查询：{query}")
        return {
            "session_id": session_id,
            "message": "查询任务已开始，请稍后..."
        }
    else:
         # 同步执行，等待返回结果
         run_query_graph(session_id,query,is_stream)
         result=get_task_result(task_id=session_id,key="answer")
         logger.info(f"{session_id}同步查询完成：{query}")
         return {
             "done_list": [],
             "session_id": session_id,
             "message": "结果处理完成！",
             "answer": result,
         }

@app.get("/stream/{session_id}")
def stream(session_id: str,request: Request):
    logger.info(f"开始sse查询：{session_id}")
    return StreamingResponse(
        sse_generator(session_id=session_id, request=request),
        media_type="text/event-stream"
    )


if __name__ == '__main__':
    uvicorn.run( app, host="127.0.0.1", port=8860)