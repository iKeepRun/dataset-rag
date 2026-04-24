
import os
import shutil
import uuid
from datetime import datetime
from typing import List, Dict, Any

import uvicorn

from app.core.logger import logger
from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.core.logger import PROJECT_ROOT
from app.import_process.agent.state import create_default_state, get_default_state
from app.import_process.agent.main_graph import kb_import_graph
from app.utils.task_utils import update_task_status, TASK_STATUS_FAILED, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, \
    add_running_task, add_done_task, get_task_status, get_done_task_list, get_running_task_list

app= FastAPI(
    title="知识图谱导入服务",
    description="知识图谱导入服务，提供知识图谱导入功能node_entry ==> pdf_to_md ==> md_img_process ==> md_split ==> item_name_recognition ==> bge-embedding ==> import_milvus",
    # version="0.1.0",
)

# 2. CORS 配置
# 允许前端页面（如运行在 5500 端口）访问后端 API（8000 端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许所有来源（生产环境建议指定具体域名）
    allow_credentials=True,
    allow_methods=["*"], # 允许所有 HTTP 方法
    allow_headers=["*"], # 允许所有请求头
)

def run_main_graph(task_id: str,local_file_path: str,local_dir: str):
    """
    运行知识图谱导入主流程
    :param state: 状态字典
    :return:
    """
    try:
        logger.info(f"任务开始执行：{task_id}")
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        state = get_default_state()
        state['task_id']=task_id
        state['local_file_path']=local_file_path
        state['local_dir']=local_dir
        for event  in kb_import_graph.stream(state):
            for node_name, node_state in event.items():
                logger.info(f"节点{node_name}执行完毕,参数状态:{node_state}")

        update_task_status(task_id, TASK_STATUS_COMPLETED)
        logger.info(f"图任务执行完成：{task_id}")
    except Exception as e:
        logger.exception(f"任务{task_id}执行异常：{str(e)}")
        update_task_status(task_id, TASK_STATUS_FAILED)


# 导入静态页面  ../page/import.html
@app.get("/import",response_class=FileResponse)
async def get_import_page():
    import_html_path =    PROJECT_ROOT / "app"/"import_process"/"page"/ "import.html"
    if not import_html_path.exists():
        logger.error(f"导入页面不存在：{import_html_path}")
        raise ValueError(f"导入页面不存在：{import_html_path}")
    return FileResponse(import_html_path)


@app.post("/upload", summary="文件上传接口", description="上传文件到服务器，并启动异步任务处理")
async def upload_file(background_tasks: BackgroundTasks, files: List[UploadFile]=File(...)):
   # 构建基础路径
   today_str= datetime.now().strftime("%Y%m%d")
   base_path= PROJECT_ROOT / "output" / today_str

   task_ids=[]
   for file in files:
       # 创建随机的任务id
       task_id=str(uuid.uuid4())
       # 添加到任务id列表返回给前端
       task_ids.append(task_id)

       local_dir= base_path / task_id
       local_file_path= local_dir/ file.filename
       local_dir.mkdir(parents=True, exist_ok=True)
       # 记录任务状态
       add_running_task(task_id, "upload_file")

       # 将上传的文件写入到 local_file_path文件中
       with open(local_file_path, "wb") as buffer:
           shutil.copyfileobj(file.file, buffer)


       # 运行知识图谱导入主流程，将任务加到队列中
       background_tasks.add_task(run_main_graph, str(task_id),str(local_file_path),str(local_dir))
       logger.info(f"{task_id}完成文件的上传并开启了异步任务！")
   # 返回结果
       add_done_task(task_id, "upload_file")
   return {
                "code":2000,
                "msg": f"文件上传成功，一共{len( files)}个文件！",
                "task_ids": task_ids
            }


# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8787)