from aiohttp.web_fileresponse import FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.logger import logger
from app.utils.path_util import PROJECT_ROOT

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
def getPage(page):
    file_path=PROJECT_ROOT / "app" / "query_process" / "page" / "chat.html"
    if not file_path.exists():
        logger.error(f"页面不存在：{file_path}")
        raise HTTPException(status_code=404, detail=f"页面不存在：{file_path}")
    # 返回静态页面
    return FileResponse(file_path)



# 用户提问
