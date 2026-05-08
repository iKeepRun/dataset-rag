from modelscope.hub.snapshot_download import snapshot_download

# rerank重排序模型下载脚本
local_dir = r"D:\ai_models\modelscope_cache\models\rerank"

snapshot_download(
    model_id="BAAI/bge-reranker-large",
    cache_dir=local_dir,
)

print("下载完成，模型目录：", local_dir)