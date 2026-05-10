from modelscope import snapshot_download
from torch.mtia import snapshot

model_dir=snapshot_download("BAAI/bge-m3",cache_dir='/Users/czq/ai_models/modelscope_cache/models')
print(f"模型已经下载到：{model_dir}")