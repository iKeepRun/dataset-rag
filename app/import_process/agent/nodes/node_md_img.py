# 读取图片并转换为 base64
import base64
import re
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Tuple

from minio.deleteobjects import DeleteObject

from app.clients.minio_utils import get_minio_client
from app.conf.lm_config import lm_config
from app.conf.minio_config import minio_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.lm.lm_utils import get_llm_client
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.utils.task_utils import add_running_task, add_done_task

# 模块级变量：用于跟踪API调用频率，避免重复初始化
_vl_request_times = deque()


def step_01_validate_file(state:ImportGraphState):
    # 1.获取上一节点的md_path的路径进行校验
    md_path= state['md_path']
    md_content = state['md_content']
    if not md_path:
        logger.error("没有输入文件，无法解析")
        raise ValueError("没有输入文件，无法解析")

    md_path_obj=Path(state['md_path'])
    if not md_path_obj.exists():
        logger.error(f"md文件地址{md_path}不存在")
        raise FileNotFoundError(f"md文件地址{md_path}不存在")

    # 判断md_content内容是否有值（处理直接上传md文件的情况）
    if not md_content:
        with open(md_path, 'r', encoding='utf-8') as f:
            state['md_content'] = f.read()

    imgs_dir_obj = md_path_obj.parent / 'images'
    return  md_path_obj , md_content ,imgs_dir_obj


def extract_image_info(md_content: str, context_chars: int = 100) -> list[Dict]:
    """
    从 Markdown 内容中提取所有图片信息，包括位置、上下文

    Args:
        md_content: Markdown 文件内容
        context_chars: 上下文字符数，默认100

    Returns:
        图片信息列表，每个元素包含：
        - img_url: 图片路径/URL
        - start_pos: 匹配开始位置
        - end_pos: 匹配结束位置
        - full_match: 完整匹配文本
        - context_before: 上文
        - context_after: 下文
        - alt_text: 替代文本
    """
    images = []

    # 正则表达式：匹配 ![alt](url) 格式
    img_pattern_md = r'!\[([^\]]*)\]\(([^)]+)\)'
    # 正则表达式：匹配 <img src="url" /> 格式
    img_pattern_html = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'

    # 使用 finditer 获取位置信息
    for match in re.finditer(img_pattern_md, md_content):
        alt_text = match.group(1)
        img_url = match.group(2)
        start_pos = match.start()
        end_pos = match.end()

        # 提取上下文
        context_start = max(0, start_pos - context_chars)
        context_end = min(len(md_content), end_pos + context_chars)
        context_before = md_content[context_start:start_pos]
        context_after = md_content[end_pos:context_end]

        images.append({
            'img_url': img_url,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'full_match': match.group(0),
            'context_before': context_before,
            'context_after': context_after,
            'alt_text': alt_text,
            'type': 'markdown'
        })

    # 处理 HTML 格式的图片标签
    for match in re.finditer(img_pattern_html, md_content):
        img_url = match.group(1)
        start_pos = match.start()
        end_pos = match.end()

        context_start = max(0, start_pos - context_chars)
        context_end = min(len(md_content), end_pos + context_chars)
        context_before = md_content[context_start:start_pos]
        context_after = md_content[end_pos:context_end]

        images.append({
            'img_url': img_url,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'full_match': match.group(0),
            'context_before': context_before,
            'context_after': context_after,
            'alt_text': '',
            'type': 'html'
        })

    logger.info(f"提取到 {len(images)} 个图片")
    return images


def generate_image_description(image_path: Path, context_before: str,
                               context_after: str) -> str:
    """
    调用视觉大模型生成图片描述

    Args:
        image_path: 图片文件路径
        context_before: 上文内容
        context_after: 下文内容

    Returns:
        图片描述文本
    """

    try:
        # logger.info(f"视觉模型配置 - 模型: {lm_config.lv_model}")
        # logger.info(f"视觉模型配置 - BaseURL: {lm_config.base_url}")
        # logger.info(f"视觉模型配置 - API Key前缀: {lm_config.api_key[:10] if lm_config.api_key else 'None'}...")

        # 应用 API 速率限制：每60秒最多允许10次请求，防止触发限流
            # request_times: 用于记录请求时间的双端队列，以跟踪API调用频率
            # max_requests: 在指定时间窗口内允许的最大请求次数，设置为10次
            # window_seconds: 时间窗口的大小，单位为秒，设置为60秒
        apply_api_rate_limit(request_times=_vl_request_times, max_requests=10, window_seconds=60)

        context=(context_before,context_after)
        # 构建提示词
        prompt = load_prompt(name='image_summary',root_folder=image_path,image_content=context)
        # 初始化视觉大模型客户端
        vl_client=get_llm_client(model=lm_config.lv_model)


        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')

        # 调用视觉模型
        response = vl_client.invoke([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ])

        description = response.content.strip().replace("\n", "")
        logger.debug(f"图片描述生成成功: {description[:50]}...")
        return description

    except Exception as e:
        logger.error(f"生成图片描述失败: {e}")
        return ""


def upload_to_minio(image_path: Path,stem:str) -> str:
    """
    上传图片到 MinIO 并返回访问 URL

    Args:
        image_path: 图片文件路径

    Returns:
        MinIO 访问 URL，失败返回 None
    """
    try:
        minio_client = get_minio_client()

        if not minio_client:
            logger.error("MinIO 客户端未初始化")
            return None

        logger.info(f"upload_to_minio方法开始执行，图片路径为：{image_path}")


        # 构造完整的对象路径
        full_object_path = f"{minio_config.minio_img_dir}/{stem}/{image_path.name}"

        # 获取需要删除的旧对象
        object_list=minio_client.list_objects(
            bucket_name=minio_config.bucket_name,
            prefix=f"{minio_config.minio_img_dir[1:]}/{stem}",   # 注意去掉前缀的 /
        )
        delete_objs=[DeleteObject(object.object_name) for object in object_list]
        logger.info(f"已经删除{stem}文件夹下的{len(delete_objs)}个旧对象：")
        minio_client.remove_objects(
            bucket_name=minio_config.bucket_name,
            delete_object_list=delete_objs
        )

        # 上传文件
        minio_client.fput_object(
            bucket_name=minio_config.bucket_name,
            object_name=full_object_path,
            file_path=str(image_path)
        )

        # 构造访问 URL
        endpoint = minio_config.endpoint
        protocol = 'https://' if minio_config.minio_secure else 'http://'

        if not endpoint.startswith(('http://', 'https://')):
            endpoint = f"{protocol}{endpoint}"

        url = f"{endpoint}/{minio_config.bucket_name}/{full_object_path}"
        logger.info(f"图片上传成功: {url}")
        return url

    except Exception as e:
        logger.error(f"上传图片到 MinIO 失败: {e}")
        return None


def process_single_image(img_info: Dict, imgs_dir_obj: Path,stem:str) -> Tuple[str, str]:
    """
    处理单个图片：上传到 MinIO 并生成描述

    Args:
        img_info: 图片信息字典
        imgs_dir_obj: 图片文件夹路径对象

    Returns:
        (minio_url, img_desc) 元组
    """
    img_url = img_info['img_url']

    # 解析图片文件路径
    img_file_path = Path(img_url)
    if not img_file_path.is_absolute():
        img_file_path = imgs_dir_obj.parent / img_file_path

    if not img_file_path.exists():
        logger.warning(f"图片文件不存在: {img_file_path}")
        return None, None

    # 1. 上传到 MinIO
    minio_url = upload_to_minio(img_file_path,stem)
    if not minio_url:
        return None, None

    # 2. 生成图片描述
    img_desc = generate_image_description(
        img_file_path,
        img_info['context_before'],
        img_info['context_after']
    )

    return minio_url, img_desc


def replace_images_in_markdown(md_content: str, images: list[Dict],
                               imgs_dir_obj: Path,stem:str) -> str:
    """
    替换 Markdown 中的所有图片为新的格式

    Args:
        md_content: 原始 Markdown 内容
        images: 图片信息列表
        imgs_dir_obj: 图片文件夹路径对象

    Returns:
        替换后的 Markdown 内容
    """
    # 按位置倒序排序，避免替换时位置偏移
    sorted_images = sorted(images, key=lambda x: x['start_pos'], reverse=True)
    process_cache = {}
    for img_info in sorted_images:
        img_url=img_info['img_url']
        if img_url in process_cache:
            minio_url, img_desc = process_cache[img_url]
        else:
            minio_url, img_desc = process_single_image(img_info, imgs_dir_obj,stem=stem)
            process_cache[img_url] = (minio_url, img_desc)

        if minio_url and img_desc:
            # 替换为新格式: ![描述](URL)
            new_format = f"![{img_desc}]({minio_url})"

            # 使用位置进行精确替换
            start = img_info['start_pos']
            end = img_info['end_pos']
            md_content = md_content[:start] + new_format + md_content[end:]

            logger.info(f"图片已替换: {img_info['img_url']} -> {new_format}")
        else:
            logger.warning(f"图片处理失败，跳过: {img_info['img_url']}")

    return md_content


def node_md_img(state:ImportGraphState)->ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """

    # 1. 记录开始任务的日志和任务状态的配置
    func_name = sys._getframe().f_code.co_name
    logger.info(f"节点{func_name}执行,参数状态:{state}")
    add_running_task(state['task_id'], func_name)

    # 2. 验证文件和提取基础信息
    md_path_obj, md_content, imgs_dir_obj = step_01_validate_file(state)
    if not md_path_obj:
        logger.error(f"{imgs_dir_obj.name}文件夹不存在，没有图片需要处理，执行下一个节点")
        return state

    # 检查图片文件夹是否存在
    if not imgs_dir_obj.exists():
        logger.warning(f"图片文件夹不存在: {imgs_dir_obj}")
        return state

    # 3. 提取所有图片信息（包含位置和上下文）
    # images 列表中的每个元素是一个字典，包含：
    # - img_url: 图片的相对路径或 URL
    # - start_pos: 图片标记在 md_content 中的起始位置
    # - end_pos: 图片标记在 md_content 中的结束位置
    # - full_match: 完整的图片标记文本（如 ![alt](url) 或 <img ...>）
    # - context_before: 图片标记前的上下文字符串
    # - context_after: 图片标记后的上下文字符串
    # - alt_text: 图片的替代文本（alt 属性）
    # - type: 图片标记类型 ('markdown' 或 'html')
    images = extract_image_info(md_content, context_chars=100)

    if not images:
        logger.info("未检测到图片，跳过处理")
        return state

    logger.info(f"找到 {len(images)} 个图片，开始处理...")

    # 4. 处理所有图片并替换
    updated_md_content = replace_images_in_markdown(md_content, images, imgs_dir_obj,stem=md_path_obj.stem)

    # 5. 保存更新后的内容
    state['md_content'] = updated_md_content

    md_path_obj = Path(state['md_path'])
    new_md_path = md_path_obj.with_stem(md_path_obj.stem + '_new')

    with open(new_md_path, 'w', encoding='utf-8') as f:
        f.write(updated_md_content)
    # 更新节点状态
    state['md_path']=str(new_md_path)
    logger.info(f"已保存新的 Markdown 文件: {new_md_path}")

    add_done_task(state['task_id'], func_name)
    logger.info(f"节点{func_name}执行完毕,参数状态:{state}")
    return state


"""
node_md_img 节点方法调用流程图及说明

1. node_md_img(state: ImportGraphState) -> ImportGraphState
   [入口] 主节点函数，负责协调整个图片处理流程。
   |
   +-- 1.1 初始化与日志记录
   |     - 获取当前函数名 func_name
   |     - 记录开始执行日志
   |     - add_running_task(state['task_id'], func_name): 标记任务开始
   |
   +-- 1.2 文件验证与信息提取
   |     - 调用 step_01_validate_file(state)
   |       |
   |       +-- 检查 md_path 是否存在且有效
   |       +-- 如果 md_content 为空，读取文件内容
   |       +-- 构建 imgs_dir_obj (图片目录路径对象)
   |       |
   |     - 返回: md_path_obj, md_content, imgs_dir_obj
   |
   +-- 1.3 前置检查
   |     - 如果 md_path_obj 无效或 imgs_dir_obj 不存在:
   |       - 记录警告/错误日志
   |       - 直接返回 state (跳过图片处理)
   |
   +-- 1.4 图片信息提取
   |     - 调用 extract_image_info(md_content, context_chars=100)
   |       |
   |       +-- 使用正则表达式匹配 Markdown 格式图片: ![alt](url)
   |       +-- 使用正则表达式匹配 HTML 格式图片: <img src="url" ...>
   |       +-- 提取每张图片的:
   |           - img_url: 图片路径
   |           - start_pos/end_pos: 在原文中的位置
   |           - context_before/context_after: 前后各100字符上下文
   |           - alt_text: 替代文本
   |           - type: 标记类型 ('markdown' 或 'html')
   |       |
   |     - 如果 images 列表为空:
   |       - 记录日志 "未检测到图片"
   |       - 返回 state
   |
   +-- 1.5 图片处理与替换
   |     - 调用 replace_images_in_markdown(md_content, images, imgs_dir_obj)
   |       |
   |       +-- 将 images 按 start_pos 倒序排序 (避免替换时位置偏移)
   |       +-- 遍历每个 img_info:
   |           |
   |           +-- 调用 process_single_image(img_info, imgs_dir_obj)
   |               |
   |               +-- 解析图片绝对路径 img_file_path
   |               +-- 检查文件是否存在，不存在则返回 (None, None)
   |               |
   |               +-- 调用 upload_to_minio(img_file_path)
   |                   |
   |                   +-- 获取 MinIO 客户端
   |                   +-- 上传文件到 MinIO Bucket
   |                   +-- 构造并返回公开访问 URL (minio_url)
   |                   |
   |               +-- 调用 generate_image_description(img_file_path, context_before, context_after)
   |                   |
   |                   +-- 加载提示词模板 load_prompt(name='image_summary', ...)
   |                   +-- 获取视觉大模型客户端 get_llm_client(model=lm_config.lv_model)
   |                   +-- 读取图片文件并转换为 Base64
   |                   +-- 调用 VL 模型 API (invoke)，传入提示词和图片 Base64
   |                   +-- 解析响应，返回图片描述文本 (img_desc)
   |                   |
   |               +-- 返回: (minio_url, img_desc)
   |           |
   |           +-- 如果 minio_url 和 img_desc 均有效:
   |               - 构造新格式: ![img_desc](minio_url)
   |               - 根据 start_pos/end_pos 替换原 Markdown 内容中的图片标记
   |           +-- 否则:
   |               - 记录警告日志，跳过该图片
   |       |
   |     - 返回: updated_md_content (替换后的 Markdown 内容)
   |
   +-- 1.6 保存结果
   |     - 更新 state['md_content'] = updated_md_content
   |     - 构造新文件路径 new_md_path (原文件名加 _new 后缀)
   |     - 将 updated_md_content 写入 new_md_path
   |     - 记录保存成功日志
   |
   +-- 1.7 收尾工作
   |     - add_done_task(state['task_id'], func_name): 标记任务完成
   |     - 记录结束执行日志
   |     - 返回 state
"""



### 测试代码
if __name__ == '__main__':
    state = create_default_state()
    state['md_path']='D:\\_project\\pycharm-projects\\dataset-rag\\output\\40725707-9e0a-4867-8cc2-28760d3fa052\\hak180产品安全手册\\hak180产品安全手册.md'
    state['md_content']='# HAK 180 烫金机\n\n# 产品安全手册（简体中文）\n\n感谢您购买 HAK 180 烫金机。\n\n在使用本设备之前，请先阅读本手册，包括所有预防措施。阅读本手册后，请妥善保管。\n\n有关使用本设备的更多信息，请参阅使用说明书，其可在兄弟 (中国)商业有限公司技术服务支持网站 http://www.95105369.com/Web/Manuals.aspx 上找到。建议您先通读使用说明书，再使用本设备。如需获得常见问题解答、故障排除和说明书，请访问http://www.95105369.com。\n\n# 对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。\n\n•\t对于保养、调整或维修事宜，请联系 Brother 呼叫中心或您当地的Brother 经销商。  \n•\t如果本设备工作不正常或发生任何错误，请关闭本设备，拔下所有电缆，然后联系 Brother 呼叫中心或您当地的 Brother 经销商。  \n•\t本文档中提供的信息可能会随时更改，恕不另行通知。  \n•\t严禁未经授权擅自复制或重制本文档的任何部分或全部内容。  \n•\t请注意，对于使用通过本设备制作的产品造成的任何损坏或利润损失，或者故障、维修导致的数据消失或更改，或者第三方提出的任何索赔，我们不承担任何责任。\n\n# 警告\n\n# 不遵守说明和警告可能导致人员死亡或严重受伤。遵守这些指引以避免冒烟、发热、爆炸、火灾或人员受伤的风险。\n\n# 设备\n\n•\t请先阅读这本手册，再尝试操作本设备或尝试进行任何维护。不按照这些说明操作可能会提高发生人员受伤或财产损坏（包括火灾、触电、烧伤或窒息所致）的风险。对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。  \n•\t请勿在未去除所有包装材料的情况下使用本设备，包括本设备内部的任何附加的包装材料。否则可能会产生火灾的风险。  \n•\t请勿拆解本设备。拆解本设备可能会导致火灾或触电。  \n•\t请勿尝试自行维修本设备。打开或拆下盖子可能使您接触到危险电压点以及带来其他风险，并且可能使您的保修失效。对于所有维修事宜，请联系 Brother 呼叫中心或您当地的 Brother 经销商。  \n•\t请在以下环境使用本设备：温度保持在 $1 0 ~ ^ { \\circ } \\mathsf { C }$ 和 ${ } ^ { 3 2 } { } ^ { \\circ } \\mathsf { C }$ 之间，湿度保持在 $20 \\%$ 和 $80 \\%$ 之间，无冷凝。  \n•\t请勿使本设备受到阳光直射、过热、接触明火、腐蚀性气体、湿气或灰尘。否则可能产生触电、短路或火灾的风险，从而导致损坏设备和/或导致设备无法运行。  \n•\t请勿将设备放在加热器、空调、电风扇或水附近。否则当水（包括加热/空调/通风设备所产生的冷凝水）接触本设备时可能产生短路或火灾的风险。  \n•\t如果设备变得异常高温、冒烟、产生任何强烈味道，或者如果您意外在设备上倒入任何液体，请立即从电源插座拔掉设备的插头。请联系 Brother 呼叫中心或您当地的 Brother 经销商。  \n•\t如果设备跌落或者已损坏，则有触电的可能性。请从电源插座中拔掉设备的插头，然后联系 Brother 呼叫中心或您当地的 Brother 经销商。  \n•\t如果水、其他液体或金属物体进入设备内部，请立即从电源插座中拔掉设备的插头，然后联系 Brother 呼叫中心或您当地的 Brother经销商。  \n•\t请勿在卡纸或有纸张散落在设备内部的情况下尝试使用本设备。纸张与定影单元长时间接触可能导致火灾。  \n•\t请勿使用任何易燃物品、任何类型的喷雾剂包含酒精或氨水的有机溶剂/液体来清洁本设备的内部或外部。否则可能导致火灾。请改用无绒干抹布。有关如何清洁本设备的说明，请参阅使用说明书。\n\n•\t请勿将本设备放在化学品附近，或者将本设备放置在可能会泼溅到化学品的位置。万一化学品接触本设备，则存在火灾或触电的风险。特别是有机溶剂或液体（如苯、油漆稀释剂、抛光剂或除臭剂）可能导致塑料盖和/或电缆溶解或分解，从而产生火灾或触电的风险。这些化学品或其他化学品可能导致本设备故障或褪色。  \n•\t本设备的包装中使用了塑料袋。塑料袋并不是玩具。为避免窒息的危险，请将这些塑料袋远离婴儿和儿童，并正确弃置这些塑料袋。  \n•\t对于使用起搏器的用户：\n\n本设备可能会产生弱磁场。如果您在本设备附近感觉到起搏器工作不正常，请远离本设备，并立即咨询医生。\n\n•\t使用本设备之后短时间内，本设备的一些内部零件仍然处于极热状态。打开前盖时，请勿触摸以灰色标记的区域。存在烧伤的风险。先等待设备冷却下来，再触摸设备的内部零件。\n\n![](images/ac26d5ab3a9f599eb2f58c2f2cb89f009fd2172b49782804756ea10c7256d4b4.jpg)\n\n![](images/3161a58d2459d3bd06765cddbe05dd4c037093c2aa66c41f1964c8f77924dc0f.jpg)  \n儎\u245fഴḽ䆜\u0a80ᛞ࠽व䀜\u1aae儎\u245fⲺ䇴༽䜞ԬȾ\n\n![](images/66ee4447cdd36e786369677a3a3aa8c36cedbfd2cdc10dde42ad9da98edefeab.jpg)\n\n# 电源线\n\n•\t本设备通过 AC 220 V-240 V 50/60 Hz 电源供电。  \n请勿将本设备连接到直流电源或逆变器（直流交流变换器）。存在火灾或触电的风险。  \n•\t请勿用湿手触摸插头。这样可能导致触电。如果不确定您拥有哪种类型的电源，请联系合格的电工。  \n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。  \n•\t设备内部有高压电极。\n\n先拔掉电源线，再清洁设备内部。拔出电源线时，不要拉电线，而是捏住插头往外拔。存在发生火灾、触电或设备故障的风险。  \n•\t请勿将任何物体压在电源线上。  \n•\t请勿将本设备放在人们可能踏过电源线的位置。  \n•\t请勿将本设备放置在会使得拉伸或拉紧电源线的位置，否则电源线可能会磨损或损坏。  \n•\t始终确保插头已完全插入。如果电源线磨损或损坏，请勿使用设备或用手触摸电源线。如果拔出设备的电源插头，请勿触摸损坏/磨损的部分。  \n•\t请勿让设备压在电源线上。  \n•\t请勿在雷暴天气期间使用本设备。存在闪电导致触电的潜在风险。  \n•\t请勿使用任何非指定的电缆。否则可能导致火灾或人员受伤。必须按照使用说明书正确安装。  \n•\t请勿让任何金属硬件或任何类型的液体落在设备的电源插头上。否则可能导致触电或火灾。  \n•\tBrother 强烈建议您不要使用任何类型的延长线。  \n•\t定期拔出电源插头进行清洁。使用干布清洁插头插脚根部以及插脚之间的位置。如果电源插头长时间插入在电源插座中，灰尘会堆积在插头插脚周围，这可能会导致短路，从而引起火灾。  \n•\t本设备装有接地的插头。此插头只能插入接地的电源插座中。这是一项安全功能。如果您无法将插头插入到插座中，请让电工更换过时的插座。请勿试图破坏接地插头的作用。\n\n# 注意\n\n# 不遵守说明和警告可能导致人员中度或严重受伤。遵守这些指引以避免人员受伤。\n\n# 设备\n\n•\t将本设备放置在平整、水平且稳定的表面上（如桌面），避免震动和冲击。  \n•\t将本设备放置在通风良好的环境中。  \n•\t为了防止人员受伤，请谨慎操作，避免将手指放置在图中所示的区域中。\n\n![](images/249fa4657bc1fd69ca12d3735f5dae5c4452840980633d314aa69411a3e31a44.jpg)\n\n![](images/91797d135ffab0d98bad74abdf8ce1c2c019b1e303a11072728e898f20a1688d.jpg)\n\n# 电源线\n\n•\t如果您长时间不会使用本设备，请从电源插座中拔掉电源线以确保安全。  \n•\t本设备必须安装在可轻松使用电源插座的位置附近。如果发生意外情况，必须从电源插座中拔掉电源线以完全关闭电源。  \n•\t请勿将手放在纸张边缘。纸张锋利的边缘可能导致受伤。\n\n# 为设备选择一个安全的位置\n\n•\t提起本设备时，请使用双手抓稳本设备的两侧。如果抓住的是进纸托板和出纸盒，它们可能会掉下来。必须通过将双手放在本设备下面来搬运本设备。\n\n![](images/53af8eff82c97e9620326cf4dd4a58924107e96c14d8c43354e9a1daacea5a67.jpg)\n\n确保本设备的任何部位均未伸出设备所在的桌面或支架。特别是当本设备位于桌面、支架等边缘时，请勿让出纸盒打开。确保本设备位于平整、水平且稳定的表面上，避免震动。不遵守这些预防措施可能导致设备跌落，从而导致用户的人身伤害以及设备严重损坏。\n\n![](images/3848fdbce626b32de93029aba36a6282ee540069b9ef08543b1b185c30d6c91e.jpg)\n\n# 重要事项\n\n# “ 重要事项 ” 表示可能导致财产损失或本设备功能丧失的潜在危险情况。\n\n# 设备\n\n如果遵守了操作说明进行操作，但是设备不能正确运行，请仅调整操作说明中涵盖的控制。错误调整其他控制可能导致损坏并且通常需要合格技术进行全面工作以将本设备恢复到正常操作。Brother不建议使用 Brother 正品烫金膜盒以外的其他品牌烫金膜盒。如果使用与本设备不兼容的耗材导致损坏本设备的任何零件，由此导致的任何维修可能不在保修范围内。\n\n# 电源线\n\n请勿将设备连接到受墙壁开关或自动计时器控制的电源插座，或者与大型设备或需要大量电力的其他设备连接到同一个电路中。否则可能会损坏电源。电源损坏还可能会从本设备的内存中删除信息，并且反复打开/关闭电源可能会损坏本设备。\n\n# 警告标签\n\n请勿撕下或损坏设备上的任何注意/ 警告标签以及序列号标签。\n\n# 设备保修和责任\n\n本手册中的任何内容都将不会影响现有设备保修，也不应被视为授予任何其他设备保修。不遵循本手册中的安全说明可能导致本设备的保修失效。\n\n# 设备和电源线\n\n•\t请仅使用本设备随附的电源线。  \n•\t不要在本设备周围放置任何物体。在紧急情况下，此类物体会阻碍接近电源插座。必须保证在需要时可以拔出设备的插头。  \n•\t请遵守所有适用法规来处理本设备。\n\n产品中有害物质的名称及含量  \n\n<table><tr><td>型号</td><td colspan="6">有害物质</td></tr><tr><td>HAK180</td><td>铅</td><td>汞</td><td>镉</td><td>六价铬</td><td>多溴联苯</td><td>多溴二苯醚</td></tr><tr><td>部件名称</td><td>(Pb)</td><td>(Hg)</td><td>(Cd)</td><td>(Cr(VI))</td><td>(PBB)</td><td>(PBDE)</td></tr><tr><td>框架L单元</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>框架R单元</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>中框架单元</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>框架</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>顶盖单元</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>进纸器单元</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>热熔器</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>盖板</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>标签</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>金属薄片保持单元</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>主电路板</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>低压电源电路板</td><td>×</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>选配件</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>包装材料</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr></table>\n\n本表格依据 SJ/T 11364 的规定编制。\n\n○：表示该有害物质在该部件所有均质材料中的含量均在GB/T26572 规定的限量要求以下。  \n×：表示该有害物质至少在该部件的某一均质材料中的含量超出GB/T 26572 规定的限量要求。\n\n（由于技术的原因暂时无法实现替代或减量化）'

    node_md_img(state)

    # images=extract_image_info(state['md_content'])
    #
    # print( json.dumps(images,indent=4,ensure_ascii=False))
