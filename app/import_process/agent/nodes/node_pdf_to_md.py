import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from turtledemo.penrose import start

import requests
from requests import session
from sympy.utilities.misc import func_name

from app.conf.mineru_config import mineru_config
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.path_util import PROJECT_ROOT

from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger

def step_1_validate_path(state):
    """
    进行路径校验 pdf_file 没有直接抛异常  local_dir 没有则创建默认值
    :param state:
    :return:
    """
    logger.debug("md转pdf节点下，开始校验文件路径")
    pdf_file=state['pdf_path']
    local_dir=state['local_dir']
    if not pdf_file:
        #抛出参数异常
        raise ValueError("没有输入文件，无法解析")
    if not local_dir:
        local_dir=PROJECT_ROOT/ "output"
    pdf_path_obj=Path(pdf_file)
    local_dir_obj=Path(local_dir)
    if not pdf_path_obj.exists():
        raise FileNotFoundError("文件不存在")

    if not local_dir_obj.exists():
        # 文件输出路径不存在，则创建
        Path(local_dir).mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, local_dir_obj

def step_2_upload_and_get_zip(pdf_path_obj):
    token = mineru_config.api_key
    url = mineru_config.base_url
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "files": [
            # {"name": f"{pdf_path_obj.name}"}
            {"name": pdf_path_obj.name}
        ],
        "model_version": "vlm"
    }

    response = requests.post(url=f"{url}/file-urls/batch", headers=header, json=data)

    if response.status_code != 200 or response.json()['code'] != 0:
        logger.error(f"MinerU 文件批量上传解析API请求失败: {response.json()['msg']}")
        raise RuntimeError("MinerU 文件批量上传解析API请求失败")

    upload_url=response.json()['data']['file_urls'][0]
    batch_id=response.json()['data']['batch_id']

    # 使用纯净的网络请求，避免被代理
    http_session=requests.session()
    http_session.trust_env=False

    with requests.session() as http_session:
        with open(pdf_path_obj, 'rb') as f:
            res_upload = http_session.put(upload_url, data=f.read())
            if res_upload.status_code != 200:
                logger.error(f"MinerU 文件上传API请求失败: {res_upload.json()['msg']}")
                raise RuntimeError("MinerU 文件上传API请求失败")


    # 获取zip下载地址
    url = f"{url}/extract-results/batch/{batch_id}"
    start_time=time.time()
    # 超时时间
    timeout=600
    # 循环间隔时间
    sleep_time=3
    while True:
        if time.time()-start_time>timeout:
            logger.error(f"MinerU 文件解析API请求超时")
            raise RuntimeError("MinerU 文件解析API请求超时")
        res = requests.get(url, headers=header)
        if res.status_code != 200:
            # 服务器异常重试
            if 500 <= res.status_code < 600:
                time.sleep(sleep_time)
                continue
            raise RuntimeError("MinerU 文件解析API请求失败")

        # 其他异常直接抛出异常
        if res.json()['code']!=0:
            logger.error(f"MinerU 获取解析结果API请求失败: {res.json()['msg']}")
            raise RuntimeError("MinerU 获取解析结果API请求失败")
        extract_result=res.json()["data"]["extract_result"][0]
        if extract_result['state'] =='done':
            full_zip_url=extract_result["full_zip_url"]
            logger.debug(f"MinerU 文件解析完成，耗时：{time.time()-start_time},结果:{full_zip_url}")
            return full_zip_url
        else:
            time.sleep(sleep_time)

def step_3_download_and_unzip(zip_url, local_dir_obj, stem):
    """
    下载zip并解压
    :param zip_url: 下载的路径
    :param local_dir_obj: 存储的路径
    :param stem: 原文件名
    :return: md文件的绝对路径，便于后面读取
    """

    response=requests.get(zip_url)
    if response.status_code != 200:
        logger.error(f"下载文件失败: {response.status_code}")
        raise RuntimeError("下载文件失败")
    zip_down_dir=local_dir_obj/f"{stem}.zip"
    with open(zip_down_dir, 'wb') as f:
        f.write(response.content)
    # 解压zip
    # 解压的路径
    extract_target_dir=local_dir_obj/stem
    if extract_target_dir.exists():
        # 递归删除目录及文件
       shutil.rmtree(extract_target_dir)
    # 创建解压路径文件夹
    Path(extract_target_dir).mkdir(parents=True, exist_ok=True)
    # 解压zip
    with zipfile.ZipFile(zip_down_dir, 'r') as zip_ref:
        zip_ref.extractall(extract_target_dir)
    # 解压以后得文件名有多种可能
    md_file_list=list(extract_target_dir.rglob("*.md"))

    if not md_file_list:
        logger.error(f"MinerU 解析结果错误，没有找到对应的md文件")
        raise RuntimeError("MinerU 解析结果错误，没有找到对应的md文件")
    # 存储最终md文件
    target_md_file=None

    for md_file in md_file_list:
        if md_file.name== stem+".md":
            target_md_file=md_file
            break
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower()=="full.md":
                target_md_file=md_file
                break
    if not target_md_file:
       target_md_file=md_file_list[0]

    if target_md_file.stem!=stem:
       target_md_file=target_md_file.rename(target_md_file.with_name(f"{stem}.md"))

    final_md_str_path=str(target_md_file.resolve())
    return  final_md_str_path

def node_pdf_to_md(state:ImportGraphState)->ImportGraphState:
    """
    LangGraph工作流节点：PDF转MD核心处理节点
    核心流程：路径校验 → MinerU上传解析 → 结果下载解压 → 读取MD内容并更新工作流状态
    参数：state-工作流状态对象，需包含pdf_path/local_dir/task_id
    返回：更新后的工作流状态，新增md_path/md_content
    步骤：1. 记录开始任务的日志和任务状态的配置
         2. 路径参数校验pdf_path和local_dir(第一个节点仅仅校验了字面层面)
         3. 使用mineru批量解析pdf文件获取到md（申请-》上传-》获取），返回xxx.zip下载地址
         4. 下载zip包，解压并提取到 local_dir 地址
         5. md_path地址赋值，读取md文件内容赋值md_content
         6. 记录结束任务的日志和任务状态的配置
    """
    # 1. 记录开始任务的日志和任务状态的配置
    func_name=sys._getframe().f_code.co_name
    logger.info(f"节点{func_name}执行,参数状态:{state}")
    add_running_task(state['task_id'], func_name)
    try:
        # 2. 路径参数校验pdf_path和local_dir(第一个节点仅仅校验了字面层面)
        pdf_path_obj, local_dir_obj=step_1_validate_path(state)
        # 3. 使用mineru批量解析pdf文件获取到md（申请-》上传-》获取），返回xxx.zip下载地址
        zip_url=step_2_upload_and_get_zip(pdf_path_obj)
        # 4. 下载zip包，解压并提取到 local_dir 地址
        md_path=step_3_download_and_unzip(zip_url, local_dir_obj, pdf_path_obj.stem)
        # 5. md_path地址赋值，读取md文件内容赋值md_content
        state['md_path']=md_path
        state['local_dir']=str(local_dir_obj)

        with open(md_path, 'r', encoding='utf-8') as f:
            state['md_content']=f.read()

    except Exception as e:
        logger.error(f"节点{func_name}执行异常:{e}")
        raise
    finally:
        add_done_task(state['task_id'], func_name)
        logger.info(f"节点{func_name}执行完毕,参数状态:{state}")
    return state



if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        # local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")