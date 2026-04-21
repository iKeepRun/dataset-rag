import json
import os.path
import re
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

MAX_CONTENT_LENGTH = 2000
MIN_CONTENT_LENGTH = 500

def step_1_get_inputs(state):
    md_content = state['md_content'].replace("\r\n", "\n").replace("\r", "\n")
    # file_title = state['file_title']
    # 使用get获取参数，做空值兜底
    file_title = state.get('file_title','default_file')

    if md_content is None:
        logger.info(f">>> 节点执行终止：无有效MD内容")
        raise Exception("md文件内容为空")
    return md_content,file_title


def step_2_split_by_titles(md_content, file_title):
    """
    按标题切割
    :param md_content:
    :param file_title:
    :return:
    """

    # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
    # ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
    # #{1,6}：匹配1-6个#（对应MD1-6级标题）
    # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
    # .+：标题文字至少1个字符（避免空标题）
    title_pattern = r'^\s*#{1,6}\s+.+'
    lines = md_content.split("\n")
    # 是否为代码块
    is_code_block = False

    current_title = ''
    current_content = []
    title_count=0
    sections=[]    #存储最终返回结果

    for line in lines:
        # 去除空格
        line=line.strip()

        # 是否为代码块
        if line.startswith("```") or line.startswith("~~~"):
            is_code_block = not is_code_block
            current_content.append(line)
            continue
        # 是不是标题
        is_title = re.match(title_pattern, line)
        # 是标题且不是代码块
        if is_title and (not is_code_block):
            # 检查是不是第一次（根据title内容）
            if current_title:
                sections.append( {
                    "title": current_title,
                    "content":'\n'.join(current_content),
                    "file_title": file_title
                })

            current_title = line
            current_content=[line] # 这里不能用append 直接赋值覆盖上一次的结果，避免不同标题之间的内容混合
            title_count+=1    # 标题数量
        else:
            current_content.append(line)
    # 添加最后一行数据
    if current_title:
        sections.append({
            "title": current_title,
            "content": '\n'.join(current_content),
            "file_title": file_title
        })
    return sections,title_count,len(lines)


def split_long_content(section, max_length):
    """
    对过长的文本块进行二次切分
    :param section: 包含 title, content, file_title 的字典
    :param max_length: 最大长度限制
    :return: 切分后的列表
    """
    split_long_result = []
    
    # 检查内容是否为空或长度未超过限制（可选优化，避免不必要的处理）
    content = section.get('content', '')
    if not content or len(content) <= max_length:
        return [section]

    # 定义langchain文本切割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=int(max_length * 0.2),
        separators=["\n\n", "\n", "。", "！", "？", "；", "，"],
    )
    
    try:
        chunks = text_splitter.split_text(content)
    except Exception as e:
        logger.warning(f"文本切分失败，使用原始内容: {str(e)}")
        chunks = [content]

    for index, chunk in enumerate(chunks,start=1):
        # 跳过空块
        # if not chunk.strip():
        #     continue
        text=chunk.strip()

        split_long_result.append({
            "title": f"{section.get('title')}_{index}",
            "content": text,
            "file_title": section['file_title'],
            "parent_title": section.get("parent_title"),
            "part": index
        })
        
    # 如果切分后结果为空（极端情况），返回原始 section
    if not split_long_result:
        return [section]
        
    return split_long_result


def merge_short_section(final_sections, min_length):
    """
    合并小的文本块并且属于同一个父标题的文本
    :param section:
    :param min_length:
    :return:
    """
    merge_short_result=[]
    pre_section=None
    for section in final_sections:
        if pre_section is None:
            pre_section=section
            continue
        is_pre_short= len(pre_section['content'])<min_length
        # 判断是否属于同一个父标题,并且父标题不为空
        is_same_parent=pre_section.get('parent_title') and (pre_section['parent_title']==section['parent_title'])
        if  is_pre_short and is_same_parent:
            pre_section['content']+="\n\n"+section['content']
        else:
            merge_short_result.append(pre_section)
            pre_section=section
    # 添加最后一次结果
    if pre_section is not None:
        merge_short_result.append(pre_section)
    return merge_short_result

def step_3_refine_chunks(sections,max_length,min_length):
    final_sections=[]

    for section in sections:
        # 长度大于最大值
        if len(section['content'])>max_length:
            split_long_result=split_long_content(section,max_length)
            final_sections.extend(split_long_result)
        else:
            # 长度未超限，直接添加
            final_sections.append(section)

    final_sections=merge_short_section(final_sections,min_length)

    #补全参数
    for section in final_sections:
        section['part']=section.get('part') or 1
        section['parent_title']=section.get('parent_title') or section.get('title')
    return final_sections


def step_4_backup_section(state, final_sections):
    local_dir=state['local_dir']
    back_file_path=os.path.join(local_dir,"chunks.json")

    with open(back_file_path,"w",encoding="utf-8") as f:
        json.dump(
            final_sections,
            f,
            ensure_ascii=False,
            indent=4
        )


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    # 1. 记录开始任务的日志和任务状态的配置
    func_name = sys._getframe().f_code.co_name
    logger.info(f"节点{func_name}执行,参数状态:{state}")
    add_running_task(state['task_id'], func_name)

    try:
        # 1.参数校验
        # 作用：从状态字典提取MD内容/文件标题/Chunk最大长度，统一换行符消除系统差异，做空值兜底
        # 输出：标准化后的md_content、文件标题、单个Chunk最大长度；无有效MD内容则直接终止节点执行
        md_content, file_title = step_1_get_inputs(state)
        # 2.粗粒度切割（根据标题切割）
        sections,title_count,lines_count=step_2_split_by_titles(md_content,file_title)
        # 没有标题兜底
        if title_count==0:
            sections=[{
                "title":"没有标题",
                "content":md_content,
                "file_title":file_title
            }]
        # 3. 细粒度切割，长度大于max_content_length进行二次切割，长度小于min_content_length的进行合并
        final_sections=step_3_refine_chunks(sections,MAX_CONTENT_LENGTH,MIN_CONTENT_LENGTH)
        # 赋值
        state['chunks']=final_sections
        # 4. 数据备份
        step_4_backup_section(state,final_sections)
    except Exception as e:
        logger.error(f"节点{func_name}发生异常：{e}")
        raise   # 终止工作流
    finally:
        logger.info(f"节点{func_name}完成,数据状态{state}")
        add_done_task(state['task_id'], func_name)
    return state




if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\40725707-9e0a-4867-8cc2-28760d3fa052\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hl3040网络说明书",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        logger.info(f"文本分割节点参数状态：{result_state}")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")