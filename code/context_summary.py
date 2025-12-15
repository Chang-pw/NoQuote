from qwen_response import get_qwen_response
import re
from utils import convert_str_to_list

with open("./data/prompt/context_zh.md","r",encoding='utf-8') as f:
    label_prompt = f.read()
with open("./data/prompt/context_summary.md","r",encoding='utf-8') as f:
    summary_prompt = f.read()

def context_summary(context,summary_context=''):
    ## 第一步：总结
    if summary_context:
        summary_context = summary_context.strip()
    else:
        prompt_sm = summary_prompt+context
        summary_context = get_qwen_response(prompt_sm)

    ## 第二步：打标签
    prompt_zh = label_prompt.format(context=context)

    content = get_qwen_response(prompt_zh)
    DM_pattern = r'<DM>(.*?)</DM>'
    LB_pattern = r'<LB>(.*?)</LB>'
    try:
        dm = re.findall(DM_pattern, content, re.DOTALL)
        lb = re.findall(LB_pattern, content, re.DOTALL)
    except Exception as e:
        print(e)
        print(content)
        return '','',[]
    
    label = convert_str_to_list(str(lb))

    return summary_context,dm,label
