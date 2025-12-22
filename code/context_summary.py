from model import get_openai_api
import re
from utils import convert_str_to_list

def context_summary(context,summary_context=''):
    with open("./data/prompt/summary.md","r",encoding='utf-8') as f:
        summary_prompt = f.read()

    with open("./data/prompt/context.md","r",encoding='utf-8') as f:
        context_prompt = f.read()

    if summary_context:
        summary_context = summary_context.strip()
    else:
        prompt_sm = summary_prompt+context
        summary_context = get_openai_api(prompt_sm)

    ## Step1: analysis and deep meaning extraction
    prompt = context_prompt.format(context=context)

    content = get_openai_api(prompt)
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
