import json
import requests
from tqdm import tqdm
from qwen_response import get_qwen_response
import re
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
## prompt
mode = "en" #['ch','mo','en']
with open(f"./data/prompt/label_{mode}.md","r",encoding='utf-8') as f:
    prompt_label = f.read()
with open(f"./data/quote/judge.json","r",encoding='utf-8') as f:
    judge = json.load(f)


## label
class label_agent:
    def __init__(self):
        self.prompt = prompt_label

    def limit_content(self,poem_content,quote):
        max_context=6
        lines = poem_content.strip().split('\n')
        try:
            index = lines.index(quote)
        except Exception as e:
            # print(e)
            return poem_content

        start = max(index - max_context, 0)
        end = min(index + max_context + 1, len(lines))
        selected_lines = lines[start:end]
        return '\n'.join(selected_lines)

    def get_dm(self,quote='',author='',poem='',poem_content='',info=''):
        if mode=='ch':
            poem = self.limit_content(poem,quote)
            label_prompt = self.prompt.format(quote=quote,poem_content=poem_content,author=author,poem=poem,info=info)
        elif mode == 'mo' or 'en':
            label_prompt = self.prompt.format(quote=quote,author=author,info=info)
        times = 0
        while times<=5:
            times += 1
        # print(label_prompt)
            content = get_qwen_response(label_prompt)
            if get_qwen_response(judge+content)=="Yes":  
                aa_pattern = r'<AA>(.*?)</AA>'
                DM_pattern = r'<DM>(.*?)</DM>'
                LB_pattern = r'<LB>(.*?)</LB>'
                try:
                    aa = re.findall(aa_pattern, content, re.DOTALL)
                    dm = re.findall(DM_pattern, content, re.DOTALL)
                    lb = re.findall(LB_pattern, content, re.DOTALL)
                except Exception as e:
                    print(e)
                    print(content)
                    return '','',[]
                return aa,dm,lb
        return '','',[]

def main_ch():
    labelagent = label_agent()
    try:
        with open("./data/quote/quotes_ch_label.json","r",encoding='utf-8') as f:
            result = json.load(f)
            len_ = len(result)
    except:
        print("no file")
        result = []
        len_=1

    ## quotes_data
    with open("./data/quote/quote_ch.json","r",encoding='utf-8') as f:
        quotes = json.load(f)

    for quote in tqdm(quotes[len_-1:]):
        _ = quote['quote']
        author = quote['author']
        poem = quote['poem']
        poem_content = quote['poem_content']



        analysis,deepmeaning,label = labelagent.get_dm(_,author,poem,poem_content)
        result.append({
            'quote':_,
            'author':author,
            'poem':poem,
            'poem_content':poem_content,
            'analysis':analysis,
            'deepmeaning':deepmeaning,
            'label':label
        })

        with open("./data/quote/quotes_ch_label.json","w",encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=4)

def main_mo():
    labelagent = label_agent()
    try:
        with open("./data/quote/quotes_mo_label.json","r",encoding='utf-8') as f:
            result = json.load(f)
            len_ = len(result)
    except:
        print("no file")
        result = []
        len_=1

    ## quotes_data
    with open("./data/quote/quote_mo.json","r",encoding='utf-8') as f:
        quotes = json.load(f)

    for quote in tqdm(quotes[len_-1:]):
        _ = quote['quote']
        author = quote['author']
        info = "主题："+ quote['topic']
        analysis,deepmeaning,label = labelagent.get_dm(quote=_,author=author,info=info)
        result.append({
            'quote':_,
            'author':author,
            'topic':quote['topic'],
            'analysis':analysis,
            'deepmeaning':deepmeaning,
            'label':label
        })

        with open("./data/quote/quotes_mo_label.json","w",encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=4)

def main_en():
    labelagent = label_agent()
    try:
        with open("./data/quote/quotes_en_label.json","r",encoding='utf-8') as f:
            result = json.load(f)
        print("LOAD")
        len_ = len(result)
    except Exception as e:
        print("no file"+str(e))
        result = []
        len_=1

    ## quotes_data
    with open("./data/quote/quote_en.json","r",encoding='utf-8') as f:
        quotes = json.load(f)

    for quote in tqdm(quotes[len_-1:]):
        _ = quote['quote']
        author = quote['author']
        info = "Topic:"+ quote['topic']
        analysis,deepmeaning,label = labelagent.get_dm(quote=_,author=author,info=info)
        result.append({
            'quote':_,
            'author':author,
            'topic':quote['topic'],
            'analysis':analysis,
            'deepmeaning':deepmeaning,
            'label':label
        })

        with open("./data/quote/quotes_en_label_.json","w",encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=4)




if __name__ == "__main__":
    # main_add()
    if mode=='ch':
        main_ch()
    elif mode=='mo':
        main_mo()
    elif mode=='en':
        main_en()