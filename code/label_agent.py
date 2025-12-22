import json
from tqdm import tqdm
from model import get_openai_api
import re

## prompt
with open(f"./data/prompt/0_deep-meaning.md","r",encoding='utf-8') as f:
    prompt_0 = f.read()
with open(f"./data/prompt/1_correct.md","r",encoding='utf-8') as f:
    prompt_1 = f.read()
with open(f"./data/prompt/2_label.md","r",encoding='utf-8') as f:
    prompt_2 = f.read()

## label
class label_agent:
    def __init__(self):
        pass

    def get_dm(self,quote='',author='',info=''):
        deep_meaning_prompt = prompt_0.format(quote=quote,author=author,info=info)
        times = 0
        while times<=3:
            times += 1

            content = get_openai_api(deep_meaning_prompt)
            aa_pattern = r'<AA>(.*?)</AA>'
            DM_pattern = r'<DM>(.*?)</DM>'
            aa = re.findall(aa_pattern, content, re.DOTALL)
            dm = re.findall(DM_pattern, content, re.DOTALL)

            Info = {"quote":quote,"author":author,"info":info,"analysis":aa,"deep_meaning":dm}
        
            correct_prompt = prompt_1 + "/n Information: " + str(Info)

            if get_openai_api(correct_prompt) == "Yes":
                return aa,dm
        
        return aa,dm

    def get_lb(self,quote='',author='',info='',deep_meaning=''):
        label_prompt = prompt_2.format(quote=quote,author=author,info=info,deep_meaning=deep_meaning)
        content = get_openai_api(label_prompt)
        lb_pattern = r'<LB>(.*?)</LB>'
        lb = re.findall(lb_pattern, content, re.DOTALL)
        return lb


    def pipeline(self):
        ## quotes_data
        with open("./data/quote/quote_en.json","r",encoding='utf-8') as f:
            quotes = json.load(f)
        result = []
        for quote in tqdm(quotes):
            _ = quote['quote']
            author = quote['author']
            info = "Topic:"+ quote['topic']
            analysis,deepmeaning = self.get_dm(quote=_,author=author,info=info)
            label = self.get_lb(quote=_,author=author,info=info,deep_meaning=deepmeaning)

            result.append({
                'quote':_,
                'author':author,
                'topic':quote['topic'],
                'analysis':analysis,
                'deepmeaning':deepmeaning,
                'label':label
            })

        with open("./data/quote/quotes_en_label.json","w",encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=4)

