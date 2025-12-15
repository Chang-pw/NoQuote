import json
from model import get_openai_api
from utils import load_yaml
from tqdm import tqdm
## Import the prompt for evaluation
with open("./data/prompt/eval_novelty.md","r",encoding='utf-8') as f:
    eval_prompt_novelty = f.read() ## (Context,Quote,Deep_Meaning)
with open("./data/prompt/eval_matching.md","r",encoding='utf-8') as f:
    eval_prompt_matching = f.read() ## (Context,Quote,Deep_Meaning)

## Evaluate JSON Data
quotes_path = "./data/quote/quotes_zh_label.json"
with open(quotes_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
deepmeaning_lst = {i['quote']: i['deepmeaning'] for i in data}

## Evaluate the model
def score_quote(context,quote,deepmeaning=''):
    all_tokens = 0
    prompt_match = eval_prompt_matching.replace("<context>",context).replace("<quote>",quote).replace("<deepmeaning>",deepmeaning)
    prompt_novelty = eval_prompt_novelty.replace("<context>",context).replace("<quote>",quote).replace("<deepmeaning>",deepmeaning)
    
    result_match,usage = get_openai_api(prompt_match)
    all_tokens += usage
    result_match = load_yaml(result_match)
    result_novelty,usage = get_openai_api(prompt_novelty)
    all_tokens += usage
    result_novelty = load_yaml(result_novelty)
    
    novelty_LLM = result_novelty['novelty']['score']
    match_LLM = result_match['matching']['score']
    return novelty_LLM,match_LLM,result_match,result_novelty,all_tokens

def score_pipeline_to_final(data):
    tokens = 0
    context = data['context']
    final_quotes = data['result']['row_quotes']
    result = []
    for quote in final_quotes:
        dpm = deepmeaning_lst[quote][0]
        # print(quote,context,dpm)
        ns,ms,result_match,result_novelty,all_tokens = score_quote(context,quote,dpm)
        tokens+=all_tokens
        result.append({
            'context': context,
            'quote' : quote,
            'novelty_score' : ns,
            'match_score' : ms,
            'result_match' : result_match,
            'result_novelty' : result_novelty
        })

    return result,tokens



