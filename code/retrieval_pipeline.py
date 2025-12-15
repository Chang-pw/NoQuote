## Please set the CUDA before you run Python: CUDA_VISIBLE_DEVICES=7 python code/retrieval_pipeline.py

from logit_token_level import compute_token_level_log_ratios
from rag import dm_retrieval,hard_label_filter
from context_summary import context_summary
from utils import label_extract,detect_language
import json
from tqdm import tqdm
import argparse
from utils import load_config

config = load_config()
BASE_MODEL = config['BASE_MODEL']

# Import the model for the Label Agent
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = BASE_MODEL
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True).to('cuda')

## Import the quotes json file
quotes_path = "./data/quote/quotes_zh_label.json"
print(f"Loading quotes {quotes_path}")
with open(quotes_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
penalty_params_lst = {i['quote']: i['penalty_weights'] for i in data}
mode_lst = {i['quote']: i['mode'] for i in data}
popularity_lst = {i['quote']: i['Search_score'] for i in data}

def hard_retrieval(quotes,label_list,threshold=0.6):
    core_domains,core_insights,aa,bb,cc,dd,ee = label_extract(label_list)
    # domains = [quote['core_domains'] for quote in quotes]
    insights = [quote['core_insights'] for quote in quotes]

    # domain_bool_list = hard_label_filter(domains,core_domains,threshold=threshold)
    insights_bool_list = hard_label_filter(insights,core_insights,threshold=threshold)

    # result = [x or y for x, y in zip(domain_bool_list, insights_bool_list)]
    result = insights_bool_list
    filtered_quotes = []
    aa = []
    for i,j in zip(quotes,result):
        if j:
            filtered_quotes.append(i)
        else:
            aa.append(i)

    quotes = [i['quote'] for i in filtered_quotes]
    aa_quotes = [i['quote'] for i in aa]
    return filtered_quotes,quotes,aa,aa_quotes

def Reasonable_retrieval(context,top_n=100,threshold=0.7,fixed_info='',mode='ch'):
    print("Starting Reasonable Retrieval...")
    if fixed_info:
        sm_context = fixed_info['summary_context']
        dm_context = [fixed_info['dm_context']]
        label_list = fixed_info['label_list']
    else:
        sm_context,dm_context,label_list = context_summary(context)
        print("Context Summary Completed:",sm_context)

    print("Context Deep Meaning:", dm_context[0])

    print("Starting Hard Filtering...")
    dm_filter_result, dm_filter_quotes = dm_retrieval(dm_context,top_n=top_n)
    hd_filter_result, hd_filter_quotes,aa_result,aa_quotes = hard_retrieval(dm_filter_result,label_list,threshold=threshold)
    print("Hard Filtering Completed. Retrieved Quotes:", len(dm_filter_quotes),"-->",len(hd_filter_quotes))

    return hd_filter_quotes,sm_context,dm_filter_quotes,dm_context[0],label_list

def Novelty_score(quotes,context,agg_mode='final'):
    left = context + "正如那句经典所言："
    result = []
    print("Starting Novelty Score...")
    for quote in tqdm(quotes):
        right = quote
        penalty_param = penalty_params_lst[right]
        res = compute_token_level_log_ratios(
            left, right,
            tokenizer, model,
            device='cuda',
            agg_mode=agg_mode,
            penalty_param=penalty_param,
            mode=mode_lst[quote]
        )
        result.append({'quote': right, 'novelty': round(res['novelty_log_ratio'], 2)})
    print("Novelty score Completed.")

    return result

def Popularity_score(quotes):
    print("Starting Popularity Score...")
    for quote in quotes:
        popularity_score = popularity_lst[quote['quote']]
        quote['popularity'] = round(popularity_score, 2)

    print("Popularity Score Completed.")
    return quotes  

def Reranking(quotes, n_alpha=1.0,p_alpha=1.0):
    for quote in quotes:
        novelty_score = quote['novelty']
        popularity_score = quote['popularity']
        final_score = round(novelty_score * n_alpha + popularity_score * p_alpha, 2)
        quote['final_score'] = final_score  

    return sorted(quotes, key=lambda x: x['final_score'], reverse=True)


def pipeline(context,top_n,agg_mode='final',fixed_info={},n_alpha=1,p_alpha=1,threshold=0.7,mode='ch'):
    try:
        ## Step 1: Reasonable Retrieval
        hd_quotes,summary_context,row_quotes,dm_context,label_list = Reasonable_retrieval(context,threshold=threshold,top_n=top_n,fixed_info=fixed_info,mode=mode)
        ## Step 2: Novelty Score
        quotes = Novelty_score(hd_quotes, summary_context, agg_mode=agg_mode)
        ## Step 3: Popularity Score
        quotes = Popularity_score(quotes)
        ## Step 4: Rearanking
        final_quotes = Reranking(quotes, n_alpha=n_alpha, p_alpha=p_alpha)
        print("Pipeline Completed.")
        print("Final Quotes:", final_quotes[0]['quote'])

        result = {
            "context":context,
            'summary_context':summary_context,
            'dm_context':dm_context,
            'label_list':label_list,
            "final_quote": final_quotes[0]['quote'],
            "final_quotes":final_quotes,
            "row_quotes": row_quotes,
            "hard_quotes": hd_quotes
        }
    except Exception as e:
        print(e)
        return 'Error','Error'

    return final_quotes[0]['quote'],result

def main(data):
    if data == 'ours':
        with open("./data/test/data_ch.json",'r',encoding='utf-8') as file:
            data=json.load(file)
        name = 'ch'
    elif data == 'quoteR':
        with open("./data/test/data_quote_R.json",'r',encoding='utf-8') as file:
            data=json.load(file)
        name = 'quoteR'
    elif data == 'quill':
        with open("./data/test/data_quill.json",'r',encoding='utf-8') as file:
            data=json.load(file)
        name = 'quill'
            
    n_alpha = 1.0
    p_alpha = 0
    for item in data:
        context = item['context']
        mode = detect_language(context)
        quote,result = pipeline(context,top_n=50, agg_mode='final',n_alpha=n_alpha,p_alpha=p_alpha,mode=mode)
        item['quote']= quote
        item['result']= result

        with open(f"./data/eval/data_{name}_eval_{n_alpha}_{p_alpha}.json","w",encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
    
    for item in data:
        item.pop('result', None)  
    
    with open(f"./data/eval/data_{name}_eval_{n_alpha}_{p_alpha}_concise.json","w",encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

if __name__ == '__main__':
    
    main("quoteR")

