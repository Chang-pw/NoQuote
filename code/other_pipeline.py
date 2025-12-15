import json
from model import get_openai_api
import json
import ast
from utils import convert_str_to_list
from tqdm import tqdm
from reranker import *

def row_quote_retrieval(top_n=50):
    from score_system import score_quote
    from rag.rag_module import MyVectorDBConnector
    quotes_path = "./data/quote/quotes_zh_label.json"
    with open(quotes_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    deepmeaning_lst = {i['quote']: i['deepmeaning'] for i in data}

    ## Import the Quote KB
    vector = MyVectorDBConnector(path='./data/rag/quill_zh_row',
                                collection_name='quill_zh_row')

    with open('./data/test/data_quote_R.json','r',encoding='utf-8') as file:
        data=json.load(file)
    
    for i in tqdm(data):
        res_list=vector.search(
            query=i['context'],
            top_n=top_n
        )
        i['retrieval']=res_list['documents'][0]
    
    for i in tqdm(data):
        try:
            top5_quotes = i['retrieval'][:5]
            context = i['context']
            match_scores = 0
            novelty_scores = 0
            for quote in top5_quotes:
                deepmeaning = deepmeaning_lst[quote][0]
                novelty_LLM,match_LLM,result_match,result_novelty,all_tokens = score_quote(context,quote,deepmeaning)
                match_scores += match_LLM
                novelty_scores += novelty_LLM
            i['top5_match_score'] = match_scores / 5
            i['top5_novelty_score'] = novelty_scores / 5
        except Exception as e:
            print(f"Error: {e}")
            continue
    match_scores = [i['top5_match_score'] for i in data]
    novelty_scores = [i['top5_novelty_score'] for i in data]
    print("match_scores:",sum(match_scores)/len(match_scores))
    print("novelty_scores:",sum(novelty_scores)/len(novelty_scores))

    return data

def LLM_quote_generation(model):
    with open('./data/test/data_ch.json','r',encoding='utf-8') as file:
        data=json.load(file)
    with open('./data/prompt/LLM_generate.md','r') as f:
        prompt = f.read()

    for i in tqdm(data):
        if model=='gpt-4o':
            get_response = get_openai_api
    
        context = i['context']
        pp = prompt.format(query=context)
        response,total_tokens = get_response(pp)
        i['top5_generate_quotes'] = response
    with open(f"./data/ablation/LLM/{model}_generate_quote.json",'w',encoding='utf-8') as file:
        json.dump(data,file,ensure_ascii=False,indent=2)
            

def reranker(rerank_method):
    if rerank_method == 'bm25':
        get_rerank = bm25_reranker
    elif rerank_method == 'bge':
        get_rerank = flag_reranker
    elif rerank_method == 'qwen3':
        get_rerank = qwen3_reranker
    elif rerank_method == 'embedding':
        get_rerank = embedding_reranker
    elif rerank_method == 'self_bleu':
        get_rerank = self_bleu_reranker
    elif rerank_method == 'surprisal':
        get_rerank = surprisal_reranker
    elif rerank_method == 'kl':
        get_rerank = kl_reranker
    elif rerank_method == 'mi':
        get_rerank = mi_reranker

    with open("./data/eval/data_ch_eval_1.0_0.json",'r',encoding='utf-8') as file:
        ref_data=json.load(file)
    
    for i in tqdm(ref_data):
        try:
            context = i['context']
            result = i['result']['hard_quotes']
            rerank_result = get_rerank(result,context)
            i[f'{rerank_method}_result'] = rerank_result
        except Exception as e:
            print(e)
            continue
    with open(f"./data/ablation/reranker/{rerank_method}_ch_eval.json",'w',encoding='utf-8') as file:
        json.dump(ref_data,file,ensure_ascii=False,indent=2)


def parameter_search(novelty_weight,popularity_weight):
    with open("./data/eval/data_ch_eval_1.0_0_ref.json","r",encoding='utf-8') as f:
        data = json.load(f)
    concise = []
    for i in tqdm(data):
        try:
            result = i['result']
            final_quotes = result['final_quotes']
            for quote in final_quotes:
                quote['final_score'] = (
                    quote['novelty'] * novelty_weight +
                    quote['popularity'] * popularity_weight
                            )
            sorted_data = sorted(final_quotes, key=lambda x: x['final_score'], reverse=True)
            quotes = [i['quote'] for i in sorted_data][:5]
            result['final_quotes'] = sorted_data
            concise.append({
                "context":i['context'],
                "final_quotes":quotes
            })
        except:
            pass

    
    with open(f"./data/ablation/parameter/data_ch_eval_{novelty_weight}_{popularity_weight}_50_0.7.json",'w',encoding='utf-8') as file:
        json.dump(data,file,ensure_ascii=False,indent=2)
    with open(f"./data/ablation/parameter/data_ch_eval_{novelty_weight}_{popularity_weight}_50_0.7_concise.json",'w',encoding='utf-8') as file:
        json.dump(concise,file,ensure_ascii=False,indent=2)

        
def score_data():
    from score_system import score_quote
    quotes_path = "./data/quote/quotes_zh_label.json"
    with open(quotes_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    deepmeaning_lst = {i['quote']: i['deepmeaning'] for i in data}
    res = []
    for method in ['Qwen2.5-7b','Qwen3-0.6B']:
        data_path = f"./data/ablation/reranker/{method}_ch_eval.json" 
        with open(data_path,'r',encoding='utf-8') as f:
            data = json.load(f)
        
        for i in tqdm(data):
            try:
                result = i['result']
                context = i['context']
                final_quotes = result['final_quotes']
                final_quotes = [i['quote'] for i in final_quotes][:5]
                match_scores =0
                novelty_scores = 0
                for quote in final_quotes:
                    try:
                        deepmeaning = deepmeaning_lst[quote]
                    except:
                        deepmeaning = ['']

                    novelty_LLM,match_LLM,result_match,result_novelty,all_tokens = score_quote(context,quote,deepmeaning[0])
                    # novelty_LLM,match_LLM = 1,1
                    match_scores += match_LLM
                    novelty_scores += novelty_LLM
                i['top5_match_score'] = match_scores / len(final_quotes)
                i['top5_novelty_score'] = novelty_scores / len(final_quotes)
                with open(data_path,'w',encoding='utf-8') as file:
                    json.dump(data,file,ensure_ascii=False,indent=2)
            except Exception as e:
                print(e)
                pass

        match_scores = [i['top5_match_score'] for i in data if 'top5_match_score' in i]
        novelty_scores = [i['top5_novelty_score'] for i in data if 'top5_novelty_score' in i]
        print("match_scores:",sum(match_scores)/len(match_scores))
        print("novelty_scores:",sum(novelty_scores)/len(novelty_scores))
        res.append({
            "match_scores":sum(match_scores)/len(match_scores),
            "novelty_scores":sum(novelty_scores)/len(novelty_scores),
        })

    

if __name__=='__main__':
    task = 'score_data'

    if task == 'exp_row_quote_retrieval':
        data=row_quote_retrieval(top_n=50)
        with open('./data/ablation/row/data_quoteR_row_quote_retrieval.json','w',encoding='utf-8') as file:
            json.dump(data,file,ensure_ascii=False,indent=2)
    elif task == 'exp_LLM_quote_generation':
        LLM_quote_generation(model='gpt-4o')
    elif task == 'rerank':
        reranker(rerank_method='self_bleu')
    elif task == 'parameter_search':
        parameter_search(novelty_weight=1.0,popularity_weight=0.0)
    elif task == 'score_data':
        score_data()


