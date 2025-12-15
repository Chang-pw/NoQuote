from .rag_module import MyVectorDBConnector
import json
import ast
from utils import convert_str_to_list

## Import the Quote KB
vector = MyVectorDBConnector(path='./data/rag/quill_en',
                            collection_name='quill_en')

def dm_retrieval(query_texts,top_n=100):
    ## Deep Meaning Retrieval
    res_list=vector.search(
        query=query_texts[0],
        top_n=top_n
    )

    dms = res_list['documents'][0]
    metas = res_list['metadatas'][0]

    result = []

    for i,j in zip(dms,metas):
        label = convert_str_to_list(j['label'])
        
        result.append({
            'quote':j['quote'],
            'author':j['author'],
            'deepmeaning':i,
            'core_domains':label['core_domains'],
            'core_insights':label['core_insights'],
            'applicability':label['applicability'],
            'core_values':label['core_values'],
            'metaphors':label['metaphors'],
            'style':label['style'],
            'sentiment_tone':label['sentiment_tone']
        })

    quotes = [i['quote'] for i in result]

    return result,quotes


