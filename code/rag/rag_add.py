#给rag数据库增加数据
from rag_module import MyVectorDBConnector,get_embeddings
import json
##运行
vector=MyVectorDBConnector(path='./code/rag/model/quill_en',collection_name='quill_en')

documents=[]
metadata=[]
#收集的1w多条quote
directory='./data/quote/quotes_en_label_.json'
with open(directory,'r',encoding='utf-8') as file:
    data=json.load(file)
for dic in data:
    # dm = dic['deepmeaning'][0].strip()
    dm = dic['quote']
    # if dic['quote'] not in documents:
    documents.append(dm)
    try:
        metadata.append({
            "quote":dic['quote'],
            "author":dic['author'],
            "poem":dic['poem'],
            "label":str(dic['label'])
        })
    except:
        metadata.append({
            'quote':dic['quote'],
            'author':dic['author'],
            'label':str(dic['label'])
        })

print(documents[:2])

print(len(documents))
def batch_add_documents(collection, documents, embeddings, metadatas, ids, batch_size=5461):
    total = len(documents)
    for i in range(0, total, batch_size):
        end = i + batch_size
        collection.add(
            documents=documents[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )

documents = documents
embeddings = get_embeddings(documents,dim=256)
metadatas = metadata
ids = [f"id{i}" for i in range(len(documents))]

batch_add_documents(vector.collection, documents, embeddings, metadatas, ids)

print('done')