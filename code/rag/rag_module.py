#pip install chromadb
#向量数据库
import chromadb
from chromadb.config import Settings
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from chromadb import Documents, EmbeddingFunction, Embeddings
import os
import json
from utils import load_config
config = load_config()
EMD_MODEL = config['EMD_MODEL']
model = SentenceTransformer(EMD_MODEL)

class MyVectorDBConnector:
    def __init__(self, path,collection_name):
        #chroma_client = chromadb.Client(Settings(allow_reset=True))
        chroma_client = chromadb.PersistentClient(path=path)

        # 为了演示，实际不需要每次 reset()
        #chroma_client.reset()

        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_fn = get_embeddings

    def add_documents(self, documents,metadata):
        '''向 collection 中添加文档与向量'''
        self.collection.add(
            documents=documents,  # 文档的原文
            embeddings=get_embeddings(documents),  # 每个文档的向量
            metadatas=metadata,#还可以加metadata用来筛选
            ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
        )

    def search_author(self, query, top_n, author):
        '''检索向量数据库'''
        results = self.collection.query(
            query_embeddings=get_embeddings([query]),
            n_results=top_n,
            where={"author":author}
        )
        return results

    def search_topic(self, query, top_n, topic):
        '''检索向量数据库'''
        results = self.collection.query(
            query_embeddings=get_embeddings([query]),
            n_results=top_n,
            where={"topic":topic}
        )
        return results
    def search(self, query, top_n):
        '''检索向量数据库'''
        results = self.collection.query(
            query_embeddings=get_embeddings([query]),
            n_results=top_n
        )
        return results
class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        self.model=SentenceTransformer('acge_text_embedding')
        self.embeddings = self.model.encode(Documents, normalize_embeddings=False)
        #压缩模型维度
        self.matryoshka_dim = 256
        self.embeddings = self.embeddings[..., :self.matryoshka_dim]  # Shrink the embedding dimensions
        self.embeddings = normalize(self.embeddings, norm="l2", axis=1)
        #reshape
        reshaped_array = self.embeddings.reshape(-1, self.matryoshka_dim)
        list_of_lists = reshaped_array.tolist()
        return list_of_lists

def get_embeddings(Documents,dim=256):
    model = sentence_model
    embeddings = model.encode(Documents, normalize_embeddings=False)
    #压缩模型维度
    matryoshka_dim = dim
    embeddings = embeddings[..., :matryoshka_dim]  # Shrink the embedding dimensions
    embeddings = normalize(embeddings, norm="l2", axis=1)
    #reshape
    reshaped_array = embeddings.reshape(-1, matryoshka_dim)
    list_of_lists = reshaped_array.tolist()
    return list_of_lists








