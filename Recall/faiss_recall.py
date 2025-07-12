import os
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
import sqlite3

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 将 meta_Electronics.jsonl 封装为 SQLite 数据库
def jsonl_to_sqlite(sqlite_path):
    jsonl_path = '../data/AmazonReviews/meta_Electronics.jsonl'
    # 读取jsonl文件
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(eval(line.strip()))
    df = pd.DataFrame(data)
    # 写入sqlite数据库
    conn = sqlite3.connect(sqlite_path)
    df.to_sql('meta_electronics', conn, if_exists='replace', index=False)
    conn.close()
    print(f"已将 {jsonl_path} 封装为 {sqlite_path}")

sqlite_path = '../data/AmazonReviews/meta_Electronics.db'
if not os.path.exists(sqlite_path):
    jsonl_to_sqlite(sqlite_path)

def faiss_recall(db):
    print(db.keys())
    embeddings = np.vstack(db.iloc[:, 0].values)  # shape: (num_items, embedding_dim)

    # 构建faiss索引
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2距离
    index.add(embeddings)  # 加入所有向量
    return index

def query_embedding(query):
    model_name = "/home/users/wzr/project/predict/LLM/embedding/bge-base-en-v1.5"
    model = SentenceTransformer(model_name)  # 你可以根据实际情况选择合适的embedding模型

    query_embedding = model.encode(query)

    return query_embedding

def find_item(parent_asins, sqlite_path):
    conn = sqlite3.connect(sqlite_path)    # 连接到sqlite数据库
    meta_df = pd.read_sql_query("SELECT * FROM meta_electronics", conn)    # 读取meta_electronics表
    conn.close()

    # 查找parent_asin在parent_asins中的行
    matched_rows = meta_df[meta_df['parent_asin'].isin(parent_asins)]
    print("对应的商品信息如下：")
    print(matched_rows)

db_path = "../data/AmazonReviews/product_em.pkl"
db = pd.read_pickle(db_path)

index = faiss_recall(db)

query = 'I want a thin and light laptop that can play big games.'
query = query_embedding(query)

k = 5  # 返回前5个最相似的结果
D, I = index.search(np.array([query]), k)  # D为距离，I为索引

parent_asins = db.index[I[0]].tolist()
print("最相似的5个商品parent_asin：", parent_asins)
find_item(parent_asins)
