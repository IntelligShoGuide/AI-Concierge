import pandas as pd
import os
import torch.multiprocessing as mp
from sentence_transformers import SentenceTransformer
import torch
import gc

os.chdir('./BulidDataBase')
import utilise

def get_embedding(model_name):
    # 加载BGE-large-en-v1.5模型，Embedding维度: 768
    embedding_model = SentenceTransformer(model_name)
    embedding_model = embedding_model.half()  # 转为半精度float16，减少显存占用

    # 设置多GPU并行计算
    gpu_ids = [1, 2]  # 可用的GPU设备列表
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")

    # 如果使用多GPU，设置数据并行
    if torch.cuda.device_count() > 1:
        embedding_model = torch.nn.DataParallel(embedding_model, device_ids=gpu_ids)
        print(f"使用多GPU并行计算，设备: {gpu_ids}")
    else:
        print(f"使用单GPU: {device}")
    embedding_model.to(device)

    print(f"模型已加载到设备: {device}")

    # 如果模型被DataParallel包装
    if hasattr(embedding_model, 'module'):
        actual_model = embedding_model.module
    else:
        actual_model = embedding_model
    return actual_model

def data_process(review_path, meta_path):
    # 加载数据
    review_data = utilise.load_jsonl(review_path)
    print(f"review_data 加载完成，行数: {len(review_data)}")
    r_df = pd.DataFrame(review_data)

    del review_data
    gc.collect()
    
    meta_data = utilise.load_jsonl(meta_path)
    print(f"meta_data 加载完成，行数: {len(meta_data)}")
    m_df = pd.DataFrame(meta_data)

    del meta_data
    gc.collect()

    r_df = r_df[['parent_asin', 'title', 'text']]   # 只保留需要的列
    m_df = m_df[['parent_asin', 'average_rating', 'rating_number', 'details', 'categories', 'price', 'description', 'features']]

    # 将 review 数据按 parent_asin 分组，处理重复的行，并合并 title、text 列，将 helpful_vote 列转换为列表
    r_df = r_df.groupby('parent_asin').agg({
        'title': lambda x: ', '.join(x),
        'text': lambda x: ', '.join(x),
        # 'helpful_vote': list
    })
    r_df = r_df.rename(columns={'title': 'review_title'})   # 为了避免与m_df中的title列名冲突，更改名字
    print(f"review_data 处理完成，行数: {len(r_df)}")

    # 删除 m_df 中 'description', 'features' 都为空的数据
    m_df = m_df[~((m_df['features'].apply(len) == 0) & (m_df['description'].apply(len) == 0))]
    print(f"meta_data 处理完成，行数: {len(m_df)}")

    # 将 m_df 和 r_df 合并，没有评论的商品的合并后的缺失列用 NaN 填充
    m_df = m_df.merge(r_df, on='parent_asin', how='left')
    m_df = m_df.set_index('parent_asin')
    print(f"db 处理完成，行数: {len(m_df)}")
    return m_df

def embedding_process(db: pd.DataFrame, embedding_model):
    db = db.apply(lambda row: [str(x) for x in row.tolist()], axis=1)  # 先将元素转换成str再换成list形式
    # 为db增加一列 'embedding'，初始值为None
    for index, row in db.items():
        # 只对不为 '[]' 或 'nan' 的元素进行 embedding，并统计参与 embedding 的元素个数
        valid_elements = [x for x in row if x != '[]' and x.lower() != 'nan']
        embedding_count = len(valid_elements)
        if embedding_count > 0:
            embedding = embedding_model.encode(valid_elements)
            embedding = embedding.mean(axis=0)
        else:
            embedding = None
        db.loc[index] = embedding
    db = db.to_frame(name='embedding')
    return db


model_name = "/home/users/wzr/project/predict/LLM/embedding/bge-base-en-v1.5"
review_path = '../data/AmazonReviews/Electronics.jsonl'
meta_path = '../data/AmazonReviews/meta_Electronics.jsonl'

embedding_model = get_embedding(model_name)
product_db = data_process(review_path, meta_path)
product_em = embedding_process(product_db, embedding_model)

del product_db
gc.collect()

# 保存最终的 product_em, 格式为 (parent_asin, embedding)
product_em.to_pickle('../data/AmazonReviews/product_em.pkl')

