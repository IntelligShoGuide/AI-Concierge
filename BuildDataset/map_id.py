import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 简单映射，返回商品id表
def simple_map():
    db_path = '../data/AmazonReviews/meta_Electronics.jsonl'
    db = pd.read_json(db_path, lines=True)
    print('已经加载数据库')
    
    # 删除 db 中 'description', 'features' 都为空的数据，与product_em.pkl 保持一致
    db = db[~((db['features'].apply(len) == 0) & (db['description'].apply(len) == 0))]

    db = db[['parent_asin', 'categories']]
    # 有一些categories为空，排在最后
    db = db.sort_values(by='categories', ascending=False).reset_index(drop=True)    
    print('得到映射表')
    return db['parent_asin'].tolist()

# 使用知识图谱映射
def knowledge_map():
    db_path = '../data/AmazonReviews/product_em.pkl'
    db = pd.read_pickle(db_path)

# 用户行为数据集中id映射
def user_map(product_id):
    # 数据库中商品的数量，数据库中一共有 1,261,420 件商品，而用户行为数据集中涉及 4,162,024 件商品
    n = len(product_id)

    beavior_path = '../data/AliUserBehavior/behavior_sequence.pkl'
    ds = pd.read_pickle(beavior_path)   # 只有一列，为[[item_id, behavior]...]
    print('已经加载用户行为数据')
    
    total = 0
    id_map = {}
    k = ds.keys().item()

    # 逐行读取 ds，映射商品id
    for _, row in ds.iterrows():
        for i in range(len(row)):
            id, _, _ = row[k][i]
            if id in id_map:
                row[k][i][0] = id_map[id]
            else:
                id_map[id] = product_id[total % n]
                row[k][i][0] = id_map[id]
                total += 1
    
    print(ds.iloc[:5])
    ds.to_pickle('../data/AliUserBehavior/behavior_sequence_map.pkl')
    print('已经保存映射后的数据集')

product_id = simple_map()
user_map(product_id)
