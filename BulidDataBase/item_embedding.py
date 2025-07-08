import pandas as pd
import os

os.chdir('./BulidDataBase')
import utilise

        
# 加载数据
review_path = '../data/AmazonReviews/Electronics_200.jsonl'
meta_path = '../data/AmazonReviews/meta_Electronics_200.jsonl'

review_data = utilise.load_jsonl(review_path)
meta_data = utilise.load_jsonl(meta_path)

# 转换为DataFrame以便查看
r_df = pd.DataFrame(review_data)
m_df = pd.DataFrame(meta_data)


