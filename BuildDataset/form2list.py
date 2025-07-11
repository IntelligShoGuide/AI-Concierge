import pandas as pd
import os
from datetime import datetime

# 将当前工作目录切换为当前脚本文件所在的目录，确保后续的文件读写操作都是以该脚本所在目录为基准进行的。
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import utilise

def load_data():
    path = '../data/AliUserBehavior/UserBehavior.csv'

    data = pd.read_csv(path, header=None)
    print('已经加载数据')
    data.columns = ['user_id', 'item_id', 'category_id', 'behavior', 'timestamp']
    data = data.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    print('已经排序')
    return data

def form_sequence(data):
    user_sequences = []  # 存储 [item_id, behavior, 停留时长] 的列表

    for _, row in data.iterrows():
        # user_id = row['user_id']    # 用户ID, int
        item_ids = row['item_id']   # 商品ID, list
        behaviors = row['behavior'] # 用户行为, list
        timestamps = row['timestamp'] # 时间戳, list
        n = len(item_ids) 

        seq = []
        
        for i in range(n):
            # 判断是否为最后一个点
            if i < n - 1:
                dwell = timestamps[i+1] - timestamps[i]
                seq.append([item_ids[i], behaviors[i], dwell])
                # 如果下一个点与当前点的时间间隔大于等于300，则在此断开
                if dwell >= 300:
                    if len(seq) > 1: 
                        seq[-1][-1] = seq[-2][-1]
                    else:
                        seq[-1][-1] = 1
                    user_sequences.append([seq])
                    seq = []
            else:
                # 最后一个点
                if len(seq) > 1:    # seq里有点，则最后一个与上一个点间隔小于300，则将最后一个点与上一个点合并
                    dwell = seq[-1][-1]
                    seq.append([item_ids[i], behaviors[i], dwell])
                else:
                    seq.append([item_ids[i], behaviors[i], 1])
                user_sequences.append([seq])
            
    user_seq_df = pd.DataFrame(user_sequences, columns=['behavior_sequence'])
    print('已经完成序列化')
    return user_seq_df

data = load_data()

# 按照 user_id 分组，将除 user_id 外的所有列合并为列表
data = data.groupby('user_id').agg({
    'item_id': list,
    'category_id': list,
    'behavior': list,
    'timestamp': list
}).reset_index()

data = form_sequence(data)
data.to_pickle('../data/AliUserBehavior/behavior_sequence.pkl')
