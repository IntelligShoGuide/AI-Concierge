import pandas as pd
import os
from datetime import datetime

# 将当前工作目录切换为当前脚本文件所在的目录，确保后续的文件读写操作都是以该脚本所在目录为基准进行的。
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_data():
    path = '../data/AliUserBehavior/UserBehavior.csv'

    data = pd.read_csv(path, header=None)
    data.columns = ['user_id', 'item_id', 'category_id', 'behavior', 'timestamp']
    data = data.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)

    return data

def statistic_time_diff(data):
    # 计算相邻两行在 timestamp 上的差值（单位：秒）
    # 统计结果，时间间隔的中位数为74s
    time_diff = data['timestamp'].diff().fillna(0)

    # 统计相邻两点之间 timestamp 差值小于 300 秒的个数（不包括第一行）,即用户连续行为的时间间隔小于5分钟
    count = (time_diff[1:] < 300).sum()
    print(f"相邻两个点在 timestamp 维度上相差小于5分钟的点的个数为: {count}")

data = load_data()

# 按照 user_id 分组，将除 user_id 外的所有列合并为列表
data = data.groupby('user_id').agg({
    'item_id': list,
    'category_id': list,
    'behavior': list,
    'timestamp': list
}).reset_index()



# 还没写好 ------------------~~~~~~~~~~~~~~~

user_sequences = []  # 存储 [user_id, [item_id, behavior, 停留时长]] 的嵌套列表
flat_records = []    # 存储 [item_id, behavior, 停留时长] 的列表

for idx, row in data.iterrows():
    user_id = row['user_id']    # 用户ID, int
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
                user_sequences.append([user_id, seq])
                seq = []
        else:
            # 最后一个点
            dewell
            seq.append([item_ids[i], behaviors[i], dwell])
            if seq:
                user_sequences.append([user_id, seq])
                for rec in seq:
                    flat_records.append(rec)
            seq = []

# 转为DataFrame保存
user_seq_df = pd.DataFrame(user_sequences, columns=['user_id', 'sequence'])
flat_df = pd.DataFrame(flat_records, columns=['item_id', 'behavior', 'dwell_time'])

# 保存为csv文件
user_seq_df.to_csv('../data/AliUserBehavior/user_sequence_list.csv', index=False, encoding='utf-8-sig')
flat_df.to_csv('../data/AliUserBehavior/item_behavior_dwell.csv', index=False, encoding='utf-8-sig')

print("用户行为序列和扁平化表格已保存。")



