
def statistic_time_diff(data):
    # 计算相邻两行在 timestamp 上的差值（单位：秒）
    # 统计结果，时间间隔的中位数为74s
    time_diff = data['timestamp'].diff().fillna(0)

    # 统计相邻两点之间 timestamp 差值小于 300 秒的个数（不包括第一行）,即用户连续行为的时间间隔小于5分钟
    count = (time_diff[1:] < 300).sum()
    print(f"相邻两个点在 timestamp 维度上相差小于5分钟的点的个数为: {count}")
