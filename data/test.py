import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pdb import set_trace
from tqdm import tqdm

# 假设文件名为 "a.csv"，并包含tid, date, num_total列
file_path = 'wuhan.csv'

# 读取CSV文件
data = pd.read_csv(file_path)

# 假设date_st, date_ed 和 time_interval是已知的
date_st = "2020-01-01 00:00:00"
date_ed = "2020-01-31 23:00:00"
time_interval = 3600  # 每小时一次

# 将date列转换为datetime格式
data['date'] = pd.to_datetime(data['date'])

# 生成完整的时间范围
start_date = datetime.strptime(date_st, "%Y-%m-%d %H:%M:%S")
end_date = datetime.strptime(date_ed, "%Y-%m-%d %H:%M:%S")
time_range = pd.date_range(start=start_date, end=end_date, freq=f"{time_interval}s")

# 获取所有tid的唯一值
tids = data['tid'].unique()
n = len(tids)  # 栅格数量

# 将时间范围转换为时间戳数组
time_range_timestamps = time_range.values.astype(np.datetime64)

# 创建tid和时间戳的索引
data['timestamp'] = data['date'].values.astype(np.datetime64)

# 为每个tid生成对应的时间序列数据
tid_data = np.zeros((n, len(time_range)), dtype=np.float32)
# 遍历每个tid
for i, tid in tqdm(enumerate(tids)):
    # 获取该tid对应的所有时间戳
    tid_subset = data[data['tid'] == tid]

    # 获取该tid所有时间戳的索引位置
    time_indices = np.searchsorted(time_range_timestamps, tid_subset['timestamp'].values)

    # 填充对应位置的num_total值
    tid_data[i, time_indices] = tid_subset['num_total'].values
set_trace()
# 将数据转换为[tid, time, 1]的形状
tid_data = tid_data[..., np.newaxis]

