import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pdb import set_trace
from tqdm import tqdm

from torch.utils.data import DataLoader
from dataset.dataset import TimeSeries


def get_data(config):
    file_path = config['data']['data_path']

    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 假设date_st, date_ed 和 time_interval是已知的
    date_st = config['data']['st']
    date_ed = config['data']['ed']
    time_interval = config['data']['interval']

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
    return tid_data


def get_time_series_loader(partition, config, n_worker):
    seq_len = config.get("seq_len", 12)
    pred_len = config.get("pred_len", 12)
    frac_test = config.get("frac_test", 0.2)
    frac_valid = config.get("frac_valid", 0.1)
    frac_train = config.get("frac_train", 0.7)

    batch_size = config.get("batch_size", 32)

    shuffle_flag = False
    drop_last_flag = False
    if partition == 'train':
        shuffle_flag = True
        drop_last_flag = True

    data = get_data(config)
    dataset = TimeSeries(
        data, seq_len, pred_len, partition,
        frac_test=frac_test, frac_valid=frac_valid, frac_train=frac_train)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=n_worker,
        drop_last=drop_last_flag)
    feature = dataset.get_feature()
    feature["num_batches"] = len(data_loader)

    return data_loader, dataset, feature