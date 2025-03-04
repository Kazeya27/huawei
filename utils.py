import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pdb import set_trace
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from torch.utils.data import DataLoader
from dataset.dataset import TimeSeries
import matplotlib.pyplot as plt


# 画图
def plot_data(data, idx, date_range=None):
    if date_range is not None:
        data = data[:, date_range[0]:date_range[1]]
    plt.figure(figsize=(20, 5))
    plt.plot(data[idx, :])
    plt.show()


def data_impute(data):
    data_T = data.T
    df = pd.DataFrame(data_T)

    # --- 3. 低缺失率变量的插值 ---
    # 设定一个缺失率阈值，这里设为30%
    low_threshold = 0.3
    # 根据缺失率将变量分为低缺失率和高缺失率
    low_missing_cols = [col for col in df.columns if df[col].isna().mean() < low_threshold]
    # high_missing_cols = [col for col in df.columns if df[col].isna().mean() >= low_threshold]

    # 对低缺失率变量使用线性插值，limit_direction='both'可以保证两端也被填充
    df[low_missing_cols] = df[low_missing_cols].interpolate(method='linear', limit_direction='both')

    # --- 4. 高缺失率变量的多变量插补（MICE） ---
    # 利用所有变量的信息，对整个 DataFrame 进行多变量插补
    imputer = IterativeImputer(random_state=0, max_iter=10)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # --- 5. 还原数据形状 ---
    # 转置回原始形状 [n, t]
    data_imputed = df_imputed.T.values
    return data_imputed

def get_data(config):
    file_path = config.get("data_path")

    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 假设date_st, date_ed 和 time_interval是已知的
    date_st = config.get("st")
    date_ed = config.get("ed")
    time_interval = config.get("interval")

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
    tid_data = data_impute(tid_data)
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


if __name__ == '__main__':
    config = parse_config_file("tm")
    data = get_data(config)
    set_trace()