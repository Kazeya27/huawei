import datetime
import os
import pickle

import numpy as np
from pdb import set_trace

import torch
from fastdtw import fastdtw
from tqdm import tqdm

from libcity.data.dataset import TrafficStatePointDataset
from libcity.data.utils import generate_dataloader
from libcity.utils import ensure_dir, StandardScaler, NoneScaler


class MixerDataset(TrafficStatePointDataset):
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.load_dtw_neighbors = self.config.get('load_dtw_neighbors', True)
        self.hidden = self.config.get('hidden', 64)
        self.rp_model = self.config.get('rp_model', 'Node2Vec')
        self.dtw_cache_path = '../../pool_mixer/libcity/cache/dataset_cache/dtw_' + self.dataset + '.npy'
        node_embedding_cache_path = f'../../pool_mixer/libcity/cache/dataset_cache/embedding_{self.rp_model}_{self.dataset}_{self.hidden}.npy'
        self.node_embedding = np.load(node_embedding_cache_path)  # [N, D]
        self.time_intervals = self.config.get('time_intervals', 300)
        self.timeslots = 24 * 60 * 60 // self.time_intervals
        self.num_neighbors = self.config.get('num_neighbors', 5)
        self.hop = self.config.get('hop', 3)
        self.total_neighbors = self.num_neighbors * self.hop
        super().__init__(config)

    def _load_cache_train_val_test(self):
        """
        加载之前缓存好的训练集、测试集、验证集

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        x_mean = cat_data['x_mean']
        x_std = cat_data['x_std']
        # self.neighbors_list = cat_data['neighbors_list']
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test, x_mean, x_std

    def _split_train_val_test(self, x, y):
        """
        划分训练集、测试集、验证集，并缓存数据集

        Args:
            x(np.ndarray): 输入数据 (num_samples, input_length, ..., feature_dim)
            y(np.ndarray): 输出数据 (num_samples, input_length, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x.shape[0]
        if self.shuffle:
            num_samples_season = num_samples // 4
            num_test = round(num_samples_season * test_rate)
            num_train = round(num_samples_season * self.train_rate)
            num_val = num_samples_season - num_test - num_train
            x_train = np.zeros((num_train * 4, self.input_window, x.shape[2], x.shape[3]), dtype=np.float32)
            y_train = np.zeros((num_train * 4, self.output_window, y.shape[2], y.shape[3]), dtype=np.float32)
            x_val = np.zeros((num_val * 4, self.input_window, x.shape[2], x.shape[3]), dtype=np.float32)
            y_val = np.zeros((num_val * 4, self.output_window, y.shape[2], y.shape[3]), dtype=np.float32)
            x_test = np.zeros(((num_test - self.input_window - self.output_window) * 4, self.input_window, x.shape[2], x.shape[3]), dtype=np.float32)
            y_test = np.zeros(((num_test - self.input_window - self.output_window) * 4, self.output_window, y.shape[2], y.shape[3]), dtype=np.float32)
            st = 0
            for i in range(4):
                x_train[i * num_train:(i + 1) * num_train] = x[st:st + num_train]
                y_train[i * num_train:(i + 1) * num_train] = y[st:st + num_train]
                st += num_train
                x_val[i * num_val:(i + 1) * num_val] = x[st:st + num_val]
                y_val[i * num_val:(i + 1) * num_val] = y[st:st + num_val]
                st += num_val
                x_test[i * (num_test - self.input_window - self.output_window):
                       (i + 1) * (num_test - self.input_window - self.output_window)] = \
                    x[st:st + (num_test - self.input_window - self.output_window)]
                y_test[i * (num_test - self.input_window - self.output_window):
                       (i + 1) * (num_test - self.input_window - self.output_window)] = \
                    y[st:st + (num_test - self.input_window - self.output_window)]

                st += num_test
            del x
            del y
        else:
            num_test = round(num_samples * test_rate)
            num_train = round(num_samples * self.train_rate)
            num_val = num_samples - num_test - num_train
            # train
            x_train, y_train = x[:num_train], y[:num_train]
            # val
            x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
            # test
            x_test, y_test = x[-num_test:], y[-num_test:]
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))

        return x_train, y_train, x_val, y_val, x_test, y_test

    def load_dtw_matrix(self, df):
        if os.path.exists(self.dtw_cache_path):
            self.dist_matrix = np.load(self.dtw_cache_path)
        else:
            self.dist_matrix = self.get_dtw_matrix(df)
            np.save(self.dtw_cache_path, self.dist_matrix)
        self._logger.info("Load DTW Matrix")

    def get_dtw_matrix(self, data):
        df = data[:, :, :self.output_dim]
        # print(df.shape)
        data_mean = np.mean([df[self.timeslots * i: self.timeslots * (i + 1)] for i in
                             range(df.shape[0] // self.timeslots)], axis=0)
        # print(data_mean.shape)
        dtw_distance = np.zeros((self.num_nodes, self.num_nodes))
        for i in tqdm(range(self.num_nodes)):
            for j in range(i, self.num_nodes):
                dtw_distance[i][j], _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6)
        for i in range(self.num_nodes):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        # mean = np.mean(dtw_distance)
        # std = np.std(dtw_distance)
        # dtw_distance = (dtw_distance - mean) / std
        # dtw_distance = np.exp(-dtw_distance ** 2 / self.sigma ** 2)
        return dtw_distance

    def get_n_hop_neighbors(self, data):
        T, N, C = data.shape
        self._logger.info("searching neighbors")
        # 获取每个节点的1-hop邻居
        top_k_neighbors = self.get_topk_neighbors(self.num_neighbors * 4)
        selected_neighbors = [set(top_k_neighbors[i, :self.num_neighbors]) for i in range(N)]
        hop_neighbors = top_k_neighbors.copy()[:, :self.num_neighbors]
        rtn_neighbors = np.zeros((self.num_nodes, self.num_neighbors * self.hop), dtype=int)
        rtn_neighbors[:, :self.num_neighbors] = hop_neighbors
        for hop in range(1, self.hop):
            new_hop_neighbors = np.zeros((self.num_nodes, self.num_neighbors), dtype=int)

            for i in range(N):
                # 获取当前节点i的1-hop邻居
                current_neighbors = hop_neighbors[i]

                # 存放候选的2-hop邻居及其相关性分数
                candidate_neighbors_with_score = []
                candidate_set = set([i])
                # 对于每个1-hop邻居，找到与该邻居最相关的邻居
                for neighbor in current_neighbors:
                    # 获取该邻居的top-k邻居
                    neighbors_of_neighbor = top_k_neighbors[neighbor]

                    # 把这些邻居以及与1-hop邻居的相关性作为候选
                    for nn in neighbors_of_neighbor:
                        if nn not in selected_neighbors[i] and nn not in candidate_set:
                            candidate_neighbors_with_score.append((nn, self.dist_matrix[neighbor, nn]))
                            candidate_set.add(nn)
                # 根据与1-hop邻居的相关性排序，并选择top-k
                assert len(candidate_neighbors_with_score) >= self.num_neighbors, f"num_neighbors too low: hop={hop}, node={i}, num={len(candidate_neighbors_with_score)}"
                sorted_candidates = sorted(candidate_neighbors_with_score, key=lambda x: x[1])[:self.num_neighbors]
                new_neighbors = [x[0] for x in sorted_candidates]
                new_hop_neighbors[i] = np.array(new_neighbors, dtype=int)

                # 更新已经选出的邻居集合
                selected_neighbors[i].update(new_neighbors)
            hop_neighbors = new_hop_neighbors
            rtn_neighbors[:, hop*self.num_neighbors:(hop+1)*self.num_neighbors] = hop_neighbors
        self.neighbors_list = rtn_neighbors
        self._logger.info(f"neighbors got, shape={self.neighbors_list.shape}")

    def get_topk_neighbors(self, k):
        """
        从 DTW 距离矩阵中提取每个节点的 topk 个最相关邻居。

        参数:
        k: int
            要提取的邻居数量（topk）。

        返回:
        topk_neighbors: numpy.ndarray, shape = [N, k]
            每个节点的topk个邻居的索引矩阵。
        """
        sorted_indices = np.argsort(self.dist_matrix, axis=1)[:, 1:k+1]
        return sorted_indices

    def _generate_input_data(self, df):
        """
        根据全局参数`input_window`和`output_window`切分输入，产生模型需要的张量输入，
        即使用过去`input_window`长度的时间序列去预测未来`output_window`长度的时间序列

        Args:
            df(np.ndarray): 数据数组，shape: (len_time, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x(np.ndarray): 模型输入数据，(epoch_size, input_length, ..., feature_dim) \n
                y(np.ndarray): 模型输出数据，(epoch_size, output_length, ..., feature_dim)
        """
        num_samples = df.shape[0]
        # 预测用的过去时间窗口长度 取决于self.input_window
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_window + 1, 1, 1),)))
        # 未来时间窗口长度 取决于self.output_window
        y_offsets = np.sort(np.arange(1, self.output_window + 1, 1))
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))
        samples = max_t - min_t
        x = np.zeros((samples, self.input_window, df.shape[1], df.shape[2]), dtype=np.float32)
        y = np.zeros((samples, self.output_window, df.shape[1], df.shape[2]), dtype=np.float32)
        for i, t in enumerate(range(min_t, max_t)):
            x[i] = df[t + x_offsets, ...]
            y[i] = df[t + y_offsets, ...]
        del df
        return x, y

    def _generate_data(self):
        """
        加载数据文件(.dyna/.grid/.od/.gridod)和外部数据(.ext)，且将二者融合，以X，y的形式返回

        Returns:
            tuple: tuple contains:
                x(np.ndarray): 模型输入数据，(num_samples, input_length, ..., feature_dim) \n
                y(np.ndarray): 模型输出数据，(num_samples, output_length, ..., feature_dim)
        """
        # 处理多数据文件问题
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:  # str
            data_files = [self.data_files].copy()
        # 加载外部数据
        if self.load_external and os.path.exists(self.data_path + self.ext_file + '.ext'):  # 外部数据集
            ext_data = self._load_ext()
        else:
            ext_data = None
        filename = data_files[0]
        df = self._load_dyna(filename)  # (len_time, ..., feature_dim)
        if self.load_external:
            df = self._add_external_information(df, ext_data)
        x, y = self._generate_input_data(df)
        del df
        self._logger.info("Dataset created")
        self._logger.info("x shape: " + str(x.shape) + ", y shape: " + str(y.shape))
        return x, y

    def _get_scalar(self, scaler_type, x_mean, x_std):
        if scaler_type == "standard":
            scaler = StandardScaler(mean=x_mean, std=x_std)
            self._logger.info('StandardScaler mean: ' + str(scaler.mean) + ', std: ' + str(scaler.std))
        elif scaler_type == "none":
            scaler = NoneScaler()
            self._logger.info('NoneScaler')
        else:
            raise ValueError('Scaler type error!')
        return scaler

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # 加载数据集
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test, x_mean, x_std = self._load_cache_train_val_test()
                self.scaler = self._get_scalar(self.scaler_type, x_mean.item(), x_std.item())
            else:
                x_train, y_train, x_val, y_val, x_test, y_test = self._generate_train_val_test()
                # 数据归一化
                x_mean = x_train[..., :self.output_dim].mean()
                x_std = x_train[..., :self.output_dim].std()
                self.scaler = self._get_scalar(self.scaler_type, x_mean, x_std)
                x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
                y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
                x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
                y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
                x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
                y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
                if self.cache_dataset:
                    ensure_dir(self.cache_file_folder)
                    np.savez_compressed(
                        self.cache_file_name,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        x_val=x_val,
                        y_val=y_val,
                        # neighbors_list=self.neighbors_list,
                        x_mean=x_mean,
                        x_std=x_std,
                    )
                    self._logger.info('Saved at ' + self.cache_file_name)
        self.feature_dim = x_train.shape[-1]
        self.ext_dim = self.feature_dim - self.output_dim
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        # 转Dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "ext_dim": self.ext_dim, "timeslots": self.timeslots,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim, "node_embedding": self.node_embedding,
                "output_dim": self.output_dim, "num_batches": self.num_batches, "total_neighbors": self.total_neighbors}

    def _add_external_information(self, df, ext_data=None):
        """
        增加外部信息（一周中的星期几/day of week，一天中的某个时刻/time of day，外部数据）

        Args:
            df(np.ndarray): 交通状态数据多维数组, (len_time, num_nodes, feature_dim)
            ext_data(np.ndarray): 外部数据

        Returns:
            np.ndarray: 融合后的外部数据和交通状态数据, (len_time, num_nodes, feature_dim_plus)
        """
        num_samples, num_nodes, feature_dim = df.shape
        is_time_nan = np.isnan(self.timesolts).any()
        data_list = [df]
        if self.add_time_in_day and not is_time_nan:
            time_ind = (self.timesolts - self.timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
            del time_in_day
        if self.add_day_in_week and not is_time_nan:
            dayofweek = []
            for day in self.timesolts.astype("datetime64[D]"):
                dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
            day_in_week = np.tile(dayofweek, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(day_in_week)
            del day_in_week
        # 外部数据集
        if ext_data is not None:
            if not is_time_nan:
                indexs = []
                for ts in self.timesolts:
                    ts_index = self.idx_of_ext_timesolts[ts]
                    indexs.append(ts_index)
                select_data = ext_data[indexs]  # T * ext_dim 选出所需要的时间步的数据
                for i in range(select_data.shape[1]):
                    data_ind = select_data[:, i]
                    data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                    data_list.append(data_ind)
            else:  # 没有给出具体的时间戳，只有外部数据跟原数据等长才能默认对接到一起
                if ext_data.shape[0] == df.shape[0]:
                    select_data = ext_data  # T * ext_dim
                    for i in range(select_data.shape[1]):
                        data_ind = select_data[:, i]
                        data_ind = np.tile(data_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                        data_list.append(data_ind)
        data = np.concatenate(data_list, axis=-1, dtype=np.float32)
        del data_list
        return data


if __name__ == '__main__':
    configs = dict()
    configs['dataset'] = 'PEMSD4-flow'