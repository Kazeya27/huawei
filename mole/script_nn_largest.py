# -*- coding: utf-8 -*-
"""
@author: 
"""


import os
import copy
import argparse
import torch
from engine.util import check_dir
from engine.util import parse_config
from engine.data_io import get_seq_dim
from engine.data_io import get_largest_loader as get_loader
from engine.model import get_model
from engine.train_test import nn_train
from engine.train_test import nn_test
from engine.train_test import largest_mse_loss as loss_fun
from engine.train_test import get_largest_metric as get_metric


def main_wrapper():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name')
    parser.add_argument('--method_name')
    parser.add_argument('--n_worker', type=int)
    parser.add_argument('--gpu_id', type=int)

    args = parser.parse_args()
    data_name = args.data_name
    method_name = args.method_name
    n_worker = args.n_worker
    main(data_name, method_name, n_worker, args.gpu_id)


def main(data_name, method_name, n_worker, gpu_id):
    model_dir = os.path.join(
        '.', 'model', data_name)
    result_dir = os.path.join(
        '.', 'result', data_name)
    check_dir(model_dir)
    check_dir(result_dir)

    data_config = os.path.join(
        '.', 'config', f'{data_name}.config')
    data_config = parse_config(data_config, verbose=True)

    method_config = os.path.join(
        '.', 'config', f'{method_name}.config')
    method_config = parse_config(method_config, verbose=True)

    config = copy.deepcopy(method_config)
    config['data'] = data_config['data']
    config['data']['seq_len'] = config['model']['seq_len']
    config['model']['pred_len'] = config['data']['pred_len']
    config['model']['seq_dim'] = get_seq_dim(config)
    config['model']['num_nodes'] = config['data']['num_nodes']
    feature_variant = int(config['data']['feature_variant'])
    seq_len = int(config['data']['seq_len'])
    if feature_variant == 0:
        feat_dim = 0
    elif feature_variant == 1:
        feat_dim = seq_len * 2
    elif feature_variant == 2:
        feat_dim = 10
    config['model']['feat_dim'] = feat_dim

    if torch.cuda.is_available():
        device = 'cuda:' + str(gpu_id)
    else:
        device = 'cpu'
    print(device)

    fmt_str = '{0:04d}'
    model_path = os.path.join(
        model_dir, f'{method_name}_{fmt_str}.pt')
    result_path = os.path.join(
        result_dir, f'{method_name}_{fmt_str}.joblib')

    is_inverse = True
    model = get_model(config)
    # path = model_path.format(9999)
    # model_pkl = torch.load(path, map_location='cpu')
    # model.load_state_dict(model_pkl['model_state_dict'])
    # model = model.to(device)
    nn_train(model, model_path, result_path, config,
             get_loader, loss_fun, get_metric, is_inverse,
             device, n_worker)
    # loader_test, dataset_test = get_loader(
    #     'test', config, n_worker)
    # metric_test = nn_test(
    #         loader_test, dataset_test, model,
    #         get_metric, device, is_inverse, False)

if __name__ == '__main__':
    main_wrapper()

