# -*- coding: utf-8 -*-
"""
@author: 
"""


import argparse
import random
from script_nn_largest import main as main_nn_largest
# from script_nn_ltsf import main as main_nn_ltsf
# from script_1nn_largest import main as main_1nn_largest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_worker', type=int, default=32)
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()
    n_worker = args.n_worker

    exp_setup = []
    # data_names = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2',
    #               'Weather', 'Electricity', 'Traffic', ]
    # method_names = ['rpm_0001', 'rpm_0002', 'rpm_0001', 'rpm_0002',
    #                 'rpm_0003', 'rpm_0004', 'rpm_0005',]
    # for data_name, method_name in zip(data_names, method_names):
    #     for config_id in range(4):
    #         data_name_ = f'{data_name}_{config_id:04d}'
    #         exp_setup.append(
    #             [data_name_, method_name, main_nn_ltsf, ])

    data_names = [
        # 'sd_his_2019_agg_0000',
        'ca_his_2019_agg_0000',
        'gba_his_2019_agg_0000',
        'gla_his_2019_agg_0000',    
    ]
    # method_name = '1nn_0000'
    # for data_name in data_names:
    #     exp_setup.append(
    #         [data_name, method_name, main_1nn_largest, ])

    method_names = ['mole_0000' ]
    for data_name in data_names:
        for method_name in method_names:
            exp_setup.append(
                [data_name, method_name, main_nn_largest, ])

    # random.shuffle(exp_setup)
    for data_name, method_name, main_fun in exp_setup:
        print(data_name, method_name)
        main_fun(data_name, method_name, n_worker, args.gpu_id)


if __name__ == '__main__':
    main()

