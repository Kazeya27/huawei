#! /bin/bash

python script_nn_largest.py --method_name rpm_0006 --gpu_id 0 --n_worker 8 --data_name ca_his_2019_agg_0000

python script_nn_largest.py --method_name rpm_0006 --gpu_id 0 --n_worker 8 --data_name gla_his_2019_agg_0000

python script_nn_largest.py --method_name rpm_0006 --gpu_id 0 --n_worker 8 --data_name gba_his_2019_agg_0000

python script_nn_largest.py --method_name rpm_0006 --gpu_id 0 --n_worker 8 --data_name sd_his_2019_agg_0000