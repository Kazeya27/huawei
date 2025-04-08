import os
import numpy as np
import pandas as pd
from pdb import set_trace


def process_dataset(dataset_dir, output_dir, dataset_names, suffix, is_agg):
    subset_names = ['ca', 'gba', 'gla', 'sd', ]
    subset_districts = [None, [4, ], [7, 8, 12, ], [11, ], ]
    direction_map = {'E': 0, 'W': 1, 'S': 2, 'N': 3, }

    meta_path = os.path.join(
        dataset_dir, 'ca_meta.csv')

    for subset_name, subset_district in zip(subset_names, subset_districts):
        output_path = os.path.join(
            output_dir, f'{subset_name}_his_{suffix}.npz')
        if os.path.isfile(output_path):
            npz = np.load(output_path)
            data = npz['data']
            print(output_path, data.shape)
            continue

        subset_id = []
        lat = []
        lng = []
        direction = []
        with open(meta_path, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                line = line.split(',')
                if line[0] == 'ID':
                    continue

                district = int(line[3])
                if subset_district is None or district in subset_district:
                    subset_id.append(line[0])
                    lat.append(float(line[1]))
                    lng.append(float(line[2]))
                    direction.append(direction_map[line[8]])

        df = pd.DataFrame()
        tod = []
        dow = []
        dom = []
        moy = []

        for dataset_name in dataset_names:
            dataset_path = os.path.join(dataset_dir, dataset_name)
            df_tmp = pd.read_hdf(dataset_path)
            if subset_district is not None:
                df_tmp = df_tmp[subset_id]
            if is_agg:
                df_tmp = df_tmp.resample('15T').mean().round(0)
            df_tmp = df_tmp.fillna(0)
            tod_ = (df_tmp.index.values - df_tmp.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
            dow_ = df_tmp.index.dayofweek
            dom_ = df_tmp.index.day - 1
            moy_ = df_tmp.index.month - 1
            # set_trace()


            tod.append(tod_)
            dow.append(dow_)
            dom.append(dom_)
            moy.append(moy_)

            df = pd.concat([df, df_tmp, ], ignore_index=True)

        data = df.values

        tod = np.concatenate(tod, axis=0)
        dow = np.concatenate(dow, axis=0)
        dom = np.concatenate(dom, axis=0)
        moy = np.concatenate(moy, axis=0)

        lat = np.array(lat)
        lng = np.array(lng)
        direction = np.array(direction)
        print(output_path, data.shape)
        np.savez_compressed(
            output_path, data=data, tod=tod, dow=dow, dom=dom, moy=moy, lat=lat, lng=lng,
            direction=direction)


# %%
dataset_dir = '../LargeST/data/ca'
# This folder needs to contain "ca_his_raw_2019.h5" and "ca_meta.csv".
output_dir = './dataset/ca/'

dataset_names = [
    'ca_his_raw_2019.h5',
]

suffix = '2019_agg'
process_dataset(dataset_dir, output_dir, dataset_names, suffix, True)
