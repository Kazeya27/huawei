import pandas as pd
from tqdm import tqdm
import numpy as np
from pdb import set_trace


def fill_missing_data(df, date_st, date_ed, time_interval):
    # 将date列转为datetime类型
    df['date'] = pd.to_datetime(df['date'])

    # 生成日期范围
    date_range = pd.date_range(start=date_st, end=date_ed, freq=f'{time_interval}s')

    # 按tid进行分组
    result_df = pd.DataFrame(columns=['tid', 'date', 'num_total'])

    def process_tid_group(tid_data):
        # 获取当前tid的完整日期序列
        full_data = pd.DataFrame({'date': date_range})

        # 合并当前tid数据和完整日期数据
        merged = pd.merge(full_data, tid_data[['date', 'num_total']], on='date', how='left', sort=False)

        # 对缺失的num_total填充0
        merged['num_total'] = merged['num_total'].fillna(0).astype(int)

        # 将tid列重复填充为当前tid
        merged['tid'] = tid_data['tid'].iloc[0]

        return merged

    # 按tid分组，并处理每个组
    result_df = pd.concat([process_tid_group(group) for _, group in df.groupby('tid')], ignore_index=True)

    return result_df


if __name__ == '__main__':
    df = pd.read_csv("./ss_20171226_temp_r11.csv")
    df_sorted = df.sort_values(by=['tid', 'date_dt', 'order_id'], ascending=True)
    df_grouped = df_sorted.groupby(['tid', 'date_dt', 'order_id'], as_index=False)['num_total'].sum()

    df = df_grouped.drop_duplicates()
    # 转换date_dt列为日期格式（确保日期正确处理）
    df['date_dt'] = pd.to_datetime(df['date_dt'], format='%Y%m%d')

    # 将order_id列处理为小时（0-23）
    df['order_id'] = df['order_id'].astype(int)

    # 创建一个新的日期列，格式为yyyy-mm-dd hh:MM:ss
    df['date'] = df.apply(lambda row: row['date_dt'].replace(hour=row['order_id'], minute=0, second=0), axis=1)

    # 确保date列的数据类型为datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop(columns=['date_dt', 'order_id'])
    date_st = '2020-01-01'
    date_ed = '2020-01-31'
    time_interval = 3600  # 以秒为单位

    # 调用函数进行补零操作
    set_trace()
    # df = fill_missing_data(df, date_st, date_ed, time_interval)
    df.to_csv('./wuhan.csv', index=False)


