# az.py
# !/usr/bin/env python
# encoding: utf-8
"""
@version: v2.0
@author: zhenglinghan
@contact: 422807471@qq.com
@software: PyCharm
@file: az.py
@time: 2020/07/23 07:00
"""

import pandas as pd
import numpy as np
import gc
import warnings
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('max_colwidth', 200)
pd.set_option('display.width', 5000)

# 路径
path_train_0 = '../raw_data/train_preliminary/'  # 初赛
path_train_1 = '../raw_data/train_semi_final/'  # 复赛
path_test = '../raw_data/test/'
data_path = '../cached_data/'


def reduce_mem_usage(df, features, verbose=True):
    '''

    :param df: dataframe to reduce memory useage
    :param features: cols to reduce memory useage
    :param verbose: if print effect
    :return:
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in tqdm(features):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def occupy_nan(datalog):
    '''
    process nan
    :param datalog: datalog to fill nan
    :return: processed datalog
    '''
    for var in ['product_id', 'industry']:
        datalog[var] = datalog[var].fillna(0)  # nan value as 0
    datalog = datalog.sort_values(['user_id', 'time', 'creative_id',
                                   'ad_id']).reset_index(drop=True)
    return datalog


def clip_unknow(datalog):
    '''
    train test word procoess
    :param datalog: datalog to clip low frequency>1 words
    :return: processed datalog
    '''
    for var in tqdm([
            'creative_id', 'ad_id', 'product_id', 'product_category',
            'advertiser_id', 'industry'
    ]):
        datalog[var] = datalog[var].astype(int)
        setcom = list(
            set(datalog.query('age == age')[var])
            & (set(datalog.query('age != age')[var])))
        dfcom = pd.DataFrame(setcom, columns=[var])
        dfcom['droped'] = 0
        datalog = datalog.merge(dfcom, on=var, how='left')
        vmax = -1  # low frequency words as -1
        datalog.loc[datalog['droped'].isna(), var] = vmax
        print(var, datalog[var].nunique())
        del datalog['droped']
    return datalog


def get_patial_data(row):
    '''

    :param row: row of datalog to copy
    :return: datalog copy by click_times
    '''
    npr = np.array(datalog.iloc[row, :]).reshape(1, -1)
    click_timesi = npr[0, 3]
    npr = npr.repeat(click_timesi, axis=0)
    return npr


if __name__ == "__main__":
    # ##################################################################################################################
    # csv to hdf
    # stage1
    trainad_0 = pd.read_csv(path_train_0 + 'ad.csv')
    trainad_0 = trainad_0.replace(r'\N', np.nan)
    trainlog_0 = pd.read_csv(path_train_0 + 'click_log.csv')
    datatrainlabel_0 = pd.read_csv(path_train_0 + 'user.csv')
    print('stage1 finish')
    # stage2
    trainad_1 = pd.read_csv(path_train_1 + 'ad.csv')
    trainad_1 = trainad_1.replace(r'\N', np.nan)
    trainlog_1 = pd.read_csv(path_train_1 + 'click_log.csv')
    datatrainlabel_1 = pd.read_csv(path_train_1 + 'user.csv')
    print('stage2 finish')
    trainad = pd.concat([trainad_0, trainad_1])
    trainlog = pd.concat([trainlog_0, trainlog_1])
    datatrainlabel = pd.concat([datatrainlabel_0, datatrainlabel_1])
    del datatrainlabel_0, datatrainlabel_1
    gc.collect()
    # test
    testad = pd.read_csv(path_test + 'ad.csv')
    testad = testad.replace(r'\N', np.nan)
    testlog = pd.read_csv(path_test + 'click_log.csv')
    datatestlabel = pd.DataFrame(list(testlog['user_id'].unique()),
                                 columns=['user_id'])
    print('testfinish')
    datalabel = pd.concat([datatrainlabel, datatestlabel
                           ]).sort_values('user_id').reset_index(drop=True)
    del datatrainlabel, datatestlabel
    gc.collect()
    dataad = pd.concat([trainad,
                        testad]).drop_duplicates().reset_index(drop=True)
    del trainad, testad
    gc.collect()
    datalog = pd.concat([trainlog,
                         testlog]).drop_duplicates().reset_index(drop=True)
    print(datalog.shape)
    del trainlog, testlog
    gc.collect()
    datalog = datalog.merge(dataad, on='creative_id', how='left')
    print(datalog.shape)
    del dataad
    gc.collect()
    datalog = datalog.drop_duplicates().reset_index(drop=True)
    datalog = reduce_mem_usage(datalog, datalog.columns)
    datalog.to_hdf(data_path + 'datalog_original_stage2.h5',
                   'datalog',
                   mode='w')
    datalabel = reduce_mem_usage(datalabel, datalabel.columns)
    datalabel.to_hdf(data_path + 'datalabel_original_stage2.h5',
                     'datalabel',
                     mode='w')
    print('csv to hdf finished!')
    # ##################################################################################################################
    # preprocess before pretrain……
    datalog = pd.read_hdf(data_path + 'datalog_original_stage2.h5')
    datalabel = pd.read_hdf(data_path + 'datalabel_original_stage2.h5')
    datalog = datalog.merge(datalabel, on='user_id', how='left')
    datalog = occupy_nan(datalog)
    print(datalog.shape)
    print('occupy_nan finished!')
    gc.collect()
    datalog = clip_unknow(datalog)
    # check if still exist unique words
    for var in tqdm([
            'creative_id', 'ad_id', 'product_id', 'product_category',
            'advertiser_id', 'industry'
    ]):
        df_var = Counter(datalog[var].values)
        df_var_counts = pd.DataFrame(
            sorted(df_var.items(), key=lambda x: x[1], reverse=True))
        df_var_counts.columns = [var, 'nums']
        print(var, 'still unique shape:',
              df_var_counts.query('nums ==1').shape)
    print('clip_unknow finished!')
    gc.collect()
    datalog.to_hdf(data_path + 'data_log_p2_stage2.h5', 'datalog', mode='w')
    print('datalog preprocess finished!')
    # ##################################################################################################################
    # click_times increase datalog
    # create another datalog file
    # used in w2v pretrain
    datalog = pd.read_hdf(data_path + 'data_log_p2_stage2.h5')
    if 'age' in datalog.columns:
        del datalog['age']
    if 'gender' in datalog.columns:
        del datalog['gender']
    gc.collect()

    for var in tqdm([
            'click_times', 'creative_id', 'ad_id', 'product_id',
            'product_category', 'advertiser_id', 'industry'
    ]):
        if var == 'click_times':
            datalog[var] = datalog[var].fillna(0)
            datalog.loc[datalog[var] > 4, var] = 4  # clip
    # duplicate click_times
    # multi process to get the click_times increased datalog
    # at less 24 cores in CPU
    processnum = 24  # muliprocess
    gc.collect()
    rowlist = [i for i in range(datalog.shape[0])]
    df_append = []
    with Pool(processnum) as p:
        df_append.append(p.map(get_patial_data, rowlist))
    df_append = np.vstack(df_append)
    df_append = pd.DataFrame(df_append,
                             columns=[
                                 'time', 'user_id', 'creative_id',
                                 'click_times', 'ad_id', 'product_id',
                                 'product_category', 'advertiser_id',
                                 'industry'
                             ])
    df_append.sort_values(['user_id', 'time', 'creative_id',
                           'ad_id']).reset_index(drop=True)
    # save new datalog
    if 'click_times' in df_append.columns:
        del df_append['click_times']
    gc.collect()
    del datalog
    gc.collect()
    df_append.to_hdf(data_path + 'datalog_p2_splits.h5', 'datalog', mode='w')
    print('click_times increased datalog finished!')
