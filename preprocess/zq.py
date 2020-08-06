import os
import pandas as pd
import numpy as np
import gc
from ..txbase import Cache
from txbase.data_utils import reduce_mem_usage
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


def get_action_lst(datalog, col):
    datalog[col] = datalog[col].astype(str)
    temp_list = datalog.groupby('user_id')[col].agg(
        lambda x: ' '.join(list(x))).rename(f'{col}_list').reset_index()
    return temp_list


if __name__ == "__main__":
    # 转h5
    path_train = '../raw_data/train_preliminary/'
    path_train_semi = "../raw_data/train_semi_final/"
    path_test = '../raw_data/test/'

    datatrain = pd.read_csv(path_train + 'ad.csv')
    datatrain_semi = pd.read_csv(path_train_semi + 'ad.csv')

    datatrainlog = pd.read_csv(path_train + 'click_log.csv')
    datatrainlog_semi = pd.read_csv(path_train_semi + 'click_log.csv')

    datalabel = pd.read_csv(path_train + 'user.csv')
    datalabel_semi = pd.read_csv(path_train_semi + 'user.csv')

    print(datatrain.shape, datatrain_semi.shape)
    datatrain_all = pd.concat([datatrain, datatrain_semi], ignore_index=True)
    print(datatrain_all.shape)
    datatrain_all = datatrain_all.drop_duplicates().reset_index(drop=True)
    print(datatrain_all.shape)

    datatrain_all = datatrain_all.replace(r'\N', np.nan)
    datatrain_all['product_id'] = datatrain_all['product_id'].astype(float)
    datatrain_all['industry'] = datatrain_all['industry'].astype(float)
    datatrain_all = reduce_mem_usage(datatrain_all, verbose=True)

    print(datatrainlog.shape, datatrainlog_semi.shape)
    datatrainlog_all = pd.concat([datatrainlog, datatrainlog_semi],
                                 ignore_index=True)
    print(datatrainlog_all.shape)
    datatrainlog_all = datatrainlog_all.drop_duplicates().reset_index(
        drop=True)
    print(datatrainlog_all.shape)

    print(datalabel.shape, datalabel_semi.shape)
    datalabel_all = pd.concat([datalabel, datalabel_semi], ignore_index=True)
    print(datalabel_all.shape)
    datalabel_all = datalabel_all.drop_duplicates().reset_index(drop=True)
    print(datalabel_all.shape)

    datatrainlog_all = reduce_mem_usage(datatrainlog_all, verbose=True)
    datalabel_all = reduce_mem_usage(datalabel_all, verbose=True)

    datatest = pd.read_csv(path_test + 'ad.csv')
    datatest = datatest.replace(r'\N', np.nan)
    datatest['product_id'] = datatest['product_id'].astype(float)
    datatest['industry'] = datatest['industry'].astype(float)
    datatest = reduce_mem_usage(datatest, verbose=True)

    datatestlog = pd.read_csv(path_test + 'click_log.csv')
    datatestlog = reduce_mem_usage(datatestlog, verbose=True)

    datatrain_all.to_hdf('../raw_data/data_train_semi.h5', key='ad', mode='a')
    datatrainlog_all.to_hdf('../raw_data/data_train_semi.h5',
                            key='click_log',
                            mode='a')
    datalabel_all.to_hdf('../raw_data/data_train_semi.h5',
                         key='user',
                         mode='a')

    datatest.to_hdf('../raw_data/data_test.h5', key='ad', mode='a')
    datatestlog.to_hdf('../raw_data/data_test.h5', key='click_log', mode='a')

    path_train = 'data/'
    path_test = 'data/'

    # ad
    datatrain = pd.read_hdf(path_train + 'data_train_semi.h5', key='ad')
    datatest = pd.read_hdf(path_test + 'data_test.h5', key='ad')

    # click_log
    datatrainlog = pd.read_hdf(path_train + 'data_train_semi.h5',
                               key='click_log')
    datatestlog = pd.read_hdf(path_test + 'data_test.h5', key='click_log')

    # user
    datatrainlabel = pd.read_hdf(path_train + 'data_train_semi.h5', key='user')

    print("datatrain.shape  ", datatrain.shape)
    print("datatest.shape  ", datatest.shape)
    print("datatrainlog.shape  ", datatrainlog.shape)
    print("datatestlog.shape  ", datatestlog.shape)
    print("datatrainlabel.shape  ", datatrainlabel.shape)

    # 测试集的uid，to predict
    datatestlabel = pd.DataFrame(list(datatestlog['user_id'].unique()),
                                 columns=['user_id'])
    datalabel = pd.concat([datatrainlabel, datatestlabel], ignore_index=True)
    del datatrainlabel, datatestlabel
    gc.collect()
    dataad = pd.concat([datatrain, datatest], ignore_index=True)
    dataad = dataad.drop_duplicates().reset_index(drop=True)
    print(dataad.shape)
    del datatrain, datatest
    gc.collect()

    # datalog
    datalog = pd.concat([datatrainlog, datatestlog], ignore_index=True)
    datalog = datalog.merge(dataad, on='creative_id', how='left')
    datalog = datalog.drop_duplicates()
    datalog = datalog.sort_values(['user_id', 'time', 'creative_id',
                                   'ad_id']).reset_index(drop=True)
    del datatrainlog, datatestlog, dataad
    gc.collect()
    datalog['product_id'] = datalog['product_id'].fillna(0)
    datalog['industry'] = datalog['industry'].fillna(0)
    datalog['product_id'] = datalog['product_id'].astype(np.int32)
    datalog['industry'] = datalog['industry'].astype(np.int16)

    datalog = datalog.merge(datalabel[['user_id', 'gender', 'age']],
                            on='user_id',
                            how='left')
    datalog['gender'] = datalog['gender'].fillna(-1).astype(np.int8)
    datalog['age'] = datalog['age'].fillna(-1).astype(np.int8)
    datalog = datalog.sort_values(['user_id', 'time', 'creative_id',
                                   'ad_id']).reset_index(drop=True)
    Cache.cache_data(datalog, nm_marker='datalog_sorted_fillna_with_0_all_int')

    print("@@@ 准备带time和click_times的 序列 2020 ...")
    # 添加序列特征
    sequence_features_raw = [
        'time', 'creative_id', 'click_times', 'ad_id', 'product_id',
        'product_category', 'advertiser_id', 'industry'
    ]  # 'click_times',
    sequence_features = list(map(lambda x: x + '_list', sequence_features_raw))

    # 重新初始化datalabel
    datalabel = datalabel[['user_id', 'gender', 'age']]
    for col in sequence_features_raw:
        temp_list = get_action_lst(datalog, col)
        datalabel = datalabel.merge(temp_list, on='user_id', how='left')

    del temp_list
    gc.collect()
    # 添加序列特征
    id_list_dict = {}
    MAX_LEN = 200
    for col in tqdm(sequence_features):
        id_list, key2index = get_sequence(datalabel,
                                          col,
                                          max_len=MAX_LEN,
                                          pre_post='pre')
        id_list_dict[col] = {'id_list': id_list, 'key2index': key2index}
    print(id_list_dict['creative_id_list']['id_list'])
    print(id_list_dict['industry_list']['id_list'])
    del datalabel, datalog
    gc.collect()
    Cache.cache_data(id_list_dict, nm_marker=F'id_list_dict_max_len_200_all')