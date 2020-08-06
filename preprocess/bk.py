#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import sys

sys.path.append('../../')
import pickle
import random
import pandas as pd
from txbase import Cache
from tqdm import tqdm
import numpy as np
import gc

if not os.path.isdir('../../cached_data/5folds_4seeds_index'):
    os.mkdir('../../cached_data/5folds_4seeds_index')

##########################################
######## 获取input_data.csv ##############
#########################################

id_list_dict = Cache.reload_cache(
    file_nm='../../cached_data/CACHE_id_list_dict_150_normal.pkl',
    base_dir='',
    pure_nm=False)
datalabel = pd.read_hdf('../../cached_data/datalabel_original_stage2.h5')

df_click_times_list = pd.DataFrame(id_list_dict['click_times_list']['id_list'])
df_click_times_list['click_times_list'] = df_click_times_list.apply(
    lambda x: list(x), axis=1)
df_click_times_list = df_click_times_list[['click_times_list']]

df_time_list = pd.DataFrame(id_list_dict['time_list']['id_list'])
df_time_list['time_list'] = df_time_list.apply(lambda x: list(x), axis=1)
df_time_list = df_time_list[['time_list']]

df_creative_id_list = pd.DataFrame(id_list_dict['creative_id_list']['id_list'])
df_creative_id_list['creative_id_list'] = df_creative_id_list.apply(
    lambda x: list(x), axis=1)
df_creative_id_list = df_creative_id_list[['creative_id_list']]

df_ad_id_list = pd.DataFrame(id_list_dict['ad_id_list']['id_list'])
df_ad_id_list['ad_id_list'] = df_ad_id_list.apply(lambda x: list(x), axis=1)
df_ad_id_list = df_ad_id_list[['ad_id_list']]

df_product_id_list = pd.DataFrame(id_list_dict['product_id_list']['id_list'])
df_product_id_list['product_id_list'] = df_product_id_list.apply(
    lambda x: list(x), axis=1)
df_product_id_list = df_product_id_list[['product_id_list']]

df_advertiser_id_list = pd.DataFrame(
    id_list_dict['advertiser_id_list']['id_list'])
df_advertiser_id_list['advertiser_id_list'] = df_advertiser_id_list.apply(
    lambda x: list(x), axis=1)
df_advertiser_id_list = df_advertiser_id_list[['advertiser_id_list']]

df_product_category_list = pd.DataFrame(
    id_list_dict['product_category_list']['id_list'])
df_product_category_list[
    'product_category_list'] = df_product_category_list.apply(
        lambda x: list(x), axis=1)
df_product_category_list = df_product_category_list[['product_category_list']]

df_industry_list = pd.DataFrame(id_list_dict['industry_list']['id_list'])
df_industry_list['industry_list'] = df_industry_list.apply(lambda x: list(x),
                                                           axis=1)
df_industry_list = df_industry_list[['industry_list']]

df_click_times_list['time_list'] = df_time_list['time_list']
df_click_times_list['creative_id_list'] = df_creative_id_list[
    'creative_id_list']
df_click_times_list['ad_id_list'] = df_ad_id_list['ad_id_list']
df_click_times_list['product_id_list'] = df_product_id_list['product_id_list']
df_click_times_list['advertiser_id_list'] = df_advertiser_id_list[
    'advertiser_id_list']
df_click_times_list['product_category_list'] = df_product_category_list[
    'product_category_list']
df_click_times_list['industry_list'] = df_industry_list['industry_list']

del df_time_list, df_creative_id_list, df_ad_id_list, df_product_id_list, df_advertiser_id_list, df_product_category_list, df_industry_list
_ = gc.collect()

df_click_times_list['user_id'] = datalabel['user_id']
df_click_times_list['age'] = datalabel['age']
df_click_times_list['gender'] = datalabel['gender']

df_click_times_list.to_csv('../../cached_data/input_data.csv', index=False)

del df_click_times_list
_ = gc.collect()

#########################################################
######## 获取20分类的pkl 和5 folds index #################
#########################################################
input_data = pd.read_csv('../../cached_data/input_data.csv')
input_data['label'] = input_data['age'] - 1 + (input_data['gender'] - 1) * 10

data_all = []
for entry in tqdm(input_data.values):
    data_all.append({
        "click_times_list": entry[0],
        "time_list": entry[1],
        "creative_id_list": entry[2],
        "ad_id_list": entry[3],
        "product_id_list": entry[4],
        "advertiser_id_list": entry[5],
        "product_category_list": entry[6],
        "industry_list": entry[7],
        "user_id": entry[8],
        "label": entry[11],
    })
pickle.dump(data_all, open('../../cached_data/input_data_20class.pkl', 'wb'))

train_data = input_data.head(3000000)

from sklearn.model_selection import StratifiedKFold

N_FOLDS = 5
seed_list = [34, 200, 1111, 2020]

for seed in seed_list:
    skf = StratifiedKFold(n_splits=N_FOLDS, random_state=seed, shuffle=True)
    X = train_data[[
        'creative_id_list', 'ad_id_list', 'advertiser_id_list',
        'product_id_list', 'product_category_list', 'industry_list',
        'time_list'
    ]]
    Y = train_data[['age']]

    for fold, (train_index, val_index) in enumerate(skf.split(X, Y)):
        np.save(
            '../../cached_data/5folds_4seeds_index/seed_{}_train_index_fold_{}.npy'
            .format(seed, fold), train_index)
        np.save(
            '../../cached_data/5folds_4seeds_index/seed_{}_val_index_fold_{}.npy'
            .format(seed, fold), val_index)
        print('seed:{} --- fold:{} done!'.format(seed, fold))
