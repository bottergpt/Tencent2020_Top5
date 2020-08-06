#!/usr/bin/env python 
# encoding: utf-8 

"""
@version: v1.0
@author: zhenglinghan
@contact: 422807471@qq.com
@software: PyCharm
@file: model_result_to_npy.py
@time: 2020/7/23 16:05
"""

import sys

sys.path.append("../")
import pandas as pd
import numpy as np
from txbase import get_cur_dt_str
import glob
import os
from pathlib import Path
import pickle
from pandas.core.frame import DataFrame
import torch.nn.functional as F
from tqdm import tqdm
import torch
import glob
import gc

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('max_colwidth', 200)
pd.set_option('display.width', 5000)


def get_mean_prob(mp_lst):
    '''

    :param mp_lst: average each prob between various folds
    :return:
    '''
    NUM = len(mp_lst)
    final = 0
    for ii in mp_lst:
        final += np.load(ii)[-1000000:, :] / NUM
    return final


def load_id_and_agg_prob(base_dir, folder_nm, prob_marker='prob'):
    '''
    zq zlh models 10 folds
    '''
    emb_dir = os.path.join(base_dir, folder_nm)
    with open(f"{emb_dir}/folds.pkl", 'rb') as file:
        folds = pickle.load(file)
    prob_list = glob.glob(f"{emb_dir}/SAVE_MODEL_PROB_FOLD*")# valid
    ifreverse = False
    for i in prob_list:
        if i.find('Reversed') > -1:
            ifreverse = True
            break
    agg_prob = get_mean_prob(prob_list)  # test
    df_id = pd.read_csv(f"{emb_dir}/SAVE_all_uid_df.csv")
    df_id = df_id[-1000000:].reset_index(drop=True)  # test id
    df_ = pd.DataFrame(agg_prob, columns=[prob_marker + '_' + str(ii) for ii in range(agg_prob.shape[-1])])
    df_final_test = pd.concat([df_id, df_], axis=1).sort_values(by="user_id").reset_index(drop=True)  # test部分

    prob_oof = np.zeros((3000000, 20))
    for i in range(10):
        index = folds[i][1]
        if ifreverse:
            prob1 = np.load(f"{emb_dir}/SAVE_MODEL_PROB_FOLD{i}_Reversed.npy")
            prob2 = np.load(f"{emb_dir}/SAVE_MODEL_PROB_FOLD{i}.npy")
            prob = (prob1 + prob2) / 2
        else:
            prob = np.load(f"{emb_dir}/SAVE_MODEL_PROB_FOLD{i}.npy")
        prob_oof[index, :] = prob[index, :]
    df_id = pd.read_csv(f"{emb_dir}/SAVE_all_uid_df.csv")
    df_id = df_id[:-1000000].reset_index(drop=True)  # oof id
    df_ = pd.DataFrame(prob_oof, columns=[prob_marker + '_' + str(ii) for ii in range(prob_oof.shape[-1])])
    # concat oof prob and test prob
    # 400w,20 sort by user_id
    df_final_oof = pd.concat([df_id, df_], axis=1).sort_values(by="user_id").reset_index(drop=True)
    df_all = pd.concat([df_final_oof, df_final_test]).sort_values(by="user_id").reset_index(drop=True)
    prob = df_all.drop('user_id', axis=1).values
    # check result should be 1
    print('check:', np.mean(np.sum(prob, axis=1)))
    print('check:', df_all['user_id'].describe())
    return prob


def to_result(x):
    '''

    :param x: value before softmax to the softmax prob result of each class
    :return:
    '''
    x = F.softmax(torch.from_numpy(x), dim=0)
    return x


def get_oof_test_bk(path):
    '''
    .. bk model 5 folds
    '''
    test_lst = glob.glob(path + "test_*")
    data1 = pickle.load(open(test_lst[0], 'rb'))
    data2 = pickle.load(open(test_lst[1], 'rb'))
    data3 = pickle.load(open(test_lst[2], 'rb'))
    data4 = pickle.load(open(test_lst[3], 'rb'))
    data5 = pickle.load(open(test_lst[4], 'rb'))
    df_uid = data1[["user_id"]]
    # we do prediction of ordered and reversed sequence of test sample and output non-softmax result
    # and wo merge the softmax result of test result as finial prediction
    # so in this function we do not need to_result
    df_uid['predicted_avg'] = (data1['pred'] + data2['pred'] + data3['pred'] + data4['pred'] + data5['pred']) / 5
    df_uid['predicted_avg'] = df_uid['predicted_avg'].apply(lambda x: x.numpy())

    test_lst = glob.glob(path + "val_*")
    data1 = pickle.load(open(test_lst[0], 'rb'))
    data2 = pickle.load(open(test_lst[1], 'rb'))
    data3 = pickle.load(open(test_lst[2], 'rb'))
    data4 = pickle.load(open(test_lst[3], 'rb'))
    data5 = pickle.load(open(test_lst[4], 'rb'))
    # valid part we do only ordered sequence of the test sample and output non-softmax result
    # so in this function we need to_result
    data1['pred'] = data1['pred'].apply(lambda x: to_result(x))
    data2['pred'] = data2['pred'].apply(lambda x: to_result(x))
    data3['pred'] = data3['pred'].apply(lambda x: to_result(x))
    data4['pred'] = data4['pred'].apply(lambda x: to_result(x))
    data5['pred'] = data5['pred'].apply(lambda x: to_result(x))
    df_oof = pd.concat([data1, data2, data3, data4, data5])
    df_oof['predicted_avg'] = df_oof['pred'].apply(lambda x: x.numpy())
    df_all = pd.concat([df_oof[['user_id', 'predicted_avg']], df_uid[['user_id', 'predicted_avg']]])
    # check result should be 1
    print('check:', np.mean(np.sum(df_all['predicted_avg'].tolist(), axis=1)))
    print('check:', df_all['user_id'].describe())
    df_all = df_all.sort_values(by="user_id").reset_index(drop=True)
    prob_bk = np.array(df_all['predicted_avg'].tolist())
    return prob_bk


if __name__ == "__main__":
    # bk model to npy 400w，20 total 4models

    # these part contain two models with data augment!
    # 5 folds and 20 folds
    path_list_bk = ['./bk_oof/Multi_Head_ResNext_seed_34_aug/', './bk_oof/Multi_Head_ResNext_seed_1111_aug/']
    prob_bk_list = []
    for pathi in path_list_bk:
        prob_bk_list.append(get_oof_test_bk(pathi))
    gc.collect()
    # merge
    prob = (prob_bk_list[0] + prob_bk_list[1]) / 2
    np.save('../05_RESULT/META/oof0/model20.npy', prob)

    path_list_bk = ['./bk_oof/Multi_Head_ResNext_seed_34/',
                    './bk_oof/Multi_Head_ResNext_seed_1111/',
                    './bk_oof/Multi_Head_ResNext_seed_200/',
                    './bk_oof/Multi_Head_ResNext_seed_2020/']
    prob_bk_list = []
    for pathi in path_list_bk:
        prob_bk_list.append(get_oof_test_bk(pathi))
    gc.collect()
    # merge
    prob = (prob_bk_list[0] + prob_bk_list[1] + prob_bk_list[2] + prob_bk_list[3]) / 4
    np.save('../05_RESULT/META/oof1/model20.npy', prob)

    # these past contain two models without data augment!
    # 5 folds and another 5 folds
    path_list_bk = ['./bk_oof/Multi_Head_ResNet/']
    prob_bk_list = []
    for pathi in path_list_bk:
        prob_bk_list.append(get_oof_test_bk(pathi))
    gc.collect()
    # merge
    prob = prob_bk_list[0]
    np.save('../05_RESULT/META/oof2/model20.npy', prob)

    path_list_bk = ['./bk_oof/Multi_Head_ResNext/']
    prob_bk_list = []
    for pathi in path_list_bk:
        prob_bk_list.append(get_oof_test_bk(pathi))
    gc.collect()
    # merge
    prob = prob_bk_list[0]
    np.save('../05_RESULT/META/oof3/model20.npy', prob)

    # zq,zlh model to npy 400w, 20 total 8models
    # 8 models in 10 folds
    # in the submission stacking we actualy use 20 probs to stacking
    # those models belong to the 5 folds result of these 8 models with various seed
    BASE_DIR = "../05_RESULT/META/"
    path_list_zq = [fi for fi in os.listdir(BASE_DIR) if fi[:4] == "2020" and os.path.isdir(os.path.join(BASE_DIR, fi))]
    for i in path_list_zq:
        print(i)
    prob_zq_lst = []
    for prob_nm_i in path_list_zq:
        prob_zq_lst.append(load_id_and_agg_prob(base_dir=BASE_DIR, folder_nm=prob_nm_i))
    probi = np.zeros((4000000, 20))
    for index, i in enumerate(prob_zq_lst):
        np.save(BASE_DIR + f'../05_RESULT/META/oof{index + 4}/model20.npy', probi)

    # total 12 models oof0~11 to stacking!
