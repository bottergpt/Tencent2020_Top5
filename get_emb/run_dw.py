# -*- encoding: utf-8 -*-
'''
@File      :   run_dw.py
@Time      :   2020/07/23 15:31:06
@Author    :   zhangqibot 
@Version   :   0.1
@Contact   :   hi@zhangqibot.com
@Desc      :   D2V
'''

import os
import sys
from gensim.models import Word2Vec
from ..txbase import reduce_mem, logger, Cache
import numpy as np
import multiprocessing
from tqdm import tqdm
import gc
import random
import pandas as pd
from collections import defaultdict


def deepwalk(df_raw,
             sentence_id,
             word_id,
             emb_size=128,
             path_length=20,
             window=10,
             epoch=10,
             n_jobs=16,
             method='cbow',
             hs=0,
             negative=10,
             dropna=False,
             seed=666,
             return_model=False):
    """
    Paramter:
    ----------
    Return:
    ----------
    Example:
    ----------
    res_dic = deepwalk(df_raw=datalog.head(1000),sentence_id='user_id',
                       word_id='creative_id',emb_size=16,
                       path_length=20,dropna=False,return_model=False)
    """
    if method.lower() in ['sg', 'skipgram']:
        sg = 1
        logger.info("## Use skip-gram ##")
    elif method.lower() in ['cbow']:
        sg = 0
        logger.info("## Use CBOW ##")
    else:
        raise NotImplementedError

    logger.info(f"========== DeepWalk:  {sentence_id} {word_id} ==========")
    df = df_raw[[sentence_id, word_id]].copy()
    if df[sentence_id].isnull().sum() > 0:
        logger.warning("NaNs exist in sentence_id column!!")
    if dropna:
        df = df.dropna(subset=[sentence_id, word_id])
    else:
        df = df.fillna('NULL_zhangqibot')
    df = df.astype(str)
    # 构建U-I关系图
    dic = defaultdict(set)
    for item in df.values:
        dic['item_' + item[1]].add('user_' + item[0])
        dic['user_' + item[0]].add('item_' + item[1])
    dic_cont = {}
    for key in dic:
        dic[key] = list(dic[key])
        dic_cont[key] = len(dic[key])
    print("Creating...")
    # 构建路径
    sentences = []
    length = []
    for key in dic:
        sentence = [key]
        while len(sentence) != path_length:
            rdm_index = random.randint(0, dic_cont[sentence[-1]] - 1)
            key = dic[sentence[-1]][rdm_index]
            # 又回到了自身，所以break。可以优化，加入try again
            if len(sentence) >= 2 and key == sentence[-2]:
                break
            else:
                sentence.append(key)
        sentences.append(sentence)
        length.append(len(sentence))
        if len(sentences) % 100000 == 0:
            print(len(sentences))
    print(np.mean(length))  # 平均长度
    print(len(sentences))
    # 训练Deepwalk模型
    print('Training...')
    random.shuffle(sentences)
    model = Word2Vec(
        sentences,
        size=emb_size,
        window=window,
        workers=n_jobs,
        min_count=1,  # 最低词频. min_count>1会出现OOV # TODO
        sg=sg,  # 1 for skip-gram; otherwise CBOW.
        hs=hs,  # If 1, hierarchical softmax will be used for model training
        negative=negative,  # hs=1 + negative 负采样
        iter=epoch,
        seed=seed)
    print('Outputing...')
    #     values = set(df[sentence_id].values)
    #     w2v = []
    #     for v in values:
    #         a = [v]
    #         a.extend(model.wv[f'user_{v}'])
    #         w2v.append(a)
    #     sentence_emb_df = pd.DataFrame(w2v)
    #     names = [sentence_id]
    #     for i in range(emb_size):
    #         names.append(f'deepwalk_emb_{sentence_id}_{word_id}_{emb_size}_{i+1}')
    #     sentence_emb_df.columns = names
    #     deepwalk_emb_cols = [
    #         col for col in sentence_emb_df.columns if 'deepwalk_emb_' in col
    #     ]
    #     sentence_emb_df[deepwalk_emb_cols] = sentence_emb_df[
    #         deepwalk_emb_cols].astype(np.float32)
    #     sentence_emb_df=None # no need to get sentence_emb
    print("Get Word Embedding...")
    all_words_vocabulary = set(df[word_id].values)
    emb_dict = {}  # word_emb_dict
    for word_i in all_words_vocabulary:
        emb_dict[word_i] = model.wv[f'item_{word_i}']
    print("Done!")
    return {"word_emb_dict": emb_dict}


def run_dw(sentence_id,
           word_id,
           emb_size,
           path_length=50,
           window=20,
           epoch=10,
           n_jobs=40,
           seed=666,
           method='cbow',
           marker="DW_CBOW_hs0ng15"):

    marker = marker + f'_path_length{path_length}'
    emb_name = f'EMB_DICT_ZQ_DW_{marker}_{window}WINDOW_{epoch}EPOCH_{sentence_id}_{word_id}'
    print(emb_name)
    res_dict = deepwalk(df_raw=datalog,
                        sentence_id=sentence_id,
                        word_id=word_id,
                        emb_size=emb_size,
                        path_length=path_length,
                        window=window,
                        epoch=epoch,
                        n_jobs=n_jobs,
                        method=method,
                        seed=seed,
                        hs=0,
                        negative=15)
    Cache.cache_data(res_dict, nm_marker=emb_name)


if __name__ == "__main__":
    ##################################################################################################

    # 'ZQ_DW_15WINDOW_10EPOCH',
    # 'ZQ_DW_RM_CNT1_PATH100_50WINDOW_10EPOCH',
    # 换成：
    # ZQ_DW_DW_CBOW_hs0ng15_path_length50_10WINDOW_10EPOCH
    # ZQ_DW_DW_CBOW_hs0ng15_path_length50_30WINDOW_10EPOCH

    datalog = Cache.reload_cache(
        file_nm='datalog_sorted_fillna_with_0_all_int',
        base_dir='../cached_data/',
        pure_nm=True)
    path_length = 50
    for WINDOW in [10, 30]:
        print("###" * 35)
        print("window==", WINDOW)
        print("###" * 35)
        for word_id in ['creative_id', 'ad_id', 'product_id', 'advertiser_id']:
            run_dw(sentence_id='user_id',
                   word_id=word_id,
                   emb_size=128,
                   path_length=path_length,
                   window=WINDOW)
        run_dw(sentence_id='user_id',
               word_id='industry',
               emb_size=64,
               path_length=path_length,
               window=WINDOW)
        run_dw(sentence_id='user_id',
               word_id='product_category',
               emb_size=8,
               path_length=path_length,
               window=WINDOW)
