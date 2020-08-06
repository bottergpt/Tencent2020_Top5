# -*- encoding: utf-8 -*-
'''
@File      :   run_glove.py
@Time      :   2020/07/23 14:46:31
@Author    :   zhangqibot 
@Version   :   0.1
@Contact   :   hi@zhangqibot.com
@Desc      :   run glove: ZQ_GLOVE_0625_50WINDOW_10EPOCH
'''

import glove as glv
import os
from gensim.models import Word2Vec
from base import reduce_mem, logger
import numpy as np
import multiprocessing
from tqdm import tqdm
from collections import Counter
import gc
import pandas as pd
from ..txbase import Cache, logger


def glove_pro(df_raw,
              sentence_id,
              word_id,
              emb_size=128,
              window=50,
              dropna=False,
              n_jobs=16,
              learning_rate=0.05,
              epoch=8,
              return_model=False):
    """
    conda create -y -n TF1.14 python=3.6 
    pip install glove_python
    ------
    test_glove = datalog.head(10000)
    sentence_id = 'user_id'
    word_id = 'industry'

    res = glove_pro(test_glove, sentence_id, word_id, emb_size=32, 
                  window=20, dropna=False, n_jobs=16, 
                  learning_rate=0.05, 
                  epoch=8,return_model=True)
    res.keys()
    res['sentence_emb_df'].info()
    res['model'].most_similar("6", number=10)

    """
    list_col_nm = f'{sentence_id}__{word_id}_list'
    if (n_jobs is None) or (n_jobs <= 0):
        n_jobs = multiprocessing.cpu_count()
    logger.info(f"========== GloVE: {sentence_id} {word_id} ==========")
    df = df_raw[[sentence_id, word_id]].copy()
    if df[sentence_id].isnull().sum() > 0:
        logger.warning("NaNs exist in sentence_id column!!")
    if dropna:
        df = df.dropna(subset=[sentence_id, word_id])
    else:
        df = df.fillna('NULL_zhangqibot')
    df = df.astype(str)
    tmp = df.groupby(sentence_id,
                     as_index=False)[word_id].agg({list_col_nm: list})
    sentences = tmp[list_col_nm].values.tolist()
    all_words_vocabulary = df[word_id].unique().tolist()
    del tmp[list_col_nm]
    gc.collect()

    matrix = glv.Corpus()
    matrix.fit(corpus=sentences, window=window)
    model = glv.Glove(no_components=emb_size,
                      learning_rate=learning_rate,
                      alpha=0.75,
                      max_count=100,
                      max_loss=10.0,
                      random_state=666)
    model.fit(matrix.matrix, epochs=epoch, no_threads=n_jobs, verbose=1)
    model.add_dictionary(matrix.dictionary)
    # get word embedding matrix
    emb_dict = {}
    for word_i in all_words_vocabulary:
        if word_i in model.dictionary:
            emb_dict[word_i] = model.word_vectors[model.dictionary[word_i]]
        else:
            emb_dict[word_i] = np.zeros(emb_size, dtype="float32")
    return {"word_emb_dict": emb_dict}


def run_glove(sentence_id,
              word_id,
              emb_size=128,
              window=50,
              epoch=10,
              marker="0625"):

    emb_name = f'EMB_DICT_ZQ_GLOVE_{marker}_{window}WINDOW_{epoch}EPOCH_{sentence_id}_{word_id}'
    print(emb_name)
    res_dict = glove_pro(datalog,
                         sentence_id,
                         word_id,
                         emb_size=emb_size,
                         window=window,
                         dropna=False,
                         n_jobs=30,
                         learning_rate=0.05,
                         epoch=epoch,
                         return_model=False)
    Cache.cache_data(res_dict, nm_marker=emb_name)


if __name__ == "__main__":
    datalog = Cache.reload_cache(
        file_nm='datalog_sorted_fillna_with_0_all_int',
        base_dir="../cached_data",
        pure_nm=True)
    for col in tqdm([
            'creative_id', 'ad_id', 'product_id', 'product_category',
            'advertiser_id', 'industry'
    ]):
        vc = datalog[col].value_counts()
        vc = vc.rename(f'{col}_cnt').reset_index()
        vc = vc.rename(columns={'index': col})
        # all_1_ids = vc[vc==1].index.tolist()
        # datalog[f'{col}_fill_cnt1'] = datalog[col].swifter.apply(lambda x: -1 if x in all_1_ids else x)
        datalog = datalog.merge(vc, how='left', on=col)
    MIN_COUNT = 1
    for col in tqdm([
            'creative_id', 'ad_id', 'product_id', 'product_category',
            'advertiser_id', 'industry'
    ]):
        datalog[col] = np.where(datalog[f'{col}_cnt'] <= MIN_COUNT, -1,
                                datalog[col])
    marker = "RM_CNT1"
    sentence_id = 'user_id'
    for word_id in tqdm(
        ['creative_id', 'ad_id', 'advertiser_id', 'product_id']):
        run_glove(sentence_id, word_id, emb_size=128, marker=marker)
    run_glove(sentence_id, word_id='industry', emb_size=64, marker=marker)
