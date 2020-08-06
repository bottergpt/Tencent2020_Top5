# -*- encoding: utf-8 -*-
'''
@File      :   run_d2v.py
@Time      :   2020/07/23 15:42:15
@Author    :   zhangqibot 
@Version   :   0.1
@Contact   :   hi@zhangqibot.com
@Desc      :   d2v
'''
import sys
sys.path.append("..")
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
from gensim.models import Word2Vec
import numpy as np
import multiprocessing
from tqdm import tqdm
import gc
import random
import pandas as pd
from txbase import reduce_mem, logger, Cache
from collections import defaultdict


def d2v_pro(df_raw,
            sentence_id,
            word_id,
            emb_size=64,
            window=150,
            dropna=False,
            n_jobs=16,
            method='dm',
            hs=0,
            negative=10,
            epoch=10,
            return_model=False):
    """
    Paper: https://cs.stanford.edu/~quocle/paragraph_vector.pdf
    Gensim: https://radimrehurek.com/gensim/models/doc2vec.html
    Now, set min_count=1 to avoid OOV...
    How to deal with oov in a more appropriate way...
    Paramter:
    ----------
    df_raw: DataFrame contains columns named sentence_id and word_id
    sentence_id: like user ID, will be coerced into str
    word_id: like item ID, will be coerced into str
    emb_size: default 8
    dropna: default False, nans will be filled with 'NULL_zhangqibot'. if True, nans will all be dropped.
    n_jobs: 16 cpus to use as default
    method: 'dm' or 'dbow'
        dm ({1,0}, optional) – Defines the training algorithm. 
        If dm=1, ‘distributed memory’ (PV-DM) is used. (better)
        Otherwise, distributed bag of words (PV-DBOW) is employed.
    hs: ({1,0}, optional)
        If 1, hierarchical softmax will be used for model training. 
        If set to 0, and negative is non-zero, negative sampling will be used.
    negative: (int, optional)
        If > 0, negative sampling will be used, 
        the int for negative specifies how many “noise words” should be drawn (usually between 5-20). 
        If set to 0, no negative sampling is used.
    epoch: iter : int, optional,default 10
        Number of iterations (epochs) over the corpus.
    return_model: default True
    Return:
    ----------
    Example:

    ----------
    """
    if method.lower() in ['dm', 'pvdm']:
        dm = 1
        logger.info("## Use PV-DM ##")
    elif method.lower() in ['dbow', 'pvdbow']:
        dm = 0
        logger.info("## Use PV-DBOW ##")
    else:
        raise NotImplementedError
    list_col_nm = f'{sentence_id}__{word_id}_list'
    if (n_jobs is None) or (n_jobs <= 0):
        n_jobs = multiprocessing.cpu_count()
    logger.info(f"========== Doc2Vec:  {sentence_id} {word_id} ==========")
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
    sentences = [
        TaggedDocument(words=wdi[1], tags=[wdi[0]])
        for wdi in tmp[[sentence_id, list_col_nm]].values
    ]
    all_words_vocabulary = df[word_id].unique().tolist()
    all_sentences_vocabulary = df[sentence_id].unique().tolist()

    del tmp[list_col_nm]
    gc.collect()
    model = Doc2Vec(
        sentences,
        size=emb_size,
        window=window,
        workers=n_jobs,
        min_count=1,  # 最低词频. min_count>1会出现OOV
        dm=
        dm,  # dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
        hs=hs,  # If 1, hierarchical softmax will be used for model training
        negative=negative,  # hs=1 + negative 负采样
        epochs=epoch,
        seed=0)
    ## get word embedding matrix ##
    word_emb_dict = {}
    for word_i in all_words_vocabulary:
        word_emb_dict[word_i] = model.wv[word_i]
# 不应该缺测
#         if word_i in model.wv:
#             emb_dict[word_i] = model.wv[word_i]
#         else:
#             emb_dict[word_i] = np.zeros(emb_size)

## get sentence embedding matrix ##
    sentence_emb_dict = {}
    for sentence_i in all_sentences_vocabulary:
        sentence_emb_dict[sentence_i] = model.docvecs[sentence_i]


# 不应该缺测
#         if sentence_i in model.docvecs:
#             sentence_emb_dict[sentence_i] = model.docvecs[sentence_i]
#         else:
#             sentence_emb_dict[sentence_i] = np.zeros(emb_size)
    sentence_emb_df = pd.DataFrame.from_dict(sentence_emb_dict,
                                             orient='index',
                                             dtype='float32').reset_index()
    col_nm = [sentence_id] + [f'doc_emb_{i}' for i in range(1, emb_size + 1)]
    sentence_emb_df.columns = col_nm

    if not return_model:
        model = None
    return {
        "word_emb_dict": word_emb_dict,
        "sentence_emb_df": sentence_emb_df,
        'model': model
    }


def run_d2v(sentence_id, word_id, marker, epoch=10, window=30, emb_size=128):
    emb_name = f'EMB_DICT_ZQ_D2V_{marker}_{window}WINDOW_{epoch}EPOCH_{sentence_id}_{word_id}'
    print(emb_name)
    if word_id == 'industry':
        epoch = 8
    res_dict = d2v_pro(datalog,
                       sentence_id=sentence_id,
                       word_id=word_id,
                       emb_size=emb_size,
                       dropna=False,
                       n_jobs=48,
                       hs=1,
                       window=window,
                       negative=10,
                       epoch=epoch,
                       return_model=False)

    Cache.cache_data(res_dict, nm_marker=emb_name)


if __name__ == "__main__":
    # ZQ_D2V_WITH_SENTENCE_EMB_30WINDOW_10EPOCH
    marker = "WITH_SENTENCE_EMB"
    sentence_id = 'user_id'
    for word_id in ['creative_id', 'ad_id', 'product_id', 'advertiser_id']:
        run_d2v(sentence_id, word_id, marker=marker, emb_size=128)
    run_d2v(sentence_id, word_id='industry', marker=marker, emb_size=64)
