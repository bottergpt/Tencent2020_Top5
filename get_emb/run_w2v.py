import os
import sys
from gensim.models import Word2Vec
import numpy as np
from ..txbase import Cache, logger, show_all_feas, reduce_mem
import multiprocessing
from tqdm import tqdm
from collections import Counter
import gc
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from collections import defaultdict
from joblib import Parallel, delayed
from functools import partial


class LossCallback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''
    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1


def get_sentence_emb(sentence, model, emb_size):
    """
    sentence: 1d list
    """
    all_sum = 0
    cnt = 0
    for ii in sentence:
        if ii in model.wv:
            all_sum += model.wv[ii]
            cnt += 1
        if cnt == 0:
            sentence_emb = None  # np.zeros(emb_size, dtype='float32')
        else:
            sentence_emb = all_sum / cnt
    return sentence_emb


def get_sep_lst(td_lst, sent_cid, threshold=7):
    """
    td_lst: list, days to last action
    sent_cid: corresponding action ID lst
    threshold: '>=threshold' as the session marker
    """
    sep = [0]
    tmp_sep = []
    for i, val in enumerate(td_lst):
        if val >= threshold:
            xx = sent_cid[sep[-1]:i]
            if len(xx) > 1:
                tmp_sep.append(xx)
            sep.append(i)
    xx = sent_cid[sep[-1]:]
    if len(xx) > 1:
        tmp_sep.append(xx)
    return tmp_sep


def get_session_df(sub_df, session_id, word_id):
    session_df = sub_df.apply(lambda x: get_sep_lst(x[session_id], x[word_id]),
                              axis=1).sum()  # .tolist()
    return session_df


def concat_list(res_lst_i):
    concated_res_lst = []
    for i_sentence in res_lst_i:
        concated_res_lst += i_sentence
    return concated_res_lst


def get_sub_df_lst(df, partition):
    """
    df = pd.DataFrame({'a':np.random.randn(100),'b':list(range(100))})
    get_sub_df_lst(df,partition=20)
    """
    sub_df_list = []
    length = len(df)
    N = len(df) // partition  # n_elements in one partition
    for i in range(partition - 1):
        sub_df_list.append(df.iloc[i * N:(i + 1) * N])
    sub_df_list.append(df.iloc[N * (partition - 1):])
    return sub_df_list


def w2v_pro(df_raw,
            sentence_id,
            word_id,
            session_id=None,
            emb_size=128,
            window=150,
            dropna=False,
            n_jobs=16,
            min_count=1,
            method='skipgram',
            hs=0,
            negative=10,
            epoch=10,
            partition=40,
            return_model=False):
    """
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
    min_count: min_count
    method: 'sg'/'skipgram' or 'cbow'
        sg : {0, 1}, optional
            Training algorithm: 1 for skip-gram; otherwise CBOW.
    hs : {0, 1}, optional
        If 1, hierarchical softmax will be used for model training.
        If 0, and `negative` is non-zero, negative sampling will be used.
    negative : int, optional
        If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
        should be drawn (usually between 5-20).
        If set to 0, no negative sampling is used.
    epoch: iter : int, optional,default 10
        Number of iterations (epochs) over the corpus.
    return_model: default False
    Return:
    ----------
    Example:
    def run_w2v(sentence_id,word_id,emb_size=128):
        res_dict= w2v_pro(datalog,sentence_id=sentence_id,word_id=word_id,
                          emb_size=emb_size,dropna=False,n_jobs=-1, 
                          method='cbow', hs=0,negative=10,epoch=10,
                          return_model=False)
        Cache.cache_data(res_dict,nm_marker=f'EMB_DICT_W2V_CBOW_10EPOCH_{sentence_id}_{word_id}')

    sentence_id='user_id'
    for word_id in tqdm(['creative_id', 'ad_id', 'product_id', 'advertiser_id']):
        run_w2v(sentence_id,word_id,emb_size=128)

    run_w2v(sentence_id,word_id='product_category',emb_size=8)
    run_w2v(sentence_id,word_id='industry',emb_size=64)
    ----------
    """

    if method.lower() in ['sg', 'skipgram']:
        sg = 1
        logger.info("## Use skip-gram ##")
    elif method.lower() in ['cbow']:
        sg = 0
        logger.info("## Use CBOW ##")
    else:
        raise NotImplementedError
    list_col_nm = f'{sentence_id}__{word_id}_list'
    if (n_jobs is None) or (n_jobs <= 0):
        n_jobs = multiprocessing.cpu_count()
    logger.info(f"========== W2V:  {sentence_id} {word_id} ==========")
    if session_id is not None:
        col2keep = [sentence_id, word_id, session_id]
    else:
        col2keep = [sentence_id, word_id]
    df = df_raw[col2keep].copy()
    if df[sentence_id].isnull().sum() > 0:
        logger.warning("NaNs exist in sentence_id column!!")
    if dropna:
        df = df.dropna(subset=[sentence_id, word_id])
    else:
        df = df.fillna('NULL_zhangqibot')
    df[[sentence_id, word_id]] = df[[sentence_id, word_id]].astype(str)
    all_words_vocabulary = df[word_id].unique().tolist()

    if session_id is not None:
        print("Get sentences by session ...")
        df = df.groupby(sentence_id)[[word_id, session_id]].agg(list)
        sub_df_list = get_sub_df_lst(df, partition=partition)
        del df
        gc.collect()
        res_lst = Parallel(n_jobs=partition)(
            delayed(get_session_df)(sub_df, session_id, word_id)
            for sub_df in sub_df_list)
        # res_lst = Parallel(n_jobs=partition)(delayed(concat_list)(res_lst_i) for res_lst_i in res_lst)
        sentences = []
        for res in res_lst:
            sentences += res
    else:
        print("Get sentences by naive ...")
        tmp = df.groupby(sentence_id,
                         as_index=False)[word_id].agg({list_col_nm: list})
        sentences = tmp[list_col_nm].values.tolist()
        del tmp[list_col_nm]
        gc.collect()

    # word=>sentence 倒排索引
    sentences_iv = defaultdict(set)
    for idx, line in enumerate(sentences):
        for iid in line:
            sentences_iv[iid].add(idx)

    model = Word2Vec(
        sentences,
        size=emb_size,
        window=window,
        workers=n_jobs,
        min_count=min_count,  # 最低词频. min_count>1会出现OOV
        sg=sg,  # 1 for skip-gram; otherwise CBOW.
        hs=hs,  # If 1, hierarchical softmax will be used for model training
        negative=negative,  # hs=1 + negative 负采样
        iter=epoch,
        compute_loss=True,
        callbacks=[LossCallback()],
        seed=0)

    # get word embedding matrix
    emb_dict = {}
    for word_i in all_words_vocabulary:
        if word_i in model.wv:
            emb_dict[word_i] = model.wv[word_i]
        else:  # min_count>1会出现OOV
            cnt = 0
            s_emb = 0
            for sid in sentences_iv[word_i]:  # 这个低频词在哪些句子中，用句子的平均代替
                sid_emb = get_sentence_emb(sentences[sid], model, emb_size)
                if sid_emb is not None:
                    cnt += 1
                    s_emb += sid_emb
            if cnt > 0:
                s_emb /= cnt
            else:
                s_emb = np.zeros(emb_size, dtype='float32')
            emb_dict[word_i] = s_emb
            # emb_dict[word_i] = np.zeros(emb_size, dtype='float32')
    return {"word_emb_dict": emb_dict}


def run_w2v_session_based(sentence_id, word_id, window, emb_size, marker):
    epoch = 10
    emb_name = f'EMB_DICT_ZQ_CBOW_{marker}_{window}WINDOW_{epoch}EPOCH_{sentence_id}_{word_id}'
    print(emb_name)
    if word_id not in ['creative_id', 'ad_id', 'advertiser_id', 'product_id']:
        epoch = 7
        print(f"@@@Train {word_id}, use 7 epoch...")

    res_dict = w2v_pro(datalog,
                       sentence_id,
                       word_id,
                       session_id='time_delta',
                       emb_size=emb_size,
                       window=window,
                       dropna=False,
                       n_jobs=30,
                       min_count=3,
                       method='cbow',
                       hs=0,
                       negative=12,
                       epoch=epoch,
                       partition=40,
                       return_model=False)
    Cache.cache_data(res_dict, nm_marker=emb_name)


def run_w2v(datalog,
            sentence_id,
            word_id,
            window,
            emb_size,
            hs=0,
            method='CBOW',
            epoch=12,
            marker="MIN_CNT3_hs0_ng15"):
    emb_name = f'EMB_DICT_ZQ_{method}_{marker}_{window}WINDOW_{epoch}EPOCH_{sentence_id}_{word_id}'
    print(emb_name)
    if word_id not in ['creative_id', 'ad_id', 'advertiser_id', 'product_id']:
        epoch = 8
        print(f"@@@Train {word_id}, use 8 epoch...")
    res_dict = w2v_pro(datalog,
                       sentence_id=sentence_id,
                       word_id=word_id,
                       emb_size=emb_size,
                       window=window,
                       dropna=False,
                       n_jobs=-1,
                       min_count=3,
                       method=method,
                       hs=hs,
                       negative=15,
                       epoch=epoch,
                       return_model=False)
    Cache.cache_data(res_dict, nm_marker=emb_name)


if __name__ == "__main__":

    ##################################################################################################
    # 'ZQ_CBOW_MIN_CNT3_hs0_ng15_20WINDOW_10EPOCH'
    # 'ZQ_CBOW_MIN_CNT3_hs0_ng15_30WINDOW_10EPOCH'
    datalog = Cache.reload_cache(
        file_nm='datalog_sorted_fillna_with_0_all_int',
        base_dir='../cached_data/',
        pure_nm=True)
    marker = "MIN_CNT3_hs0_ng15"
    sentence_id = 'user_id'
    for window in [20, 30]:
        for word_id in ['creative_id', 'ad_id', 'advertiser_id', 'product_id']:
            run_w2v(datalog,
                    sentence_id,
                    word_id,
                    window=window,
                    emb_size=128,
                    marker=marker)
        run_w2v(datalog,
                sentence_id,
                word_id="industry",
                window=window,
                emb_size=64,
                marker=marker)
        run_w2v(datalog,
                sentence_id,
                word_id="product_category",
                window=window,
                emb_size=8,
                marker=marker)
    ##################################################################################################
    # 'ZQ_CBOW_0623_150WINDOW_10EPOCH',
    marker = "0623"
    sentence_id = 'user_id'
    for word_id in ['creative_id', 'ad_id', 'advertiser_id', 'product_id']:
        run_w2v(datalog,
                sentence_id,
                word_id,
                window=150,
                emb_size=128,
                method='CBOW',
                epoch=10,
                marker=marker)
    run_w2v(datalog,
            sentence_id,
            word_id="industry",
            window=150,
            emb_size=64,
            method='CBOW',
            epoch=10,
            marker=marker)
    run_w2v(datalog,
            sentence_id,
            word_id="product_category",
            window=150,
            emb_size=8,
            method='CBOW',
            epoch=10,
            marker=marker)
    ##################################################################################################
    #  'ZQ_CBOW_HS1_100WINDOW_10EPOCH',
    marker = "HS1"
    sentence_id = 'user_id'
    for word_id in ['creative_id', 'ad_id', 'advertiser_id', 'product_id']:
        run_w2v(datalog,
                sentence_id,
                word_id,
                window=100,
                emb_size=128,
                hs=1,
                method='CBOW',
                epoch=10,
                marker=marker)
    run_w2v(datalog,
            sentence_id,
            word_id="industry",
            window=100,
            emb_size=64,
            hs=1,
            method='CBOW',
            epoch=10,
            marker=marker)
    run_w2v(datalog,
            sentence_id,
            word_id="product_category",
            window=100,
            emb_size=8,
            hs=1,
            method='CBOW',
            epoch=10,
            marker=marker)

    ##################################################################################################

    #  'ZQ_SG_MIN_CNT3_hs0_90WINDOW_10EPOCH',
    #  'ZQ_SG_MIN_CNT3_hs1_30WINDOW_10EPOCH',  => 'ZQ_SG_MIN_CNT3_hs0_30WINDOW_10EPOCH'
    #  'ZQ_SG_MIN_CNT5_hs0_10WINDOW_10EPOCH',  => "ZQ_SG_MIN_CNT3_hs0_10WINDOW_10EPOCH"
    #  'ZQ_SG_RM_CNT1_30WINDOW_10EPOCH'  => "ZQ_SG_MIN_CNT3_hs0_30WINDOW_10EPOCH"

    marker = "MIN_CNT3_hs0"
    sentence_id = 'user_id'
    for window in [10, 30, 90]:
        for word_id in ['creative_id', 'ad_id', 'advertiser_id', 'product_id']:
            run_w2v(datalog,
                    sentence_id,
                    word_id,
                    window=window,
                    emb_size=128,
                    hs=0,
                    method='SG',
                    epoch=10,
                    marker=marker)
        run_w2v(datalog,
                sentence_id,
                word_id="industry",
                window=window,
                emb_size=64,
                hs=0,
                method='SG',
                epoch=10,
                marker=marker)
        run_w2v(datalog,
                sentence_id,
                word_id="product_category",
                window=window,
                emb_size=8,
                hs=0,
                method='SG',
                epoch=10,
                marker=marker)

    ##################################################################################################
    #  'ZQ_CBOW_SESSION_BASED_MIN_CNT3_hs0_10WINDOW_10EPOCH',
    #  'ZQ_CBOW_SESSION_BASED_MIN_CNT3_hs0_30WINDOW_10EPOCH',
    datalog["time_shift1"] = datalog.groupby('user_id')['time'].shift(
        1).fillna(method='bfill')
    datalog['time_delta'] = datalog["time"] - datalog["time_shift1"]

    del datalog["time_shift1"]
    gc.collect()

    sentence_id = 'user_id'
    marker = "SESSION_BASED_MIN_CNT3_hs0"

    for WINDOW in [10, 30]:  # 5, 10, 20, 30,
        print("###" * 35)
        print("window==", WINDOW)
        print("###" * 35)
        for word_id in ['creative_id', 'ad_id', 'advertiser_id', 'product_id']:
            run_w2v_session_based(sentence_id,
                                  word_id,
                                  window=WINDOW,
                                  emb_size=128,
                                  marker=marker)
        run_w2v_session_based(sentence_id,
                              word_id="industry",
                              window=WINDOW,
                              emb_size=64,
                              marker=marker)

    ##################################################################################################
    # ZQ_CBOW_MIN_CNT3_shuffled_hs0_ng15_30WINDOW_12EPOCH
    datalog = datalog.sample(frac=1, random_state=666).reset_index(drop=True)
    marker = "MIN_CNT3_shuffled_hs0_ng15"
    sentence_id = 'user_id'
    for word_id in ['creative_id', 'ad_id', 'advertiser_id', 'product_id']:
        run_w2v(datalog,
                sentence_id,
                word_id,
                window=30,
                emb_size=128,
                marker=marker)
    run_w2v(datalog,
            sentence_id,
            word_id="industry",
            window=30,
            emb_size=64,
            marker=marker)

    run_w2v(datalog,
            sentence_id,
            word_id="product_category",
            window=30,
            emb_size=8,
            marker=marker)
    ##################################################################################################
