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

import sys
sys.path.append("../")
from txbase import Cache
from txbase import logger
import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
import gc
import multiprocessing
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

data_path = '../cached_data/'

# three method to pretrain w2v embedding
# turn datalog to embedding index seqs

if __name__ == "__main__":
    # ######################################################################################################################
    # normal datalog embedding pretrain
    # base datalog
    datalog = pd.read_hdf(data_path + 'data_log_p2_stage2.h5')

    def w2v_pro_normal(df_raw,
                       sentence_id,
                       word_id,
                       emb_size=128,
                       window=20,
                       dropna=False,
                       n_jobs=16,
                       method='cbow',
                       hs=1,
                       negative=0,
                       epoch=10):
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
        return_model: default True
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
        model = Word2Vec(
            sentences,
            size=emb_size,
            window=window,
            workers=n_jobs,
            min_count=1,  # 最低词频. min_count>1会出现OOV
            sg=sg,  # 1 for skip-gram; otherwise CBOW.
            hs=hs,  # If 1, hierarchical softmax will be used for model training
            negative=negative,  # hs=1 + negative 负采样
            iter=epoch,
            seed=0)

        # get word embedding matrix
        emb_dict = {}
        for word_i in all_words_vocabulary:
            if word_i in model.wv:
                emb_dict[word_i] = model.wv[word_i]
            else:
                emb_dict[word_i] = np.zeros(emb_size)
        # get sentence embedding matrix
        emb_matrix = []
        for seq in sentences:
            vec = []
            for w in seq:
                if w in model.wv:
                    vec.append(model.wv[w])
            if len(vec) > 0:
                emb_matrix.append(np.mean(vec, axis=0))
            else:
                emb_matrix.append([0] * emb_size)
        emb_matrix = np.array(emb_matrix)
        for i in range(emb_size):
            tmp[f'{sentence_id}_{word_id}_emb_{i}'] = emb_matrix[:, i]
        return {"word_emb_dict": emb_dict}

    # %%

    def run_w2v(sentence_id, word_id, emb_size=256):
        '''

        :param sentence_id: sentence groupby key
        :param word_id: col as word
        :param emb_size: output embedding size used in w2v
        :return:
        '''
        # large window embedding
        window = 150
        res_dict = w2v_pro_normal(datalog,
                                  sentence_id=sentence_id,
                                  word_id=word_id,
                                  window=window,
                                  emb_size=emb_size,
                                  dropna=False,
                                  n_jobs=12,
                                  epoch=5)
        epoch = 10
        method = 'CBOW'
        author = 'AZ'
        marker = 'TXBASE'
        Cache.cache_data(
            res_dict,
            nm_marker=
            f'EMB_DICT_{author}_{method}_{marker}_{window}WINDOW_{epoch}EPOCH_{sentence_id}_{word_id}'
        )
        del res_dict
        gc.collect()

    sentence_id = 'user_id'
    # various embeddim according to the dictionary size
    emb_size_dict = {
        'time': 8,
        'creative_id': 256,
        'ad_id': 256,
        'product_id': 32,
        'advertiser_id': 64,
        'product_category': 8,
        'industry': 16
    }
    for word_id in tqdm([
            'time', 'creative_id', 'ad_id', 'product_id', 'advertiser_id',
            'product_category', 'industry'
    ]):
        run_w2v(sentence_id, word_id, emb_size=emb_size_dict[word_id])
    print('TXBASE emb w2v_finished!')

    # ######################################################################################################################
    # item datalog embedding pretrain
    gc.collect()
    # calculate time diff
    datalog['time_diff'] = datalog.groupby('user_id')['time'].diff(1)
    datalog['time_diff'] = datalog['time_diff'].fillna(0)

    for var in [
            'time_diff', 'time', 'creative_id', 'ad_id', 'product_id',
            'advertiser_id', 'product_category', 'industry'
    ]:
        datalog[var] = datalog[var].astype(int).astype(str)

    def w2v_pro_item(df_raw,
                     sentence_id,
                     word_id,
                     emb_size=128,
                     window=100,
                     dropna=False,
                     n_jobs=16,
                     method='skipgram',
                     hs=0,
                     negative=5,
                     epoch=10):
        """
        Now, set min_count=1 to avoid OOV...
        How to deal with oov in a more appropriate way...
        Paramter:
        ----------
        df_raw: DataFrame contains columns named sentence_id and word_id
        sentence_id: like user ID, will be coerced into str
        word_id: like item ID, will be coerced into str tuple (item_key,item_category)
                 item_category will not in word Dictionary only as item in sequence
        emb_size: default 8
        dropna: default False, nans will be filled with 'NULL_zhangqibot'. if True, nans will all be dropped.
        n_jobs: 16 cpus to use as default
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
        return_model: default True
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
        list_col_nm = f'{sentence_id}__{word_id[0]}_{word_id[1]}_list'
        if (n_jobs is None) or (n_jobs <= 0):
            n_jobs = multiprocessing.cpu_count()
        logger.info(
            f"========== W2V:  {sentence_id} {word_id[0]} {word_id[1]} =========="
        )
        df = df_raw[[sentence_id, word_id[0], word_id[1]]].copy()
        if df[sentence_id].isnull().sum() > 0:
            logger.warning("NaNs exist in sentence_id column!!")
        if dropna:
            df = df.dropna(subset=[sentence_id, word_id[0], word_id[1]])
        else:
            df = df.fillna('NULL_zhangqibot')
        # item embedding
        # change sentence as below
        # word_i,category_n,word_j,category_m,word_k,category_n
        # dictionary set = words set + categories set
        # model can get both words embedding vector and categories embedding vector
        max_key_item = 10000000
        df[word_id[1]] = df[word_id[1]].astype(
            int
        ) + max_key_item  # make sure no same word in words and categories
        df['concat_item'] = df[word_id[0]].astype(str) + ' ' + df[
            word_id[1]].astype(str)
        tmp = df.groupby(sentence_id)['concat_item'].agg(
            lambda x: (' '.join(list(x))).split(' ')).reset_index()
        tmp.columns = [sentence_id, list_col_nm]
        sentences = tmp[list_col_nm].values.tolist()
        all_words_vocabulary_0 = df[word_id[0]].astype(str).unique().tolist()
        all_words_vocabulary_1 = df[word_id[1]].astype(
            str).unique().tolist()  # +了10000000
        del tmp[list_col_nm]
        gc.collect()
        model = Word2Vec(
            sentences,
            size=emb_size,
            window=window,
            workers=n_jobs,
            min_count=1,  # 最低词频. min_count>1会出现OOV
            sg=sg,  # 1 for skip-gram; otherwise CBOW.
            hs=hs,  # If 1, hierarchical softmax will be used for model training
            negative=negative,  # hs=1 + negative 负采样
            iter=epoch,
            seed=0)

        # get word embedding matrix
        emb_dict0 = {}
        for word_i in all_words_vocabulary_0:
            if word_i in model.wv:
                emb_dict0[word_i] = model.wv[word_i]
            else:
                emb_dict0[word_i] = np.zeros(emb_size)
        emb_dict1 = {}
        for word_i in all_words_vocabulary_1:
            if word_i in model.wv:
                word_i_dict = str(int(int(word_i) - 10000000))
                emb_dict1[word_i_dict] = model.wv[word_i]
            else:
                word_i_dict = str(int(int(word_i) - 10000000))
                emb_dict1[word_i_dict] = np.zeros(emb_size)
        return {"word_emb_dict": emb_dict0}, {"word_emb_dict": emb_dict1}

    def run_w2v(sentence_id, word_id, emb_size=256, epoch=10):
        window = 60
        res_dict0, res_dict1 = w2v_pro_item(datalog,
                                            sentence_id=sentence_id,
                                            word_id=word_id,
                                            window=window,
                                            emb_size=emb_size,
                                            dropna=False,
                                            n_jobs=12,
                                            epoch=epoch)
        epoch = epoch
        method = 'cbow'
        author = 'AZ'
        marker = 'CONCAT_' + word_id[1]
        Cache.cache_data(
            res_dict0,
            nm_marker=
            f'EMB_DICT_{author}_{method}_{marker}_{window}WINDOW_{epoch}EPOCH_{sentence_id}_{word_id[0]}'
        )
        del res_dict0, res_dict1
        # do not use category embedding
        # Cache.cache_data(res_dict1,
        #                  nm_marker=f'EMB_DICT_{author}_{method}_{marker}_{window}WINDOW_{epoch}EPOCH_{sentence_id}_{word_id[1]}')
        gc.collect()

    sentence_id = 'user_id'
    emb_size = 128
    gc.collect()
    '''
    '''
    # product category as category
    # time_diff as category
    for word_id in tqdm([('creative_id', 'product_category'),
                         ('ad_id', 'product_category'),
                         ('advertiser_id', 'product_category'),
                         ('product_id', 'product_category'),
                         ('product_category', 'product_category'),
                         ('industry', 'product_category'),
                         ('creative_id', 'time_diff'), ('ad_id', 'time_diff'),
                         ('advertiser_id', 'time_diff'),
                         ('product_id', 'time_diff'),
                         ('product_category', 'time_diff'),
                         ('industry', 'time_diff')]):
        run_w2v(sentence_id, word_id, emb_size=emb_size, epoch=10)
    print('item w2v_finished!')

    del datalog
    gc.collect()

    # ######################################################################################################################
    # click_times increase datalog embedding pretrain
    # base datalog
    datalog = pd.read_hdf(data_path + 'data_log_p2_stage2.h5')

    def run_w2v(sentence_id, word_id, emb_size=256):
        window = 60
        res_dict = w2v_pro_normal(datalog,
                                  sentence_id=sentence_id,
                                  word_id=word_id,
                                  window=60,
                                  emb_size=emb_size,
                                  dropna=False,
                                  n_jobs=24,
                                  epoch=10)
        epoch = 10
        method = 'CBOW'
        author = 'AZ'
        marker = 'CLICK_TIMES_INCREASED'

        Cache.cache_data(
            res_dict,
            nm_marker=
            f'EMB_DICT_{author}_{method}_{marker}_{window}WINDOW_{epoch}EPOCH_{sentence_id}_{word_id}'
        )
        del res_dict
        gc.collect()

    sentence_id = 'user_id'
    emb_size_dict = {
        'time': 8,
        'creative_id': 256,
        'ad_id': 256,
        'product_id': 32,
        'advertiser_id': 64,
        'product_category': 8,
        'industry': 16
    }
    for word_id in tqdm([
            'time', 'creative_id', 'ad_id', 'product_id', 'advertiser_id',
            'product_category', 'industry'
    ]):
        run_w2v(sentence_id, word_id, emb_size=emb_size_dict[word_id])
    print('TXBASE emb w2v_finished!')

    # ######################################################################################################################
    # get_zlh_bk use index seq
    # base datalog
    datalog = pd.read_hdf(data_path + 'data_log_p2_stage2.h5')
    for var in tqdm([
            'click_times', 'creative_id', 'ad_id', 'product_id',
            'product_category', 'advertiser_id', 'industry'
    ]):
        datalog[var] = datalog[var].astype(str)
    features = [
        'click_times', 'time', 'creative_id', 'ad_id', 'product_id',
        'advertiser_id', 'product_category', 'industry'
    ]

    datalabel = pd.read_hdf(data_path + 'datalabel_original.h5')

    for var in features:
        datalog[var] = datalog[var].astype(str)

    sequence_features = list(map(lambda x: x + '_list', features))

    # add sequence
    def get_action_lst(datalog, col):
        temp_list = datalog.groupby('user_id')[col].agg(
            lambda x: ' '.join(list(x))).rename(f'{col}_list').reset_index()
        return temp_list

    # datalabel userid as key
    datalabel = datalabel[['user_id', 'gender', 'age']]
    for col in features:
        temp_list = get_action_lst(datalog, col)
        datalabel = datalabel.merge(temp_list, on='user_id', how='left')

    def get_sequence(data, col, max_len=None):
        '''

        :param data: datalabel
        :param col: col data sequence to dictionary key index sequence
        :param max_len: padding each sequnce to maxlen
        :return:
        '''
        key2index = {}

        def split(x):
            key_ans = x.split()
            for key in key_ans:
                if key not in key2index:
                    # Notice : input value 0 is a special "padding",
                    # so we do not use 0 to encode valid feature for sequence input
                    key2index[key] = len(key2index) + 1  # 从1开始，0用于padding
            return list(map(lambda x: key2index[x], key_ans))

        # preprocess the sequence feature
        id_list = list(map(split, data[col].values))
        id_list_length = np.array(list(map(len, id_list)))
        # max_len = max(genres_length)
        if max_len is None:
            max_len = int(np.percentile(id_list_length, 99))
        # pre padding , 0 before sequence
        id_list = pad_sequences(id_list,
                                maxlen=max_len,
                                padding='pre',
                                truncating='pre')
        return id_list, key2index

    id_list_dict = {}
    for col in tqdm(sequence_features):
        id_list, key2index = get_sequence(datalabel, col, max_len=150)
        # dict ,id_list as key index sequence key2index as words -> key index
        id_list_dict[col] = {'id_list': id_list, 'key2index': key2index}

    Cache.cache_data(id_list_dict, nm_marker='id_list_dict_150_normal')

    # ##################################################################################################################
    # get time embedding
    import datetime
    # during 2019-09-01 to 2019-11-30
    id_list_dict = Cache.reload_cache(file_nm=data_path +
                                      'CACHE_id_list_dict_150_normal.pkl',
                                      base_dir='',
                                      pure_nm=False)

    class strTimeEmb(object):
        '''
    	# time 中一些特征做onehot encoding
        周x
        是否是周末
        月
        月第x周
        教师节 中秋节 16日 9.29调休 10.1假期 10.7重阳节 10.12调休 10.28寒衣节 11.8立冬 11.17学生日 11.28感恩节
        '''
        def __init__(self, daynow):
            self.daynow = int(daynow)
            self.month = 0
            self.day = 0

        def getday(self):
            '''

            :return: day in month
            '''
            if self.month == 0:
                self.day = self.daynow
            elif self.month == 1:
                self.day = self.daynow - 30
            else:
                self.day = self.daynow - 61

        def getweekday(self):
            '''

            :return: data in week
            '''

            dayi = (self.daynow % 7) - 1
            if dayi == 0:
                dayi = 7
            if dayi < 0:
                dayi = 6
            return np.eye(7)[dayi - 1]

        def getifweekend(self):
            '''

            :return: if weekend
            '''
            if self.daynow % 7 < 2:
                return np.array([1])
            else:
                return np.array([0])

        def getmonth(self):
            '''

            :return: month
            '''
            if self.daynow <= 30:
                month = 0
            elif self.daynow > 61:
                month = 2
            else:
                month = 1
            self.month = month
            return np.eye(3)[month]

        def getweekofmonth(self):
            '''

            :return: th week in month
            '''
            year = 2019
            end = int(
                datetime.datetime(year, self.month + 9,
                                  self.day).strftime("%W"))
            begin = int(
                datetime.datetime(year, self.month + 9, 1).strftime("%W"))
            return np.eye(6)[end - begin]

        def getfestival(self):
            '''

            :return: same festival entity ->one-hot result
            '''
            nr = np.zeros((12, ))
            if self.daynow == 10:
                nr[0] = 1
            if self.daynow >= 13 and self.daynow <= 15:
                nr[1] = 1
            if self.daynow == 16:
                nr[2] = 1
            if self.daynow == 29:
                nr[3] = 1
            if self.daynow >= 31 and self.daynow <= 37:
                nr[4] = 1
            if self.daynow == 37:
                nr[5] = 1
            if self.daynow == 42:
                nr[6] = 1
            if self.daynow == 58:
                nr[7] = 1
            if self.daynow == 69:
                nr[8] = 1
            if self.daynow == 72:
                nr[9] = 1
            if self.daynow == 79:
                nr[10] = 1
            if self.daynow == 89:
                nr[11] = 1
            return nr

        def getemb(self):
            '''

            :return: one-hot result
            '''
            nrweekday = self.getweekday()
            nrweekend = self.getifweekend()
            nrmonth = self.getmonth()
            self.getday()
            nrweekofmonth = self.getweekofmonth()
            nrfestival = self.getfestival()
            return np.hstack(
                [nrweekday, nrweekend, nrmonth, nrweekofmonth, nrfestival])

    days_emb = {}
    for dayi, indexi in id_list_dict['time_list']['key2index'].items():
        stremb = stemb = strTimeEmb(dayi)
        days_emb[dayi] = stremb.getemb()
    days_emb_save = {}
    days_emb_save['word_emb_dict'] = days_emb
    author = 'AZ'
    method = 'timeemb'
    sentence_id = 'user_id'
    word_id = 'time'
    Cache.cache_data(
        days_emb_save,
        nm_marker=f'EMB_DICT_{author}_{method}_{sentence_id}_{word_id}')
