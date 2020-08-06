from . import Cache
import numpy as np


class EmbBatchLoader:
    def __init__(self,
                 all_emb_cols,
                 emb_base_dir=None,
                 key2index=None,
                 outer_emb=False):
        """
        outer_emb: 设置该参数为True，如果是导入外部的embedding，
            会造成key2index和word_emb_dict对不齐，词表不一样。
            默认会认为用最大的值来填充低频词。
        """
        self.all_emb_cols = all_emb_cols
        self.all_emb_cols_backup = all_emb_cols
        self.emb_base_dir = emb_base_dir
        self.key2index = key2index
        self.outer_emb = outer_emb

    def _get_max_index(self, word_emb_dict):
        '''
        原先功能定位oov = max_index 后改为 =-1 此函数弃用
        :param word_emb_dict:
        :return:
        '''
        return str(sorted(map(int, list(word_emb_dict.keys())))[-1])

    def get_emb_matrix(self, word_emb_dict, key2index_col):  # modify by zlh
        """
        prepare embedding for NN
        initializing the embedding... id => emb vectors
        the id is your own label encoding mapping...which stored in the self.key2index[col]
        """
        if self.outer_emb:
            # self._get_max_index(word_emb_dict)  # 阿郑的是“max“为低频词
            key_to_represent_rare = '-1'
        else:
            key_to_represent_rare = '-1'  # 我的是”-1“为低频词
        for _, k in word_emb_dict.items():
            break
        emb_size = k.shape[0]
        voc_size = len(key2index_col)
        # 真实的词表，编码始于1，你准备input sequence的时候编码的词表，GetSeqFeas.ipynb
        # 100个词，编码就是1-100，所以初始化要+1
        emb_matrix = np.zeros((voc_size + 1, emb_size))  # 0如何优化，mean?
        # emb 中必须要有'-1'
        if '-1' not in word_emb_dict.keys():
            #  emb中无-1 为全词表数据！需要自行计算均值emb vec
            # 为embi 添加一个embedding
            # 求词表与embi key的差
            set_drop_words = list(
                set(word_emb_dict.keys()).difference(set(
                    key2index_col.keys())))
            if len(set_drop_words) > 0:
                # 这些词的vector求均值作为这个oov词的embedding vector
                vector_low_frequency_words = np.zeros((emb_size, ))
                for w in set_drop_words:
                    vector_low_frequency_words += word_emb_dict[w]
                vector_low_frequency_words = vector_low_frequency_words / len(
                    set_drop_words)
                # emb添加一个key value
                word_emb_dict['-1'] = vector_low_frequency_words
                print(' file has ' + str(len(set_drop_words)) + \
                      ' low frequency words and fill vector as:', vector_low_frequency_words)
        for k, idx in key2index_col.items():
            try:
                emb_matrix[idx, :] = word_emb_dict[k]
            except KeyError:  # 如果k不在不在word_emb_dict中，则默认用max_key_to_represent_rare填充
                #                 print('find oov:',(k, idx))
                emb_matrix[idx, :] = word_emb_dict[key_to_represent_rare]
        emb_matrix = np.float32(emb_matrix)
        return emb_matrix

    def load_batch_embedding(self, emb_base_name, pure_nm):
        """
        批量导入embedding，目前对于一组embedding就self.all_emb_cols个变量
        """
        emb_dict = {}
        for col in self.all_emb_cols:
            file_nm = F'{emb_base_name}_{col}'
            try:
                emb_dict[col] = Cache.reload_cache(
                    file_nm=file_nm,
                    pure_nm=pure_nm,
                    base_dir=self.emb_base_dir)['word_emb_dict']
            except FileNotFoundError as e:
                print("[Error]" + " = =" * 30)
                print("ErrorMessage: ", e)
                print("col: ", col)
                print("file_nm:", file_nm)
                print("[Error]" + " = =" * 30)
        print(f"Raw self.all_emb_cols: {self.all_emb_cols}")
        self.all_emb_cols = list(emb_dict.keys())
        print(f"Updated self.all_emb_cols: {self.all_emb_cols}")
        return emb_dict

    def load_emb_dict_with_raw_embs(self,
                                    marker=None,
                                    emb_base_name=None,
                                    sentence_id='user_id',
                                    pure_nm=True):

        if emb_base_name is None:
            if marker is None:
                raise ValueError(
                    "marker can't be None if emb_base_name is None!!")
            else:
                if marker.endswith("_advertiser_id") or marker.endswith(
                        "_user_id"):
                    # marker中包括了sentence_id，目前sentence_id只有_advertiser_id和_user_id
                    emb_base_name = F'EMB_DICT_{marker}'
                else:
                    # marker中不包括sentence_id，需要添加
                    emb_base_name = F'EMB_DICT_{marker}_{sentence_id}'
        else:
            emb_base_name = emb_base_name.rstrip('_')  # 对于一组embedding一致的名称
        emb_dict_with_raw_embs = self.load_batch_embedding(
            emb_base_name, pure_nm)
        return emb_dict_with_raw_embs

    def get_batch_emb_matrix(self,
                             marker=None,
                             emb_base_name=None,
                             sentence_id='user_id',
                             pure_nm=True):

        emb_dict_with_raw_embs = self.load_emb_dict_with_raw_embs(
            marker=marker,
            emb_base_name=emb_base_name,
            sentence_id=sentence_id,
            pure_nm=pure_nm)
        emb_matrix_ready_dict = {}
        for col in self.all_emb_cols:
            emb_matrix_ready_dict[col] = self.get_emb_matrix(
                emb_dict_with_raw_embs[col], key2index_col=self.key2index[col])
        print("-" * 6)
        print("Done!")
        # restore all_emb_cols to all_emb_cols_backup
        self.all_emb_cols = self.all_emb_cols_backup
        return emb_matrix_ready_dict

    def get_batch_emb_matrix_by_absolute_path(self,
                                              absolute_path_with_placeholder):
        emb_matrix_ready_dict = {}
        for col in self.all_emb_cols:
            path = absolute_path_with_placeholder.format(col)
            try:
                i_raw_embs = Cache.reload_cache(
                    file_nm=path, base_dir=self.emb_base_dir)['word_emb_dict']
                emb_matrix_ready_dict[col] = self.get_emb_matrix(
                    i_raw_embs, key2index_col=self.key2index[col])
            except FileNotFoundError as e:
                print("[Error]" + " = =" * 30)
                print("ErrorMessage: ", e)
                print("col: ", col)
                print("file_nm:", path)
                print("[Error]" + " = =" * 30)
        print(f"Raw self.all_emb_cols: {self.all_emb_cols}")
        self.all_emb_cols = list(emb_matrix_ready_dict.keys())
        print(f"Updated self.all_emb_cols: {self.all_emb_cols}")
        print("-" * 6)
        print("Done!")
        # restore all_emb_cols to all_emb_cols_backup
        self.all_emb_cols = self.all_emb_cols_backup
        return emb_matrix_ready_dict
