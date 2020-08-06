"""
model_cnn_lstm.py
# az model
"""
import sys
sys.path.append("../../")
import gc
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from txbase import Cache,reduce_mem,show_all_feas,rm_feas, get_cur_dt_str
from txbase.attention import Attention
from collections import OrderedDict
from tensorflow.python.keras.layers import CuDNNLSTM, CuDNNGRU
import tensorflow.keras.backend as K
from tensorflow import keras
import random
import argparse
import pickle
from txbase.emb import EmbBatchLoader
import logging

##############################################################################
# 传参。例如：
# /root/miniconda3/envs/TF_2.2/bin/python -u run_by_fold.py --fold 4 --tm_now ${tm_now}
# 如果跑五折，--fold 如果不在 [0,1,2,3,4] 中，则全部五折会被一起训练。
##############################################################################

parser = argparse.ArgumentParser(description="kfold: 0,1,2,3,4...")
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--tm_now', type=str, default="19920911")
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--used_seq',
                    type=str,
                    default="id_list_dict_max_len_200_all"
                    )  # maxseqlen = 200 without oov preprocess
parser.add_argument('--pre_post', type=str, default="pre")
parser.add_argument('--seed', type=int, default=149)
parser.add_argument('--cvseed', type=int, default=2020)

args = parser.parse_args()
CUR_FOLD = args.fold
tm_now = args.tm_now.replace(":", "").replace("-", "")
EPOCHS = args.epochs
SEED = args.seed
CV_SEED = args.cvseed

random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(tf.__version__)
print(gpus)
tf.debugging.set_log_device_placement(True)
# distribution = tf.distribute.MirroredStrategy()
distribution = tf.distribute.experimental.CentralStorageStrategy()

##############################################################################
# 定义一些参数：
##############################################################################
isTEST = False

ID_LST_NM = args.used_seq
ID_LST_ZERO_PRE_POST = args.pre_post

ZQ_EMB_LST = [
    "ZQ_CBOW_MIN_CNT3_hs0_ng15_20WINDOW_10EPOCH",
    "ZQ_CBOW_MIN_CNT3_hs0_ng15_30WINDOW_10EPOCH",
    "AZ_cbow_CONCAT_time_diff_60WINDOW_10EPOCH",
    "ZQ_SG_MIN_CNT3_hs0_90WINDOW_10EPOCH", "AZ_timeemb"
]

LSTM_UNIT = 512
USE_SEQ_LENGTH = 170  # <= 最大序列长度
GlobalSeqLength = USE_SEQ_LENGTH
NUM_WORKERS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
LABEL_CLASS = {'age': 10, 'gender': 2, 'age_gender': 20}
CUR_LABEL = 'age_gender'  # 'age_gender'  # 'age'
NUM_CLASSES = LABEL_CLASS[CUR_LABEL]
N_FOLDS = 10
BATCH_SIZE = 512 * NUM_WORKERS
VERBOSE = 2
USE_TRAINED_CKPT = False  # 是否用ckpt做预测
CKPT_BASE_DIR = "../MODELS/"
INPUT_DATA_BASE_DIR = "../../cached_data/"  # : 复赛数据
AUTHOR = "AZ"  # 用的谁的输入序列 需要标记一下 方便ensemble
YOUR_MARKER = f"2020_DataAug_CNN_LSTM_SEED{SEED}"
# 【断点训练】：如果是断点训练，必须设置 tm_now 保持一致即可，其他的别改

TRAIN_MARKER = f"{tm_now}_{CUR_LABEL}_{YOUR_MARKER}_{AUTHOR}"  # Important!

# 去掉 product_id
EMB_keys2do = [
    'creative_id', 'ad_id', 'product_id', 'advertiser_id', 'product_category',
    'industry', "time"
]

TRAINABLE_DICT = {col: False for col in EMB_keys2do}
# tf2.0 中解冻这几个层会有bug
# TRAINABLE_DICT['industry'] = True
# TRAINABLE_DICT['product_category'] = True

##############################################################################

print("###" * 35)
print(f"@@@ SEED: {SEED}")
print(f"@@@ CV_SEED: {CV_SEED}")
print(f"@@@ CUR_FOLD to train: {CUR_FOLD}/{N_FOLDS}")
print(f"@@@ tm_now: {tm_now}")
print(f"@@@ ID_LST_NM: {ID_LST_NM}")
print(f"@@@ ID_LST_ZERO_PRE_POST: {ID_LST_ZERO_PRE_POST}")
print(f"@@@ ZQ_EMB_LST: {ZQ_EMB_LST}")
print("@@@ TRAIN_MARKER: ", TRAIN_MARKER)
print("@@@ CUR_LABEL: ", CUR_LABEL)
print("@@@ EPOCHS: ", EPOCHS)
print("@@@ NUM_WORKERS: ", NUM_WORKERS)
print("@@@ Cards to use:", os.environ["CUDA_VISIBLE_DEVICES"])
print("@@@ BATCH_SIZE: ", BATCH_SIZE)
print("@@@ EMB_keys2do: ", EMB_keys2do)
print("@@@ NUM_CLASSES: ", NUM_CLASSES)
print("@@@ USE_SEQ_LENGTH: ", USE_SEQ_LENGTH)
print("@@@ INPUT_DATA_BASE_DIR:", INPUT_DATA_BASE_DIR)
print("@@@ TRAINABLE_DICT:", TRAINABLE_DICT)
print("###" * 35)

##############################################################################


def load_idlist(id_list_nm='id_list_dict', zero_pre_post='pre'):
    """
    zero_pre_post: "pre"表示序列开头填充0，"post"表示序列尾部填充0
    """
    # id_list_dict: 包含padding后的序列特征字典以及词表
    id_list_dict = Cache.reload_cache(file_nm=id_list_nm,
                                      base_dir=INPUT_DATA_BASE_DIR,
                                      pure_nm=True)
    # truncate:
    if USE_SEQ_LENGTH < 200:
        if zero_pre_post == 'pre':  # 前面填充0，从后序开始截断：-USE_SEQ_LENGTH:
            for col in EMB_keys2do:
                id_list_dict[col + "_list"]['id_list'] = id_list_dict[
                    col + "_list"]['id_list'][:, -USE_SEQ_LENGTH:]

        elif zero_pre_post == 'post':  # 后面填充0，从前序开始截断：0:USE_SEQ_LENGTH
            for col in EMB_keys2do:
                id_list_dict[col + "_list"]['id_list'] = id_list_dict[
                    col + "_list"]['id_list'][:, 0:USE_SEQ_LENGTH]
        else:
            raise NotImplementedError

    KEY2INDEX_DICT = {}  # 每个序列特征的词表组成的字典
    SEQ_LENTH_DICT = {}  # 存放每个序列截断长度的字典 一般都是一样的，比如这里是 150

    for key in EMB_keys2do:
        KEY2INDEX_DICT[key] = id_list_dict[f'{key}_list']['key2index']
        SEQ_LENTH_DICT[key] = id_list_dict[f'{key}_list']['id_list'].shape[-1]

    if len(set(SEQ_LENTH_DICT.values())) == 1:
        print("GlobalSeqLength:", SEQ_LENTH_DICT[key])
    else:
        print(
            "GlobalSeqLength is Not Unique!!! If you are sure, comment the line after to avoid exception."
        )
        raise

    input_dict_all = {}
    for col in EMB_keys2do:
        input_dict_all[col] = id_list_dict[col + '_list']['id_list']

    return input_dict_all, KEY2INDEX_DICT


def load_datalabel():
    '''
    :return: train datalabel and matrix to save modelresult
    '''
    datalabel = Cache.reload_cache(file_nm='datalabel_with_seq_length',
                                   base_dir=INPUT_DATA_BASE_DIR,
                                   pure_nm=True)
    if datalabel['age'].min() == 1:
        datalabel['age'] = datalabel['age'] - 1
    if datalabel['gender'].min() == 1:
        datalabel['gender'] = datalabel['gender'] - 1
    assert datalabel['age'].min() == 0
    assert datalabel['gender'].min() == 0

    datalabel = datalabel[['user_id', 'gender', 'age']]
    traindata = datalabel.loc[~datalabel['age'].isna()].reset_index(drop=True)
    testdata = datalabel.loc[datalabel['age'].isna()].copy().reset_index(
        drop=True)

    traindata['age'] = traindata['age'].astype(np.int8)
    traindata['gender'] = traindata['gender'].astype(np.int8)
    traindata['age_gender'] = traindata['gender'] * 10 + traindata['age']
    # gender = 0, age => 0~9
    # gender = 1, age+=10 => 10~19
    print(
        f"traindata['age_gender'].unique(): {sorted(traindata['age_gender'].unique())}"
    )
    print(traindata.shape, testdata.shape)

    # init array to store oof and model prob
    train_shape = traindata.shape[0]
    test_shape = testdata.shape[0]
    model_prob = np.zeros((train_shape + test_shape, NUM_CLASSES, N_FOLDS),
                          dtype='float32')

    all_uid_df = datalabel[['user_id']].copy()  # to save the model_prob
    train_uid_df = traindata[['user_id']].copy()  # to save the oof_prob

    if not isTEST:
        os.makedirs(f"../../05_RESULT/META/{TRAIN_MARKER}", exist_ok=True)
        os.makedirs("../../05_RESULT/SUB", exist_ok=True)
        all_uid_df.to_csv(
            f"../../05_RESULT/META/{TRAIN_MARKER}/SAVE_all_uid_df.csv",
            index=False)
        train_uid_df.to_csv(
            f"../../05_RESULT/META/{TRAIN_MARKER}/SAVE_train_uid_df.csv",
            index=False)
    return traindata, model_prob


def _load_embs(zq_emb_lst=None, az_emb_lst=None):
    '''
    :param zq_emb_lst: embedding files group name
    :param az_emb_lst: all embedding in same path so useless
    :return:
    '''
    ALL_EMB_MATRIX_READY_DICT = {}
    # #################################################################################
    # LOAD ZQ EMB
    # #################################################################################
    EBL = EmbBatchLoader(all_emb_cols=EMB_keys2do, key2index=KEY2INDEX_DICT)
    for ii, i_emb in enumerate(zq_emb_lst, 1):
        ALL_EMB_MATRIX_READY_DICT[
            f'emb_matrix_ready_dict_{ii}'] = EBL.get_batch_emb_matrix(
                marker=i_emb)
    return ALL_EMB_MATRIX_READY_DICT


def print_emb_matrix_dict_info(all_emb):
    '''
    :param all_emb: embeddin file stats info
    :return:
    '''
    for k, v in all_emb.items():
        print("-" * 116)
        print(f"● {k}:  {list(v.keys())}")
        print("-" * 116)
        for ik, iv in v.items():
            print(f"  {ik:20}: {iv.shape}")


def concat_byid_emb(zq_emb_lst, az_emb_lst=None):
    """
    分IDconcat
    all_emb_dict: {'emb_nm1':{'creative_id':xxx, 'ad_id':yyy ...},
              'emb_nm2':{'creative_id':xxx, 'ad_id':yyy ...},
              ......
              }
    """
    all_emb_dict = _load_embs(zq_emb_lst, az_emb_lst)
    print_emb_matrix_dict_info(all_emb_dict)
    concated_id_emb_dict = defaultdict(list)
    for _, emb_dict in all_emb_dict.items():
        for id_nm, id_emb in emb_dict.items():
            concated_id_emb_dict[id_nm].append(id_emb)

    for id_nm, emb_lst in concated_id_emb_dict.items():
        concated_id_emb_dict[id_nm] = np.concatenate(emb_lst,
                                                     axis=1)  # 水平concat
        print(f"●{id_nm}: {concated_id_emb_dict[id_nm].shape}")
    return concated_id_emb_dict


def get_seq_input_layers(cols):
    '''
    :param cols: get keras input layers
    :return:
    '''
    print("Prepare input layer:", cols)
    inputs_dict = OrderedDict()
    for col in cols:
        inputs_dict[col] = keras.Input(shape=(GlobalSeqLength, ),
                                       dtype="int32",
                                       name=col)
    return inputs_dict


def get_emb_layer(emb_matrix, seq_length=None, trainable=False):
    '''
    :param emb_matrix: pretrain embedding matrix as the weight of embedding layer
    :param seq_length: max seqlen
    :param trainable: if freeze the embedding layer
    :return:
    '''
    if seq_length is None:
        seq_length = GlobalSeqLength  # 如果未指定 seq_length 就用 GlobalSeqLength
    embedding_dim = emb_matrix.shape[-1]
    input_dim = emb_matrix.shape[0]
    emb_layer = keras.layers.Embedding(input_dim,
                                       embedding_dim,
                                       input_length=seq_length,
                                       weights=[emb_matrix],
                                       dtype="float32",
                                       trainable=trainable)
    return emb_layer


def get_callbacks(count):
    '''
    :param count:
    :return:
    '''
    earlystop_callback = keras.callbacks.EarlyStopping(
        monitor="val_acc",
        min_delta=0.00001,
        patience=3,
        verbose=1,
        mode="max",
        baseline=None,
        restore_best_weights=True,
    )
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_acc',
                                                           factor=0.25,
                                                           patience=1,
                                                           min_delta=2e-4,
                                                           min_lr=2e-5)
    callbacks = [earlystop_callback, reduce_lr_callback]
    return callbacks


# #################################################################################
# MODEL
# #################################################################################
def create_model_AZ_cnn_lstm(rnn_unit, concated_id_emb_dict):
    '''
    :param rnn_unit: cudnnlstm units
    :param concated_id_emb_dict: col input to use
    :return:
    '''
    # input layers no mask add
    inputs_dict = get_seq_input_layers(cols=EMB_keys2do)
    inputs_all = list(inputs_dict.values())
    # feature filter conv setting
    conv1d_info_dict = {
        'creative_id': 256,
        'ad_id': 128,
        'advertiser_id': 128,
        'industry': 64,
        'product_category': 64,
        'product_id': 128,
        'time': 32,
        'click_times': -1
    }
    layers2concat = []
    for id_nm, emb_matrix in concated_id_emb_dict.items():
        if id_nm != 'click_times':
            print(id_nm, 'get embedding!')
            emb_layer = get_emb_layer(emb_matrix,
                                      trainable=TRAINABLE_DICT[id_nm])
            x = emb_layer(inputs_dict[id_nm])
            if conv1d_info_dict[id_nm] > -1:
                cov_layer = keras.layers.Conv1D(
                    filters=conv1d_info_dict[id_nm],
                    kernel_size=1,
                    activation='relu')
                x = cov_layer(x)
            layers2concat.append(x)
    # embedding all connected
    concat_emb = keras.layers.concatenate(layers2concat)
    # 4 route cnn
    xconv1 = keras.layers.Conv1D(filters=x.shape[-1].value,
                                 kernel_size=1,
                                 padding='same',
                                 activation='relu')(concat_emb)
    xconv2 = keras.layers.Conv1D(filters=256,
                                 kernel_size=1,
                                 padding='same',
                                 activation='relu')(concat_emb)
    xconv3 = keras.layers.Conv1D(filters=128,
                                 kernel_size=1,
                                 padding='same',
                                 activation='relu')(concat_emb)
    xconv4 = keras.layers.Conv1D(filters=64,
                                 kernel_size=1,
                                 padding='same',
                                 activation='relu')(concat_emb)
    concat_emb = keras.layers.concatenate([xconv1, xconv2, xconv3, xconv4])
    # pre lstm&cnn structure
    xprelstm = keras.layers.Bidirectional(
        CuDNNLSTM(rnn_unit, return_sequences=True))(concat_emb)
    xpreconv = keras.layers.Conv1D(filters=256,
                                   kernel_size=1,
                                   padding='same',
                                   activation='relu')(concat_emb)
    xconcat = keras.layers.concatenate([xprelstm, xpreconv])
    # last lstm&cnn structure
    xlastlstm = keras.layers.Conv1D(filters=256,
                                    kernel_size=1,
                                    padding='same',
                                    activation='relu')(xconcat)
    xlastconv = keras.layers.Bidirectional(
        CuDNNLSTM(rnn_unit, return_sequences=True))(xconcat)
    # poolings
    # max&mean
    xmaxpoolconv = keras.layers.GlobalMaxPooling1D()(xlastconv)
    xmaxpoollstm = keras.layers.GlobalMaxPooling1D()(xlastlstm)
    xmeanpoolconv = keras.layers.GlobalAveragePooling1D()(xlastconv)
    xmeanpoollstm = keras.layers.GlobalAveragePooling1D()(xlastlstm)
    concat_emb = keras.layers.concatenate(
        [xmaxpoolconv, xmaxpoollstm, xmeanpoolconv, xmeanpoollstm])
    concat_all = keras.layers.Dropout(0.3)(concat_emb)
    # Dense layers
    x = keras.layers.Dense(512)(concat_all)
    x = keras.layers.PReLU()(x)
    x = keras.layers.Dense(512)(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.Dense(256)(x)
    x = keras.layers.PReLU()(x)
    # 20 classes
    outputs_all = keras.layers.Dense(NUM_CLASSES,
                                     activation='softmax',
                                     name='age_gender')(x)
    model = keras.Model(inputs_all, outputs_all)
    model.summary()
	# return compiled model tf 2.0
	model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                loss='sparse_categorical_crossentropy',
                metrics=['acc'])
    return model


def get_input_by_index(input_dict_all, idx_lst, do_aug=False):
    '''
    train test valid split
    :param input_dict_all:
    :param idx_lst:
    :param do_aug:
    :return:
    '''
    input_dict_trn = {}
    for col in EMB_keys2do:
        input_dict_trn[col] = input_dict_all[col][idx_lst]
    if do_aug:  # 数据增强：
        for key, arr in input_dict_trn.items():
            print("---" * 30)
            print(f"@@@ process: {key} ...")
            print(f"@@@ shape before: {input_dict_trn[key].shape}")
            arr_reverse = []
            for row in range(arr.shape[0]):
                rowi = arr[row, :]
                arr_reverse.append(rowi[::-1][:np.sum(rowi > 0)])
            arr_reverse = pad_sequences(arr_reverse,
                                        maxlen=USE_SEQ_LENGTH,
                                        padding='pre',
                                        truncating='pre')
            input_dict_trn[key] = np.vstack([arr, arr_reverse])
            print(f"@@@ shape after: {input_dict_trn[key].shape}")
            print("---" * 30)
    return input_dict_trn


def get_reverse_input_dict(input_dict_all):
    '''
    data augment reverse
    :param input_dict_all:
    :return:
    '''
    input_dict_all_reverse = {}
    for key, arr in input_dict_all.items():
        print("---" * 30)
        print(f"@@@ process: {key} ...")
        arr_reverse = []
        for row in range(arr.shape[0]):
            rowi = arr[row, :]
            arr_reverse.append(
                rowi[::-1][:np.sum(rowi > 0)])  # reverse the sequence
        #  still use pre padding
        arr_reverse = pad_sequences(arr_reverse,
                                    maxlen=USE_SEQ_LENGTH,
                                    padding='pre',
                                    truncating='pre')
        input_dict_all_reverse[key] = arr_reverse
        print("---" * 30)
    return input_dict_all_reverse


def shuffle_xy(input_dict_trn, y_true_trn):
    '''
    shuffle data
    :param input_dict_trn:
    :param y_true_trn:
    :return:
    '''
    shuffled_idx = np.random.permutation(len(y_true_trn))
    y_true_trn = y_true_trn[shuffled_idx]
    input_dict_trn_sfd = {}
    for key, arr in input_dict_trn.items():
        input_dict_trn_sfd[key] = arr[shuffled_idx]
    return input_dict_trn_sfd, y_true_trn


if __name__ == "__main__":
    print("###" * 35)
    print("@@@Load id_list_dict...")
    print("###" * 35)

    input_dict_all, KEY2INDEX_DICT = load_idlist(
        id_list_nm=ID_LST_NM, zero_pre_post=ID_LST_ZERO_PRE_POST)
    # #################################################################################
    print("###" * 35)
    print("@@@Load datalabel...")
    print("###" * 35)
    traindata, model_prob = load_datalabel()
    # #################################################################################
    print("###" * 35)
    print("@@@Load Embedding...")
    print('ALL_EMB_COLS_TO_USE:', EMB_keys2do)
    print("###" * 35)
    # #################################################################################

    CONCATED_ID_EMB_DICT = concat_byid_emb(zq_emb_lst=ZQ_EMB_LST)

    # #################################################################################
    # 10折开始啦~
    # #################################################################################
    print("###" * 35)
    print(f"{N_FOLDS} Fold Training Start...")
    print("###" * 35)
    score_val = []
    skf = StratifiedKFold(n_splits=N_FOLDS, random_state=CV_SEED, shuffle=True)
    folds = list(skf.split(traindata, traindata[CUR_LABEL]))
    if not isTEST:
        with open(f"../../05_RESULT/META/{TRAIN_MARKER}/folds.pkl",
                  'wb') as file:
            pickle.dump(folds, file)
    for count, (train_index, test_index) in enumerate(folds):
        print("###" * 35)
        if (CUR_FOLD >= 0) and (CUR_FOLD < N_FOLDS):
            if count < CUR_FOLD:
                print("Skip...")
                continue
            if count > CUR_FOLD:
                print("Break...")
                break
        else:
            print(
                f"CUR_FOLD={CUR_FOLD}, Full 5 folds will be trained in this loop..."
            )
        print("FOLD | ", count)
        print("###" * 35)
        do_aug = True
        input_dict_trn = get_input_by_index(input_dict_all,
                                            train_index,
                                            do_aug=do_aug)
        y_true_trn = traindata[CUR_LABEL].values[train_index]
        if do_aug:
            y_true_trn = np.hstack([y_true_trn, y_true_trn])
        # shuffle...
        print("@@@ do shuffle_xy ...")
        input_dict_trn, y_true_trn = shuffle_xy(input_dict_trn, y_true_trn)
        input_dict_val = get_input_by_index(input_dict_all, test_index)
        y_true_val = traindata[CUR_LABEL].values[test_index]
        del input_dict_all
        gc.collect()
		try:
			del model
			gc.collect()
			K.clear_session()
		except:
			pass
		with distribution.scope():
			model = create_model_AZ_cnn_lstm(rnn_unit=LSTM_UNIT, concated_id_emb_dict=CONCATED_ID_EMB_DICT)
        # save memory but need run each fold at each time
        # del CONCATED_ID_EMB_DICT
        # gc.collect()
        print("###" * 35)
        print("Train from Scratch...")
        print("###" * 35)
        callbacks = get_callbacks(count)
        hist = model.fit(input_dict_trn,
                         y_true_trn,
                         epochs=EPOCHS,
                         batch_size=BATCH_SIZE,
                         verbose=VERBOSE,
                         callbacks=callbacks,
                         validation_data=(input_dict_val, y_true_val))
        print("###" * 35)
        print(hist.history)
        print("###" * 35)
        score_val.append(np.max(hist.history["val_acc"]))
        del input_dict_trn, input_dict_val
        gc.collect()
        print("###" * 35)

        # #################################################################################
        # Original Seq
        # #################################################################################

        print(f"Make Prediction...Fold-{count}")
        input_dict_all, _ = load_idlist(id_list_nm=ID_LST_NM,
                                        zero_pre_post=ID_LST_ZERO_PRE_POST)
        pred_all = model.predict(input_dict_all,
                                 batch_size=BATCH_SIZE,
                                 verbose=VERBOSE)
        pred_all = np.float32(pred_all)
        model_prob[:, :, count] = pred_all
        np.save(
            f"../../05_RESULT/META/{TRAIN_MARKER}/SAVE_MODEL_PROB_FOLD{count}",
            pred_all)

        # #################################################################################
        # Reversed
        # #################################################################################

        print(f"Make Reverse Prediction...Fold-{count}")
        input_dict_all = get_reverse_input_dict(input_dict_all)
        pred_all = model.predict(input_dict_all,
                                 batch_size=BATCH_SIZE,
                                 verbose=VERBOSE)
        pred_all = np.float32(pred_all)
        model_prob[:, :, count] = pred_all
        np.save(
            f"../../05_RESULT/META/{TRAIN_MARKER}/SAVE_MODEL_PROB_FOLD{count}_Reversed",
            pred_all)
        print("Done Prediction!")
        print("###" * 35)
    print(f"offline score mean: {score_val}")
    print(f"offline score by folds: {np.mean(score_val)}")
    print("All Done!")
    # we do the training with data augment
    # wo do the predicting of all train,test samples twice, ordered and reversed