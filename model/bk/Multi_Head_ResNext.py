#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append('../../')
import time
import pickle
import random
import pandas as pd
from pandas.core.frame import DataFrame
from txbase import Cache
from tqdm import tqdm
import numpy as np
import gc
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

##用于模型可复现
seed_num = 1111
torch.manual_seed(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.backends.cudnn.deterministic = True

inputlist = [
    'creative_id', 'ad_id', 'advertiser_id', 'product_id', 'product_category',
    'industry', 'time'
]

train_able_dict = ['creative_id', 'ad_id', 'advertiser_id', 'product_id']

##超参数
BATCH_SIZE = 512
SEQ_LENGTH = 150
DROPOUT = 0.3
NUM_CLASS = 20
EPOCHS = 30
LR = 1e-3

device = torch.device("cuda:0")
device_ids = [0]

##############################
######## 获取emb #############
##############################

seq_length_creative_id = 150  # 序列都padding到了150
id_list_dict = Cache.reload_cache(
    file_nm='../../cached_data/CACHE_id_list_dict_150_normal.pkl',
    base_dir='',
    pure_nm=False)

# 定义需要的输入
cols_to_emb = [
    'creative_id', 'ad_id', 'advertiser_id', 'product_id', 'product_category',
    'industry', 'time'
]
# 定义emb 文件路径
path_list = ['../../cached_data/']
# 定义最大emb_size
max_embs = {
    'creative_id': 2000,
    'ad_id': 2000,
    'advertiser_id': 2000,
    'product_id': 2000,
    'product_category': 2000,
    'industry': 2000,
    'click_times': 600,
    'time': 600
}
# 定义随机抽几个emb
max_nums = {
    'creative_id': 3,
    'ad_id': 3,
    'advertiser_id': 3,
    'product_id': 3,
    'product_category': 2,
    'industry': 3,
    'click_times': 3,
    'time': 2
}
# 定义必须要用的emb
special_userlist = {
    'creative_id': [
        '../../cached_data/CACHE_EMB_DICT_ZQ_SG_RM_CNT1_30WINDOW_10EPOCH_user_id_creative_id.pkl',
        '../../cached_data/CACHE_EMB_DICT_AZ_CBOW_CLICK_TIMES_INCREASED_60WINDOW_10EPOCH_user_id_creative_id.pkl',
        '../../cached_data/CACHE_EMB_DICT_ZQ_SG_MIN_CNT3_hs0_10WINDOW_10EPOCH_user_id_creative_id.pkl'
    ],
    'ad_id': [
        '../../cached_data/CACHE_EMB_DICT_ZQ_SG_RM_CNT1_30WINDOW_10EPOCH_user_id_ad_id.pkl',
        '../../cached_data/CACHE_EMB_DICT_AZ_CBOW_CLICK_TIMES_INCREASED_60WINDOW_10EPOCH_user_id_ad_id.pkl',
        '../../cached_data/CACHE_EMB_DICT_ZQ_SG_MIN_CNT3_hs0_10WINDOW_10EPOCH_user_id_ad_id.pkl'
    ],
    'advertiser_id': [
        '../../cached_data/CACHE_EMB_DICT_ZQ_SG_RM_CNT1_30WINDOW_10EPOCH_user_id_advertiser_id.pkl',
        '../../cached_data/CACHE_EMB_DICT_AZ_CBOW_CLICK_TIMES_INCREASED_60WINDOW_10EPOCH_user_id_advertiser_id.pkl',
        '../../cached_data/CACHE_EMB_DICT_ZQ_SG_MIN_CNT3_hs0_10WINDOW_10EPOCH_user_id_advertiser_id.pkl'
    ],
    'product_id': [
        '../../cached_data/CACHE_EMB_DICT_AZ_CBOW_CLICK_TIMES_INCREASED_60WINDOW_10EPOCH_user_id_product_id.pkl',
        '../../cached_data/CACHE_EMB_DICT_ZQ_SG_MIN_CNT3_hs0_10WINDOW_10EPOCH_user_id_product_id.pkl',
        '../../cached_data/CACHE_EMB_DICT_ZQ_SG_MIN_CNT3_hs0_30WINDOW_10EPOCH_user_id_product_id.pkl'
    ],
    'product_category': [
        '../../cached_data/CACHE_EMB_DICT_AZ_CBOW_CLICK_TIMES_INCREASED_60WINDOW_10EPOCH_user_id_product_category.pkl',
        '../../cached_data/CACHE_EMB_DICT_ZQ_CBOW_HS1_100WINDOW_10EPOCH_user_id_product_category.pkl'
    ],
    'industry': [
        '../../cached_data/CACHE_EMB_DICT_AZ_CBOW_CLICK_TIMES_INCREASED_60WINDOW_10EPOCH_user_id_industry.pkl',
        '../../cached_data/CACHE_EMB_DICT_ZQ_SG_MIN_CNT3_hs0_10WINDOW_10EPOCH_user_id_industry.pkl',
        '../../cached_data/CACHE_EMB_DICT_ZQ_SG_MIN_CNT3_hs0_30WINDOW_10EPOCH_user_id_industry.pkl'
    ],
    'click_times': [],
    'time': [
        '../../cached_data/CACHE_EMB_DICT_AZ_CBOW_CLICK_TIMES_INCREASED_60WINDOW_10EPOCH_user_id_time.pkl',
        '../../cached_data/CACHE_EMB_DICT_AZ_CBOW_TXBASE_150WINDOW_10EPOCH_user_id_time.pkl'
    ],
}


class get_embedding_tool:
    '''
    用于读取embedding矩阵
    
    给定最大每个col 的embedding size {col:max_embs}
    给定最大每个col 所用到的embedding 文件个数 {col:max_nums}
    给定本次筛选用到的列[coli,colj]
    给定文件路径集合
    给定必须要用的emb字典
    example：
    cols_to_emb = ['creative_id','ad_id','advertiser_id','product_id','product_category','industry','click_times','time']
    path_list=['/home/zq/code_base/Tencent2020/zlh/cached_data/',
          '/home/zq/code_base/Tencent2020/zhangqibot/cached_data/']#
    '''
    def __init__(self, max_embs, max_nums, use_cols, path_list, spec_emb_dict):
        self.max_embs = max_embs
        self.max_nums = max_nums
        self.use_cols = use_cols
        self.path_list = path_list
        self.spec_emb_dict = spec_emb_dict

    def get_emb_matrix(self, word_emb_dict, key2index, len_words):
        '''
        word_emb_dict:每个词的vector
        key2index:词表：索引
        len_words:embed_matrix的行维度
        '''
        for _, k in word_emb_dict.items():
            break
        emb_size = k.shape[0]  # embedding_size
        emb_matrix = np.zeros((len_words, emb_size), dtype=np.float32)  # 总矩阵大小
        for k, idx in key2index.items():
            if k in word_emb_dict:
                emb_matrix[idx, :] = word_emb_dict[k]  # 填入
        return emb_matrix

    def get_batch_emb_matrix(self,
                             files,
                             emblist,
                             id_list_dict,
                             emb_name,
                             if_print=False,
                             max_embs=max_embs):
        '''
        emblist:所有的embs 来自各个文件夹，zq,zlh
        id_list_dict:zlh的词表
        emb_name:col+'_list'
        # len_word:这个col在zlh词表里的词汇量+1
        max_embs:最大emb size总合
        '''
        emb_matrix_all = []
        low_frequency_words = str(
            max(list(map(int, id_list_dict[emb_name]['key2index'].keys()))))
        print(emb_name, 'has low frequency words fill as :',
              low_frequency_words)
        len_word = len(id_list_dict[emb_name]['key2index'].keys()) + 1
        sum_embs = 0  # 当前的总emb_size
        for index, embi in enumerate(emblist):
            for _, k in embi.items():
                break
            emb_sizei = k.shape[0]
            sum_embs += emb_sizei
            # 判断是否是zq embedding 低频词在不在id_list_dict[emb_name]['key2index']里
            if low_frequency_words not in embi.keys():
                # 为embi 添加一个embedding
                # 求词表与embi key的差
                set_drop_words = list(
                    set(embi.keys()).difference(
                        set(id_list_dict[emb_name]['key2index'].keys())))
                if len(set_drop_words) > 0:
                    # 这些词的vector求均值
                    vector_low_frequency_words = np.zeros((emb_sizei, ))
                    for w in set_drop_words:
                        vector_low_frequency_words += embi[w]
                    vector_low_frequency_words = vector_low_frequency_words / len(
                        set_drop_words)
                    # emb添加一个key value
                    embi[low_frequency_words] = vector_low_frequency_words
                    if if_print:
                        print(
                            index, ' file has ' + str(len(set_drop_words)) +
                            ' low frequency words and fill vector as :',
                            vector_low_frequency_words)
                else:
                    if if_print:
                        print(
                            index,
                            ' file has no low_frequency words vector to fill!')
            # 添加完成后正常获取embvector
            emb_matrix = self.get_emb_matrix(
                embi, id_list_dict[emb_name]['key2index'], len_word)
            emb_matrix = emb_matrix.astype(np.float32)
            emb_matrix_all.append(emb_matrix)
            if sum_embs >= max_embs:
                print('reach max_embs !')
                print('now embs files:', files[:index + 1])
                break
        print(emb_name, 'has emb_matrix shape:', sum_embs, 'total nums:',
              len(emb_matrix_all))
        return emb_matrix_all

    def random_get_embedding_fun(self, id_list_dict):
        emb_matrix_dict = {}
        for col in self.use_cols:
            col_file_names = []
            sepc_embs = self.spec_emb_dict[col]  # 必须要用
            # 随机抽一些embedding 优先抽最大个数个 再在后续不断拼到dict中达到max_embs就停止
            # 文件名对应的表示是user_id_xx
            for indexpath, pathi in enumerate(self.path_list):
                for filei in os.listdir(pathi):
                    if filei.find('user_id_' + col) > -1:
                        col_file_names.append(pathi + filei)
            if len(sepc_embs) > 0:
                # 排它
                col_file_names = list(
                    set(col_file_names).difference(set(sepc_embs)))
            random.shuffle(col_file_names)
            select_nums = min(
                [len(col_file_names),
                 self.max_nums[col] - len(sepc_embs)])  # 再选入的个数
            file_to_load = col_file_names[:select_nums]  # 再选入的emb
            file_to_load = sepc_embs + file_to_load
            emblist = []
            for filei in file_to_load:
                try:
                    emb_i = Cache.reload_cache(file_nm=filei,
                                               base_dir='',
                                               pure_nm=False)['word_emb_dict']
                    emblist.append(emb_i)
                except:
                    print('missing! ', filei)
            print('processing {} shape {}'.format(col, len(emblist)))
            print(file_to_load)  # 选中的file
            emb_matrix_all = self.get_batch_emb_matrix(
                file_to_load,
                emblist,
                id_list_dict,
                col + '_list',
                max_embs=self.max_embs[col])  # id_list_dict 外部传入
            emb_matrix_dict[col] = emb_matrix_all  # 一个list
            del emb_matrix_all, emblist
            gc.collect()
        # key 是列名 value是一个list 里面有这个列所属的各种embedding矩阵 按照词表*emb_size的
        return emb_matrix_dict


gt = get_embedding_tool(max_embs=max_embs,
                        max_nums=max_nums,
                        use_cols=cols_to_emb,
                        path_list=path_list,
                        spec_emb_dict=special_userlist)
emb_matrix_dict = gt.random_get_embedding_fun(id_list_dict)

###############################
######## 定义模型 #############
###############################

Kernel_Sizes_List = [1, 3, 5, 7]  #4路Resnext的卷积尺寸


#From SENet, 通道（emb_size维度）的注意力机制
class se_layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(se_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResNeXtBlock(nn.Module):
    def __init__(self, input_size, output_size, kernelsize, padding, expansion,
                 num_groups, is_short_cut):
        super(ResNeXtBlock, self).__init__()
        places = output_size // expansion
        self.is_short_cut = is_short_cut
        self.se_layer = se_layer(channel=output_size, reduction=16)

        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels=input_size,
                      out_channels=places,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm1d(places),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=places,
                      out_channels=places,
                      kernel_size=kernelsize,
                      padding=padding,
                      bias=False,
                      groups=num_groups),
            nn.BatchNorm1d(places),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=places,
                      out_channels=output_size,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm1d(output_size),
        )

        self.relu = nn.ReLU(inplace=True)

        if is_short_cut:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels=input_size,
                          out_channels=output_size,
                          kernel_size=1,
                          padding=0), nn.BatchNorm1d(output_size))

        self.pooling_layer = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):

        residual = x
        out = self.bottleneck(x)
        out = self.se_layer(out)
        if self.is_short_cut:
            out += self.shortcut(residual)
        out = self.relu(out)
        out = self.pooling_layer(out)

        return out


def conv1x1_layer(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=1), nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True))


class TencentModel3(nn.Module):
    def __init__(self):
        super(TencentModel3, self).__init__()

        self.dropout = nn.Dropout(DROPOUT)

        self.conv_head_0 = conv1x1_layer(in_channels=556, out_channels=128)
        self.conv_head_1 = conv1x1_layer(in_channels=556, out_channels=128)
        self.conv_head_2 = conv1x1_layer(in_channels=556, out_channels=128)
        self.conv_head_3 = conv1x1_layer(in_channels=320, out_channels=128)
        self.conv_head_4 = conv1x1_layer(in_channels=24, out_channels=64)
        self.conv_head_5 = conv1x1_layer(in_channels=192, out_channels=128)
        self.conv_head_6 = conv1x1_layer(in_channels=40, out_channels=64)

        self.LSTM = nn.LSTM(input_size=768,
                            hidden_size=256,
                            bidirectional=True,
                            dropout=0.2,
                            num_layers=2,
                            batch_first=True)



        self.Res_conv_layer_1 = nn.Sequential(
                                              ResNeXtBlock(input_size =512 , output_size = 256, kernelsize =Kernel_Sizes_List[0], padding = 0 , \
                                                           expansion =2 , num_groups =32, is_short_cut = True),

                                             )

        self.Res_conv_layer_2 = nn.Sequential(
                                              ResNeXtBlock(input_size =512 , output_size = 256, kernelsize =Kernel_Sizes_List[1], padding = 1 , \
                                                           expansion =2 , num_groups =32, is_short_cut = True),
                                             )

        self.Res_conv_layer_3 = nn.Sequential(
                                              ResNeXtBlock(input_size =512 , output_size = 256, kernelsize =Kernel_Sizes_List[2], padding = 2 , \
                                                           expansion =2 , num_groups =32, is_short_cut = True),
                                             )

        self.Res_conv_layer_4 = nn.Sequential(
                                              ResNeXtBlock(input_size =512 , output_size = 256, kernelsize =Kernel_Sizes_List[3], padding = 3 , \
                                                           expansion =2 , num_groups =32, is_short_cut = True),
                                             )

        #1024，256
        self.fc_final = nn.Sequential(nn.Linear(1024, 256),
                                      nn.BatchNorm1d(256),
                                      nn.RReLU(inplace=True),
                                      nn.Linear(256, NUM_CLASS))

    def forward(self, emb_layer_mat):

        #每个emb layer concat 所有 emb dict

        emb_layer0 = torch.cat(
            (emb_layer_mat[0][0], emb_layer_mat[0][1], emb_layer_mat[0][2]),
            dim=2)

        emb_layer1 = torch.cat(
            (emb_layer_mat[1][0], emb_layer_mat[1][1], emb_layer_mat[1][2]),
            dim=2)

        emb_layer2 = torch.cat(
            (emb_layer_mat[2][0], emb_layer_mat[2][1], emb_layer_mat[2][2]),
            dim=2)

        emb_layer3 = torch.cat(
            (emb_layer_mat[3][0], emb_layer_mat[3][1], emb_layer_mat[3][2]),
            dim=2)

        emb_layer4 = torch.cat((emb_layer_mat[4][0], emb_layer_mat[4][1]),
                               dim=2)

        emb_layer5 = torch.cat(
            (emb_layer_mat[5][0], emb_layer_mat[5][1], emb_layer_mat[5][2]),
            dim=2)

        emb_layer6 = torch.cat((emb_layer_mat[6][0], emb_layer_mat[6][1]),
                               dim=2)

        emb_layer0_out = self.conv_head_0(emb_layer0.permute(0, 2, 1))
        emb_layer1_out = self.conv_head_1(emb_layer1.permute(0, 2, 1))
        emb_layer2_out = self.conv_head_2(emb_layer2.permute(0, 2, 1))
        emb_layer3_out = self.conv_head_3(emb_layer3.permute(0, 2, 1))
        emb_layer4_out = self.conv_head_4(emb_layer4.permute(0, 2, 1))
        emb_layer5_out = self.conv_head_5(emb_layer5.permute(0, 2, 1))
        emb_layer6_out = self.conv_head_6(emb_layer6.permute(0, 2, 1))
        concat_all = torch.cat(
            (emb_layer0_out, emb_layer1_out, emb_layer2_out, emb_layer3_out,
             emb_layer4_out, emb_layer5_out, emb_layer6_out),
            dim=1)
        #         #用于LSTM的多卡训练
        #         if not hasattr(self, '_flattened'):
        #             self.LSTM.flatten_parameters()
        #         setattr(self, '_flattened', True)

        concat_all, (hn, cn) = self.LSTM(concat_all.permute(0, 2, 1))

        concat_all = self.dropout(concat_all)

        concat_all = concat_all.permute(0, 2, 1)

        out1 = self.Res_conv_layer_1(concat_all)
        out2 = self.Res_conv_layer_2(concat_all)
        out3 = self.Res_conv_layer_3(concat_all)
        out4 = self.Res_conv_layer_4(concat_all)

        out_all = torch.cat((out1, out2, out3, out4), dim=1)
        out_reshape = out_all.view(out_all.size(0), -1)

        out_all = self.fc_final(out_reshape)

        return out_all


###############################
######## Dataset ##############
###############################


class TenCentDataset(Dataset):
    def __init__(self, input_data=None):
        super().__init__()
        self.data = input_data

    def __len__(self):
        return len(self.data)

    def remove_str(self, str_list):
        str_list = str_list[1:-1].split(',')
        np_list = [int(i) for i in str_list]
        return np_list

    def __getitem__(self, index):
        entry = self.data[index]
        creative_id = entry['creative_id_list']
        ad_id = entry['ad_id_list']
        product_id = entry['product_id_list']
        advertiser_id = entry['advertiser_id_list']
        industry = entry['industry_list']
        product_category = entry['product_category_list']
        time = entry['time_list']
        click_times = entry['click_times_list']
        user_id = entry['user_id']

        y_label = entry['label']

        creative_id = torch.LongTensor(self.remove_str(creative_id))
        ad_id = torch.LongTensor(self.remove_str(ad_id))
        advertiser_id = torch.LongTensor(self.remove_str(advertiser_id))
        product_id = torch.LongTensor(self.remove_str(product_id))
        product_category = torch.LongTensor(self.remove_str(product_category))
        industry = torch.LongTensor(self.remove_str(industry))
        time = torch.LongTensor(self.remove_str(time))

        return creative_id, ad_id, product_id, advertiser_id, industry, product_category, time, user_id, y_label


#############################################
######## 做分类Label Smoothing ##############
#############################################


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.05, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(
            log_preds, target, reduction=self.reduction)


###############################
######## 模型训练 ##############
###############################


class Tencent2020:
    def __init__(self, emb_layer, train_data, val_data, test_data, folds,
                 is_resume):
        # Model
        self.model = TencentModel3()

        self.model = self.model.to(device)

        self.emb_layer = emb_layer
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.folds = folds
        self.is_resume = is_resume

        self.loss_func = LabelSmoothingCrossEntropy().to(device)

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=0.001)

        self.scheduler_ReduceLROnPlateauLR = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode='max', factor=0.1, patience=0, verbose=True)

    def train(self):

        iter_wrapper = lambda x: tqdm(x, total=len(self.train_data))
        start_epoch = -1
        best_valid = 0.
        min_lr = 1e-7

        if self.is_resume:
            print('Let Continue!')
            checkpoint = torch.load(PATH_CHECKPOINT)  # 加载断点

            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_valid = checkpoint['best_valid']

        for epoch in range(start_epoch + 1, EPOCHS):

            print('=========================')
            print('Processing Epoch {}'.format(epoch))
            print('=========================')

            loss_per_epoch, train_n_batch = 0., 0.

            for index, data in iter_wrapper(enumerate(self.train_data)):

                creative_id, ad_id, product_id, advertiser_id, industry, product_category, time, user_id, y_label = data


                advertiser_id, product_id, product_category, industry, time = advertiser_id.to(device,non_blocking=True),\
                                                                              product_id.to(device,non_blocking=True), \
                                                                              product_category.to(device,non_blocking=True), \
                                                                              industry.to(device,non_blocking=True), \
                                                                              time.to(device,non_blocking=True)

                self.model.train()
                self.optim.zero_grad()

                #获取embedding抽取的向量
                inputlist_tensor = [
                    creative_id, ad_id, advertiser_id, product_id,
                    product_category, industry, time
                ]
                emb_layer_mat = []
                for index, input_col in enumerate(inputlist_tensor):
                    emb_layer_col_mat = {}
                    for j in range(len(self.emb_layer[index])):
                        if index in [2, 3, 4, 5, 6]:
                            self.emb_layer[index][j] = self.emb_layer[index][
                                j].to(device, non_blocking=True)
                        emb_layer_col_mat[j] = self.emb_layer[index][j](
                            input_col)
                        emb_layer_col_mat[j] = emb_layer_col_mat[j].to(
                            device, non_blocking=True)
                    emb_layer_mat.append(emb_layer_col_mat)

                output = self.model(emb_layer_mat)
                y_label = y_label.to(device, non_blocking=True)

                y_label = y_label.long()

                loss = self.loss_func(output, y_label)

                loss_per_epoch += loss.item()
                train_n_batch += 1

                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 10.)  # 梯度裁剪

                self.optim.step()

                del creative_id, ad_id, product_id, advertiser_id, industry, product_category, time, y_label
                _ = gc.collect()

            if self.val_data is not None:  # Do Validation

                valid_score, valid_loss = self.evaluate(self.val_data, epoch)
                print('evaluate done!')
                if valid_score > 0.48:
                    self.test(self.test_data, epoch)

                if valid_score > best_valid:
                    best_valid = valid_score

            self.scheduler_ReduceLROnPlateauLR.step(valid_score)

            if self.optim.param_groups[0]['lr'] < min_lr:
                print("stopping")
                break

            torch.cuda.empty_cache()

    def evaluate(self, loader, epoch):

        self.model.eval()
        user_id_list, true_y, pred_y = [], [], []
        loss_all, num_batch = 0., 0.
        with torch.no_grad():
            for index, datum_tuple in enumerate(loader):
                creative_id, ad_id, product_id, advertiser_id, industry, product_category, time, user_id, y_label = datum_tuple


                advertiser_id, product_id, product_category, industry, time = advertiser_id.to(device,non_blocking=True), \
                                                                              product_id.to(device,non_blocking=True), \
                                                                              product_category.to(device,non_blocking=True), \
                                                                              industry.to(device,non_blocking=True), \
                                                                              time.to(device,non_blocking=True)

                #获取embedding抽取的向量
                inputlist_tensor = [
                    creative_id, ad_id, advertiser_id, product_id,
                    product_category, industry, time
                ]
                emb_layer_mat = []
                for index, input_col in enumerate(inputlist_tensor):
                    emb_layer_col_mat = {}
                    for j in range(len(self.emb_layer[index])):
                        if index in [2, 3, 4, 5, 6]:
                            self.emb_layer[index][j] = self.emb_layer[index][
                                j].to(device, non_blocking=True)
                        emb_layer_col_mat[j] = self.emb_layer[index][j](
                            input_col)
                        emb_layer_col_mat[j] = emb_layer_col_mat[j].to(
                            device, non_blocking=True)
                    emb_layer_mat.append(emb_layer_col_mat)

                output = self.model(emb_layer_mat)
                y_label = y_label.to(device, non_blocking=True)

                y_label = y_label.long()

                loss = self.loss_func(output, y_label)
                loss_all += loss.item()
                num_batch += 1

                pred_y.extend(list(output.cpu().detach().numpy()))
                true_y.extend(list(y_label.cpu().detach().numpy()))
                user_id_list.extend(list(user_id.numpy()))

                del creative_id, ad_id, product_id, advertiser_id, industry, product_category, time, y_label
                _ = gc.collect()

        pred = np.argmax(np.array(pred_y), 1)
        true = np.array(true_y).reshape((-1, ))
        acc_score = accuracy_score(true, pred)

        loss_valid = loss_all / num_batch

        output_data = DataFrame({'user_id': user_id_list, 'pred': pred_y})

        if acc_score > 0.48:

            if not os.path.isdir('../../oof/bk_oof/Multi_Head_ResNext'):
                os.mkdir('../../oof/bk_oof/Multi_Head_ResNext')

            pickle.dump(
                output_data,
                open(
                    '../../oof/bk_oof/Multi_Head_ResNext/val_{}_folds_{}.pkl'.
                    format(epoch, self.folds), 'wb'))

        del pred, true, pred_y, true_y
        _ = gc.collect()

        return acc_score, loss_valid

    def test(self, loader, epoch):

        self.model.eval()
        user_id_list, pred_y = [], []
        with torch.no_grad():
            for index, datum_tuple in enumerate(loader):
                creative_id, ad_id, product_id, advertiser_id, industry, product_category, time, user_id, _ = datum_tuple


                advertiser_id, product_id, product_category, industry, time = advertiser_id.to(device,non_blocking=True), \
                                                                              product_id.to(device,non_blocking=True), \
                                                                              product_category.to(device,non_blocking=True), \
                                                                              industry.to(device,non_blocking=True), \
                                                                              time.to(device,non_blocking=True)

                #获取embedding抽取的向量
                inputlist_tensor = [
                    creative_id, ad_id, advertiser_id, product_id,
                    product_category, industry, time
                ]
                emb_layer_mat = []
                for index, input_col in enumerate(inputlist_tensor):
                    emb_layer_col_mat = {}
                    for j in range(len(self.emb_layer[index])):
                        if index in [2, 3, 4, 5, 6]:
                            self.emb_layer[index][j] = self.emb_layer[index][
                                j].to(device, non_blocking=True)
                        emb_layer_col_mat[j] = self.emb_layer[index][j](
                            input_col)
                        emb_layer_col_mat[j] = emb_layer_col_mat[j].to(
                            device, non_blocking=True)
                    emb_layer_mat.append(emb_layer_col_mat)

                output = self.model(emb_layer_mat)

                pred_y.extend(list(output.cpu().detach().numpy()))
                user_id_list.extend(list(user_id.numpy()))

        output_data = DataFrame({'user_id': user_id_list, 'pred': pred_y})

        if not os.path.isdir('../../oof/bk_oof/Multi_Head_ResNext'):
            os.mkdir('../../oof/bk_oof/Multi_Head_ResNext')

        pickle.dump(
            output_data,
            open(
                '../../oof/bk_oof/Multi_Head_ResNext/test_{}_folds_{}.pkl'.
                format(epoch, self.folds), 'wb'))


##########################################################################################
##########################################################################################
##########################################################################################


def find(index, myList):
    for i in index:
        yield myList[i]


if __name__ == '__main__':

    data = pickle.load(open('../../cached_data/input_data_20class.pkl',
                            'rb'))  #读数据

    ##获取emb_layer
    emb_layer = []
    for index, col in enumerate(inputlist):
        emb_layer_col = {}
        for indexj, matrixi in enumerate(emb_matrix_dict[col]):
            emb_layer_col[indexj] = nn.Embedding.from_pretrained(
                torch.from_numpy(matrixi))
            if col in train_able_dict:
                emb_layer_col[indexj].weight.requires_grad = False
            else:
                emb_layer_col[indexj].weight.requires_grad = True

        emb_layer.append(emb_layer_col)

    seed = 34
    for folds in range(5):
        print('This is fold: ', folds)
        train_idx = list(
            np.load(
                '../../cached_data/5folds_4seeds_index/seed_{}_train_index_fold_{}.npy'
                .format(seed, folds)))
        val_idx = list(
            np.load(
                '../../cached_data/5folds_4seeds_index/seed_{}_val_index_fold_{}.npy'
                .format(seed, folds)))

        train_df, val_df, test_df = list(find(train_idx, data)), list(
            find(val_idx, data)), data[3000000:]
        dataset_train, dataset_valid, dataset_test = TenCentDataset(
            input_data=train_df), TenCentDataset(
                input_data=val_df), TenCentDataset(input_data=test_df)

        train_data = DataLoader(dataset_train,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                drop_last=True,
                                num_workers=2,
                                pin_memory=True)
        val_data = DataLoader(dataset_valid,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              drop_last=False,
                              num_workers=2,
                              pin_memory=True)
        test_data = DataLoader(dataset_test,
                               batch_size=BATCH_SIZE,
                               shuffle=False,
                               drop_last=False,
                               num_workers=2,
                               pin_memory=True)

        is_resume = False

        My_model = Tencent2020(emb_layer, train_data, val_data, test_data,
                               folds, is_resume)  #实例化模型

        My_model.train()

        print('done')
