# -*- coding: utf-8 -*-
# @Time    : 2022/7/5 13:41
# @Author  : kevin
# @Version : python 3.7
# @Desc    : train

import gc
import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.layers import DNN
from deepctr_torch.layers.interaction import BiInteractionPooling
from deepctr_torch.layers.sequence import AttentionSequencePoolingLayer
from deepctr_torch.models.basemodel import BaseModel
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from utils import get_logger

pd.set_option('display.max_columns', None)
# data_path
raw_data_path = '../../data'
reform_data_path = '../../data/reform_data'
feature_data_path = '../../data/feature_data'
model_path = '../../data/model'
log_path = '../../data/log'

logger = get_logger(f"{log_path}/mmoe_0807.txt")
logger.info("start training...")

train = pd.read_pickle(f"{feature_data_path}/train_v0.pkl").reset_index(drop=True)
test = pd.read_pickle(f"{feature_data_path}/test_v0.pkl").reset_index(drop=True)
df = pd.concat([train, test], ignore_index=True)
print("数据集维度:", train.shape, test.shape, df.shape)
del train, test
gc.collect()

## 特征列定义
play_cols = ['is_finish', 'play_times', 'stay_times', 'play', 'stay']
y_list = ['read_comment', 'like', 'click_avatar', 'favorite', 'forward', 'comment', 'follow']

## 离散和连续特征
sparse_features = ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'keyword1', 'tag1']
dense_features = [x for x in df.columns if x not in sparse_features + ['date_'] + play_cols + y_list]
logger.info("sparse_fea: {}, dense_fea: {}".format(len(sparse_features), len(dense_features)))


def get_word2id(df, col, is_del=True):
    cnt_dict0 = dict(Counter(df[col]))
    if is_del:
        cnt_dict = {k: v for k, v in cnt_dict0.items() if v >= 2}
        word2id = {k: (i + 2) for i, k in enumerate(cnt_dict.keys())}
    else:
        word2id = {k: i for i, k in enumerate(cnt_dict0.keys())}
    print("{}, {} -> {}".format(col, len(cnt_dict0), len(word2id)))
    return word2id


userid_2_id = get_word2id(df, 'userid', is_del=True)
feedid_2_id = get_word2id(df, 'feedid', is_del=True)
device_2_id = get_word2id(df, 'device', is_del=False)
authorid_2_id = get_word2id(df, 'authorid', is_del=True)
bgm_song_id_2_id = get_word2id(df, 'bgm_song_id', is_del=True)
bgm_singer_id_2_id = get_word2id(df, 'bgm_singer_id', is_del=True)
keyword1_2_id = get_word2id(df, 'keyword1', is_del=True)
tag1_2_id = get_word2id(df, 'tag1', is_del=True)
print(len(userid_2_id), len(feedid_2_id), len(device_2_id), len(authorid_2_id), len(bgm_song_id_2_id),
      len(bgm_singer_id_2_id), len(keyword1_2_id), len(tag1_2_id))

pickle.dump([userid_2_id, feedid_2_id, device_2_id, authorid_2_id, bgm_song_id_2_id, bgm_singer_id_2_id, keyword1_2_id,
             tag1_2_id], open(f'{feature_data_path}/all_word2id.pkl', 'wb'))

## 分别获取embedding
## feed侧的embedding
fid_kw_tag_word_emb = pd.read_pickle(f'{feature_data_path}/fid_kw_tag_word_emb_final')
fid_mmu_emb = pd.read_pickle(f"{feature_data_path}/fid_mmu_emb_final.pkl")
fid_w2v_emb = pd.read_pickle(f"{feature_data_path}/fid_w2v_emb_final.pkl")
fid_prone_emb = pd.read_pickle(f"{feature_data_path}/fid_prone_emb_final.pkl")
print(fid_kw_tag_word_emb.shape, fid_mmu_emb.shape, fid_w2v_emb.shape, fid_prone_emb.shape)

## 合并
fid_2_emb_df = fid_kw_tag_word_emb
fid_2_emb_df = fid_2_emb_df.merge(fid_mmu_emb, how='left', on=['feedid'])
fid_2_emb_df = fid_2_emb_df.merge(fid_w2v_emb, how='left', on=['feedid'])
fid_2_emb_df = fid_2_emb_df.merge(fid_prone_emb, how='left', on=['feedid'])
fid_2_emb_df.fillna(0.0, inplace=True)
print("feedid embedding shape:", fid_2_emb_df.shape)

## userid侧的embedding
uid_2_emb_df = pd.read_pickle(f'{feature_data_path}/uid_prone_emb_final.pkl')
print("userid embedding shape:", uid_2_emb_df.shape)
## authorid侧的embedding
aid_2_emb_df = pd.read_pickle(f'{feature_data_path}/aid_prone_emb_final.pkl')
print("authorid embedding shape: ", aid_2_emb_df.shape)

## 制作hash
fid_2_emb = {}
for line in (fid_2_emb_df.values):
    fid_2_emb[int(line[0])] = line[1:].astype(np.float32)
uid_2_emb = {}
for line in (uid_2_emb_df.values):
    uid_2_emb[int(line[0])] = line[1:].astype(np.float32)
aid_2_emb = {}
for line in (aid_2_emb_df.values):
    aid_2_emb[int(line[0])] = line[1:].astype(np.float32)

## 删除，减少内存消耗
del fid_kw_tag_word_emb, fid_mmu_emb, fid_w2v_emb, fid_prone_emb
# del fid_2_emb_df, uid_2_emb_df, aid_2_emb_df
gc.collect()
gc.collect()

## 打印长度
print('feedid', len(fid_2_emb), len(fid_2_emb[54042]))
print('userid', len(uid_2_emb), len(uid_2_emb[0]))
print('authorid', len(aid_2_emb), len(aid_2_emb[0]))
# print('userid_date_', len(uiddate_2_bertemb), len(uiddate_2_bertemb[(0, 8)]))

emb_fea_nums = len(fid_2_emb[54042]) + len(uid_2_emb[0]) + len(aid_2_emb[0])
logger.info("embedding features nums: {}".format(emb_fea_nums))

manual_fea = pickle.load(open(f'{feature_data_path}/singer_col_stat_feas.pkl', 'rb'))

for x in manual_fea:
    print(len(x), type(x))


class MyDataset(Dataset):
    def __init__(self, df, sparse_cols, dense_cols, labels, word2id_list, uid_2_emb=None, fid_2_emb=None,
                 aid_2_emb=None, uiddate_2_bertemb=None, manual_fea=None, uid_date_2_fid_hist=None,
                 uid_date_2_aid_hist=None, ):
        self.sparse_features = df[sparse_cols].values
        self.dense_features = df[dense_cols].values
        self.dates = df['date_'].values
        self.labels = df[labels].values

        self.word2id_list = word2id_list

        self.uid_2_emb = uid_2_emb
        self.fid_2_emb = fid_2_emb
        self.aid_2_emb = aid_2_emb
        self.uiddate_2_bertemb = uiddate_2_bertemb
        self.manual_fea = manual_fea
        self.mf_size = [41, 30, 33, 32, 32, 32, 32]
        self.df_len = df.shape[0]

    def __len__(self):
        return self.df_len

    def __getitem__(self, item):
        # 标签信息，日期信息
        label = self.labels[item]
        date_ = self.dates[item]
        # Sparse特征
        sparse_f = self.sparse_features[item]
        uid, fid, device, aid, bgm_song, bgm_singer, kw1, tag1 = [int(x) for x in sparse_f]
        # Dense特征
        dense_f = list(self.dense_features[item])
        # manual feature
        mf_list = [uid, fid, aid, bgm_song, bgm_singer, kw1, tag1]
        for idx, mf in enumerate(self.manual_fea):
            dense_f.extend(list(mf.get((mf_list[idx], date_), [0.0] * self.mf_size[idx])))

        # embedding特征
        all_emb_f = list(self.uid_2_emb.get(uid, [0.0] * 128))
        all_emb_f.extend(list(self.fid_2_emb.get(fid, [0.0] * 576)))
        all_emb_f.extend(list(self.aid_2_emb.get(aid, [0.0] * 64)))

        sparse_f = [self.word2id_list[idx].get(int(sparse_f[idx]), 1) for idx in range(len(sparse_f))]

        return (torch.FloatTensor(sparse_f + dense_f + all_emb_f),
                torch.FloatTensor(label),)


word2id_list = pickle.load(open(f"{feature_data_path}/all_word2id.pkl", 'rb'))


def get_loader(df, batch_size=20480, train_mode=True, n_cpu=14):
    ds = MyDataset(df, sparse_cols=sparse_features, dense_cols=dense_features, labels=y_list, word2id_list=word2id_list,
                   uid_2_emb=uid_2_emb, fid_2_emb=fid_2_emb, aid_2_emb=aid_2_emb, uiddate_2_bertemb=None,
                   manual_fea=manual_fea, uid_date_2_fid_hist=None, uid_date_2_aid_hist=None)
    if train_mode:
        sampler = RandomSampler(ds)
    else:
        sampler = SequentialSampler(ds)
    my_loader = DataLoader(ds, sampler=sampler, batch_size=batch_size, num_workers=n_cpu, pin_memory=True)
    return my_loader


dense_fea_nums = sum([41, 30, 33, 32, 32, 32, 32])
print("dense_fea_nums:", dense_fea_nums)
print("emb_fea_nums:", emb_fea_nums)
print(len(sparse_features), len(dense_features))
print(len(sparse_features) + len(dense_features) + dense_fea_nums + emb_fea_nums)

## 划分训练测试集
train = df[df['date_'] <= 14].reset_index(drop=True)
test = df[df['date_'] == 15].reset_index(drop=True)
del df
gc.collect()
print("train & test shape:", train.shape, test.shape)


# 模型：DNN作为主编码器
class MMOE_DNN(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, use_fm=False, use_din=False, embed_dim=32,
                 dnn_hidden_units=(256, 128), l2_reg_linear=0.001, l2_reg_embedding=0.01, l2_reg_dnn=0.0,
                 init_std=0.001, seed=1024, dnn_dropout=0.5, dnn_activation='relu', dnn_use_bn=True, task='binary',
                 device='cpu', gpus=None, num_tasks=4, num_experts=16, expert_dim=32, ):
        super(MMOE_DNN, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                       l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                       device=device, gpus=gpus)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if self.use_fm:
            self.fm = BiInteractionPooling()
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, activation=dnn_activation,
                           l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std,
                           device=device)
            self.dnn_aux = DNN(embed_dim * 2, dnn_hidden_units[-2:], activation=dnn_activation, l2_reg=l2_reg_dnn,
                               dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)
        self.use_din = use_din
        if self.use_din:
            self.feedid_emb_din = self.embedding_dict.feedid
            # self.LSTM_din = nn.LSTM(input_size=embed_dim,hidden_size=embed_dim,num_layers=1,batch_first=True,bidirectional=False)
            self.attention = AttentionSequencePoolingLayer(att_hidden_units=(64, 64),
                                                           embedding_dim=embed_dim,
                                                           att_activation='Dice',
                                                           return_score=False,
                                                           supports_masking=False,
                                                           weight_normalization=False)
        # 专家设置
        self.input_dim = dnn_hidden_units[-1]
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.num_tasks = num_tasks

        # expert-kernel
        self.expert_kernel = nn.Linear(self.input_dim, num_experts * expert_dim)

        # TODO 每个任务的单独变换
        # self.gate_mlp = nn.ModuleList([DNN_head(232) for i in range(num_tasks)])
        # 每个任务的gate-kernel
        self.gate_kernels = nn.ModuleList(
            [nn.Linear(self.input_dim, num_experts, bias=False) for i in range(num_tasks)])
        self.cls = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.expert_dim, 128), nn.ReLU(), nn.Linear(128, 1)) for i in
             range(self.num_tasks)])
        self.gate_softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.to(device)

    def forward(self):
        pass


# submit

for date_ in [14]:
    start_time = time.time()
    logger.info("********* train date_ is {} *********".format(date_))

    # For debug
    train_idx = train[train['date_'] != date_].index
    valid_idx = train[train['date_'] == date_].inedx
    ## 开始训练模型
    emb_size = 48
    actions = ['read_comment', 'like', 'click_avatar', 'favorite', 'forward', 'comment', 'follow']
    # count unique features for each sparse field, and record dense features field name
    fixlen_feature_columns = [SparseFeat(feat, max(list(word2id_list[i].values())) + 1, embedding_dim=emb_size) for
                              i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1) for feat in
                                                                        dense_features + ['dense_{}'.format(i) for i in
                                                                                          range(dense_fea_nums)] + [
                                                                            'emb_{}'.format(i) for i in
                                                                            range(emb_fea_nums)]]
    logger.info("fixlen_fea nums: {}".format(len(fixlen_feature_columns)))

    # 所有特征列，dnn和linear都一样
    dnn_feature_columns = fixlen_feature_columns  # for DNN
    linear_feature_columns = fixlen_feature_columns  # for Embedding
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    logger.info("feature nums is {}".format(len(feature_names)))

    ## 验证集上的标签
    val_x = train.iloc[valid_idx][['userid'] + actions].reset_index(drop=True)
    for col in val_x.columns:
        val_x[col] = val_x[col].astype(np.int32)
    logger.info("valid df shape is {}".format(val_x.shape))
    # get 数据加载器
    train_loader = get_loader(train.iloc[train_idx].reset_index(drop=True))
    valid_loader = get_loader(train.iloc[valid_idx].reset_index(drop=True), train_mode=False)
    test_loader = get_loader(test, train_mode=False)
    logger.info(
        "train_loader len {}, valid_loader len {}, test_loader len {},".format(len(train_loader), len(valid_loader),
                                                                               len(test_loader)))
    # DEVICE
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        logger.info('cuda ready...')
        device = 'cuda:0'

    # 定义模型
