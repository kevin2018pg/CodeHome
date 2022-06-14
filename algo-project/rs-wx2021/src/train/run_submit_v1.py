# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 20:50
# @Author  : west
# @File    : run_submit.py
# @Version : python 3.6
# @Desc    : 训练


import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../model'))
sys.path.append(os.path.join(BASE_DIR, '../'))

import pandas as pd
import numpy as np
import tensorflow as tf
import random
import argparse
import warnings
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from mmoe_v1 import MMOE
from tensorflow.python.keras.models import save_model

# from tensorflow.keras.models import save_model, load_model

parser = argparse.ArgumentParser(description='log')

parser.add_argument("--k", metavar='Mode', type=int, default=0)
args = parser.parse_args()
seed = args.k
print(seed)

# seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
# < 2.0 version
# tf.set_random_seed(seed)
# > 2.0 version
tf.random.set_seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# < 2.0 version
# config = tf.ConfigProto()
# > 2.0 version
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# < 2.0 version
# sess = tf.Session(config=config)
# > 2.0 version
sess = tf.compat.v1.Session(config=config)
warnings.filterwarnings('ignore')

targets = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'follow', 'comment']
sparse_features = ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id', 'bgm_singer_id'] + ['cluster_0',
                                                                                                'cluster_1',
                                                                                                'cluster_2',
                                                                                                'cluster_3',
                                                                                                'cluster_4',
                                                                                                'cluster_5',
                                                                                                'cluster_0_authorid',
                                                                                                'cluster_1_authorid',
                                                                                                'cluster_2_authorid',
                                                                                                'cluster_3_authorid',
                                                                                                'cluster_4_authorid',
                                                                                                'cluster_5_authorid']

varlen_sparse_features = ['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list', 'machine_tag_list']
dense_features = ['videoplayseconds'] + ['userid_feedid_emb_{}'.format(i) for i in range(16)]

# vocab size
vocab = {'userid': 250249, 'feedid': 112872, 'device': 3, 'authorid': 18789, 'bgm_song_id': 25160,
         'bgm_singer_id': 17501, 'manual_keyword_list': 27271, 'machine_keyword_list': 27264, 'manual_tag_list': 353,
         'machine_tag_list': 346,
         'cluster_0': 10, 'cluster_1': 100, 'cluster_2': 500, 'cluster_3': 1000, 'cluster_4': 50, 'cluster_5': 5000,
         'cluster_0_authorid': 10, 'cluster_1_authorid': 100, 'cluster_2_authorid': 500, 'cluster_3_authorid': 1000,
         'cluster_4_authorid': 50, 'cluster_5_authorid': 5000}
max_len = {'manual_keyword_list': 5, 'machine_keyword_list': 5, 'manual_tag_list': 5, 'machine_tag_list': 3}

epochs = 5
batch_size = 1024 * 4
embedding_dim = 256

train = pd.read_pickle(BASE_DIR[:-9] + 'data/train_on.pkl')

ff = pd.read_pickle(BASE_DIR[:-9] + 'data/user_feedid_w2v.pkl')
train = pd.merge(train, ff, on='userid', how='left')

fixlen_feature_columns = [DenseFeat(feat, 1) for feat in dense_features] + [
    SparseFeat(feat, vocabulary_size=vocab[feat], embedding_dim=embedding_dim) for feat in sparse_features]
varlen_feature_columns = [
    VarLenSparseFeat(SparseFeat(feat, vocabulary_size=vocab[feat], embedding_dim=embedding_dim), maxlen=max_len[feat],
                     combiner='mean') for feat in varlen_sparse_features]

dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
feature_names = get_feature_names(dnn_feature_columns)

train_model_input = {name: train[name] for name in feature_names}
for feat in varlen_sparse_features:
    train_model_input[feat] = np.asarray(train_model_input[feat].to_list()).astype(np.int32)

train_labels = [train[y].values for y in targets]

train_model = MMOE(dnn_feature_columns, num_tasks=len(targets), expert_dim=16, dnn_hidden_units=(256, 256),
                   tasks=['binary'] * len(targets), task_dnn_units=(128, 128))

train_model.compile("adagrad", loss='binary_crossentropy', )
for epoch in range(epochs):
    history = train_model.fit(train_model_input, train_labels, batch_size=batch_size, epochs=1, verbose=1)
    if epoch == 4:
        save_model(train_model, BASE_DIR[:-9] + 'data/model/model_{}_run{}.h5'.format(epoch, seed))
