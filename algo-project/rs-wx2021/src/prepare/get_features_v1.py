# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 17:35
# @Author  : west
# @File    : get_features.py
# @Version : python 3.6
# @Desc    : 生成特征


import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../model'))

import pandas as pd
import numpy as np
import warnings
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 显示宽度
pd.set_option('display.width', None)

warnings.filterwarnings('ignore')

## feed info
feed_info = pd.read_csv(BASE_DIR[:-11] + 'data/feed_info.csv')

feed_info.drop(['ocr', 'asr', 'description', 'description_char', 'ocr_char', 'asr_char'], axis=1, inplace=True)
feed_info[['bgm_song_id', 'bgm_singer_id']] += 1
feed_info[['bgm_song_id', 'bgm_singer_id']] = feed_info[['bgm_song_id', 'bgm_singer_id']].fillna(0)
feed_info[['bgm_song_id', 'bgm_singer_id']] = feed_info[['bgm_song_id', 'bgm_singer_id']].astype('int')

print("feed: {}".format(feed_info.shape))
print(feed_info.head())
# sys.exit(1)

targets = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'follow', 'comment']
sparse_features = ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id', 'bgm_singer_id']
# 可变长度特征
varlen_sparse_features = ['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list']


def get_vocab_size(feat):
    return max(feed_info[feat].apply(lambda x: max(x) if list(x) else 0)) + 1


def get_maxlen(feat):
    return max(feed_info[feat].apply(lambda x: len(x)))


# 提取id，补齐最大长度
for feat in varlen_sparse_features:
    feed_info[feat] = feed_info[feat].apply(
        lambda x: np.fromstring(x, "int", sep=';')[:5] if x is not np.NAN else np.array([], dtype=np.int32))
    feed_info[feat] = pad_sequences(feed_info[feat], maxlen=get_maxlen(feat), padding='post', dtype=np.int32,
                                    value=0).tolist()

# 处理可变特征 machine_tag_list（去除概率值与其他特征保持统一）
feed_info['machine_tag_list'] = feed_info['machine_tag_list'].apply(
    lambda x: x.strip().split(';')[:3] if x is not np.NAN else np.array([], dtype=np.int32))
feed_info['machine_tag_list'] = feed_info['machine_tag_list'].apply(
    lambda x: np.asarray([i.split(' ')[0] for i in x], dtype=float))
feed_info['machine_tag_list'] = pad_sequences(feed_info['machine_tag_list'], maxlen=get_maxlen('machine_tag_list'),
                                              padding='post', dtype=np.int32, value=0).tolist()
varlen_sparse_features.append('machine_tag_list')

print(feed_info.head())
# sys.exit(1)

# log平滑
feed_info['videoplayseconds'] = feed_info['videoplayseconds'].fillna(0)
feed_info['videoplayseconds'] = np.log(feed_info['videoplayseconds'] + 1.0)


# feed_cluster = pd.read_csv(BASE_DIR[:-11] + 'data/feature/feedid_cluster.csv')
# feed_info = pd.merge(feed_info, feed_cluster, on='feedid', how='left')
# author_cluster = pd.read_csv(BASE_DIR[:-11] + 'data/feature/authorid_cluster.csv')
# feed_info = pd.merge(feed_info, author_cluster, on='authorid', how='left')

# 压缩数据类型
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


feed_info = reduce_mem_usage(feed_info)
feed_info.to_pickle(BASE_DIR[:-11] + 'data/train_data/feed.pkl')

## user info
user_action = pd.read_csv(BASE_DIR[:-11] + 'data/user_action.csv')
# user_action2 = pd.read_csv(BASE_DIR[:-11] + 'data/wedata/wechat_algo_data2/user_action.csv')
# user_action = pd.concat([user_action, user_action2])
# del user_action2
print("user: {}".format(user_action.shape))
print(user_action.head())

# merge info
user_action = pd.merge(user_action, feed_info, on='feedid', how='left')
print("total data: {}".format(user_action.shape))
print(user_action.head())

##
from gensim.models import Word2Vec


def emb(df, f1, f2):
    emb_size = 16
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]

    # model = Word2Vec(sentences, size=emb_size, window=6, min_count=5, sg=0, hs=0, seed=1, iter=5)  # cbow
    model = Word2Vec(sentences, vector_size=emb_size, window=6, min_count=5, sg=0, hs=0, seed=1, epochs=5)  # cbow
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            # if w in model.wv.vocab:
            if w in model.wv.key_to_index:
                vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
    emb_matrix = np.array(emb_matrix)
    for i in range(emb_size):
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    return tmp


# user feed id embedding
user_df_ = emb(user_action, f1='userid', f2='feedid')
user_df_ = reduce_mem_usage(user_df_)
# print(user_df_.shape)
# print(user_df_.head())
# sys.exit(1)
user_df_.to_pickle(BASE_DIR[:-11] + 'data/train_data/user_feedid_w2v.pkl')
print("total embedding: {}".format(user_df_.shape))
print(user_df_.head())

## 整体数据
data = user_action
data.to_pickle(BASE_DIR[:-11] + 'data/train_data/train_on.pkl')

# 训练数据划分
train = data[data['date_'] < 14]
val = data[data['date_'] == 14]
train.to_pickle(BASE_DIR[:-11] + 'data/train_data/train_off.pkl')
val.to_pickle(BASE_DIR[:-11] + 'data/train_data/val_off.pkl')
del data
del train
del val
