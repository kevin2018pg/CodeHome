# -*- coding: utf-8 -*-
# @Time    : 2022/6/30 15:57
# @Author  : kevin
# @Version : python 3.7
# @Desc    : fea


import gc
import logging
import time

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import word2vec
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils import reduce_mem, ProNE

# data_path
raw_data_path = '../../data'
reform_data_path = '../../data/reform_data'
feature_data_path = '../../data/feature_data'
model_path = '../../data/model'

train = pd.read_csv(raw_data_path + '/user_action.csv')
print("origin train:", train.shape)
train.drop_duplicates(['userid', 'feedid'], inplace=True)
print("del dup train:", train.shape)

# play stay 单位ms
train['play'] = train['play'] / 1000.0
train['stay'] = train['stay'] / 1000.0
train['play'] = train['play'].apply(lambda x: min(x, 180.0))
train['stay'] = train['stay'].apply(lambda x: min(x, 180.0))

# 测试集
test_a = pd.read_csv(f'{raw_data_path}/test_a.csv')
# test_b = pd.read_csv(data_path + 'test_b.csv')
# print(test_a.shape, test_b.shape)

feed_info = pd.read_csv(f'{raw_data_path}/feed_info.csv')
feed_info['videoplayseconds'] = feed_info['videoplayseconds'].apply(lambda x: min(x, 60))
print("缺失值情况：\n", feed_info.isnull().sum())

## 填充缺失值
# string特征
for col in ['description', 'ocr', 'asr', 'description_char', 'ocr_char', 'asr_char', 'machine_keyword_list',
            'manual_keyword_list', 'manual_tag_list', 'machine_tag_list']:
    feed_info[col] = feed_info[col].fillna('')
# int id特征
for col in ['bgm_song_id', 'bgm_singer_id']:
    feed_info[col] = feed_info[col].fillna(-1)

# reduce memory
train = reduce_mem(train, train.columns)
test_a = reduce_mem(test_a, test_a.columns)

# export
train.to_pickle(f"{reform_data_path}/user_action.pkl")
test_a.to_pickle(f"{reform_data_path}/test_a.pkl")
feed_info.to_pickle(f"{reform_data_path}/feed_info.pkl")

## User侧的GNN特征
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15

## 读取训练集（多次使用）
train = pd.read_pickle(reform_data_path + '/user_action.pkl')
test = pd.read_pickle(reform_data_path + '/test_a.pkl')
test['date_'] = max_day
print("train test shape:", train.shape, test.shape)

## 合并处理
df = pd.concat([train, test], ignore_index=True)
print("concat shape:", df.shape)

feed_info = pd.read_pickle(reform_data_path + '/feed_info.pkl')[['feedid', 'authorid', 'videoplayseconds']]
df = df.merge(feed_info, how='left', on=['feedid'])
df['play_times'] = df['play'] / df['videoplayseconds']
print("merge shape:", df.shape)


def get_proNE_embedding(df, col1, col2, emb_size=64):
    ### userid-feedid二部图
    uid_lbl, qid_lbl = LabelEncoder(), LabelEncoder()
    df['new_col1'] = uid_lbl.fit_transform(df[col1])
    df['new_col2'] = qid_lbl.fit_transform(df[col2])
    new_uid_max = df['new_col1'].max() + 1
    df['new_col2'] += new_uid_max

    ## 构建图
    G = nx.Graph()
    G.add_edges_from(df[['new_col1', 'new_col2']].values)
    model = ProNE(G, emb_size=emb_size, n_iter=6, step=12)
    features_matrix = model.fit(model.mat, model.mat)
    model.chebyshev_gaussian(model.mat, features_matrix, model.step, model.mu, model.theta)
    ## 得到proNE的embedding
    emb = model.transform()

    ## for userid
    uid_emb = emb[emb['nodes'].isin(df['new_col1'])]
    uid_emb['nodes'] = uid_lbl.inverse_transform(uid_emb['nodes'])  # 得到原id
    uid_emb.rename(columns={'nodes': col1}, inplace=True)
    for col in uid_emb.columns[1:]:
        uid_emb[col] = uid_emb[col].astype(np.float32)
    user_prone_emb = uid_emb[uid_emb.columns]
    user_prone_emb = user_prone_emb.reset_index(drop=True)
    user_prone_emb.columns = [col1] + ['prone_emb{}'.format(i) for i in range(emb_size)]

    ## for feedid
    fid_emb = emb[emb['nodes'].isin(df['new_col2'])]
    fid_emb['nodes'] = qid_lbl.inverse_transform(fid_emb['nodes'] - new_uid_max)  ## 还原需要减掉
    fid_emb.rename(columns={'nodes': col2}, inplace=True)
    for col in fid_emb.columns[1:]:
        fid_emb[col] = fid_emb[col].astype(np.float32)
    feed_prone_emb = fid_emb[fid_emb.columns]
    feed_prone_emb = feed_prone_emb.reset_index(drop=True)
    feed_prone_emb.columns = [col2] + ['prone_emb{}'.format(i) for i in range(emb_size)]
    print(user_prone_emb.shape, feed_prone_emb.shape)
    return user_prone_emb, feed_prone_emb


user_prone_emb1, feed_prone_emb = get_proNE_embedding(df[['userid', 'feedid']], col1='userid', col2='feedid',
                                                      emb_size=64)
user_prone_emb2, auth_prone_emb = get_proNE_embedding(df[['userid', 'authorid']], col1='userid', col2='authorid',
                                                      emb_size=64)

user_prone_emb2.columns = ['userid'] + ['prone_emb{}'.format(i) for i in range(64, 128)]
user_prone_emb = user_prone_emb1.merge(user_prone_emb2, how='left', on=['userid'])
print("graph shape:", user_prone_emb.shape, feed_prone_emb.shape, auth_prone_emb.shape)
user_prone_emb.to_pickle(f"{feature_data_path}/uid_prone_emb_final.pkl")
feed_prone_emb.to_pickle(f"{feature_data_path}/fid_prone_emb_final.pkl")
auth_prone_emb.to_pickle(f"{feature_data_path}/aid_prone_emb_final.pkl")

## 多模态特征
feed_emb = pd.read_csv(f'{raw_data_path}/feed_embeddings.csv')
print("多模态 shape:", feed_emb.shape)
time.sleep(0.5)
feedid_list, emb_list = [], []
for line in tqdm(feed_emb.values):
    fid, emb = int(line[0]), [float(x) for x in line[1].split()]
    feedid_list.append(fid)
    emb_list.append(emb)

feedid_emb = np.array(emb_list, dtype=np.float32)
emb_size = 192

# feedid_emb = feedid_emb - feedid_emb.mean(0, keepdims=True)
# ss = StandardScaler()
# feedid_emb = ss.fit_transform(feedid_emb)
# print(feedid_emb.shape)

# pca = PCA(n_components=emb_size)
# fid_emb = pca.fit_transform(feedid_emb)

svd = TruncatedSVD(n_components=emb_size)
fid_emb = svd.fit_transform(feedid_emb)
print("多模态降维 shape:", fid_emb.shape)
fid_emb = fid_emb.astype(np.float32)
fid_mmu_emb = pd.concat(
    [feed_emb[['feedid']], pd.DataFrame(fid_emb, columns=['mmu_emb{}'.format(i) for i in range(emb_size)])], axis=1)
fid_mmu_emb.to_pickle(f"{feature_data_path}/fid_mmu_emb_final.pkl")
# print(svd.explained_variance_ratio_)
# print(np.cumsum(svd.explained_variance_ratio_))
print("方差信息保留：", svd.explained_variance_ratio_.sum())

## feedid的word2vec特征
# 使用前面的训练集测试集合并处理
df = pd.concat([train, test], ignore_index=True)
print(df.shape)
feed_info = pd.read_pickle(reform_data_path + '/feed_info.pkl')[['feedid', 'videoplayseconds']]
df = df.merge(feed_info, how='left', on=['feedid'])
df['play_times'] = df['play'] / df['videoplayseconds']

# 用户历史n天的 feedid序列（每隔5天）
user_fid_list = []
n_day = 5
for target_day in range(6, 17):
    left, right = max(target_day - n_day, 1), target_day - 1
    tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
    user_dict = tmp.groupby('userid')['feedid'].agg(list)
    user_fid_list.extend(user_dict.values.tolist())
    # tmp = tmp[tmp['play_times'] >= 1.0].reset_index(drop=True)
    # print(tmp.shape)
    # user_dict = tmp.groupby('userid')['feedid'].agg(list)
    # user_fid_list.extend(user_dict.values.tolist())
    print("生成user历史feed序列->", target_day, left, right, len(user_dict))

## 训练word2vec
print("number of sentence {}".format(len(user_fid_list)))
emb_size = 128
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = word2vec.Word2Vec(user_fid_list, min_count=1, window=20, vector_size=emb_size, sg=1, workers=14, epochs=10)
model.save(f"{model_path}/w2v_model_128d.model")
# model = word2vec.Word2Vec.load(f"{model_path}/w2v_model_128d.model")    # 测试加载

## 将每个feedi的向量保存为pickle
feed_emb = pd.read_csv(f'{raw_data_path}/feed_embeddings.csv')[['feedid']]
w2v_fid_mat = []
emb_size = 128
null_cnt = 0
for fid in tqdm(feed_emb.feedid.values):
    try:
        emb = model.wv[fid]
    except:
        emb = np.zeros(emb_size)
        null_cnt += 1
    w2v_fid_mat.append(emb)

print("feedid缺失w2v个数：", null_cnt)
w2v_fid_mat = np.array(w2v_fid_mat, dtype=np.float32)
fid_w2v_emb = pd.concat([feed_emb, pd.DataFrame(w2v_fid_mat, columns=['w2v_emb{}'.format(i) for i in range(emb_size)])],
                        axis=1)
fid_w2v_emb.to_pickle(f"{feature_data_path}/fid_w2v_emb_final.pkl")

## Feed_info的预处理
feed_info = pd.read_pickle(reform_data_path + '/feed_info.pkl')
manual_kw = feed_info['manual_keyword_list'].apply(lambda x: x.split(';'))
machine_kw = feed_info['machine_keyword_list'].apply(lambda x: x.split(';'))
manual_tag = feed_info['manual_tag_list'].apply(lambda x: x.split(';'))


def func(x):
    if len(x) == 0:
        return ['-1']
    return [_.split()[0] for _ in x.split(';') if float(_.split()[1]) >= 0.5]


machine_tag = feed_info['machine_tag_list'].apply(lambda x: func(x))

all_kw = []  # 关键词：机器和人工去重
assert len(manual_kw) == len(machine_kw)
for i in (range(len(manual_kw))):
    tmp = set(manual_kw[i] + machine_kw[i])
    tmp = [x.strip() for x in tmp if x != '' and x != '-1']
    if len(tmp) == 0:
        tmp = ['-1']
    all_kw.append(' '.join(tmp))

all_tag = []  # tag标签：机器和人工去重
assert len(manual_tag) == len(machine_tag)
for i in (range(len(manual_tag))):
    tmp = set(manual_tag[i] + machine_tag[i])
    tmp = [x.strip() for x in tmp if x != '' and x != '-1']
    if len(tmp) == 0:
        tmp = ['-1']
    all_tag.append(' '.join(tmp))

assert len(all_kw) == len(all_tag)

## 处理keyword
print("****** 处理keyword *******")
emb_size = 48
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=20)
all_kw_mat = tfidf_vectorizer.fit_transform(all_kw)
kw1 = np.array(all_kw_mat.argmax(axis=1)).reshape(-1)

svd = TruncatedSVD(n_components=emb_size)
all_kw_mat = svd.fit_transform(all_kw_mat)
print(all_kw_mat.shape)
print("方差信息保留：", svd.explained_variance_ratio_.sum())

## 处理tag
print("****** 处理tag *******")
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=10)
all_tag_mat = tfidf_vectorizer.fit_transform(all_tag)
tag1 = np.array(all_tag_mat.argmax(axis=1)).reshape(-1)

svd = TruncatedSVD(n_components=emb_size)
all_tag_mat = svd.fit_transform(all_tag_mat)
print(all_tag_mat.shape)
print("方差信息保留：", svd.explained_variance_ratio_.sum())

## 处理words
print("****** 处理words *******")
all_words = feed_info['description'] + ' ' + feed_info['ocr'] + ' ' + feed_info['asr']
all_words = [' '.join(x.split()[:100]) for x in all_words.values.tolist()]
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=20)
all_words_mat = tfidf_vectorizer.fit_transform(all_words)

svd = TruncatedSVD(n_components=emb_size)
all_words_mat = svd.fit_transform(all_words_mat)
print(all_words_mat.shape)
print("方差信息保留：", svd.explained_variance_ratio_.sum())

## 处理chars
print("****** 处理chars *******")
all_chars = feed_info['description_char'] + ' ' + feed_info['ocr_char'] + ' ' + feed_info['asr_char']
all_chars = [' '.join(x.split()[:100]) for x in all_chars.values.tolist()]
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=20)
all_chars_mat = tfidf_vectorizer.fit_transform(all_chars)

svd = TruncatedSVD(n_components=emb_size)
all_chars_mat = svd.fit_transform(all_chars_mat)
print(all_chars_mat.shape)
print("方差信息保留：", svd.explained_variance_ratio_.sum())

all_kw_mat = all_kw_mat.astype(np.float32)
all_tag_mat = all_tag_mat.astype(np.float32)
all_words_mat = all_words_mat.astype(np.float32)
all_chars_mat = all_chars_mat.astype(np.float32)

fid_kw_tag_word_emb = pd.concat(
    [feed_info[['feedid']], pd.DataFrame(all_kw_mat, columns=['kw_emb{}'.format(i) for i in range(emb_size)]),
     pd.DataFrame(all_tag_mat, columns=['tag_emb{}'.format(i) for i in range(emb_size)]),
     pd.DataFrame(all_words_mat, columns=['word_emb{}'.format(i) for i in range(emb_size)]),
     pd.DataFrame(all_chars_mat, columns=['char_emb{}'.format(i) for i in range(emb_size)]), ], axis=1)
fid_kw_tag_word_emb.to_pickle(f'{feature_data_path}/fid_kw_tag_word_emb_final.pkl')


## 相同字数占比, desc, ocr, asr字数
def funct(row):
    desc = row['description_char']
    ocr = row['ocr_char']
    desc, ocr = set(desc.split()), set(ocr.split())
    return len(desc & ocr) / (min(len(desc), len(ocr)) + 1e-8)


feed_info['desc_ocr_same_rate'] = feed_info.apply(lambda row: funct(row), axis=1)
feed_info['desc_len'] = feed_info['description_char'].apply(lambda x: len(x.split()))
feed_info['asr_len'] = feed_info['asr_char'].apply(lambda x: len(x.split()))
feed_info['ocr_len'] = feed_info['ocr_char'].apply(lambda x: len(x.split()))

feed_info['keyword1'] = kw1
feed_info['tag1'] = tag1
feed_info['all_keyword'] = all_kw
feed_info['all_tag'] = all_tag

# def get_tag_top1(x):
#     try:
#         tmp = sorted([(int(x_.split()[0]), float(x_.split()[1]))  for x_ in x.split(';') if len(x_) > 0],
#                        key=lambda x: x[1], reverse=True)
#     except:
#         return 0
#     return tmp[0][0]
# feed_info['tag_m1'] =  feed_info['machine_tag_list'].apply(lambda x: get_tag_top1(x))


feed_info.drop(columns=['description', 'ocr', 'asr', 'manual_keyword_list', 'machine_keyword_list', 'manual_tag_list',
                        'machine_tag_list', 'description_char', 'ocr_char', 'asr_char'], inplace=True)
feed_info['bgm_song_id'] = feed_info['bgm_song_id'].astype(np.int32)
feed_info['bgm_singer_id'] = feed_info['bgm_singer_id'].astype(np.int32)
feed_info.to_pickle(f'{feature_data_path}/feed_info.pkl')

## CTR特征    stat统计特征
## 读取训练集（多次使用）
max_day = 15
train = pd.read_pickle(reform_data_path + '/user_action.pkl')
test = pd.read_pickle(reform_data_path + '/test_a.pkl')
test['date_'] = max_day
print("train test shape:", train.shape, test.shape)

## 合并处理
df = pd.concat([train, test], ignore_index=True)
print("concat shape:", df.shape)
del train, test
gc.collect()
# feed侧信息
feed_info = pd.read_pickle(feature_data_path + '/user_action.pkl')
feed_info.drop(columns=['all_keyword', 'all_tag'], inplace=True)
print(feed_info.shape)
df = df.merge(feed_info, on='feedid', how='left')

## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
df['is_finish'] = (df['play'] >= df['videoplayseconds']).astype('int8')
df['play_times'] = df['play'] / df['videoplayseconds']
df['stay_times'] = df['stay'] / df['videoplayseconds']

play_cols = ['is_finish', 'play_times', 'stay_times', 'play', 'stay']
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
cols_y = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'tag1', 'keyword1', 'date_'] + y_list
cols_play = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'tag1', 'keyword1', 'date_'] + play_cols

df_target = df[cols_y]
df_target = reduce_mem(df_target, [col for col in df_target.columns.tolist() if col not in y_list])
for col in y_list:
    df_target[col] = df_target[col].astype(np.float32)
df_play = df[cols_play]
df_play = reduce_mem(df_play, [col for col in cols_play if col not in play_cols])
for col in play_cols:
    df_play[col] = df_play[col].astype(np.float32)

print(df_target.info())
print(df_play.info())

## 统计历史n天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
n_day = 5
max_day = 15
all_stat_cols = [['userid'], ['feedid'], ['authorid'], ['bgm_song_id'], ['bgm_singer_id'], ['tag1'], ['keyword1'],
                 ['userid', 'tag1'], ['userid', 'keyword1'], ['userid', 'authorid']]


def get_ctr_fea(df, all_stat_cols):
    def in_func(stat_cols):
        f = '_'.join(stat_cols)
        print('======== ' + f + ' =========')
        stat_df = pd.DataFrame()
        for target_day in range(6, max_day + 1):
            left, right = max(target_day - n_day, 1), target_day - 1
            tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
            tmp['date_'] = target_day
            g = tmp.groupby(stat_cols)
            feats = []
            for y in y_list:
                tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')
                tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')
                # tmp['{}_{}day_{}'.format(f, n_day, y) + '_all_count'] = g[y].transform('count')
                # tmp['{}_{}day_{}'.format(f, n_day, y) + '_label_count'] = g[y].transform('sum')
                # HP = HyperParam(1, 1)
                # HP.update_from_data_by_moment(tmp['{}_{}day_{}'.format(f, n_day, y) + '_all_count'].values,
                # tmp['{}_{}day_{}'.format(f, n_day, y) + '_label_count'].values)
                # tmp['{}_{}day_{}_ctr'.format(f, n_day, y)] = (tmp['{}_{}day_{}'.format(f, n_day,
                #                                                                        y) + '_label_count'] + HP.alpha) / (
                #                                                          tmp['{}_{}day_{}'.format(f, n_day,
                #                                                                                   y) + '_all_count'] + HP.alpha + HP.beta)
                feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])
            tmp = tmp[stat_cols + ['date_'] + feats].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
            stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
        stat_df = reduce_mem(stat_df, stat_df.columns)
        m, n = stat_df.shape
        stat_df.to_pickle(f"{feature_data_path}/{f}_{m}_{n}_{n_day}days_ctr_fea.pkl")

    n_jobs = len(all_stat_cols)
    all_stat_df = Parallel(n_jobs=n_jobs)(delayed(in_func)(col) for col in all_stat_cols)


tmp_res = get_ctr_fea(df_target, all_stat_cols[:7])
tmp_res = get_ctr_fea(df_target, all_stat_cols[7:])


def get_stat_fea(df, all_stat_cols):
    def in_func(stat_cols):
        f = '_'.join(stat_cols)
        print('======== ' + f + ' =========')
        stat_df = pd.DataFrame()
        for target_day in tqdm(range(6, max_day + 1)):
            left, right = max(target_day - n_day, 1), target_day - 1

            tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
            tmp['date_'] = target_day
            tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')

            g = tmp.groupby(stat_cols)
            tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')  # 观看完成率

            # 特征列
            feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]

            for x in play_cols[1:]:
                for stat in ['max', 'mean', 'sum']:
                    tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)
                    feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))

            tmp = tmp[stat_cols + ['date_'] + feats].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
            stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)

        stat_df = reduce_mem(stat_df, stat_df.columns)

        m, n = stat_df.shape
        stat_df.to_pickle(f"{feature_data_path}/{f}_{m}_{n}_{n_day}day_stat_fea.pkl")

    n_jobs = len(all_stat_cols)
    all_stat_df = Parallel(n_jobs=n_jobs)(delayed(in_func)(col) for col in all_stat_cols)


get_stat_fea(df, all_stat_cols[:7])
get_stat_fea(df, all_stat_cols[7:])

## 全局统计特征
count_feas = []
for f in tqdm(['userid', 'feedid', 'authorid', 'tag1', 'keyword1', 'bgm_song_id', 'bgm_singer_id']):
    df[f + '_count_global'] = df[f].map(df[f].value_counts())
