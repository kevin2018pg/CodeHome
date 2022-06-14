# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 20:35
# @Author  : west
# @File    : reform_w2v_feature.py
# @Version : python 3.6
# @Desc    : feed user author的w2v特征（step3）
import csv
import gc
import os
import pickle
import sys
from itertools import zip_longest

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

csv.field_size_limit(sys.maxsize)
sys.path.append('../')
from utils import reduce_mem_usage, dict2model

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

global size, epoch, opt, w_model, W2V_DATA_PATH
size = 64
epoch = 30
opt = 0  # 1: 采用hierarchical softmax训练模型; 0: 使用负采样。
opt_map = {1: 'hs', 0: 'ns'}
w_model = 1  # 1: skip-gram; 0: CBOW.
model_map = {1: 'sg', 0: 'cbow'}

RAW_DATA_PATH = '../../data'
REFORM_DATA_PATH = '../../data/reform_data'
# RAW_DATA_PATH_pri = '/home/tione/notebook/wbdc2021/data/wedata/wechat_algo_data1/'
W2V_DATA_PATH = f'{REFORM_DATA_PATH}/w2v_models_{model_map[w_model]}_{opt_map[opt]}_dim{size}_epoch{epoch}'

if not os.path.exists(W2V_DATA_PATH):
    os.makedirs(W2V_DATA_PATH)

# 聚类特征
CLS_COLS = ['feed_manu_tag_tfidf_cls_32', 'feed_machine_tag_tfidf_cls_32', 'feed_manu_kw_tfidf_cls_22',
            'feed_machine_kw_tfidf_cls_17', 'feed_description_tfidf_cls_18', 'author_manu_tag_tfidf_cls_19',
            'author_machine_tag_tfidf_cls_21', 'author_manu_kw_tfidf_cls_18', 'author_machine_kw_tfidf_cls_18',
            'author_description_tfidf_cls_18']

# TODO 补充特征字段为空
text_cols = []  # ['bgm_song_id', 'bgm_singer_id'] # +CLS_COLS


def get_sentences():
    """
    根据用户交互行为生成序列feedid、authorid、userid序列，用于后续word2vec embedding
    """
    # feed聚类数据
    feed = pd.read_pickle(f'{REFORM_DATA_PATH}/feed_author_text_features_fillna_by_author_clusters.pkl')
    feed = feed[['feedid', 'authorid'] + text_cols]
    # 使用authorid来填充bgm缺失
    # feed.loc[feed.bgm_song_id.isna(), 'bgm_song_id'] = (feed.loc[feed.bgm_song_id.isna(), 'authorid']*-1).astype(int).astype(str)
    # feed.loc[feed.bgm_singer_id.isna(), 'bgm_singer_id'] = (feed.loc[feed.bgm_singer_id.isna(), 'authorid']*-1).astype(int).astype(str)
    feed = feed.astype(str)
    # user数据
    # 初赛用户行为数据
    user_act = pd.read_csv(f'{RAW_DATA_PATH}/user_action.csv', header=0)[['userid', 'feedid', 'date_']]
    user_act[['userid', 'feedid']] = user_act[['userid', 'feedid']].astype(str)
    # TODO 复赛用户行为数据
    # user_act1 = pd.read_csv(f'{RAW_DATA_PATH}/user_action.csv', header=0)[['userid', 'feedid', 'date_']]
    # user_act1[['userid', 'feedid']] = user_act1[['userid', 'feedid']].astype(str)
    # user_act = user_act.append(user_act1)

    # test数据
    # 初赛a榜数据
    testa = pd.read_csv(f'{RAW_DATA_PATH}/test_a.csv', header=0)[['userid', 'feedid']]
    testa[['userid', 'feedid']] = testa[['userid', 'feedid']].astype(str)
    # 初赛b榜数据
    testb = pd.read_csv(f'{RAW_DATA_PATH}/test_b.csv', header=0)[['userid', 'feedid']]
    testb[['userid', 'feedid']] = testb[['userid', 'feedid']].astype(str)
    # TODO 复赛a榜数据
    # testa2 = pd.read_csv(f'{RAW_DATA_PATH}/test_a.csv', header=0)[['userid', 'feedid']]
    # testa2[['userid', 'feedid']] = testa2[['userid', 'feedid']].astype(str)
    # test = testa.append(testb).append(testa2)
    test = testa.append(testb)
    test['date_'] = 15
    test = test[user_act.columns]

    # 纵向合并
    user_act = user_act.append(test)
    # 关联得到所有用户行为的feed author
    df_act_feed = user_act.merge(feed, on='feedid', how='left').sort_values(by='date_', ascending=True).reset_index(
        drop=True)
    del user_act, test
    gc.collect()
    # 分别得到用户的物料和作者序列
    feed_text = df_act_feed.groupby('userid')[['feedid', 'authorid'] + text_cols].agg(list).reset_index()
    # 得到物料曝光的用户序列
    user_text1 = df_act_feed.groupby('feedid')['userid'].agg(list).reset_index()
    user_text1['size'] = user_text1['userid'].apply(len)
    user_text1 = user_text1.query('size>=2')
    # 得到视频作者曝光的用户序列
    user_text2 = df_act_feed.groupby('authorid')['userid'].agg(list).reset_index()
    user_text2['size'] = user_text2['userid'].apply(len)
    user_text2 = user_text2.query('size>=2')

    return feed_text, [list(user_text1['userid']), list(user_text2['userid'])]


def fit_w2v():
    """
    对userid序列、feedid序列、authorid序列进行word2vec embedding
    """
    texts, text_userids = get_sentences()  # texts：用户浏览的物料和作者序列；text_userids[0]：物料曝光的用户序列，text_userids[1]：视频作者曝光的用户序列
    feed_cols = ['feedid', 'authorid'] + text_cols
    user_cols = ['feedid', 'authorid']

    for col in feed_cols:  # 生成物料序列，作者序列
        print(f"Training {col} models >>> >>>")
        sentences = list(texts[col])
        sentences_rmdup = []
        # 连续多个重复仅保留两个
        for s in sentences:  # 对应每个用户
            if (len(s) == 1):  # 一条曝光跳过
                continue
            elif len(s) == 2:
                sentences_rmdup.append(s)
            else:  # 多条曝光三条内去重
                new_s = [i for i, j in zip_longest(s, s[2:]) if i != j]
                if len(new_s) >= 2:
                    sentences_rmdup.append(new_s)
        w2v = Word2Vec(sentences_rmdup, vector_size=size, window=7, min_count=1, sg=w_model, hs=opt, workers=48,
                       epochs=epoch)
        pickle.dump(w2v, open(f'{W2V_DATA_PATH}/{col}_w7_iter10.{size}d.pkl', 'wb'))

    for i in range(len(user_cols)):  # 生成用户序列
        print(f"Training {user_cols[i]} userid models >>> >>>")
        text_userids_rmdup = []
        # 连续多个重复仅保留两个
        for s in text_userids[i]:
            if (len(s) == 1):
                continue
            elif len(s) == 2:
                text_userids_rmdup.append(s)
            else:
                new_s = [i for i, j in zip_longest(s, s[2:]) if i != j]
                if len(new_s) >= 2:
                    text_userids_rmdup.append(new_s)

        w2v = Word2Vec(text_userids_rmdup, vector_size=size, window=10, min_count=1, sg=w_model, hs=opt, workers=48,
                       epochs=epoch)
        pickle.dump(w2v, open(f'{W2V_DATA_PATH}/userid_by_{user_cols[i]}_w10_iter10.{size}d.pkl', 'wb'))


def mysplit2(x):
    lst = []
    for i in x:
        lst.extend(i)
    return lst


def text_embedding():
    """
    根据用户行为对 'manual_tag_list', 'manual_keyword_list'中的词进行embedding.
    """
    feed = pd.read_pickle(f'{REFORM_DATA_PATH}/feed_author_text_features_fillna_by_author_clusters.pkl')[
        ['feedid', 'authorid', 'manual_tag_list', 'manual_keyword_list', 'machine_tag_list', 'machine_keyword_list']]
    feed.loc[(feed.manual_tag_list.isnull()) | (feed.manual_tag_list == ''), 'manual_tag_list'] = (
            feed.loc[(feed.manual_tag_list.isnull()) | (feed.manual_tag_list == ''), 'authorid'] * -1).astype(str)
    feed.loc[(feed.manual_keyword_list.isnull()) | (feed.manual_keyword_list == ''), 'manual_keyword_list'] = (
            feed.loc[(feed.manual_keyword_list.isnull()) | (feed.manual_keyword_list == ''), 'authorid'] * -1).astype(
        str)
    feed.loc[(feed.machine_tag_list.isnull()) | (feed.machine_tag_list == ''), 'machine_tag_list'] = (
            feed.loc[(feed.machine_tag_list.isnull()) | (feed.machine_tag_list == ''), 'authorid'] * -1).astype(str)
    feed.loc[(feed.machine_keyword_list.isnull()) | (feed.machine_keyword_list == ''), 'machine_keyword_list'] = (
            feed.loc[(feed.machine_keyword_list.isnull()) | (feed.machine_keyword_list == ''), 'authorid'] * -1).astype(
        str)

    feed['manual_tag_list'] = feed[['manual_tag_list']].applymap(lambda x: x.split())
    feed['manual_keyword_list'] = feed[['manual_keyword_list']].applymap(lambda x: x.split())
    feed['machine_tag_list'] = feed[['machine_tag_list']].applymap(lambda x: x.split())
    feed['machine_keyword_list'] = feed[['machine_keyword_list']].applymap(lambda x: x.split())
    feed['feed_id'] = feed['feed_id'].astype(str)

    # 初赛用户行为数据
    user_act = pd.read_csv(f'{RAW_DATA_PATH}/user_action.csv', header=0)[['userid', 'feedid', 'date_']]
    user_act[['userid', 'feedid']] = user_act[['userid', 'feedid']].astype(str)
    # 复赛a榜数据
    testa2 = pd.read_csv(f'{RAW_DATA_PATH}/test_a.csv', header=0)[['userid', 'feedid']]
    testa2[['userid', 'feedid']] = testa2[['userid', 'feedid']].astype(str)

    # TODO 复赛用户行为数据
    # user_act1 = pd.read_csv(f'{RAW_DATA_PATH_pri}/user_action.csv', header=0)[['userid', 'feedid', 'date_']]
    # user_act1[['userid', 'feedid']] = user_act1[['userid', 'feedid']].astype(str)
    # user_act = user_act.append(user_act1)

    # 初赛a榜数据
    testa = pd.read_csv(f'{RAW_DATA_PATH}/test_a.csv', header=0)[['userid', 'feedid']]
    testa[['userid', 'feedid']] = testa[['userid', 'feedid']].astype(str)
    # 初赛b榜数据
    testb = pd.read_csv(f'{RAW_DATA_PATH}/test_b.csv', header=0)[['userid', 'feedid']]
    testb[['userid', 'feedid']] = testb[['userid', 'feedid']].astype(str)
    test = testa.append(testb).append(testa2)
    test['date_'] = 15
    test = test[user_act.columns]

    user_act = user_act.append(test)
    user_act = user_act.merge(feed, on='feedid', how='inner')
    user_act = user_act.sort_values(by='date_', ascending=True)

    user_info = user_act.groupby('userid')[
        ['manual_tag_list', 'manual_keyword_list', 'machine_tag_list', 'machine_keyword_list']].agg(
        lambda x: list(x)).reset_index()
    del user_act
    gc.collect()

    user_info['manual_tag_list'] = user_info['manual_tag_list'].apply(lambda x: mysplit2(x))
    user_info['manual_keyword_list'] = user_info['manual_keyword_list'].apply(lambda x: mysplit2(x))
    user_info['machine_tag_list'] = user_info['machine_tag_list'].apply(lambda x: mysplit2(x))
    user_info['machine_keyword_list'] = user_info['machine_keyword_list'].apply(lambda x: mysplit2(x))

    # 训练word2vec
    manu_tag_w2v = Word2Vec(list(user_info['manual_tag_list']), vector_size=64, window=15, min_count=1, sg=w_model,
                            hs=opt,
                            workers=16, epochs=10)
    pickle.dump(manu_tag_w2v, open(f'{W2V_DATA_PATH}/manu_tag_w15_iter10.64d.pkl', 'wb'))

    # 训练word2vec
    manu_kw_w2v = Word2Vec(list(user_info['manual_keyword_list']), vector_size=64, window=15, min_count=1, sg=w_model,
                           hs=opt,
                           workers=16, epochs=10)
    pickle.dump(manu_kw_w2v, open(f'{W2V_DATA_PATH}/manu_kw_w15_iter10.64d.pkl', 'wb'))

    # 训练word2vec
    machine_tag_w2v = Word2Vec(list(user_info['machine_tag_list']), vector_size=64, window=15, min_count=1, sg=w_model,
                               hs=opt, workers=16, epochs=10)
    pickle.dump(machine_tag_w2v, open(f'{W2V_DATA_PATH}/machine_tag_w15_iter10.64d.pkl', 'wb'))

    # 训练word2vec
    machine_kw_w2v = Word2Vec(list(user_info['machine_keyword_list']), vector_size=64, window=15, min_count=1,
                              sg=w_model,
                              hs=opt, workers=16, epochs=10)
    pickle.dump(machine_kw_w2v, open(f'{W2V_DATA_PATH}/machine_kw_w15_iter10.64d.pkl', 'wb'))


def sent2vec(x, mm):
    vec = np.zeros(64)
    n = 0
    for word in x:
        try:
            vec += mm.wv.get_vector(word)
            n += 1
        except:
            pass
    vec = vec / n
    return vec


def get_feed_tag_kw_avg_emb():
    """
    生成feed的tag、keyword average embedding.
    """
    feed = pd.read_pickle(f'{REFORM_DATA_PATH}/feed_author_text_features_fillna_by_author_clusters.pkl')[
        ['feedid', 'authorid', 'manual_tag_list', 'manual_keyword_list', 'machine_tag_list', 'machine_keyword_list']]
    feed.loc[(feed.manual_tag_list.isnull()) | (feed.manual_tag_list == ''), 'manual_tag_list'] = (
            feed.loc[(feed.manual_tag_list.isnull()) | (feed.manual_tag_list == ''), 'authorid'] * -1).astype(str)
    feed.loc[(feed.manual_keyword_list.isnull()) | (feed.manual_keyword_list == ''), 'manual_keyword_list'] = (
            feed.loc[(feed.manual_keyword_list.isnull()) | (feed.manual_keyword_list == ''), 'authorid'] * -1).astype(
        str)
    feed.loc[(feed.machine_tag_list.isnull()) | (feed.machine_tag_list == ''), 'machine_tag_list'] = (
            feed.loc[(feed.machine_tag_list.isnull()) | (feed.machine_tag_list == ''), 'authorid'] * -1).astype(str)
    feed.loc[(feed.machine_keyword_list.isnull()) | (feed.machine_keyword_list == ''), 'machine_keyword_list'] = (
            feed.loc[(feed.machine_keyword_list.isnull()) | (feed.machine_keyword_list == ''), 'authorid'] * -1).astype(
        str)

    feed['manual_tag_list'] = feed[['manual_tag_list']].applymap(lambda x: x.split())
    feed['manual_keyword_list'] = feed[['manual_keyword_list']].applymap(lambda x: x.split())
    feed['machine_tag_list'] = feed[['machine_tag_list']].applymap(lambda x: x.split())
    feed['machine_keyword_list'] = feed[['machine_keyword_list']].applymap(lambda x: x.split())

    manu_tag_w2v = pickle.load(open(f'{W2V_DATA_PATH}/manu_tag_w15_iter10.64d.pkl', 'rb'))
    manu_kw_w2v = pickle.load(open(f'{W2V_DATA_PATH}/manu_kw_w15_iter10.64d.pkl', 'rb'))
    machine_tag_w2v = pickle.load(open(f'{W2V_DATA_PATH}/machine_tag_w15_iter10.64d.pkl', 'rb'))
    machine_kw_w2v = pickle.load(open(f'{W2V_DATA_PATH}/machine_kw_w15_iter10.64d.pkl', 'rb'))

    feed['manu_tag_emb'] = feed['manual_tag_list'].apply(lambda x: sent2vec(x, manu_tag_w2v))
    feed['manu_kw_emb'] = feed['manual_keyword_list'].apply(lambda x: sent2vec(x, manu_kw_w2v))
    feed[[f'manu_tag_emb_{i}' for i in range(64)]] = feed['manu_tag_emb'].apply(pd.Series)
    feed[[f'manu_kw_emb_{i}' for i in range(64)]] = feed['manu_kw_emb'].apply(pd.Series)

    reduce_mem_usage(feed[['feedid'] + [f'manu_tag_emb_{i}' for i in range(64)]]).to_pickle(
        f'{W2V_DATA_PATH}/feed_manu_tag_emb_df.64d.pkl')
    reduce_mem_usage(feed[['feedid'] + [f'manu_kw_emb_{i}' for i in range(64)]]).to_pickle(
        f'{W2V_DATA_PATH}/feed_manu_kw_emb_df.64d.pkl')
    feed = reduce_mem_usage(feed)

    # 转换为gensim model
    _ = dict2model(feed, idcol='feedid', cols=[f'manu_tag_emb_{i}' for i in range(64)],
                   save_name=f'{W2V_DATA_PATH}/feed_manu_tag_emb.64d.pkl')
    _ = dict2model(feed, idcol='feedid', cols=[f'manu_kw_emb_{i}' for i in range(64)],
                   save_name=f'{W2V_DATA_PATH}/feed_manu_kw_emb.64d.pkl')


if __name__ == '__main__':
    fit_w2v()
