# -*- coding: utf-8 -*-
# @Time    : 2022/4/28 14:16
# @Author  : west
# @File    : inference.py
# @Version : python 3.6
# @Desc    : 推理

import gc
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from deepctr_torch.inputs import SparseFeat

from model.mmoe_model import MOE
from utils import reduce_mem_usage

sys.path.append('src/train')
import feature_preprocess_util

logger = logging.getLogger(__name__)


def get_test_feats(test_path):
    """
    读取测试集数据并join特征，生成测试集dataloader
    """
    df_test = pd.read_csv(test_path, header=0)[['userid', 'feedid', 'device']]
    df_id = df_test.copy()
    feed = pd.read_pickle(FEED_PATH)[FEED_FEATURES]

    # label encoder category features
    for feat in FEED_FEATURES:
        if (feat != 'feedid') and (feat in LBE_MODEL):
            feed[feat] = LBE_MODEL[feat].transform(feed[feat].astype(str)) + 1
    feed = reduce_mem_usage(feed)
    df_test = df_test[['userid', 'feedid']].merge(feed, on='feedid', how='left')
    df_test['feedid'] = (LBE_MODEL['feedid'].transform(df_test['feedid'].astype(str)) + 1).astype(np.int32)
    df_test['userid'] = (LBE_MODEL['userid'].transform(df_test['userid'].astype(str)) + 1).astype(np.int32)
    test_X = {name: df_test[name].values for name in USED_FEATURES}
    _moe = MOE(dnn_hidden_units=(512, 256, 128), linear_feature_columns=LINEAR_FEATURES,
               dnn_feature_columns=DNN_FEATURES, task='binary', dnn_dropout=0., l2_reg_embedding=0., l2_reg_dnn=0.,
               l2_reg_linear=0., device='cpu', seed=1233, num_tasks=7, pretrained_user_emb_weight=None,
               pretrained_author_emb_weight=None, pretrained_feed_emb_weight=None, )
    # 测试集
    online_test_loader = feature_preprocess_util.get_loader(test_X, _moe, y=None, batch_size=100000, num_workers=8)
    del _moe
    gc.collect()
    return df_id, online_test_loader


def infer_once(test_loader, model_path):
    """
    模型预测
    """
    moe = MOE(dnn_hidden_units=(784, 512, 128),
              linear_feature_columns=LINEAR_FEATURES,
              dnn_feature_columns=DNN_FEATURES, task='binary',
              dnn_dropout=0., l2_reg_embedding=0.01, l2_reg_dnn=0.001,
              device='cuda:0', seed=1233, num_tasks=7,
              pretrained_user_emb_weight=[user_emb_weight],
              pretrained_author_emb_weight=[author_emb_weight],
              pretrained_feed_emb_weight=[feed_emb_weight, official_feed_weight], )
    state_dic = torch.load(model_path)
    moe.load_state_dict(state_dic)
    pred_arr = moe.predict(test_loader)
    del moe
    gc.collect()
    torch.cuda.empty_cache()
    return pred_arr


def infer_test(test_path):
    test_sub, test_loader = get_test_feats(test_path)
    pred_arr = np.zeros((test_sub.shape[0], 7))
    t1 = time.time()
    for model_path in MODEL_LIST:
        pred_arr += infer_once(test_loader, model_path)
    t2 = time.time()
    t3 = t2 - t1
    logger.info(f"Model Inference cost {t3} secs.")
    pred_arr = pred_arr / len(MODEL_LIST)
    df_res = pd.DataFrame(pred_arr)
    df_res.columns = ["read_comment", "like", "click_avatar", "forward", 'favorite', 'comment', 'follow']
    test_sub = pd.concat([test_sub, df_res], axis=1)
    test_sub.loc[test_sub.device == 1, 'read_comment'] = 0
    test_sub[['userid', 'feedid', "read_comment", "like", "click_avatar", "forward", 'favorite', 'comment',
              'follow']].to_csv(OUT_PATH + '/result.csv', header=True, index=False)


if __name__ == '__main__':
    logger.info("Start infer >>> >>> ")
    s1 = time.time()
    FEED_FEATURES = ['feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'videoplayseconds_bin', 'bgm_na',
                     'videoplayseconds', 'tag_manu_machine_corr'] + ['feed_machine_tag_tfidf_cls_32',
                                                                     'feed_machine_kw_tfidf_cls_17',
                                                                     'author_machine_tag_tfidf_cls_21',
                                                                     'author_machine_kw_tfidf_cls_18']
    # 全部特征
    USED_FEATURES = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'videoplayseconds_bin', 'bgm_na',
                     'videoplayseconds', 'tag_manu_machine_corr'] + ['feed_machine_tag_tfidf_cls_32',
                                                                     'feed_machine_kw_tfidf_cls_17',
                                                                     'author_machine_tag_tfidf_cls_21',
                                                                     'author_machine_kw_tfidf_cls_18']

    RAW_DATA_PATH = 'data'
    REFORM_DATA_PATH = 'data/reform_data'
    LBE_MODEL_PATH = '../data/reform_data/label_encoder_models/lbe_dic_all.pkl'
    MODEL_PATH = 'data/model/'
    TRAIN_DATA_PATH = 'data/train_data'
    OUT_PATH = 'data/submission'
    # 模型文件
    MODEL_LIST = [os.path.join(MODEL_PATH, mpath) for mpath in os.listdir(MODEL_PATH) if mpath.startswith('npseed')]
    logger.info(f"Total {len(MODEL_LIST)} models.")
    # feed特征文件
    FEED_PATH = f'{REFORM_DATA_PATH}/feed_author_text_features_fillna_by_author_clusters.pkl'
    # label encoder模型文件
    LBE_MODEL = pickle.load(open(LBE_MODEL_PATH, 'rb'))
    # 读取特征名文件
    linear_feature_columns = pickle.load(open(TRAIN_DATA_PATH + '/linear_feature.pkl', 'rb'))
    dnn_feature_columns = pickle.load(open(TRAIN_DATA_PATH + '/dnn_feature.pkl', 'rb'))
    linear_feature_columns = [f for f in linear_feature_columns if f.name in USED_FEATURES]
    dnn_feature_columns = [f for f in dnn_feature_columns if f.name in USED_FEATURES]
    features = []
    for f in linear_feature_columns:
        if isinstance(f, SparseFeat):
            features.append(SparseFeat(f.name, f.vocabulary_size, 128))
        else:
            features.append(f)

    LINEAR_FEATURES = features
    DNN_FEATURES = features

    # 加载预训练权重（sg_ns_64_epoch30）
    USER_EMB_PATH = f'{REFORM_DATA_PATH}/w2v_models_sg_ns_dim64_epoch30/userid_by_feedid_w10_iter10.64d.pkl'  # feedid下user序列离线embedding
    AUTHOR_EMB_PATH = f'{REFORM_DATA_PATH}/w2v_models_sg_ns_dim64_epoch30/authorid_w7_iter10.64d.pkl'  # userid浏览作者序列离线embedding
    FEED_EMB_PATH = f'{REFORM_DATA_PATH}/w2v_models_sg_ns_dim64_epoch30/feedid_w7_iter10.64d.pkl'  # userid浏览feed序列离线embedding
    OFFICIAL_EMB_PATH = f'{REFORM_DATA_PATH}/official_feed_emb.d512.pkl'

    # 加载预训练embedding weight matrix
    global user_emb_weight, author_emb_weight, feed_emb_weight, official_feed_weight
    user_emb_weight = feature_preprocess_util.load_feature_pretrained_embedding(LBE_MODEL['userid'], USER_EMB_PATH,
                                                                                padding=True)
    author_emb_weight = feature_preprocess_util.load_feature_pretrained_embedding(LBE_MODEL['authorid'],
                                                                                  AUTHOR_EMB_PATH, padding=True)
    feed_emb_weight = feature_preprocess_util.load_feature_pretrained_embedding(LBE_MODEL['feedid'], FEED_EMB_PATH,
                                                                                padding=True)
    # TODO official_feed_weight保存的是模型，后期加载方式与其他有区别
    official_feed_weight = feature_preprocess_util.load_feature_pretrained_embedding(LBE_MODEL['feedid'],
                                                                                     OFFICIAL_EMB_PATH, padding=True)
    test_path = sys.argv[1]
    # test_path = '/home/tione/notebook/wbdc2021/data/wedata/wechat_algo_data2/test_a.csv'
    infer_test(test_path)
    s2 = time.time()
    s = s2 - s1
    logger.info(f"Inference Done! Total cost {s} secs.")
