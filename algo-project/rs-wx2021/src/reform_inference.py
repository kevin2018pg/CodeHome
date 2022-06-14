# -*- coding: utf-8 -*-
# @Time    : 2022/4/28 14:16
# @Author  : west
# @File    : reform_inference.py
# @Version : python 3.6
# @Desc    : 推理

import logging
import os
import pickle
import time

from deepctr_torch.inputs import SparseFeat

logger = logging.getLogger(__name__)

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
                     'videoplayseconds'] + ['feed_machine_tag_tfidf_cls_32', 'feed_machine_kw_tfidf_cls_17',
                                            'author_machine_tag_tfidf_cls_21', 'author_machine_kw_tfidf_cls_18']
    RAW_DATA_PATH = '../../data'
    REFORM_DATA_PATH = '../../data/reform_data'
    LBE_MODEL_PATH = '../../data/reform_data/label_encoder_models/lbe_dic_all.pkl'
    MODEL_PATH = '../data/model/'
    TRAIN_DATA_PATH = '../../data/train_data'
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

    # 加载预训练权重
