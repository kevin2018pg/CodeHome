# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 21:28
# @Author  : west
# @File    : model_train.py
# @Version : python 3.6
# @Desc    : 训练模块

import gc
import pickle
import sys

sys.path.append('../')
import feature_preprocess_util
from model.mmoe_model import MOE

import logging

logger = logging.getLogger(__name__)

from deepctr_torch.inputs import SparseFeat
import torch
import numpy as np
import random


def train_single_model(args, np_rd_seed=2345, rd_seed=2345, torch_seed=1233):
    np.random.seed(np_rd_seed)
    random.seed(rd_seed)
    moe = MOE(dnn_hidden_units=args['hidden_units'], linear_feature_columns=args['linear_feature_columns'],
              dnn_feature_columns=args['dnn_feature_columns'], task='binary', dnn_dropout=args['dropout'],
              l2_reg_embedding=args['l2_reg_embedding'], l2_reg_dnn=args['l2_reg_dnn'], device=device, seed=torch_seed,
              num_tasks=args['num_tasks'], pretrained_user_emb_weight=[user_emb_weight],
              pretrained_author_emb_weight=[author_emb_weight],
              pretrained_feed_emb_weight=[feed_emb_weight, official_feed_weight], )
    moe.compile(optimizer=args['optimizer'], learning_rate=args['learning_rate'], loss="bcelogit",
                metrics=["bce", 'auc', 'uauc'])
    metric = moe.fit(online_train_loader, batch_size=args['batch_size'], validation_data=None, epochs=args['epochs'],
                     val_userid_list=None, lr_scheduler=args['lr_scheduler'], scheduler_epochs=args['scheduler_epochs'],
                     scheduler_method=args['scheduler_method'], num_warm_epochs=args['num_warm_epochs'],
                     reduction=args['reduction'], task_dict=args['task_dict'], task_weight=args['task_weight'],
                     verbose=2, early_stop_uauc=0.55)
    torch.save(moe.state_dict(), f'{MODEL_SAVE_PATH}/npseed{np_rd_seed}_rdseed{rd_seed}_torchseed{torch_seed}')
    del moe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    global MODEL_SAVE_PATH
    MODEL_SAVE_PATH = '../../data/model'
    RAW_DATA_PATH = '../../data'
    REFORM_DATA_PATH = '../../data/reform_data'
    TRAIN_DATA_PATH = '../../data/train_data'

    pretrained_models = {
        'sg_ns_64_epoch30': {
            'official_feed': f'{REFORM_DATA_PATH}/official_feed_emb.d512.pkl',  # feed多模态embedding，存储feedid对应的w2v特征
            # 'feedid': f'{REFORM_DATA_PATH}/w2v_models_sg_ns_dim64_epoch30/feedid_w7_iter10.64d.filled_cold.pkl',
            'feedid': f'{REFORM_DATA_PATH}/w2v_models_sg_ns_dim64_epoch30/feedid_w7_iter10.64d.pkl',
            # userid浏览feed序列离线embedding
            # 'authorid': f'{REFORM_DATA_PATH}/w2v_models_sg_ns_dim64_epoch30/authorid_w7_iter10.64d.filled_cold.pkl',
            'authorid': f'{REFORM_DATA_PATH}/w2v_models_sg_ns_dim64_epoch30/authorid_w7_iter10.64d.pkl',
            # userid浏览作者序列离线embedding
            'userid_by_feed': f'{REFORM_DATA_PATH}/w2v_models_sg_ns_dim64_epoch30/userid_by_feedid_w10_iter10.64d.pkl'
            # feedid下user序列离线embedding
        }
    }

    USED_FEATURES = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'videoplayseconds_bin', 'bgm_na',
                     'videoplayseconds', 'tag_manu_machine_corr'] + ['feed_machine_tag_tfidf_cls_32',
                                                                     'feed_machine_kw_tfidf_cls_17',
                                                                     'author_machine_tag_tfidf_cls_21',
                                                                     'author_machine_kw_tfidf_cls_18']
    args = {}
    args['USED_FEATURES'] = USED_FEATURES
    args['REFORM_DATA_PATH'] = REFORM_DATA_PATH
    global hidden_units
    hidden_units = (784, 512, 128)
    args['hidden_units'] = hidden_units
    args['batch_size'] = 40000
    args['emb_dim'] = 128
    args['learning_rate'] = 0.06
    args['lr_scheduler'] = True
    args['epochs'] = 2
    args['scheduler_epochs'] = 3
    args['num_warm_epochs'] = 0
    args['scheduler_method'] = 'cos'
    args['use_bn'] = True
    args['reduction'] = 'sum'
    args['optimizer'] = 'adagrad'
    args['num_tasks'] = 7
    args['early_stop_uauc'] = 0.689
    args['num_workers'] = 7
    args['dropout'] = 0.0
    args['l2_reg_dnn'] = 0.001
    args['l2_reg_embedding'] = 0.01
    args['task_dict'] = {
        0: 'read_comment',
        1: 'like',
        2: 'click_avatar',
        3: 'forward',
        4: 'favorite',
        5: 'comment',
        6: 'follow'
    }
    args['task_weight'] = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1
    }
    args['pretrained_model'] = pretrained_models['sg_ns_64_epoch30']

    logger.info("Parameters: ")
    logger.info(args)

    # 加载全部特征
    linear_feature_columns = pickle.load(open(TRAIN_DATA_PATH + '/linear_feature.pkl', 'rb'))
    dnn_feature_columns = pickle.load(open(TRAIN_DATA_PATH + '/dnn_feature.pkl', 'rb'))
    # 使用其中部分特征
    linear_feature_columns = [f for f in linear_feature_columns if f.name in USED_FEATURES]
    dnn_feature_columns = [f for f in dnn_feature_columns if f.name in USED_FEATURES]
    features = []
    for f in linear_feature_columns:
        if isinstance(f, SparseFeat):
            features.append(SparseFeat(f.name, f.vocabulary_size, args['emb_dim']))
        else:
            features.append(f)

    linear_feature_columns = features
    dnn_feature_columns = features
    args['linear_feature_columns'] = linear_feature_columns
    args['dnn_feature_columns'] = dnn_feature_columns

    # 加载数据集和encoder模型
    lbe_dict = pickle.load(open(REFORM_DATA_PATH + '/label_encoder_models/lbe_dic_all.pkl', 'rb'))

    train_x = pickle.load(open(TRAIN_DATA_PATH + '/train_x.pkl', 'rb'))
    train_y = pickle.load(open(TRAIN_DATA_PATH + '/train_y.pkl', 'rb'))
    val_x = pickle.load(open(TRAIN_DATA_PATH + '/val_x.pkl', 'rb'))
    val_y = pickle.load(open(TRAIN_DATA_PATH + '/val_y.pkl', 'rb'))

    # 选取数据集中使用的特征
    train_X = {f.name: train_x[f.name] for f in dnn_feature_columns}
    val_X = {f.name: val_x[f.name] for f in dnn_feature_columns}

    # 加载预训练embedding weight matrix
    global user_emb_weight, author_emb_weight, feed_emb_weight, official_feed_weight
    user_emb_weight = feature_preprocess_util.load_feature_pretrained_embedding(lbe_dict['userid'],
                                                                                args['pretrained_model'][
                                                                                    'userid_by_feed'], padding=True)
    # user_by_author_emb_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['userid'],
    #                                                                          args['pretrained_model'][
    #                                                                              'userid_by_author'], padding=True)
    author_emb_weight = feature_preprocess_util.load_feature_pretrained_embedding(lbe_dict['authorid'],
                                                                                  args['pretrained_model']['authorid'],
                                                                                  padding=True)
    feed_emb_weight = feature_preprocess_util.load_feature_pretrained_embedding(lbe_dict['feedid'],
                                                                                args['pretrained_model']['feedid'],
                                                                                padding=True)
    # feed_emb_weight_eges = preprocess.load_feature_pretrained_embedding(lbe_dict['feedid'],
    #                                                                     '../my_data/eges/feedid_eges0_emb.pkl',
    #                                                                     padding=True)
    # TODO official_feed_weight保存的是模型，后期加载方式与其他有区别
    official_feed_weight = feature_preprocess_util.load_feature_pretrained_embedding(lbe_dict['feedid'],
                                                                                     args['pretrained_model'][
                                                                                         'official_feed'], padding=True)
    logger.info('All used features:')
    logger.info(train_X.keys())

    device = 'gpu'
    if device == 'gpu' and torch.cuda.is_available():
        # print('cuda ready...')
        device = 'cuda:1'
    else:
        device = 'cpu'
    logger.info(f'device:{device}')

    # 临时加载模型取特征id
    _moe = MOE(dnn_hidden_units=args['hidden_units'], linear_feature_columns=linear_feature_columns,
               dnn_feature_columns=dnn_feature_columns, task='binary', dnn_dropout=0., l2_reg_embedding=0.,
               l2_reg_dnn=0., l2_reg_linear=0., device=device, seed=1233, num_tasks=args['num_tasks'],
               pretrained_user_emb_weight=None, pretrained_author_emb_weight=None, pretrained_feed_emb_weight=None, )

    # 用于线上预测的训练集, 初赛+复赛+初复赛验证集
    online_train_X = {}
    for col in train_X:
        online_train_X[col] = np.concatenate((train_X[col], val_X[col]), axis=0)
    online_train_y = np.concatenate((train_y, val_y), axis=0)
    online_train_loader = feature_preprocess_util.get_dataloader(online_train_X, _moe, online_train_y,
                                                                 batch_size=args['batch_size'], num_workers=7)
    del _moe
    gc.collect()
    # 测试
    # train_single_model(args, np_rd_seed=2345, rd_seed=2345, torch_seed=1233)
    for _ in range(50):
        seed1 = random.randint(1, 100000)
        seed2 = random.randint(1, 100000)
        seed3 = random.randint(1, 100000)
        logger.info("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.info("np random seed = " + str(seed1))
        logger.info("random seed = " + str(seed2))
        logger.info("torch random seed = " + str(seed3))
        train_single_model(args, np_rd_seed=seed1, rd_seed=seed2, torch_seed=seed3)
