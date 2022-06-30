# -*- coding: utf-8 -*-
# @Time    : 2022/4/5 15:35
# @Author  : west
# @File    : generate_train_dat.py
# @Version : python 3.6
# @Desc    : 生成训练验证数据集

import pickle

import feature_preprocess_util


def process_pipe(feed_path, user_act_path, used_columns, used_sparse_cols, used_dense_cols, emb_dim=16,
                 is_training=True, test_data=False):
    data = feature_preprocess_util.preprocess(feed_path, user_act_path)  # 数据预处理，user-feed去重（保留一条user-feed正行为样本）
    data_ds = feature_preprocess_util.down_sample(data, used_columns, sample_method=None, neg2pos_ratio=300,
                                                  user_samp='random', by_date=None,
                                                  is_training=is_training)  # 采样：随机采样/按用户采样
    # 生成训练集/验证集/测试集/特征字段/onehot编码
    if (list(data_ds.head(2)['date_'])[0] == 15):  # test_data
        X_dic, y_arr, linear_feats, dnn_feats, lbe_dict = feature_preprocess_util.process_features(data_ds,
                                                                                                   used_sparse_cols,
                                                                                                   used_dense_cols,
                                                                                                   actions=ACTIONS,
                                                                                                   emb_dim=emb_dim,
                                                                                                   use_tag_text=None,
                                                                                                   use_kw_text=None,
                                                                                                   feed_history=None,
                                                                                                   author_history=None,
                                                                                                   use_din=False,
                                                                                                   max_seq_length=128,
                                                                                                  behavior_feature_list=[
                                                                                                      'feedid',
                                                                                                      'authorid'], )
        return [(X_dic, y_arr, linear_feats, dnn_feats, lbe_dict)]

    else:  # train data
        train_data = data_ds.query('date_<14')
        val_data = data_ds.query('date_==14')
        X_dic_train, y_arr_train, linear_feats, dnn_feats, lbe_dict = feature_preprocess_util.process_features(
            train_data,
            used_sparse_cols,
            used_dense_cols,
            actions=ACTIONS,
            emb_dim=emb_dim,
            use_tag_text=None,
            use_kw_text=None,
            feed_history=None,
            author_history=None,
            use_din=False,
            max_seq_length=128,
            behavior_feature_list=[
                'feedid',
                'authorid'], )
        X_dic_val, y_arr_val, linear_feats, dnn_feats, lbe_dict = feature_preprocess_util.process_features(val_data,
                                                                                                           used_sparse_cols,
                                                                                                           used_dense_cols,
                                                                                                           actions=ACTIONS,
                                                                                                           emb_dim=emb_dim,
                                                                                                           use_tag_text=None,
                                                                                                           use_kw_text=None,
                                                                                                           feed_history=None,
                                                                                                           author_history=None,
                                                                                                           use_din=False,
                                                                                                           max_seq_length=128,
                                                                                                          behavior_feature_list=[
                                                                                                              'feedid',
                                                                                                              'authorid'], )
        return [(X_dic_train, y_arr_train, linear_feats, dnn_feats, lbe_dict),
                (X_dic_val, y_arr_val, linear_feats, dnn_feats, lbe_dict)]


if __name__ == '__main__':
    CLS_COLS = ['feed_manu_tag_tfidf_cls_32', 'feed_machine_tag_tfidf_cls_32', 'feed_manu_kw_tfidf_cls_22',
                'feed_machine_kw_tfidf_cls_17', 'feed_description_tfidf_cls_18', 'author_manu_tag_tfidf_cls_19',
                'author_machine_tag_tfidf_cls_21', 'author_manu_kw_tfidf_cls_18', 'author_machine_kw_tfidf_cls_18',
                'author_description_tfidf_cls_18']  # 聚类特征

    TOPIC_COLS = ['feed_manu_tag_topic_class', 'feed_machine_tag_topic_class', 'feed_manu_kw_topic_class',
                  'feed_machine_kw_topic_class', 'feed_description_topic_class', 'author_description_topic_class',
                  'author_manu_kw_topic_class', 'author_machine_kw_topic_class', 'author_manu_tag_topic_class',
                  'author_machine_tag_topic_class']  # 主题特征

    SPARSE_COLS = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'videoplayseconds_bin',
                   'bgm_na'] + CLS_COLS + TOPIC_COLS
    DENSE_COLS = ['videoplayseconds', 'tag_manu_machine_corr']
    ACTIONS = ["read_comment", "like", "click_avatar", "forward", 'favorite', 'comment', 'follow']

    USED_COLUMNS = SPARSE_COLS + DENSE_COLS + ACTIONS

    RAW_DATA_PATH = '../../data'
    REFORM_DATA_PATH = '../../data/reform_data'
    TRAIN_DATA_PATH = '../../data/train_data'

    test_a = RAW_DATA_PATH + '/test_a.csv'
    user_action = RAW_DATA_PATH + '/user_action.csv'
    feed_info = REFORM_DATA_PATH + '/feed_author_text_features_fillna_by_author_clusters.pkl'
    # 去重/采样/生成样本特征+onhot 编码
    train, val = process_pipe(feed_info, user_action, USED_COLUMNS, SPARSE_COLS, DENSE_COLS)

    pickle.dump(train[0], open(f'{TRAIN_DATA_PATH}/train_x.pkl', 'wb'))
    pickle.dump(train[1], open(f'{TRAIN_DATA_PATH}/train_y.pkl', 'wb'))

    pickle.dump(val[0], open(f'{TRAIN_DATA_PATH}/val_x.pkl', 'wb'))
    pickle.dump(val[1], open(f'{TRAIN_DATA_PATH}/val_y.pkl', 'wb'))

    # 特征columns存储
    print("linear_feature: ", train[2])
    print("dnn_feature: ", train[3])
    pickle.dump(train[2], open(f'{TRAIN_DATA_PATH}/linear_feature.pkl', 'wb'))
    pickle.dump(train[3], open(f'{TRAIN_DATA_PATH}/dnn_feature.pkl', 'wb'))
