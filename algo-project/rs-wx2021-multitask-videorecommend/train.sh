# 特征处理
cd ./src/prepare/
python text_tfidf_feature.py &&
  python text_cluster_feature.py &&
  python feed_user_author_w2v_feature.py &&

  # 生成训练数据
  cd ./src/train/
python generate_train_dat.py &&

  # 训练最终模型
  python model_train.py
