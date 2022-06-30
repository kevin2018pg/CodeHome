## **1. 环境依赖**
- pandas==1.0.5
- numpy==1.19.5
- numba==0.53.1
- scipy==1.5.0
- torch==1.4.0
- python==3.6.5
- gensim==3.8.0
- deepctr-torch==0.2.7
- transformers==3.1.0
- bayesian-optimization==1.2.0
- tensorflow==2.5.0
- tensorflow-estimator==2.5.0

## **2. 目录**


```
./
├── README.md
├── requirements.txt, python package requirements 
├── init.sh, script for installing package requirements
├── train.sh, script for preparing train/inference data and training models, including pretrained models
├── inference.sh, script for inference 
├── src
│   ├── prepare, codes for preparing train/inference dataset
|       ├──Step1
|       ├──Step2
|       ├──Step3
|       ├──Step4
|       ├──Step5
│   ├── model, codes for model architecture
|       ├──moe
|   ├── train, codes for training
|       ├──preprocess
|       ├──generate_train_data
|       ├──opt_moe
|       ├──train
|   ├── inference.py, main function for inference on test dataset
|   ├── utils.py, some utils functions
├── data
│   ├── dataset
│       ├── data1
│       ├── data2
│   ├── train data and features for training models
│   ├── submission, prediction result after running inference.sh
│   ├── model, model files (e.g. pytorch trained model state dict)

```

## **3. 运行**
- 进入目录
- 安装环境：source init.sh
- 预测并生成结果文件：sh inference.sh ./data/test_b.csv
- 数据准备和模型训练：sh train.sh

## **4. 模型及特征**
- 模型：Multi-perceptron DNN
- 参数：
    - batch_size: 40000
    - emded_dim: 128
    - num_epochs: 2
    - learning_rate: 0.06
    
- 特征：userid, feedid, authorid, bgm_singer_id, bgm_song_id, videoplayseconds, feed和author的tag、keyword聚类特征，
      以及user、feed、author的word2vec Embedding特征;


## **5. 算法性能**
- 资源配置：2*P40_48G显存_14核CPU_112G内存
- 预测耗时
    - 单模总预测时长: 3418 s
    - 单个目标行为2000条样本的平均预测时长: 228 ms


## **6. 代码说明**
模型预测部分代码位置如下：

| 路径 | 行数 | 内容 |
| :--- | :--- | :--- |
| src/inference.py | 71 | `pred_arr = moe.predict(test_loader)`|

## **7. 相关文献**
无
