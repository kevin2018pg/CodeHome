# RankMixer

这个目录现在实现的是字节跳动论文 **RankMixer: Scaling Up Ranking Models in Industrial Recommenders** 的一个可运行工程版骨架，而不是旧的 logistic ranker demo。

论文来源：

- arXiv: [2507.15551](https://arxiv.org/abs/2507.15551)
- PDF: [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/pdf/2507.15551)

## 当前实现范围

已对齐的核心结构：

- 多个语义特征组先做 tokenization
- `Multi-Head Token Mixing`
- `Per-token FFN`
- 可选 `Sparse-MoE` per-token FFN
- 默认支持 `Dense-Training / Sparse-Inference (DTSI)` 风格路由
- block 内残差连接与 layer norm
- 最后 `mean pooling` 输出排序分数

当前没有复刻论文中的生产级部分：

- trillion 级训练数据与在线 serving 工程优化
- quantization / 高 MFU CUDA kernel / 自定义推理图优化
- 复杂序列模块与字节内部特征系统

## 文件说明

- `feature_config.json`: 特征分组配置，决定 token 数和每组字段
- `rank_mixer/features.py`: 配置加载与 schema 编码
- `rank_mixer/model.py`: RankMixer 主模型、Token Mixing、Per-token FFN、Sparse-MoE
- `rank_mixer/data.py`: 合成排序数据、DataLoader
- `train.py`: 训练入口
- `infer.py`: 推理入口
- `test_pipeline.py`: 端到端 smoke pipeline

## 安装

```powershell
pip install -r requirements.txt
```

## 训练

```powershell
python train.py --generate-data --epochs 5 --device cpu
```

启用论文里的 Sparse-MoE 版本：

```powershell
python train.py --generate-data --epochs 5 --use-sparse-moe --device cpu
```

关闭 `Dense-Training / Sparse-Inference`，改成训练时也做稀疏路由：

```powershell
python train.py --generate-data --epochs 5 --use-sparse-moe --disable-dtsi --device cpu
```

## 推理

```powershell
python infer.py --input-path artifacts/data/test.csv --model-path artifacts/model/rankmixer.pt
```

## 测试

```powershell
python test_pipeline.py
```
