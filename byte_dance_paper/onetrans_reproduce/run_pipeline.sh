#!/usr/bin/env bash
# =============================================================================
# OneTrans 完整训练 + 推理流程
#
# 用法：
#   bash run_pipeline.sh                    # mock 数据快速验证
#   bash run_pipeline.sh --mode parquet     # 真实 Parquet 数据训练
#   bash run_pipeline.sh --mode parquet --skip_train   # 跳过训练，只跑推理
#
# 环境要求：
#   - Python 3.8+，已安装 requirements.txt 依赖
#   - 有 GPU 自动用 GPU，没有自动用 CPU
# =============================================================================

set -euo pipefail   # 任意命令失败立即退出

# ─────────────────────────────────────────────────────────────────────────────
# 0. 解析参数
# ─────────────────────────────────────────────────────────────────────────────
MODE="mock"           # mock | parquet
SKIP_TRAIN=false
SKIP_INCR=false
SKIP_INFER=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)        MODE="$2";      shift 2 ;;
        --skip_train)  SKIP_TRAIN=true; shift ;;
        --skip_incr)   SKIP_INCR=true;  shift ;;
        --skip_infer)  SKIP_INFER=true; shift ;;
        *) echo "[错误] 未知参数: $1"; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# 1. 环境检测
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " OneTrans Pipeline  $(date '+%Y-%m-%d %H:%M:%S')"
echo " 工作目录: $SCRIPT_DIR"
echo " 数据模式: $MODE"
echo "============================================================"

# 检测 Python
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)
if [[ -z "$PYTHON" ]]; then
    echo "[错误] 找不到 Python，请先安装 Python 3.8+"
    exit 1
fi
echo "[环境] Python: $($PYTHON --version)"

# 检测 GPU（有 GPU 用 GPU，没有用 CPU）
if $PYTHON -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_COUNT=$($PYTHON -c "import torch; print(torch.cuda.device_count())")
    GPU_NAME=$($PYTHON  -c "import torch; print(torch.cuda.get_device_name(0))")
    DEVICE="cuda"
    echo "[环境] GPU: ${GPU_COUNT}x ${GPU_NAME}  ✓"
else
    DEVICE="cpu"
    echo "[环境] 未检测到 GPU，使用 CPU 训练（速度较慢）"
fi

# 检测依赖
echo "[环境] 检查依赖..."
$PYTHON -c "import torch, numpy, tqdm, sklearn, pandas, pyarrow, yaml" 2>/dev/null || {
    echo "[提示] 缺少依赖，正在安装..."
    pip install -r requirements.txt -q
}
echo "[环境] 依赖检查完成 ✓"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 2. 目录准备
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR="./checkpoints"
PREDICT_DIR="./predictions"
LOG_DIR="./logs"

mkdir -p "$CHECKPOINT_DIR" "$PREDICT_DIR" "$LOG_DIR"

BEST_MODEL="$CHECKPOINT_DIR/best_model.pt"
EMB_MODEL="$CHECKPOINT_DIR/emb_update_latest.pt"
HEAD_MODEL="$CHECKPOINT_DIR/head_update_latest.pt"
PREDICT_OUT="$PREDICT_DIR/predictions_$(date '+%Y%m%d_%H%M').csv"

# ─────────────────────────────────────────────────────────────────────────────
# 3. 全量训练
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SKIP_TRAIN" == "false" ]]; then
    echo "------------------------------------------------------------"
    echo "[Step 1/4] 全量训练"
    echo "------------------------------------------------------------"

    if [[ "$MODE" == "mock" ]]; then
        # ── mock 模式：随机数据，快速验证模型结构 ──
        echo "[训练] 使用模拟数据（mock 模式）"
        $PYTHON train.py \
            --data_mode   mock \
            --n_train     100000 \
            --n_val       20000 \
            --ns_dim      8 \
            --seq_lengths 50 10 20 \
            --seq_dims    3 3 2 \
            --d_model     128 \
            --n_heads     4 \
            --n_layers    4 \
            --L_NS        8 \
            --epochs      10 \
            --batch_size  512 \
            --lr          3e-4 \
            --save_path   "$BEST_MODEL" \
            2>&1 | tee "$LOG_DIR/train_full.log"

    else
        # ── parquet 模式：真实数据训练 ──
        TRAIN_DIR="${TRAIN_DIR:-./data/parquet/train}"
        VAL_DIR="${VAL_DIR:-./data/parquet/val}"
        FEATURE_CFG="${FEATURE_CFG:-./feature_config.yaml}"

        echo "[训练] 使用真实数据（parquet 模式）"
        echo "  训练集: $TRAIN_DIR"
        echo "  验证集: $VAL_DIR"
        echo "  特征配置: $FEATURE_CFG"

        # 训练前验证数据格式
        echo "[训练] 验证数据格式..."
        $PYTHON check_parquet.py \
            --data_dir      "$TRAIN_DIR" \
            --feature_config "$FEATURE_CFG" \
            2>&1 | tee "$LOG_DIR/check_data.log"

        $PYTHON train.py \
            --data_mode     parquet \
            --train_dir     "$TRAIN_DIR" \
            --val_dir       "$VAL_DIR" \
            --feature_config "$FEATURE_CFG" \
            --d_model       256 \
            --n_heads       8 \
            --n_layers      6 \
            --L_NS          8 \
            --epochs        5 \
            --batch_size    2048 \
            --num_workers   4 \
            --lr            0.001 \
            --save_path     "$BEST_MODEL" \
            2>&1 | tee "$LOG_DIR/train_full.log"
    fi

    echo "[Step 1/4] 全量训练完成 ✓  checkpoint: $BEST_MODEL"
else
    echo "[Step 1/4] 跳过全量训练（--skip_train）"
    if [[ ! -f "$BEST_MODEL" ]]; then
        echo "[错误] 找不到 checkpoint: $BEST_MODEL，请先运行全量训练"
        exit 1
    fi
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 4. 增量训练（仅 parquet 模式且有近期数据时运行）
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SKIP_INCR" == "false" && "$MODE" == "parquet" ]]; then
    echo "------------------------------------------------------------"
    echo "[Step 2/4] 增量训练（Embedding + Head）"
    echo "------------------------------------------------------------"

    RECENT_DIR="${RECENT_DIR:-./data/parquet/recent}"
    FEATURE_CFG="${FEATURE_CFG:-./feature_config.yaml}"

    if [[ -d "$RECENT_DIR" ]]; then
        # Step 2a：更新 hotel_id Embedding（冷启动 + 新酒店）
        echo "[增量] mode=emb：更新 hotel_id Embedding..."
        $PYTHON train_embedding.py \
            --mode            emb \
            --base_checkpoint "$BEST_MODEL" \
            --data_dir        "$RECENT_DIR" \
            --feature_config  "$FEATURE_CFG" \
            --emb_features    hotel_id \
            --save_path       "$EMB_MODEL" \
            --epochs          1 \
            --batch_size      512 \
            --lr              0.01 \
            2>&1 | tee "$LOG_DIR/train_emb.log"
        echo "[增量] Embedding 更新完成 ✓  checkpoint: $EMB_MODEL"

        # Step 2b：更新 Task Head + Calibration（应对分布漂移）
        echo "[增量] mode=head：更新 Task Head + Calibration..."
        $PYTHON train_embedding.py \
            --mode            head \
            --base_checkpoint "$BEST_MODEL" \
            --data_dir        "$RECENT_DIR" \
            --feature_config  "$FEATURE_CFG" \
            --save_path       "$HEAD_MODEL" \
            --epochs          1 \
            --batch_size      512 \
            --lr              0.001 \
            2>&1 | tee "$LOG_DIR/train_head.log"
        echo "[增量] Head 更新完成 ✓  checkpoint: $HEAD_MODEL"
    else
        echo "[增量] 未找到近期数据目录 $RECENT_DIR，跳过增量训练"
        echo "       （如需增量训练，请设置环境变量 RECENT_DIR=/path/to/recent/data）"
    fi
else
    echo "[Step 2/4] 跳过增量训练"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 5. 选择推理用的 checkpoint（优先级：head > emb > full）
# ─────────────────────────────────────────────────────────────────────────────
if [[ -f "$HEAD_MODEL" ]]; then
    INFER_MODEL="$HEAD_MODEL"
    echo "[推理] 使用 Head 增量模型: $INFER_MODEL"
elif [[ -f "$EMB_MODEL" ]]; then
    INFER_MODEL="$EMB_MODEL"
    echo "[推理] 使用 Embedding 增量模型: $INFER_MODEL"
else
    INFER_MODEL="$BEST_MODEL"
    echo "[推理] 使用全量模型: $INFER_MODEL"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 6. 离线批量推理
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SKIP_INFER" == "false" ]]; then
    echo "------------------------------------------------------------"
    echo "[Step 3/4] 离线批量推理"
    echo "------------------------------------------------------------"

    if [[ "$MODE" == "mock" ]]; then
        # mock 模式：用模拟数据推理（直接用 Python 验证模型可以跑通）
        echo "[推理] mock 模式：验证模型前向推理..."
        $PYTHON - <<'PYEOF'
import torch
import sys
sys.path.insert(0, ".")
from train import load_checkpoint
from data  import get_dataloader

ckpt_path = "./checkpoints/best_model.pt"
model = load_checkpoint(ckpt_path, "cpu")

# 从 checkpoint 里读取训练时的实际维度，避免硬编码不一致
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
kwargs      = ckpt["model_kwargs"]
seq_lengths = kwargs["L_S_init"]          # 总长度（不直接用）
seq_dims    = kwargs["seq_input_dims"]    # 训练时的实际 seq_dims
ns_dim      = kwargs["total_ns_dim"]

# 用与训练相同的 seq_dims / ns_dim 构造 mock 数据
loader, _ = get_dataloader(
    mode="mock", n_samples=100, ns_dim=ns_dim,
    seq_lengths=[50, 10, 20], seq_dims=seq_dims,
    batch_size=32, shuffle=False,
)
batch    = next(iter(loader))
ns_inputs = batch["ns_inputs"]
seqs      = batch["sequences"]
ts        = batch["timestamps"]
masks     = batch["seq_masks"]

with torch.no_grad():
    logits, _ = model(ns_inputs, seqs, ts, masks)

for name, logit in logits.items():
    score = torch.sigmoid(logit)
    print(f"  {name}: shape={score.shape}, mean={score.mean():.4f}, "
          f"range=[{score.min():.4f}, {score.max():.4f}]")
print("推理验证通过 ✓")
PYEOF

    else
        # parquet 模式：对测试集批量推理，输出 CSV
        TEST_DIR="${TEST_DIR:-./data/parquet/test}"
        FEATURE_CFG="${FEATURE_CFG:-./feature_config.yaml}"

        if [[ -d "$TEST_DIR" ]]; then
            echo "[推理] 对测试集批量推理..."
            echo "  测试集: $TEST_DIR"
            echo "  输出:   $PREDICT_OUT"

            $PYTHON infer.py \
                --checkpoint     "$INFER_MODEL" \
                --data_dir       "$TEST_DIR" \
                --feature_config "$FEATURE_CFG" \
                --output         "$PREDICT_OUT" \
                --batch_size     2048 \
                --num_workers    4 \
                2>&1 | tee "$LOG_DIR/infer.log"

            echo "[推理] 完成 ✓  结果: $PREDICT_OUT"
        else
            echo "[推理] 未找到测试集目录 $TEST_DIR，跳过批量推理"
            echo "       （如需推理，请设置环境变量 TEST_DIR=/path/to/test/data）"
        fi
    fi
else
    echo "[Step 3/4] 跳过推理（--skip_infer）"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 7. 汇总
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " Pipeline 完成  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "产出文件："
[[ -f "$BEST_MODEL"  ]] && echo "  全量模型:    $BEST_MODEL"
[[ -f "$EMB_MODEL"   ]] && echo "  Emb 增量:    $EMB_MODEL"
[[ -f "$HEAD_MODEL"  ]] && echo "  Head 增量:   $HEAD_MODEL"
[[ -f "$PREDICT_OUT" ]] && echo "  推理结果:    $PREDICT_OUT"
echo ""
echo "日志目录: $LOG_DIR/"
echo ""
echo "下一步（服务部署）："
echo "  # 打包 TorchServe .mar"
echo "  python torchserve/build_mar.py --checkpoint $INFER_MODEL --output_dir ./mar_output"
echo ""
echo "  # 或使用热加载服务（需配置 HDFS）"
echo "  # 参考 serving/model_loader.py"
