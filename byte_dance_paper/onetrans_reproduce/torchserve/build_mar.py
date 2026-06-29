# coding: utf-8
"""
OneTrans .mar 打包脚本

对应你之前的 torch_model_archiver.py，把模型权重 + 代码 + 配置打成 .mar 文件。

前置条件：
    pip install torchserve torch-model-archiver

用法：
    cd onetrans_reproduce
    python torchserve/build_mar.py \
        --checkpoint  ./checkpoints/best_model.pt \
        --output_dir  ./mar_output

打包完成后启动服务：
    torchserve --start \
        --model-store ./mar_output \
        --models onetrans=OneTrans_ranking.mar \
        --ts-config torchserve/config.properties

测试接口：
    curl -X POST http://localhost:8080/predictions/onetrans \
         -H "Content-Type: application/json" \
         -d @torchserve/sample_request.json
"""

import argparse
import os
import subprocess
import sys


# extra-files：handler 之外需要打包的文件（handler 通过 --handler 单独指定，不重复打包）
EXTRA_FILES = [
    "model.py",
    "tokenizer.py",
    "data.py",
    "feature_config.yaml",
    "torchserve/setup_config_onetrans.json",
]


def build_mar(checkpoint: str, output_dir: str, model_name: str = "OneTrans_ranking",
              version: str = "1.0"):
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.isfile(checkpoint), f"找不到 checkpoint: {checkpoint}"

    # build_mar.py 在 torchserve/ 子目录下，项目根目录是上一级
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 收集 extra-files
    extra = []
    for rel_path in EXTRA_FILES:
        abs_path = os.path.join(project_root, rel_path)
        assert os.path.isfile(abs_path), f"找不到文件: {abs_path}"
        extra.append(abs_path)

    handler_path = os.path.join(script_dir, "onetrans_handler.py")
    extra_str    = ",".join(extra)

    cmd = [
        "torch-model-archiver",
        "--model-name",      model_name,
        "--version",         version,
        "--serialized-file", os.path.abspath(checkpoint),
        "--handler",         handler_path,
        "--extra-files",     extra_str,
        "--export-path",     os.path.abspath(output_dir),
        "--force",
    ]

    print("执行命令：")
    print("  " + " \\\n    ".join(cmd))
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("打包失败！错误信息：")
        print(result.stderr)
        sys.exit(1)

    mar_path = os.path.join(output_dir, f"{model_name}.mar")
    print(f"\n打包成功：{mar_path}")
    print(f"文件大小：{os.path.getsize(mar_path) / 1024 / 1024:.1f} MB")
    print()
    print("启动 TorchServe：")
    print(f"  torchserve --start \\")
    print(f"    --model-store {os.path.abspath(output_dir)} \\")
    print(f"    --models onetrans={model_name}.mar \\")
    print(f"    --ts-config torchserve/config.properties")
    print()
    print("测试接口：")
    print(f"  curl -X POST http://localhost:8080/predictions/onetrans \\")
    print(f"       -H 'Content-Type: application/json' \\")
    print(f"       -d @torchserve/sample_request.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OneTrans .mar 打包脚本")
    p.add_argument("--checkpoint",  type=str, required=True, help="训练好的 checkpoint 路径")
    p.add_argument("--output_dir",  type=str, default="./mar_output", help="输出目录")
    p.add_argument("--model_name",  type=str, default="OneTrans_ranking", help=".mar 文件名（不含扩展名）")
    p.add_argument("--version",     type=str, default="1.0", help="模型版本号")
    args = p.parse_args()
    build_mar(args.checkpoint, args.output_dir, args.model_name, args.version)
