"""
OneTrans 推理服务热加载模块

职责：
  1. 启动时从 HDFS 加载 active 模型
  2. 后台线程定期轮询 latest.json，发现新版本时热替换模型
  3. 推理时用读写锁保证线程安全（替换期间不中断服务）

使用方式（在你的推理服务 main.py 里）：

    from serving.model_loader import ModelLoader

    loader = ModelLoader(
        hdfs_base="hdfs://namenode:9000/models/onetrans",
        feature_config="feature_config.yaml",
        poll_interval=60,   # 每60秒检查一次新版本
        device="cuda",
    )
    loader.start()          # 启动后台轮询线程

    # 推理时
    model = loader.get_model()   # 始终拿到最新版本，线程安全
    preds, _ = model(ns_inputs, seqs, ts, masks)

    # 服务关闭时
    loader.stop()

与 TorchServe 集成：
    在 onetrans_handler.py 的 initialize 里用 ModelLoader 替换直接 torch.load，
    handler 每次 inference 调用 loader.get_model() 即可。
"""

import glob
import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    线程安全的模型热加载器。

    内部维护一个 _model 引用和一个读写锁：
    - 推理线程调用 get_model() 获取当前模型（加读锁，几乎无开销）
    - 后台线程发现新版本时，加载完成后原子替换 _model（加写锁，毫秒级）
    """

    def __init__(
        self,
        hdfs_base: str,
        feature_config: Optional[str] = None,
        poll_interval: int = 60,
        device: str = "cpu",
        local_cache_dir: str = "/tmp/onetrans_model_cache",
    ):
        """
        Args:
            hdfs_base:       HDFS 根目录，如 hdfs://namenode:9000/models/onetrans
            feature_config:  本地 feature_config.yaml 路径（推理服务本地存一份）
            poll_interval:   轮询间隔（秒），建议 60~300
            device:          推理设备
            local_cache_dir: 下载的模型文件缓存目录
        """
        self.hdfs_base       = hdfs_base.rstrip("/")
        self.feature_config  = feature_config
        self.poll_interval   = poll_interval
        self.device          = device
        self.local_cache_dir = local_cache_dir
        os.makedirs(local_cache_dir, exist_ok=True)

        self._model: Optional[nn.Module] = None
        self._current_active: str = ""          # 当前加载的 active 路径
        self._lock = threading.RLock()           # 读写都用同一把锁（模型替换很快）
        self._stop_event = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None

    # ──────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────

    def start(self):
        """启动：先同步加载一次，再启动后台轮询线程"""
        logger.info("ModelLoader 启动，hdfs_base=%s", self.hdfs_base)
        self._reload_if_needed()   # 同步加载，确保服务启动时模型可用

        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="ModelLoader-Poll",
            daemon=True,           # daemon=True：主进程退出时自动结束
        )
        self._poll_thread.start()
        logger.info("后台轮询线程已启动，间隔=%ds", self.poll_interval)

    def stop(self):
        """停止后台轮询线程"""
        self._stop_event.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
        logger.info("ModelLoader 已停止")

    def get_model(self) -> nn.Module:
        """
        获取当前最新模型，线程安全。
        推理时调用此方法，不要缓存返回值（每次调用保证拿到最新版本）。
        """
        with self._lock:
            if self._model is None:
                raise RuntimeError("模型尚未加载，请先调用 start()")
            return self._model

    def get_status(self) -> dict:
        """返回当前加载状态，用于健康检查接口"""
        with self._lock:
            return {
                "loaded": self._model is not None,
                "active_path": self._current_active,
                "device": self.device,
            }

    # ──────────────────────────────────────────────────────────────────────
    # 内部逻辑
    # ──────────────────────────────────────────────────────────────────────

    def _poll_loop(self):
        """后台轮询：每 poll_interval 秒检查一次 latest.json"""
        while not self._stop_event.is_set():
            try:
                self._reload_if_needed()
            except Exception as e:
                logger.warning("轮询检查失败（将在下次重试）: %s", e)
            self._stop_event.wait(self.poll_interval)

    def _reload_if_needed(self):
        """
        读取 latest.json，如果 active 路径变了则热加载新模型。
        """
        meta = self._fetch_latest_json()
        if meta is None:
            return

        new_active = meta.get("active", "")
        if not new_active:
            logger.warning("latest.json 中 active 字段为空，跳过")
            return

        with self._lock:
            if new_active == self._current_active:
                return   # 没有变化，不需要重新加载

        logger.info("发现新版本: %s → %s", self._current_active or "(初次加载)", new_active)

        # 下载到本地临时文件（下载期间不持锁，不影响推理）
        local_path = self._download(new_active)
        if local_path is None:
            return

        # 加载模型（加载期间不持锁）
        new_model = self._load_model(local_path)
        if new_model is None:
            return

        # 原子替换（持锁时间极短，只是赋值）
        with self._lock:
            old_model = self._model
            self._model = new_model
            self._current_active = new_active

        # 释放旧模型内存
        if old_model is not None:
            del old_model
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        logger.info("模型热加载完成: %s", new_active)

    def _fetch_latest_json(self) -> Optional[dict]:
        """从 HDFS 读取 latest.json"""
        hdfs_path  = f"{self.hdfs_base}/latest.json"
        local_path = os.path.join(self.local_cache_dir, "latest.json")
        try:
            subprocess.run(
                ["hdfs", "dfs", "-get", "-f", hdfs_path, local_path],
                check=True, capture_output=True, timeout=30,
            )
            with open(local_path) as f:
                return json.load(f)
        except subprocess.CalledProcessError as e:
            logger.warning("读取 latest.json 失败: %s", e.stderr.decode())
            return None
        except Exception as e:
            logger.warning("解析 latest.json 失败: %s", e)
            return None

    def _download(self, hdfs_path: str) -> Optional[str]:
        """下载模型文件到本地缓存"""
        fname      = os.path.basename(hdfs_path)
        local_path = os.path.join(self.local_cache_dir, fname)

        # 已经缓存过则跳过下载
        if os.path.exists(local_path):
            logger.info("使用本地缓存: %s", local_path)
            return local_path

        try:
            logger.info("下载模型: %s", hdfs_path)
            subprocess.run(
                ["hdfs", "dfs", "-get", hdfs_path, local_path],
                check=True, capture_output=True, timeout=300,
            )
            return local_path
        except subprocess.CalledProcessError as e:
            logger.error("下载失败: %s\n%s", hdfs_path, e.stderr.decode())
            return None

    def _load_model(self, local_path: str) -> Optional[nn.Module]:
        """从本地文件加载模型"""
        try:
            ckpt = torch.load(local_path, map_location=self.device, weights_only=False)
            # 动态 import，避免循环依赖；去重插入，防止多次调用重复添加
            import sys
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from model import OneTrans
            model = OneTrans(**ckpt["model_kwargs"])
            model.load_state_dict(ckpt["state_dict"])
            model.to(self.device).eval()
            logger.info("模型加载成功: epoch=%s metrics=%s", ckpt.get("epoch"), ckpt.get("metrics"))
            # 加载成功后清理旧缓存（保留最近 3 个版本）
            self._cleanup_cache(keep=3)
            return model
        except Exception as e:
            logger.error("模型加载失败: %s", e)
            return None

    def _cleanup_cache(self, keep: int = 3):
        """清理本地缓存目录，只保留最新的 keep 个 .pt 文件"""
        pt_files = sorted(
            glob.glob(os.path.join(self.local_cache_dir, "*.pt")),
            key=os.path.getmtime,
        )
        for old_file in pt_files[:-keep]:
            try:
                os.remove(old_file)
                logger.debug("清理旧缓存: %s", old_file)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 本地文件模式（不依赖 HDFS，用于测试或小规模部署）
# ---------------------------------------------------------------------------

class LocalModelLoader(ModelLoader):
    """
    从本地目录热加载，不依赖 HDFS。
    适用于：单机部署、测试环境、没有 HDFS 的场景。

    目录约定：
        model_dir/
          latest.json    ← 训练脚本写入（同 HDFS 格式）
          *.pt           ← 模型文件

    用法：
        loader = LocalModelLoader(
            model_dir="./checkpoints",
            poll_interval=30,
        )
        loader.start()
    """

    def __init__(self, model_dir: str, poll_interval: int = 30, device: str = "cpu"):
        # 复用父类逻辑，但把 hdfs_base 设为本地路径
        super().__init__(
            hdfs_base=model_dir,
            poll_interval=poll_interval,
            device=device,
            local_cache_dir=model_dir,
        )
        self.model_dir = model_dir

    def _fetch_latest_json(self) -> Optional[dict]:
        """直接读本地 latest.json"""
        path = os.path.join(self.model_dir, "latest.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("读取 latest.json 失败: %s", e)
            return None

    def _download(self, path: str) -> Optional[str]:
        """本地模式直接返回路径，不需要下载"""
        if path.startswith("hdfs://"):
            logger.error("LocalModelLoader 不支持 HDFS 路径: %s，请使用 ModelLoader", path)
            return None
        return path if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# 训练脚本写 latest.json 的工具函数（本地模式）
# ---------------------------------------------------------------------------

def write_local_latest(model_dir: str, mode: str, model_path: str):
    """
    训练完成后调用，更新本地 latest.json。
    train.py 和 train_embedding.py 在不使用 HDFS 时调用此函数。

    Args:
        model_dir:  checkpoints 目录
        mode:       "full" / "emb" / "head"
        model_path: 新保存的模型文件路径
    """
    latest_path = os.path.join(model_dir, "latest.json")
    if os.path.exists(latest_path):
        with open(latest_path) as f:
            meta = json.load(f)
    else:
        meta = {"full": "", "emb": "", "head": ""}

    meta[mode]        = model_path
    meta["updated_at"] = datetime.now().isoformat()
    meta["active"]    = meta.get("head") or meta.get("emb") or meta.get("full") or ""

    with open(latest_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("latest.json 已更新: active=%s", meta["active"])
