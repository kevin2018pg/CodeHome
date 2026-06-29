"""
algo_trainer — 算法自动化训练 & 预测服务

训练接口（均支持 auto_tune 参数，true 时自动搜索最优超参）：
- POST /train            : 上传CSV文件训练
- POST /train_text       : 直接传入CSV文本训练（适合 OpenClaw/飞书）
- POST /train_from_path  : 传入服务器本地路径训练（适合大数据）

预测接口：
- POST /predict          : 上传CSV文件预测，返回结果CSV
- POST /predict_text     : 直接传入CSV文本预测，返回JSON（适合 OpenClaw/飞书）
- POST /predict_from_path: 传入服务器本地路径预测，返回JSON

模型管理：
- GET  /models           : 查看所有已训练的模型列表
- GET  /models/{id}      : 查看某个模型的详细元信息
- DELETE /models/{id}    : 删除模型
"""

import io
import json
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

app = FastAPI(title="Algo Trainer — 算法自动化训练预测服务", version="2.0.0")

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

TaskType = Literal["classification", "regression"]


@app.get("/healthCheck", summary="健康检查")
def health():
    return {"status": "ok"}


# ─────────────────────────── 内部工具函数 ───────────────────────────

def _meta_path(model_id: str) -> Path:
    return MODEL_DIR / f"{model_id}.json"


def _model_path(model_id: str) -> Path:
    return MODEL_DIR / f"{model_id}.pkl"


def _save_model(model: xgb.XGBModel, model_id: str, meta: dict):
    with open(_model_path(model_id), "wb") as f:
        pickle.dump(model, f)
    with open(_meta_path(model_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_model(model_id: str) -> tuple[xgb.XGBModel, dict]:
    if not _model_path(model_id).exists():
        raise HTTPException(status_code=404, detail=f"模型 '{model_id}' 不存在，请先训练")
    with open(_model_path(model_id), "rb") as f:
        model = pickle.load(f)
    with open(_meta_path(model_id), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV解析失败: {e}")


def _read_csv_text(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV文本解析失败: {e}")


def _read_csv_path(file_path: str) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"文件不存在: {file_path}")
    try:
        return pd.read_csv(p)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV读取失败: {e}")


# ─────────────────────────── 核心业务逻辑 ───────────────────────────

def _calc_metrics(model, X_val, y_val, task: str, y_full) -> dict:
    """计算验证集指标，y_full 用于判断总类别数"""
    metrics: dict = {}
    if task == "classification":
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)
        metrics["accuracy"] = round(float(accuracy_score(y_val, y_pred)), 4)
        try:
            n_cls = len(np.unique(y_full))
            if n_cls == 2:
                metrics["auc"] = round(float(roc_auc_score(y_val, y_prob[:, 1])), 4)
            else:
                metrics["auc"] = round(float(roc_auc_score(y_val, y_prob, multi_class="ovr")), 4)
        except Exception:
            pass
    else:
        y_pred = model.predict(X_val)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_val, y_pred))), 4)
    return metrics


def _do_train(
    df: pd.DataFrame,
    id_col: str,
    label_col: str,
    task: str,
    model_id: Optional[str],
    owner: Optional[str],
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    test_size: float,
    auto_tune: bool = False,
    n_trials: int = 30,
) -> dict:
    """
    训练核心逻辑。
    auto_tune=True 时忽略传入的超参，使用 Optuna 自动搜索最优超参后训练。
    """
    if id_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"找不到ID列 '{id_col}'，文件中的列为: {list(df.columns)}")
    if label_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"找不到标签列 '{label_col}'，文件中的列为: {list(df.columns)}")
    if task not in ("classification", "regression"):
        raise HTTPException(status_code=400, detail="task 只支持 classification 或 regression")

    feature_names = [c for c in df.columns if c not in (id_col, label_col)]
    X = df[feature_names].values.astype(np.float32)
    y = df[label_col].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    is_classification = (task == "classification")
    n_classes = len(np.unique(y)) if is_classification else None

    auto_tune_info: Optional[dict] = None

    if auto_tune:
        if not OPTUNA_AVAILABLE:
            raise HTTPException(status_code=500, detail="auto_tune=true 需要安装 optuna：pip install optuna")

        def objective(trial: "optuna.Trial") -> float:
            p = dict(
                n_estimators=trial.suggest_int("n_estimators", 50, 500),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                random_state=42, n_jobs=-1,
            )
            if is_classification:
                p["objective"] = "binary:logistic" if n_classes == 2 else "multi:softprob"
                if n_classes and n_classes > 2:
                    p["num_class"] = n_classes
                m = xgb.XGBClassifier(**p)
                m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                prob = m.predict_proba(X_val)
                try:
                    if n_classes == 2:
                        return float(roc_auc_score(y_val, prob[:, 1]))
                    else:
                        return float(roc_auc_score(y_val, prob, multi_class="ovr"))
                except Exception:
                    return float(accuracy_score(y_val, m.predict(X_val)))
            else:
                p["objective"] = "reg:squarederror"
                m = xgb.XGBRegressor(**p)
                m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                return -float(np.sqrt(mean_squared_error(y_val, m.predict(X_val))))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        auto_tune_info = {
            "n_trials": n_trials,
            "best_trial": study.best_trial.number,
            "best_value": round(study.best_value, 4),
            "top5_trials": sorted(
                [{"trial": t.number, "value": round(t.value, 4), "params": t.params}
                 for t in study.trials if t.value is not None],
                key=lambda x: x["value"], reverse=True
            )[:5],
        }
        n_estimators = best_params["n_estimators"]
        max_depth = best_params["max_depth"]
        learning_rate = best_params["learning_rate"]
        subsample = best_params["subsample"]
        extra_params = {k: v for k, v in best_params.items()
                        if k not in ("n_estimators", "max_depth", "learning_rate", "subsample")}
    else:
        extra_params = {}

    params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                  subsample=subsample, random_state=42, n_jobs=-1, **extra_params)

    if is_classification:
        params["objective"] = "binary:logistic" if n_classes == 2 else "multi:softprob"
        if n_classes and n_classes > 2:
            params["num_class"] = n_classes
        params["eval_metric"] = "auc" if n_classes == 2 else "mlogloss"
        model = xgb.XGBClassifier(**params)
    else:
        params["objective"] = "reg:squarederror"
        params["eval_metric"] = "rmse"
        model = xgb.XGBRegressor(**params)

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    metrics = _calc_metrics(model, X_val, y_val, task, y)

    mid = model_id or str(uuid.uuid4())[:8]
    meta: dict = {
        "model_id": mid,
        "owner": owner or "unknown",
        "task": task,
        "id_col": id_col,
        "label_col": label_col,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "hyperparams": {"n_estimators": n_estimators, "max_depth": max_depth,
                        "learning_rate": learning_rate, "subsample": subsample, **extra_params},
        "metrics": metrics,
        "auto_tune": auto_tune_info,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_model(model, mid, meta)
    return meta


def _do_predict(
    df: pd.DataFrame,
    model_id: str,
    id_col: str,
    output_prob: bool,
) -> pd.DataFrame:
    """预测核心逻辑，返回结果 DataFrame"""
    model, meta = _load_model(model_id)
    expected_features: list[str] = meta["feature_names"]

    if id_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"找不到ID列 '{id_col}'，文件中的列为: {list(df.columns)}")

    missing = set(expected_features) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"预测数据缺少特征列: {sorted(missing)}，该模型训练时的特征为: {expected_features}"
        )

    id_series = df[id_col].reset_index(drop=True)
    X = df[expected_features].values.astype(np.float32)
    is_classifier = isinstance(model, xgb.XGBClassifier)

    result_df = pd.concat(
        [pd.DataFrame({id_col: id_series}), df[expected_features].reset_index(drop=True)],
        axis=1
    )

    if is_classifier and output_prob:
        probs = model.predict_proba(X)
        if probs.shape[1] == 2:
            result_df["pred_prob"] = probs[:, 1]
        else:
            for i in range(probs.shape[1]):
                result_df[f"pred_prob_class{i}"] = probs[:, i]
        result_df["pred_label"] = model.predict(X)
    else:
        result_df["prediction"] = model.predict(X)

    return result_df


def _df_to_csv_response(df: pd.DataFrame, model_id: str) -> Response:
    return Response(
        content=df.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=algo_trainer_predict_{model_id}.csv"},
    )


# ─────────────────────────── Pydantic 请求体（训练参数共享基类） ───────────────────────────

class _TrainParams(BaseModel):
    """训练参数基类，供 TrainTextBody / TrainPathBody 复用"""
    id_col: str = Field("id", description="ID列名，不参与训练")
    label_col: str = Field("label", description="标签列名")
    task: TaskType = Field("classification", description="classification 或 regression")
    model_id: Optional[str] = Field(None, description="模型ID，不填则自动生成")
    owner: Optional[str] = Field(None, description="归属人/业务标识")
    n_estimators: int = Field(100, description="树的数量")
    max_depth: int = Field(6, description="树的最大深度")
    learning_rate: float = Field(0.1, description="学习率")
    subsample: float = Field(0.8, description="样本采样比例")
    test_size: float = Field(0.2, description="验证集比例")
    auto_tune: bool = Field(False, description="是否自动搜索最优超参，True时忽略上方超参")
    n_trials: int = Field(30, description="auto_tune=True时的搜索次数，建议20~50")

    def do_train(self, df: pd.DataFrame) -> dict:
        return _do_train(df, self.id_col, self.label_col, self.task,
                         self.model_id, self.owner, self.n_estimators,
                         self.max_depth, self.learning_rate, self.subsample,
                         self.test_size, self.auto_tune, self.n_trials)


class TrainTextBody(_TrainParams):
    csv_content: str = Field(..., description="CSV格式文本内容")

    model_config = {"json_schema_extra": {"example": {
        "csv_content": "id,f1,f2,label\n1,0.1,0.2,1\n2,-0.5,1.3,0",
        "model_id": "my_model_v1", "owner": "张三",
        "task": "classification", "auto_tune": False,
    }}}


class TrainPathBody(_TrainParams):
    file_path: str = Field(..., description="服务器本地CSV文件路径")

    model_config = {"json_schema_extra": {"example": {
        "file_path": "/data/train.csv", "model_id": "hotel_ctr_v1",
        "owner": "推荐组", "task": "classification", "auto_tune": False,
    }}}


class PredictTextBody(BaseModel):
    csv_content: str = Field(..., description="CSV格式文本内容")
    model_id: str = Field(..., description="训练时返回的模型ID")
    id_col: str = Field("id", description="ID列名")
    output_prob: bool = Field(False, description="分类任务是否输出概率")

    model_config = {"json_schema_extra": {"example": {
        "csv_content": "id,f1,f2\n1,0.1,0.2\n2,-0.5,1.3",
        "model_id": "my_model_v1", "output_prob": True,
    }}}


class PredictPathBody(BaseModel):
    file_path: str = Field(..., description="服务器本地CSV文件路径")
    model_id: str = Field(..., description="训练时返回的模型ID")
    id_col: str = Field("id", description="ID列名")
    output_prob: bool = Field(False, description="分类任务是否输出概率")
    save_result_path: Optional[str] = Field(None, description="结果保存路径，不填则只返回JSON")

    model_config = {"json_schema_extra": {"example": {
        "file_path": "/data/predict.csv", "model_id": "hotel_ctr_v1",
        "output_prob": True, "save_result_path": "/data/result.csv",
    }}}


# ─────────────────────────── 接口定义 ───────────────────────────

@app.get("/models", summary="查看所有已训练的模型")
def list_models():
    result = []
    for p in sorted(MODEL_DIR.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        result.append({
            "model_id": p.stem,
            "owner": meta.get("owner", "-"),
            "task": meta.get("task"),
            "n_features": meta.get("n_features"),
            "feature_names": meta.get("feature_names"),
            "metrics": meta.get("metrics"),
            "auto_tune": meta.get("auto_tune") is not None,
            "trained_at": meta.get("trained_at"),
        })
    return {"count": len(result), "models": result}


@app.get("/models/{model_id}", summary="查看某个模型的详细元信息")
def get_model_info(model_id: str):
    if not _meta_path(model_id).exists():
        raise HTTPException(status_code=404, detail=f"模型 '{model_id}' 不存在")
    with open(_meta_path(model_id), "r", encoding="utf-8") as f:
        return json.load(f)


@app.delete("/models/{model_id}", summary="删除模型")
def delete_model(model_id: str):
    mp = _model_path(model_id)
    if not mp.exists():
        raise HTTPException(status_code=404, detail=f"模型 '{model_id}' 不存在")
    mp.unlink()
    _meta_path(model_id).unlink(missing_ok=True)
    return {"message": f"模型 '{model_id}' 已删除"}


@app.post("/train", summary="【文件上传】训练模型")
async def train(
    file: UploadFile = File(..., description="带标签的CSV文件，必须包含 id 列和 label 列"),
    id_col: str = Form("id"),
    label_col: str = Form("label"),
    task: str = Form("classification"),
    model_id: Optional[str] = Form(None),
    owner: Optional[str] = Form(None),
    n_estimators: int = Form(100),
    max_depth: int = Form(6),
    learning_rate: float = Form(0.1),
    subsample: float = Form(0.8),
    test_size: float = Form(0.2),
    auto_tune: bool = Form(False, description="是否自动搜索最优超参（True时忽略上方超参设置）"),
    n_trials: int = Form(30, description="auto_tune=True时的搜索次数，建议20~50"),
):
    """
    上传CSV文件训练模型。

    - `auto_tune=false`（默认）：使用指定超参直接训练
    - `auto_tune=true`：自动搜索最优超参后训练，忽略 n_estimators/max_depth 等手动设置
    """
    df = _read_csv_bytes(await file.read())
    return _do_train(df, id_col, label_col, task, model_id, owner,
                     n_estimators, max_depth, learning_rate, subsample, test_size,
                     auto_tune=auto_tune, n_trials=n_trials)


@app.post("/predict", summary="【文件上传】预测，返回CSV文件")
async def predict(
    file: UploadFile = File(..., description="特征CSV，必须包含 id 列"),
    model_id: str = Form(...),
    id_col: str = Form("id"),
    output_prob: bool = Form(False),
):
    """上传CSV文件预测，返回带结果的CSV文件下载，适合大数据量场景。"""
    df = _read_csv_bytes(await file.read())
    return _df_to_csv_response(_do_predict(df, model_id, id_col, output_prob), model_id)


@app.post("/train_text", summary="【文本传入】训练模型（适合 OpenClaw/飞书）")
async def train_text(body: TrainTextBody):
    """直接传入CSV文本内容训练模型，无需上传文件，适合在飞书/OpenClaw中直接粘贴数据。"""
    return body.do_train(_read_csv_text(body.csv_content))


@app.post("/predict_text", summary="【文本传入】预测，返回JSON（适合 OpenClaw/飞书）")
async def predict_text(body: PredictTextBody):
    """直接传入CSV文本内容预测，结果以JSON返回，适合在飞书/OpenClaw中直接查看。"""
    result_df = _do_predict(_read_csv_text(body.csv_content), body.model_id, body.id_col, body.output_prob)
    return {"model_id": body.model_id, "total_rows": len(result_df),
            "results": result_df.to_dict(orient="records")}


@app.post("/train_from_path", summary="【本地路径】训练模型（适合大数据）")
async def train_from_path(body: TrainPathBody):
    """传入服务器本地CSV文件路径训练模型，适合大数据量（提前把文件放到服务器上）。"""
    return body.do_train(_read_csv_path(body.file_path))


@app.post("/predict_from_path", summary="【本地路径】预测（适合大数据）")
async def predict_from_path(body: PredictPathBody):
    """传入服务器本地CSV文件路径进行预测，可选择将结果保存到指定路径。"""
    result_df = _do_predict(_read_csv_path(body.file_path), body.model_id, body.id_col, body.output_prob)

    saved_path = None
    if body.save_result_path:
        save_p = Path(body.save_result_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(save_p, index=False)
        saved_path = str(save_p)

    pred_cols = [c for c in result_df.columns if c.startswith("pred") or c == "prediction"]
    return {
        "model_id": body.model_id,
        "total_rows": len(result_df),
        "saved_path": saved_path,
        "preview": result_df[[body.id_col] + pred_cols].head(10).to_dict(orient="records"),
    }
