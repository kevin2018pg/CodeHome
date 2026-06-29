"""
本地单元测试 —— 不需要启动服务，直接测试核心逻辑函数
覆盖：_do_train / _do_predict / _calc_metrics / 所有错误分支 / auto_tune
"""

import os
import shutil
import time
import traceback

import numpy as np
import pandas as pd

# 切换到项目目录，确保 models/ 路径正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 用独立的临时目录，测试完自动清理，不污染真实 models/
TEST_MODEL_DIR = "models_test"
os.makedirs(TEST_MODEL_DIR, exist_ok=True)

import main as svc
from pathlib import Path

_orig_model_dir = svc.MODEL_DIR
svc.MODEL_DIR = Path(TEST_MODEL_DIR)

_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "[PASS]" if condition else "[FAIL]"
    msg = f"  {status} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if not condition:
        _failures.append(name)
    return condition


def gen_train_df(n: int = 300, n_features: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    y = (1 / (1 + np.exp(-X[:, :3].sum(axis=1))) < 0.5).astype(int)
    return pd.DataFrame({
        "id": [f"r{i}" for i in range(n)],
        **{f"f{i}": X[:, i] for i in range(n_features)},
        "label": y,
    })


def gen_predict_df(n: int = 50, n_features: int = 5, seed: int = 99) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    return pd.DataFrame({
        "id": [f"p{i}" for i in range(n)],
        **{f"f{i}": X[:, i] for i in range(n_features)},
    })


# ─────────────────────────── 测试用例 ───────────────────────────

def test_train_basic():
    print("\n[1] _do_train 基本训练（分类）")
    df = gen_train_df()
    meta = svc._do_train(df, "id", "label", "classification",
                         "unit_cls", "tester", 50, 4, 0.1, 0.8, 0.2)
    check("返回 model_id", meta["model_id"] == "unit_cls")
    check("feature_names 正确", meta["feature_names"] == [f"f{i}" for i in range(5)])
    check("metrics 包含 auc", "auc" in meta["metrics"])
    check("auc 在合理范围", 0.5 <= meta["metrics"]["auc"] <= 1.0, str(meta["metrics"]["auc"]))
    check("auto_tune 为 None", meta["auto_tune"] is None)
    check("模型文件已保存", (svc.MODEL_DIR / "unit_cls.pkl").exists())
    check("元信息文件已保存", (svc.MODEL_DIR / "unit_cls.json").exists())


def test_train_regression():
    print("\n[2] _do_train 回归任务")
    rng = np.random.RandomState(1)
    n = 200
    X = rng.randn(n, 3).astype(np.float32)
    y = X[:, 0] * 2 + rng.randn(n).astype(np.float32) * 0.1
    df = pd.DataFrame({
        "id": range(n),
        "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
        "target": y,
    })
    meta = svc._do_train(df, "id", "target", "regression",
                         "unit_reg", None, 50, 4, 0.1, 0.8, 0.2)
    check("返回 model_id", meta["model_id"] == "unit_reg")
    check("metrics 包含 rmse", "rmse" in meta["metrics"])
    check("rmse > 0", meta["metrics"]["rmse"] > 0)


def test_predict_basic():
    print("\n[3] _do_predict 基本预测（output_prob=True）")
    df = gen_predict_df()
    result = svc._do_predict(df, "unit_cls", "id", output_prob=True)
    check("结果行数正确", len(result) == 50, f"got {len(result)}")
    check("包含 id 列", "id" in result.columns)
    check("包含 pred_prob 列", "pred_prob" in result.columns)
    check("包含 pred_label 列", "pred_label" in result.columns)
    check("id 第一行正确", result["id"].iloc[0] == "p0")
    check("pred_prob 在 [0,1]", result["pred_prob"].between(0, 1).all())
    check("pred_label 只有 0/1", set(result["pred_label"].unique()).issubset({0, 1}))


def test_predict_no_prob():
    print("\n[4] _do_predict（output_prob=False）")
    df = gen_predict_df()
    result = svc._do_predict(df, "unit_cls", "id", output_prob=False)
    check("包含 prediction 列", "prediction" in result.columns)
    check("不含 pred_prob 列", "pred_prob" not in result.columns)


def test_predict_feature_align():
    print("\n[5] 特征列顺序打乱后自动对齐")
    df = gen_predict_df()
    cols = ["id"] + list(reversed([f"f{i}" for i in range(5)]))
    result = svc._do_predict(df[cols], "unit_cls", "id", output_prob=True)
    check("预测成功（列顺序不影响结果）", "pred_prob" in result.columns)


def test_error_missing_id_col():
    print("\n[6] 错误：缺少 id 列")
    df = gen_train_df().drop(columns=["id"])
    try:
        svc._do_train(df, "id", "label", "classification", "err1", None, 50, 4, 0.1, 0.8, 0.2)
        check("应该抛出 HTTPException", False)
    except Exception as e:
        check("抛出 HTTPException 400", "400" in str(e) or "id" in str(e).lower(), str(e)[:80])


def test_error_missing_label_col():
    print("\n[7] 错误：缺少 label 列")
    df = gen_train_df().drop(columns=["label"])
    try:
        svc._do_train(df, "id", "label", "classification", "err2", None, 50, 4, 0.1, 0.8, 0.2)
        check("应该抛出 HTTPException", False)
    except Exception as e:
        check("抛出 HTTPException 400", "400" in str(e) or "label" in str(e).lower(), str(e)[:80])


def test_error_model_not_found():
    print("\n[8] 错误：预测不存在的模型")
    df = gen_predict_df()
    try:
        svc._do_predict(df, "not_exist_model", "id", False)
        check("应该抛出 HTTPException", False)
    except Exception as e:
        check("抛出 HTTPException 404", "404" in str(e), str(e)[:80])


def test_error_missing_feature():
    print("\n[9] 错误：预测数据缺少特征列")
    df = gen_predict_df().drop(columns=["f0"])
    try:
        svc._do_predict(df, "unit_cls", "id", False)
        check("应该抛出 HTTPException", False)
    except Exception as e:
        check("抛出 HTTPException 400", "400" in str(e) or "f0" in str(e), str(e)[:80])


def test_error_invalid_task():
    print("\n[10] 错误：非法 task 参数")
    df = gen_train_df()
    try:
        svc._do_train(df, "id", "label", "invalid_task", "err3", None, 50, 4, 0.1, 0.8, 0.2)
        check("应该抛出 HTTPException", False)
    except Exception as e:
        check("抛出 HTTPException 400", "400" in str(e), str(e)[:80])


def test_load_save_model():
    print("\n[11] 模型持久化：保存后重新加载")
    model, meta = svc._load_model("unit_cls")
    check("加载模型成功", model is not None)
    check("meta 包含 feature_names", "feature_names" in meta)
    check("meta feature_names 正确", meta["feature_names"] == [f"f{i}" for i in range(5)])


def test_extra_cols_ignored():
    print("\n[12] 预测数据有多余列，自动忽略")
    df = gen_predict_df()
    df["extra_col"] = 999
    result = svc._do_predict(df, "unit_cls", "id", output_prob=False)
    check("预测成功", "prediction" in result.columns)
    check("多余列不出现在结果中", "extra_col" not in result.columns)


def test_auto_tune():
    print("\n[13] auto_tune=True 自动调优（n_trials=5，快速验证）")
    df = gen_train_df(n=400)
    t0 = time.time()
    meta = svc._do_train(df, "id", "label", "classification",
                         "unit_auto", "tester", 100, 6, 0.1, 0.8, 0.2,
                         auto_tune=True, n_trials=5)
    elapsed = round(time.time() - t0, 1)

    check("返回 model_id", meta["model_id"] == "unit_auto")
    check("metrics 包含 auc", "auc" in meta["metrics"])
    check("auc 在合理范围", 0.5 <= meta["metrics"]["auc"] <= 1.0, str(meta["metrics"]["auc"]))

    at = meta.get("auto_tune")
    check("auto_tune 字段不为 None", at is not None)
    check("记录了 n_trials=5", at.get("n_trials") == 5 if at else False, str(at))
    check("记录了 best_trial", "best_trial" in (at or {}))
    check("记录了 best_value", "best_value" in (at or {}))
    check("top5_trials 有数据", len((at or {}).get("top5_trials", [])) > 0)

    hp = meta.get("hyperparams", {})
    check("hyperparams 含 colsample_bytree", "colsample_bytree" in hp, str(hp))
    check("hyperparams 含 min_child_weight", "min_child_weight" in hp)

    print(f"    耗时 {elapsed}s，AUC={meta['metrics'].get('auc')}，"
          f"最优超参: n_estimators={hp.get('n_estimators')}, "
          f"max_depth={hp.get('max_depth')}, "
          f"lr={round(hp.get('learning_rate', 0), 4)}")


# ─────────────────────────── 主流程 ───────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("algo_trainer 本地单元测试（无需启动服务）")
    print("=" * 55)

    try:
        test_train_basic()
        test_train_regression()
        test_predict_basic()
        test_predict_no_prob()
        test_predict_feature_align()
        test_error_missing_id_col()
        test_error_missing_label_col()
        test_error_model_not_found()
        test_error_missing_feature()
        test_error_invalid_task()
        test_load_save_model()
        test_extra_cols_ignored()
        test_auto_tune()

    except Exception:
        print("\n[CRASH] 测试崩溃:")
        traceback.print_exc()
    finally:
        svc.MODEL_DIR = _orig_model_dir
        shutil.rmtree(TEST_MODEL_DIR, ignore_errors=True)

    print("\n" + "=" * 55)
    if _failures:
        print(f"失败 {len(_failures)} 项: {_failures}")
    else:
        print("所有测试通过 [ALL PASS]")
    print("=" * 55)
