"""
algo_trainer 完整自动化测试脚本
覆盖所有接口：train / predict / train_text / predict_text /
             train_from_path / predict_from_path / models / delete
"""

import io
import json
import os
import time

import numpy as np
import pandas as pd
import requests

BASE_URL = "http://127.0.0.1:8000"
PASS = "[PASS]"
FAIL = "[FAIL]"


# ─────────────────────────── 数据生成 ───────────────────────────

def gen_train_df(n=500, n_features=5, seed=42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    logit = X[:, :3].sum(axis=1)
    y = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    feat = {f"f{i}": X[:, i] for i in range(n_features)}
    return pd.DataFrame({"id": [f"r{i}" for i in range(n)], **feat, "label": y})


def gen_predict_df(n=100, n_features=5, seed=99) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    feat = {f"f{i}": X[:, i] for i in range(n_features)}
    return pd.DataFrame({"id": [f"p{i}" for i in range(n)], **feat})


def df_to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def df_to_csv_str(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"  {status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


# ─────────────────────────── 各接口测试 ───────────────────────────

def test_models_empty():
    print("\n[1] GET /models（初始状态）")
    r = requests.get(f"{BASE_URL}/models")
    check("状态码 200", r.status_code == 200)
    check("返回 count 字段", "count" in r.json())


def test_train_file(model_id="test_file_model") -> str:
    print(f"\n[2] POST /train（文件上传，model_id={model_id}）")
    df = gen_train_df()
    r = requests.post(
        f"{BASE_URL}/train",
        files={"file": ("train.csv", df_to_bytes(df), "text/csv")},
        data={"model_id": model_id, "owner": "tester", "task": "classification",
              "id_col": "id", "label_col": "label"},
    )
    check("状态码 200", r.status_code == 200, r.text[:100] if r.status_code != 200 else "")
    if r.status_code == 200:
        meta = r.json()
        check("返回 model_id", meta.get("model_id") == model_id)
        check("返回 metrics.auc", "auc" in meta.get("metrics", {}), str(meta.get("metrics")))
        check("auto_tune 字段为 None", meta.get("auto_tune") is None)
        check("feature_names 正确", meta.get("feature_names") == [f"f{i}" for i in range(5)])
    return model_id


def test_predict_file(model_id: str):
    print(f"\n[3] POST /predict（文件上传，model_id={model_id}）")
    df = gen_predict_df()
    r = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("pred.csv", df_to_bytes(df), "text/csv")},
        data={"model_id": model_id, "id_col": "id", "output_prob": "true"},
    )
    check("状态码 200", r.status_code == 200, r.text[:100] if r.status_code != 200 else "")
    if r.status_code == 200:
        result = pd.read_csv(io.StringIO(r.text))
        check("结果行数正确", len(result) == 100, f"got {len(result)}")
        check("包含 id 列", "id" in result.columns)
        check("包含 pred_prob 列", "pred_prob" in result.columns)
        check("包含 pred_label 列", "pred_label" in result.columns)
        check("id 对应正确", result["id"].iloc[0] == "p0")
        check("pred_prob 在 [0,1]", result["pred_prob"].between(0, 1).all())


def test_train_text(model_id="test_text_model") -> str:
    print(f"\n[4] POST /train_text（JSON文本，model_id={model_id}）")
    df = gen_train_df(n=200)
    r = requests.post(
        f"{BASE_URL}/train_text",
        json={"csv_content": df_to_csv_str(df), "model_id": model_id,
              "owner": "openclaw", "task": "classification",
              "id_col": "id", "label_col": "label"},
    )
    check("状态码 200", r.status_code == 200, r.text[:100] if r.status_code != 200 else "")
    if r.status_code == 200:
        meta = r.json()
        check("返回 model_id", meta.get("model_id") == model_id)
        check("返回 metrics.auc", "auc" in meta.get("metrics", {}))
    return model_id


def test_predict_text(model_id: str):
    print(f"\n[5] POST /predict_text（JSON文本，model_id={model_id}）")
    df = gen_predict_df(n=10)
    r = requests.post(
        f"{BASE_URL}/predict_text",
        json={"csv_content": df_to_csv_str(df), "model_id": model_id,
              "id_col": "id", "output_prob": True},
    )
    check("状态码 200", r.status_code == 200, r.text[:100] if r.status_code != 200 else "")
    if r.status_code == 200:
        data = r.json()
        check("返回 total_rows=10", data.get("total_rows") == 10)
        check("返回 results 列表", isinstance(data.get("results"), list))
        check("results[0] 包含 id", "id" in data["results"][0])
        check("results[0] 包含 pred_prob", "pred_prob" in data["results"][0])
        check("results[0] id 对应正确", data["results"][0]["id"] == "p0")


def test_train_from_path(model_id="test_path_model") -> str:
    print(f"\n[6] POST /train_from_path（本地路径，model_id={model_id}）")
    df = gen_train_df(n=300)
    tmp_path = os.path.abspath("tmp_train.csv")
    df.to_csv(tmp_path, index=False)
    r = requests.post(
        f"{BASE_URL}/train_from_path",
        json={"file_path": tmp_path, "model_id": model_id,
              "owner": "path_tester", "task": "classification",
              "id_col": "id", "label_col": "label"},
    )
    check("状态码 200", r.status_code == 200, r.text[:100] if r.status_code != 200 else "")
    if r.status_code == 200:
        meta = r.json()
        check("返回 model_id", meta.get("model_id") == model_id)
        check("返回 metrics.auc", "auc" in meta.get("metrics", {}))
    os.remove(tmp_path)
    return model_id


def test_predict_from_path(model_id: str):
    print(f"\n[7] POST /predict_from_path（本地路径，model_id={model_id}）")
    df = gen_predict_df(n=20)
    tmp_path = os.path.abspath("tmp_predict.csv")
    result_path = os.path.abspath("tmp_result.csv")
    df.to_csv(tmp_path, index=False)
    r = requests.post(
        f"{BASE_URL}/predict_from_path",
        json={"file_path": tmp_path, "model_id": model_id,
              "id_col": "id", "output_prob": True,
              "save_result_path": result_path},
    )
    check("状态码 200", r.status_code == 200, r.text[:100] if r.status_code != 200 else "")
    if r.status_code == 200:
        data = r.json()
        check("返回 total_rows=20", data.get("total_rows") == 20)
        check("saved_path 正确", data.get("saved_path") == result_path)
        check("preview 包含 pred_prob", "pred_prob" in data["preview"][0])
        check("结果文件已保存", os.path.exists(result_path))
        if os.path.exists(result_path):
            saved = pd.read_csv(result_path)
            check("保存文件行数正确", len(saved) == 20)
    for f in [tmp_path, result_path]:
        if os.path.exists(f):
            os.remove(f)


def test_models_list(expected_ids: list):
    print(f"\n[8] GET /models（应有 {len(expected_ids)} 个模型）")
    r = requests.get(f"{BASE_URL}/models")
    check("状态码 200", r.status_code == 200)
    data = r.json()
    check(f"模型数量 >= {len(expected_ids)}", data["count"] >= len(expected_ids),
          f"got {data['count']}")
    model_ids = [m["model_id"] for m in data["models"]]
    for mid in expected_ids:
        check(f"包含模型 {mid}", mid in model_ids)


def test_model_detail(model_id: str):
    print(f"\n[9] GET /models/{model_id}")
    r = requests.get(f"{BASE_URL}/models/{model_id}")
    check("状态码 200", r.status_code == 200)
    if r.status_code == 200:
        meta = r.json()
        check("包含 feature_names", "feature_names" in meta)
        check("包含 hyperparams", "hyperparams" in meta)
        check("包含 trained_at", "trained_at" in meta)


def test_error_cases():
    print("\n[10] 错误处理测试")
    df = gen_predict_df(n=5)

    # 预测不存在的模型
    r = requests.post(f"{BASE_URL}/predict_text",
                      json={"csv_content": df_to_csv_str(df), "model_id": "not_exist"})
    check("不存在模型 -> 404", r.status_code == 404)

    # 训练缺少 label 列
    df_no_label = gen_predict_df(n=10)
    r = requests.post(f"{BASE_URL}/train_text",
                      json={"csv_content": df_to_csv_str(df_no_label),
                            "model_id": "err_test", "label_col": "label"})
    check("缺少 label 列 -> 400", r.status_code == 400)

    # 训练缺少 id 列
    df_no_id = gen_train_df(n=10).drop(columns=["id"])
    r = requests.post(f"{BASE_URL}/train_text",
                      json={"csv_content": df_to_csv_str(df_no_id),
                            "model_id": "err_test", "id_col": "id"})
    check("缺少 id 列 -> 400", r.status_code == 400)

    # 预测特征列不匹配
    df_wrong = gen_predict_df(n=5).rename(columns={"f0": "wrong_col"})
    r = requests.post(f"{BASE_URL}/predict_text",
                      json={"csv_content": df_to_csv_str(df_wrong),
                            "model_id": "test_text_model"})
    check("特征列缺失 -> 400", r.status_code == 400)

    # 本地路径不存在
    r = requests.post(f"{BASE_URL}/train_from_path",
                      json={"file_path": "/not/exist/file.csv", "model_id": "err"})
    check("文件路径不存在 -> 400", r.status_code == 400)


def test_delete_model(model_id: str):
    print(f"\n[11] DELETE /models/{model_id}")
    r = requests.delete(f"{BASE_URL}/models/{model_id}")
    check("状态码 200", r.status_code == 200)
    r2 = requests.get(f"{BASE_URL}/models/{model_id}")
    check("删除后 GET -> 404", r2.status_code == 404)


def test_auto_tune_file(model_id="test_auto_tune_model") -> str:
    print(f"\n[12] POST /train（auto_tune=true，文件上传，n_trials=5）")
    df = gen_train_df(n=500)
    t0 = time.time()
    r = requests.post(
        f"{BASE_URL}/train",
        files={"file": ("train.csv", df_to_bytes(df), "text/csv")},
        data={"model_id": model_id, "owner": "tester", "task": "classification",
              "id_col": "id", "label_col": "label",
              "auto_tune": "true", "n_trials": "5"},
    )
    elapsed = round(time.time() - t0, 1)
    check("状态码 200", r.status_code == 200, r.text[:100] if r.status_code != 200 else "")
    if r.status_code == 200:
        meta = r.json()
        check("返回 model_id", meta.get("model_id") == model_id)
        check("metrics.auc 正常", "auc" in meta.get("metrics", {}))
        at = meta.get("auto_tune")
        check("auto_tune 字段不为 None", at is not None)
        check("记录了 n_trials=5", at.get("n_trials") == 5 if at else False)
        check("记录了 best_value", "best_value" in (at or {}))
        check("top5_trials 有数据", len((at or {}).get("top5_trials", [])) > 0)
        hp = meta.get("hyperparams", {})
        check("hyperparams 含 colsample_bytree", "colsample_bytree" in hp)
        check("hyperparams 含 min_child_weight", "min_child_weight" in hp)
        print(f"    耗时 {elapsed}s，AUC={meta['metrics'].get('auc')}，"
              f"最优超参: n_estimators={hp.get('n_estimators')}, "
              f"max_depth={hp.get('max_depth')}, "
              f"lr={round(hp.get('learning_rate', 0), 4)}")
    return model_id


def test_auto_tune_text(model_id="test_auto_tune_text") -> str:
    print(f"\n[13] POST /train_text（auto_tune=true，JSON文本，n_trials=5）")
    df = gen_train_df(n=300)
    t0 = time.time()
    r = requests.post(
        f"{BASE_URL}/train_text",
        json={"csv_content": df_to_csv_str(df), "model_id": model_id,
              "owner": "openclaw", "task": "classification",
              "id_col": "id", "label_col": "label",
              "auto_tune": True, "n_trials": 5},
    )
    elapsed = round(time.time() - t0, 1)
    check("状态码 200", r.status_code == 200, r.text[:100] if r.status_code != 200 else "")
    if r.status_code == 200:
        meta = r.json()
        check("返回 model_id", meta.get("model_id") == model_id)
        at = meta.get("auto_tune")
        check("auto_tune 字段不为 None", at is not None)
        check("记录了 top5_trials", len((at or {}).get("top5_trials", [])) > 0)
        print(f"    耗时 {elapsed}s，AUC={meta['metrics'].get('auc')}")
    return model_id


# ─────────────────────────── 主流程 ───────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("algo_trainer 完整自动化测试")
    print("=" * 55)

    try:
        test_models_empty()

        mid_file = test_train_file("test_file_model")
        test_predict_file(mid_file)

        mid_text = test_train_text("test_text_model")
        test_predict_text(mid_text)

        mid_path = test_train_from_path("test_path_model")
        test_predict_from_path(mid_path)

        test_models_list([mid_file, mid_text, mid_path])
        test_model_detail(mid_file)

        test_error_cases()

        test_delete_model("test_path_model")

        mid_at_file = test_auto_tune_file("test_auto_tune_model")
        mid_at_text = test_auto_tune_text("test_auto_tune_text")

        # 用 auto_tune 训练的模型正常预测
        print(f"\n[14] 用 auto_tune 模型预测（predict_text）")
        df_pred = gen_predict_df(n=10)
        r = requests.post(f"{BASE_URL}/predict_text",
                          json={"csv_content": df_to_csv_str(df_pred),
                                "model_id": mid_at_file, "output_prob": True})
        check("状态码 200", r.status_code == 200)
        if r.status_code == 200:
            data = r.json()
            check("返回 10 行结果", data.get("total_rows") == 10)
            check("包含 pred_prob", "pred_prob" in data["results"][0])

        print("\n" + "=" * 55)
        print("所有测试完成 [ALL DONE]")
        print("=" * 55)

    except requests.exceptions.ConnectionError:
        print("\n[ERROR] 连接失败，请先启动服务:")
        print("  cd d:\\EthanProject\\algo_trainer")
        print("  py -3.14 -m uvicorn main:app --host 127.0.0.1 --port 8000")
