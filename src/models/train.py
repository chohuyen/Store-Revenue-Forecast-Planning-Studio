"""Huấn luyện + đánh giá + lưu mô hình.

Ví dụ (Windows/PyCharm, đứng ở thư mục gốc repo):
    python -m src.models.train --algo xgboost
    python -m src.models.train --algo xgboost --with-holiday-type   # What-if phân biệt được a/b/c
Kết quả đánh giá ghi vào outputs/results/model_comparison_run.csv (KHÔNG ghi đè model_comparison.csv của nhóm).
"""
import argparse
import json
import platform
import random
from datetime import datetime

import numpy as np
import pandas as pd

from src import config
from src.data.loader import load_train_store
from src.features.pipeline import RossmannDataPipeline
from src.models.bundle import ModelBundle
from src.models import evaluate as ev


def make_model(algo, seed=config.SEED):
    if algo == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(objective="reg:squarederror", learning_rate=0.1, max_depth=7,
                                subsample=0.9, colsample_bytree=0.9, random_state=seed,
                                n_jobs=-1, n_estimators=400)
    if algo == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="regression", learning_rate=0.1, max_depth=8,
                                 num_leaves=63, random_state=seed, n_estimators=400, verbose=-1)
    if algo == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=50, max_depth=15, min_samples_leaf=10,
                                     max_samples=0.1, random_state=seed, n_jobs=-1)
    if algo == "hgb":  # thay thế khi máy chưa cài xgboost (có sẵn trong scikit-learn)
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(learning_rate=0.1, max_depth=7, max_iter=400, random_state=seed)
    raise ValueError(f"algo không hợp lệ: {algo}")


def split_by_time(d):
    train = d[d["Date"] <= config.TRAIN_END]
    val = d[(d["Date"] >= config.VAL_START) & (d["Date"] <= config.VAL_END)]
    test = d[d["Date"] >= config.TEST_START]
    return train.copy(), val.copy(), test.copy()


def run(algo="xgboost", with_holiday_type=False, out_path=config.BUNDLE_PATH, evaluate=True):
    random.seed(config.SEED); np.random.seed(config.SEED)
    feats = config.FEATURES_WITH_HOLIDAY_TYPE if with_holiday_type else config.FEATURES
    cats = config.CATEGORICAL_WITH_HOLIDAY_TYPE if with_holiday_type else config.CATEGORICAL
    train_raw, store_raw = load_train_store()
    metadata = {"algo": algo, "seed": config.SEED, "features": feats,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "python": platform.python_version(), "with_holiday_type": with_holiday_type}

    if evaluate:
        pipe = RossmannDataPipeline().fit(store_raw)
        d = pipe.transform(train_raw, store_raw)
        tr, va, te = split_by_time(d)
        enc = ModelBundle.make_encoder().fit(tr[cats].astype(str))   # encoder CHỈ fit trên Train
        b = ModelBundle(pipe, enc, make_model(algo), feats, cats, algo)
        b.model.fit(b.encode(tr), tr["LogSales"])
        rows = []
        for name, pv, pt in [
            ("Baseline A (Store-DOW-Promo)", ev.predict_baseline_a(va, tr), ev.predict_baseline_a(te, tr)),
            ("Baseline B (Seasonal Naive)", ev.predict_seasonal_naive(va, d, tr), ev.predict_seasonal_naive(te, d, tr)),
            (algo, b.predict_features(va), b.predict_features(te)),
        ]:
            mv, mt = ev.evaluate_predictions(va["Sales"], pv), ev.evaluate_predictions(te["Sales"], pt)
            rows.append({"Mô hình": name, **{f"Val {k}": v for k, v in mv.items()}, **{f"Test {k}": v for k, v in mt.items()}})
        res = pd.DataFrame(rows)
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        res.to_csv(config.RESULTS_DIR / "model_comparison_run.csv", index=False, encoding="utf-8-sig")
        print(res.to_string(index=False))
        metadata.update({"n_train": len(tr), "n_val": len(va), "n_test": len(te),
                         "eval": rows[-1]})

    # Huấn luyện lại trên TOÀN BỘ dữ liệu sạch (giống notebook) để phục vụ ứng dụng
    pipe = RossmannDataPipeline().fit(store_raw)
    d_full = pipe.transform(train_raw, store_raw)
    enc = ModelBundle.make_encoder().fit(d_full[cats].astype(str))
    bundle = ModelBundle(pipe, enc, make_model(algo), feats, cats, algo, metadata)
    bundle.model.fit(bundle.encode(d_full), d_full["LogSales"])
    metadata["n_full_train"] = len(d_full)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle.save(out_path)
    (config.MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("Đã lưu:", out_path)
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="xgboost", choices=["xgboost", "lightgbm", "rf", "hgb"])
    ap.add_argument("--with-holiday-type", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--out", default=str(config.BUNDLE_PATH))
    a = ap.parse_args()
    run(a.algo, a.with_holiday_type, a.out, evaluate=not a.skip_eval)
