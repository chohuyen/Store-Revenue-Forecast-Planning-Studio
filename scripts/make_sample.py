"""Tạo data/sample (nhỏ) từ data/raw: 12 cửa hàng, dùng để demo/chạy thử. Chạy: python scripts/make_sample.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src import config
STORES = [1, 85, 262, 292, 500, 562, 817, 842, 1000, 1097, 1114, 733]
config.DATA_SAMPLE.mkdir(parents=True, exist_ok=True)
for f in ("train.csv", "test.csv"):
    d = pd.read_csv(config.DATA_RAW / f, low_memory=False)
    d[d["Store"].isin(STORES)].to_csv(config.DATA_SAMPLE / f, index=False)
pd.read_csv(config.DATA_RAW / "store.csv").query("Store in @STORES").to_csv(config.DATA_SAMPLE / "store.csv", index=False)
print("Đã tạo", config.DATA_SAMPLE)
