"""Giải nén rossmann-store-sales.zip vào data/raw và kiểm tra nhanh.
Dùng:  python scripts/prepare_data.py [đường_dẫn_zip]
Mặc định tìm data/raw/rossmann-store-sales.zip (hoặc file zip cùng tên ở thư mục gốc repo).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src import config
from src.data.loader import extract_zip

cands = [Path(sys.argv[1])] if len(sys.argv) > 1 else [config.DATA_RAW / "rossmann-store-sales.zip", config.ROOT / "rossmann-store-sales.zip"]
zp = next((p for p in cands if p.exists()), None)
if zp is None and not (config.DATA_RAW / "train.csv").exists():
    sys.exit("Không thấy file zip. Tải rossmann-store-sales.zip từ Kaggle (xem README) và đặt vào data/raw/.")
if zp is not None:
    extract_zip(zp)
for f, n in (("train.csv", 1017209), ("store.csv", 1115), ("test.csv", 41088)):
    df = pd.read_csv(config.DATA_RAW / f, low_memory=False)
    print(f"{f}: {df.shape}  {'OK' if len(df) == n else 'KHÁC số dòng kỳ vọng ' + str(n)}")
