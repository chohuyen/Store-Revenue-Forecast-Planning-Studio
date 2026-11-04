"""Nạp dữ liệu Rossmann từ data/raw (hoặc data/sample). Không dùng Google Drive."""
import zipfile
from pathlib import Path
import pandas as pd
from src import config


class DataNotFoundError(FileNotFoundError):
    pass


def find_data_dir() -> Path:
    for d in (config.DATA_RAW, config.DATA_SAMPLE):
        if (d / "store.csv").exists() and (d / "train.csv").exists():
            return d
    raise DataNotFoundError(
        "Không tìm thấy train.csv/store.csv. Đặt rossmann-store-sales.zip vào data/raw/ "
        "rồi chạy: python scripts/prepare_data.py"
    )


def extract_zip(zip_path, dest=None):
    dest = Path(dest or config.DATA_RAW)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return dest


def load_train_store(data_dir=None):
    d = Path(data_dir) if data_dir else find_data_dir()
    return pd.read_csv(d / "train.csv", low_memory=False), pd.read_csv(d / "store.csv")


def load_test(data_dir=None):
    d = Path(data_dir) if data_dir else find_data_dir()
    p = d / "test.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p, low_memory=False)
    t["Open"] = t["Open"].fillna(1)  # 11 dòng thiếu Open trong test.csv gốc
    return t


def load_history(data_dir=None):
    """Lịch sử train + test: dùng lấy kịch bản ban đầu cho Store/ngày được chọn."""
    d = Path(data_dir) if data_dir else find_data_dir()
    train = pd.read_csv(d / "train.csv", low_memory=False, usecols=lambda c: c != "Customers")
    test = load_test(d)
    if test is not None:
        test = test.drop(columns=[c for c in ("Id",) if c in test.columns])
        train = pd.concat([train, test], ignore_index=True)
    train["Date"] = pd.to_datetime(train["Date"])
    train["StateHoliday"] = train["StateHoliday"].astype(str).replace({"0.0": "0"})
    return train
