"""Forecaster: nạp bundle đã lưu, kiểm tra đầu vào, dự báo 1 hoặc nhiều dòng. KHÔNG huấn luyện lại."""
from dataclasses import dataclass
import pandas as pd
from src import config
from src.models.bundle import ModelBundle


class InputError(ValueError):
    """Lỗi đầu vào người dùng (thông báo tiếng Việt, hiển thị trực tiếp trên web)."""


@dataclass
class Scenario:
    store: int
    date: pd.Timestamp
    promo: int
    state_holiday: str
    school_holiday: int

    def as_row(self):
        return {"Store": self.store, "Date": self.date, "Promo": self.promo,
                "StateHoliday": self.state_holiday, "SchoolHoliday": self.school_holiday}


def validate_inputs(store, date, promo, state_holiday, school_holiday, valid_stores=None):
    try:
        store_i = int(store)
    except (TypeError, ValueError):
        raise InputError("Mã cửa hàng (Store) phải là số nguyên từ 1 đến 1115.")
    if valid_stores is not None and store_i not in valid_stores:
        raise InputError(f"Cửa hàng {store_i} không tồn tại. Chọn số từ 1 đến 1115.")
    try:
        dt = pd.Timestamp(date)
        if pd.isna(dt):
            raise ValueError
    except Exception:
        raise InputError("Ngày dự báo không hợp lệ. Dùng định dạng YYYY-MM-DD, ví dụ 2015-08-15.")
    if not (pd.Timestamp(config.DATE_MIN) <= dt <= pd.Timestamp(config.DATE_MAX)):
        raise InputError(f"Ngày dự báo phải nằm trong khoảng {config.DATE_MIN} đến {config.DATE_MAX}.")
    try:
        promo_i, school_i = int(promo), int(school_holiday)
    except (TypeError, ValueError):
        raise InputError("Promo và SchoolHoliday chỉ nhận giá trị 0 hoặc 1.")
    if promo_i not in (0, 1) or school_i not in (0, 1):
        raise InputError("Promo và SchoolHoliday chỉ nhận giá trị 0 hoặc 1.")
    sh = str(state_holiday)
    if sh not in config.VALID_HOLIDAYS:
        raise InputError("StateHoliday chỉ nhận một trong các mã: 0, a, b, c.")
    return Scenario(store_i, dt.normalize(), promo_i, sh, school_i)


class Forecaster:
    def __init__(self, bundle_path=config.BUNDLE_PATH):
        try:
            self.bundle = ModelBundle.load(bundle_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Chưa có mô hình đã huấn luyện (models/forecaster.joblib). "
                "Chạy: python -m src.models.train --algo xgboost")
        self.valid_stores = set(self.bundle.pipeline.store_clean_["Store"].astype(int))

    @property
    def info(self):
        m = self.bundle.metadata
        return {"algo": self.bundle.algo, "trained_at": m.get("trained_at"),
                "features": len(self.bundle.features),
                "supports_holiday_type": self.bundle.supports_holiday_type,
                "stand_in": bool(m.get("stand_in_model"))}

    def validate(self, store, date, promo, state_holiday, school_holiday):
        return validate_inputs(store, date, promo, state_holiday, school_holiday, self.valid_stores)

    def predict_scenarios(self, scenarios):
        """Dự báo nhiều kịch bản trong MỘT lần gọi; trả về (list doanh thu, bảng đặc trưng đầu vào)."""
        frame = self.bundle.pipeline.build_inference_frame([s.as_row() for s in scenarios])
        return list(self.bundle.predict_features(frame)), frame

    def predict_one(self, scenario):
        p, f = self.predict_scenarios([scenario])
        return float(p[0]), f.iloc[0]
