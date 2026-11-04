"""ModelBundle: pipeline + encoder + danh sách đặc trưng + mô hình, lưu/nạp bằng joblib."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import joblib


class ModelBundle:
    def __init__(self, pipeline, encoder, model, features, categorical, algo, metadata=None):
        self.pipeline, self.encoder, self.model = pipeline, encoder, model
        self.features, self.categorical = list(features), list(categorical)
        self.algo = algo
        self.metadata = metadata or {}

    @staticmethod
    def make_encoder():
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def encode(self, df):
        X = df[self.features].copy()
        X[self.categorical] = self.encoder.transform(X[self.categorical].astype(str))
        return X

    def predict_features(self, df):
        """df: bảng đã có đủ cột đặc trưng (đầu ra của pipeline). Trả về doanh thu (€)."""
        pred = np.expm1(self.model.predict(self.encode(df)))
        return np.maximum(pred, 0.0)

    @property
    def supports_holiday_type(self):
        return "StateHoliday" in self.features

    def save(self, path):
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path):
        return joblib.load(path)
