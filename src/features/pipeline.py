"""Pipeline tiền xử lý + Feature Engineering (dùng chung cho notebook, huấn luyện và web).

Tách từ RossmannDataPipeline trong TEST DATA.ipynb, bổ sung:
- fit() học median và LƯU bảng store đã làm sạch (bundle tự đủ để dự báo),
- build_inference_frame() tạo đặc trưng cho các dòng nhập từ web, giữ nguyên thứ tự dòng.
"""
import numpy as np
import pandas as pd


class RossmannDataPipeline:
    def __init__(self, median_competition_dist=2325.0):
        self.median_competition_dist = median_competition_dist
        self.store_clean_ = None

    def fit(self, store_raw):
        if "CompetitionDistance" in store_raw.columns:
            self.median_competition_dist = float(store_raw["CompetitionDistance"].median())
        self.store_clean_ = self.preprocess_store(store_raw)
        return self

    def preprocess_store(self, store_raw):
        s = store_raw.copy()
        s["CompetitionDistance"] = s["CompetitionDistance"].fillna(self.median_competition_dist)
        for c in ("CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"):
            s[c] = s[c].fillna(-1).astype(int)
        for c in ("Promo2SinceWeek", "Promo2SinceYear"):
            s[c] = s[c].fillna(0).astype(int)
        s["PromoInterval"] = s["PromoInterval"].fillna("None")
        s["HasCompetition"] = (s["CompetitionOpenSinceYear"] != -1).astype(int)
        return s

    def preprocess_train(self, train_raw, is_train_set=True):
        df = train_raw.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["StateHoliday"] = df["StateHoliday"].astype(str).replace({"0.0": "0"})
        if is_train_set and "Open" in df.columns and "Sales" in df.columns:
            df = df[(df["Open"] != 0) & (df["Sales"] > 0)].copy()
        return df

    def feature_engineering(self, df):
        d = df.copy()
        d["Year"] = d["Date"].dt.year
        d["Month"] = d["Date"].dt.month
        d["Day"] = d["Date"].dt.day
        d["WeekOfYear"] = d["Date"].dt.isocalendar().week.astype(int)
        d["DayOfWeek"] = d["Date"].dt.dayofweek + 1
        d["IsWeekend"] = (d["DayOfWeek"] >= 6).astype(int)
        d["IsMonthStart"] = d["Date"].dt.is_month_start.astype(int)
        d["IsMonthEnd"] = d["Date"].dt.is_month_end.astype(int)
        if "Sales" in d.columns:
            d["LogSales"] = np.log1p(d["Sales"])
        d["IsStateHoliday"] = (d["StateHoliday"] != "0").astype(int)
        d["Promo_x_SchoolHoliday"] = d["Promo"] * d["SchoolHoliday"]
        d["Promo_x_StateHoliday"] = d["Promo"] * d["IsStateHoliday"]
        return d

    def transform(self, train_raw, store_raw=None, is_train_set=True):
        """1 lệnh: làm sạch -> ghép store -> tạo đặc trưng -> sắp xếp (Date, Store)."""
        store = self.preprocess_store(store_raw) if store_raw is not None else self.store_clean_
        if store is None:
            raise ValueError("Chưa có bảng store: gọi fit(store_raw) hoặc truyền store_raw.")
        clean = self.preprocess_train(train_raw, is_train_set=is_train_set)
        out = self.feature_engineering(clean.merge(store, on="Store", how="left"))
        return out.sort_values(["Date", "Store"]).reset_index(drop=True)

    def build_inference_frame(self, rows):
        """rows: list[dict] (Store, Date, Promo, StateHoliday, SchoolHoliday) -> DataFrame đặc trưng."""
        raw = pd.DataFrame(rows)
        raw["Open"] = 1
        merged = self.preprocess_train(raw, is_train_set=False).merge(self.store_clean_, on="Store", how="left")
        if merged["StoreType"].isna().any():
            raise ValueError("Có Store không tồn tại trong store.csv.")
        return self.feature_engineering(merged)


if __name__ == "__main__":  # python -m src.features.pipeline
    from src.data.loader import load_train_store
    tr, st = load_train_store()
    out = RossmannDataPipeline().fit(st).transform(tr, st)
    print("Dữ liệu sạch:", out.shape)
    out.to_csv("data/processed/train_clean.csv", index=False)
    print("Đã lưu data/processed/train_clean.csv")
