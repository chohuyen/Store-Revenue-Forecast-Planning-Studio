"""Chỉ số đánh giá và baseline (định nghĩa khớp notebook)."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmspe(y_true, y_pred):
    """RMSPE chỉ tính trên các dòng y_true != 0."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = y_true != 0
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean(((y_true[m] - y_pred[m]) / y_true[m]) ** 2)))


def evaluate_predictions(y_true, y_pred):
    return {
        "RMSPE": rmspe(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def predict_baseline_a(split, train_split):
    t = train_split.groupby(["Store", "DayOfWeek", "Promo"])["Sales"].mean().rename("pred").reset_index()
    m = split[["Store", "DayOfWeek", "Promo"]].merge(t, on=["Store", "DayOfWeek", "Promo"], how="left")
    return m["pred"].fillna(train_split["Sales"].mean()).values


def predict_seasonal_naive(split, full, train_split):
    lag = full[["Store", "Date", "Sales"]].copy()
    lag["Date"] = lag["Date"] + pd.Timedelta(weeks=52)
    m = split[["Store", "Date"]].merge(lag.rename(columns={"Sales": "pred"}), on=["Store", "Date"], how="left")
    fb = pd.Series(predict_baseline_a(split, train_split), index=m.index)
    return m["pred"].fillna(fb).values
