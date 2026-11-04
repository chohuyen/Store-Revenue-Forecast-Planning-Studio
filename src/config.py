"""Cấu hình trung tâm. Mọi đường dẫn tính theo thư mục gốc repo (không phụ thuộc máy/Google Drive)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_SAMPLE = ROOT / "data" / "sample"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"
BUNDLE_PATH = MODELS_DIR / "forecaster.joblib"

SEED = 42

# Cách chia theo thời gian (giữ nguyên như notebook TEST DATA.ipynb)
TRAIN_END = "2015-05-07"
VAL_START, VAL_END = "2015-05-08", "2015-06-18"
TEST_START = "2015-06-19"

# 24 đặc trưng của mô hình trong notebook (KHÔNG có Customers, KHÔNG có loại ngày lễ a/b/c)
FEATURES = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday", "HasCompetition",
    "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "Year", "Month", "Day",
    "WeekOfYear", "IsWeekend", "IsMonthStart", "IsMonthEnd", "IsStateHoliday",
    "Promo_x_SchoolHoliday", "Promo_x_StateHoliday",
    "StoreType", "Assortment", "PromoInterval",
]
CATEGORICAL = ["StoreType", "Assortment", "PromoInterval"]
# Phiên bản tùy chọn: thêm loại ngày lễ để What-if phân biệt được a/b/c
FEATURES_WITH_HOLIDAY_TYPE = FEATURES + ["StateHoliday"]
CATEGORICAL_WITH_HOLIDAY_TYPE = CATEGORICAL + ["StateHoliday"]

HOLIDAY_LABELS = {
    "0": "Không phải ngày lễ",
    "a": "Public holiday (ngày lễ công cộng)",
    "b": "Easter holiday (lễ Phục sinh)",
    "c": "Christmas (Giáng sinh)",
}
VALID_HOLIDAYS = tuple(HOLIDAY_LABELS)
DATE_MIN, DATE_MAX = "2013-01-01", "2016-12-31"
DATA_LAST_DATE = "2015-09-17"  # ngày cuối trong test.csv của Kaggle
