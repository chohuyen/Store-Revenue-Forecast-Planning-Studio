"""Chạy các ca What-if mẫu bằng mô hình đang lưu và ghi outputs/results/whatif_demo_results.csv.
Chạy lại sau mỗi lần huấn luyện mô hình mới để cập nhật số liệu trong báo cáo."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src import config
from src.models.predict import Forecaster
from src.models.whatif import run_what_if

CASES = [  # (store, date, promo_goc, lễ_gốc, school, promo_mới, lễ_mới, mô tả)
    (262, "2015-07-20", 0, "0", 1, 1, "0", "Bật Promo"),
    (262, "2015-07-20", 1, "0", 1, 0, "0", "Tắt Promo"),
    (262, "2015-07-20", 0, "0", 1, 0, "a", "Đổi StateHoliday 0->a"),
    (262, "2015-07-20", 0, "0", 1, 1, "a", "Kết hợp Promo + lễ a"),
    (1, "2015-07-20", 0, "0", 1, 1, "0", "Cửa hàng khác: bật Promo"),
    (817, "2015-07-20", 0, "0", 1, 1, "0", "Cửa hàng khác: bật Promo"),
    (262, "2015-08-10", 0, "0", 1, 1, "0", "Ngày khác (trong giai đoạn test.csv)"),
    (262, "2014-03-10", 0, "0", 0, 1, "0", "Ngày khác (2014)"),
]
fc = Forecaster()
rows = []
for s, d, p0, h0, sc, p1, h1, desc in CASES:
    r = run_what_if(fc, fc.validate(s, d, p0, h0, sc), promo=p1, state_holiday=h1)
    rows.append({"Mô tả": desc, "Store": s, "Ngày": d, "Promo gốc": p0, "Lễ gốc": h0, "Promo mới": p1, "Lễ mới": h1,
                 "Doanh thu gốc (€)": round(r["base_sales"], 2), "Doanh thu What-if (€)": round(r["new_sales"], 2),
                 "Chênh lệch (€)": round(r["diff"], 2), "Tỷ lệ (%)": None if r["pct"] is None else round(r["pct"], 2),
                 "Mô hình": fc.info["algo"]})
out = pd.DataFrame(rows)
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out.to_csv(config.RESULTS_DIR / "whatif_demo_results.csv", index=False, encoding="utf-8-sig")
print(out.to_string(index=False))
