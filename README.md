# Store Revenue Forecast & Planning Studio

Ứng dụng dự báo doanh thu cửa hàng và **mô phỏng kịch bản What-if** (khuyến mãi, ngày lễ) trên bộ dữ liệu Rossmann Store Sales.
Đồ án nhóm môn *Machine Learning in Business Analytics* (MLBA).

## 1. Mục tiêu
- Dự báo doanh thu ngày của từng cửa hàng; so sánh mô hình (RMSPE, MAE, RMSE); phân tích sai số và đặc trưng (SHAP/LIME từ thực nghiệm).
- **What-if Scenario:** đổi `Promo` (0/1) và `StateHoliday` (0/a/b/c), giữ nguyên mọi đầu vào khác, so sánh doanh thu dự báo ban đầu và kịch bản mới.
- Lưu ý: What-if thể hiện **thay đổi của dự báo mô hình**, không chứng minh quan hệ nhân quả.

## 2. Dữ liệu
Rossmann Store Sales (Kaggle): 1.115 cửa hàng, 01/01/2013–31/07/2015. Nguồn và điều khoản: xem `data/raw/SOURCE.md` (giấy phép **chưa xác minh**; dữ liệu gốc không đưa lên GitHub). Từ điển dữ liệu: `docs/data_dictionary.md`.

## 3. Kiến trúc
```
data/raw ──> src/features/pipeline.py ──> src/models/train.py ──> models/forecaster.joblib
                 (làm sạch + 24 đặc trưng)   (Train/Val/Test theo thời gian)   (pipeline+encoder+model)
                                                                    │
app/app.py (Flask) ── src/models/predict.py ── src/models/whatif.py ┘   (chỉ nạp mô hình, KHÔNG huấn luyện lại)
```

## 4. Mô hình và kết quả (số liệu thực nghiệm của nhóm – `outputs/results/model_comparison.csv`)
| Mô hình | Val RMSPE | Test RMSPE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| Baseline A (Store-DOW-Promo) | 0,1531 | **0,1475** | 749,49 | 1.115,32 |
| Baseline B (Seasonal Naive) | 0,1734 | 0,1631 | 803,67 | 1.210,13 |
| LightGBM | 0,1466 | 0,1590 | 770,49 | 1.123,47 |
| XGBoost | **0,1403** | 0,1485 | **717,66** | **1.041,78** |
| Random Forest | 0,2134 | 0,2130 | 1.099,46 | 1.634,95 |

XGBoost tốt nhất trên Validation, nhưng trên Test RMSPE **chưa thấp hơn Baseline A** (MAE và RMSE thì thấp hơn). Không nên kết luận XGBoost vượt mọi baseline.

## 5. Cài đặt (Windows + PyCharm)
1. Cài Python 3.10–3.12 (https://www.python.org, tích **Add Python to PATH**).
2. PyCharm → *Open* thư mục dự án → *File → Settings → Project → Python Interpreter → Add Interpreter → Virtualenv*.
3. Mở **Terminal** trong PyCharm:
```
pip install -r requirements.txt
```
4. Đặt `rossmann-store-sales.zip` (tải từ Kaggle, xem `data/raw/SOURCE.md`) vào `data/raw/`, rồi:
```
python scripts/prepare_data.py
```
5. Huấn luyện mô hình (khoảng vài phút; kết quả đánh giá ghi vào `outputs/results/model_comparison_run.csv`):
```
python -m src.models.train --algo xgboost
```
Tùy chọn: `--algo lightgbm|rf|hgb`, `--with-holiday-type` (thêm loại ngày lễ a/b/c vào đặc trưng; **làm thay đổi số liệu so với báo cáo**, cần chạy lại và cập nhật), `--skip-eval` (chỉ huấn luyện bản cuối).

> Bản repo đi kèm có thể chứa `models/forecaster.joblib` được tạo **trong sandbox bằng HistGradientBoosting** để chạy thử giao diện (xem `models/metadata.json`: `stand_in_model: true`). Kết quả đó **không phải XGBoost**; hãy chạy lệnh ở bước 5 để thay thế.

Chạy trên Colab (không cần Google Drive): upload thư mục dự án, `!pip install -r requirements.txt`, `!python scripts/prepare_data.py`, `!python -m src.models.train --algo xgboost`.

## 6. Chạy ứng dụng
```
python -m app.app
```
Mở http://127.0.0.1:5000. Nếu thiếu `data/raw`, ứng dụng dùng `data/sample` (12 cửa hàng) để lấy kịch bản ban đầu; mô hình vẫn dự báo được cả 1.115 cửa hàng.

## 7. Cách dùng
**Dự báo:** *Dự báo doanh thu* → nhập Store (1–1115), ngày, Promo, StateHoliday, SchoolHoliday → *Chạy dự báo*. Kết quả giả định cửa hàng **mở cửa**.

**What-if:** *What-if Scenario* → nhập Store + ngày → *Tải kịch bản ban đầu* (lấy từ dữ liệu nếu có) → chọn Promo mới và StateHoliday mới → *Mô phỏng kịch bản*. Hệ thống hiển thị bảng so sánh, chênh lệch (€), tỷ lệ thay đổi (%) (không xác định nếu doanh thu ban đầu = 0), biểu đồ cột và so sánh 4 kịch bản (Promo 0/1 × lễ 0/có lễ).
Giới hạn: ngày lễ có rất ít dữ liệu doanh thu (a: 694, b: 145, c: 71 dòng huấn luyện); với 24 đặc trưng mặc định mô hình **không phân biệt a/b/c**.

API: `POST /api/predict`, `POST /api/whatif` (JSON).

## 8. Kiểm thử
```
python -m unittest discover -s tests -v
```
Bảng QA: `docs/QA_test_cases.xlsx`.

## 9. Ảnh giao diện
`docs/demo/01_tong_quan.png`, `02_du_bao.png`, `03_whatif_buoc3.png`, `04_whatif_ketqua.png`, `05_phan_tich.png` (chụp với mô hình thay thế).

## 10. Cấu trúc thư mục
```
src/{config.py, data/, features/, models/, utils/}   app/{app.py, templates/, static/}
models/  data/{raw,sample,processed}  outputs/{results,figures}  notebooks/  docs/  tests/  scripts/
```

## 11. Thành viên
| Tên | Họ tên | MSSV | Vai trò |
|---|---|---|---|
| Huyên | Lê Trần Ngọc Huyên | K244060717 | Project Coordinator – điều phối, chốt bản chính, Chương 5–6 |
| Bảo | Nguyễn Gia Bảo | K244030343 | Research & Problem Definition – Chương 1–2, kiểm thử độc lập |
| Hà | Bùi Thi Khánh Hà | K244060714 | Data & EDA – phân tích kết quả, Chương 4 |
| Vy | Lê Vy | K244161824 | ML Modeling & Evaluation – mô hình, Chương 3 |
| Lụa | K Thị Lụa | K244060724 | Web & Model Integration – web, tích hợp, demo |
| Hân | Nguyễn Phan Bảo Hân | K244161782 | Report, QA & Presentation – tham khảo, QA, slide |
