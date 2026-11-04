# Kịch bản demo (G2/G3/G6) – khoảng 10 phút

Chuẩn bị: `python -m src.models.train --algo xgboost` (một lần) → `python -m app.app` → mở http://127.0.0.1:5000. Mở sẵn ảnh/video dự phòng trong `docs/demo/`.

| # | Nội dung | Thao tác / lời nói chính |
|---|---|---|
| 1 | Giới thiệu đề tài | Bài toán dự báo doanh thu ngày cho 1.115 cửa hàng và thử kịch bản khuyến mãi/ngày lễ |
| 2 | Dữ liệu | Rossmann Store Sales; 844.338 dòng sạch; trang **Tổng quan** |
| 3 | Quy trình | Làm sạch → đặc trưng (24) → chia theo thời gian → huấn luyện → lưu bundle |
| 4 | Mô hình | Baseline A, B, LightGBM, XGBoost, Random Forest |
| 5 | Đánh giá | Trang **Phân tích kết quả**: XGBoost tốt nhất Val; Test RMSPE chưa vượt Baseline A (0,1485 vs 0,1475) nhưng MAE/RMSE tốt hơn |
| 6 | Mở web | Trang **Dự báo doanh thu** |
| 7 | Dự báo | Store 262, ngày 2015-07-20, Promo 1 → xem doanh thu và thông tin đầu vào |
| 8 | Bật/tắt Promo, đổi ngày lễ | Trang **What-if**: nhập 262 + 2015-07-20 → "Tải kịch bản ban đầu" |
| 9 | Chạy What-if | Đặt Promo mới = 1, StateHoliday mới = a → "Mô phỏng kịch bản" |
| 10 | So sánh 4 kịch bản | Đọc bảng + biểu đồ; nhấn mạnh: không phải quan hệ nhân quả; ngày lễ có ít dữ liệu |
| 11 | Kết luận | Giá trị cho quản lý + hạn chế + hướng phát triển |

**Phương án dự phòng nếu web lỗi:** (1) mở `docs/demo/*.png` hoặc video quay sẵn; (2) chạy `python scripts/generate_whatif_tables.py` và mở `outputs/results/whatif_demo_results.csv`; (3) mở `outputs/results/dashboard.html`.
**Nhật ký diễn tập (G3) – điền sau mỗi lần chạy thật:** ngày | người chạy | lỗi | thời gian. (Chưa có lần diễn tập nào được thực hiện.)
