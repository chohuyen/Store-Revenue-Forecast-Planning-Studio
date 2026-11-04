# Kết quả rà soát notebook / báo cáo (Giai đoạn 1)

Đã xác minh bằng cách chạy lại bằng `src/`:
- Chia thời gian: Train 765.470 / Val 37.472 / Test 41.396 dòng, đúng Bảng 4.1 của báo cáo.
- Baseline A và B tái lập **đúng** số của nhóm (Val RMSPE 0,153089 / 0,173401; Test 0,147522 / 0,163136).
- Encoder fit trên Train; không có `Customers` trong 24 đặc trưng; RMSPE chỉ tính trên Sales ≠ 0.

Chưa xác minh được trong môi trường này: số liệu của XGBoost/LightGBM/Random Forest (cần cài `xgboost`, `lightgbm`) – dùng số của nhóm trong `outputs/results/model_comparison.csv`.

## Vấn đề phát hiện

| # | Vấn đề | Mức độ | Đề xuất / cần chạy lại |
|---|---|---|---|
| 1 | Notebook phụ thuộc Colab + Google Drive cá nhân (ô 0, 25, 26: `drive.mount`, `/content/...`) | Cao | Dùng `src/` + `data/raw/`; notebook giữ nguyên làm tài liệu gốc |
| 2 | 24 đặc trưng chỉ có `IsStateHoliday` (0/1), **không có loại ngày lễ a/b/c** → chọn a/b/c trong What-if cho cùng một kết quả | Cao với What-if | Train với `--with-holiday-type` nếu muốn phân biệt; phải chạy lại đánh giá và cập nhật Bảng 4.3 |
| 3 | Dữ liệu ngày lễ có doanh thu rất ít (a 694, b 145, c 71) và SHAP cho thấy ngày lễ không nằm trong nhóm chính (đã ghi ở mục 4.5 DRAFT) | Trung bình | Diễn giải What-if ngày lễ thận trọng |
| 4 | Chương 3 (3.3.2) nêu target encoding và lag/rolling nhưng notebook cuối **không dùng** các đặc trưng này | Trung bình | Sửa câu chữ hoặc bỏ dòng khỏi checklist leakage (đã chỉnh trong DRAFT_v2) |
| 5 | Median CompetitionDistance tính trên toàn bộ store.csv (bảng thuộc tính tĩnh), báo cáo nói "chỉ fit trên train" | Thấp | Không gây rò rỉ thời gian (thuộc tính tĩnh); nên nói rõ trong báo cáo |
| 6 | `fillna(..., inplace=True)` trên cột (ô 3) sẽ không còn hiệu lực với pandas 3.0 | Thấp | `pipeline.py` đã viết lại không dùng inplace |
| 7 | Tập Test dài 43 ngày (19/06–31/07) trong khi báo cáo/Literature ghi "6 tuần/42 ngày" (Val là 42 ngày) | Thấp | Ghi chính xác 43 ngày |
| 8 | Đánh giá chỉ trên ngày cửa hàng mở (đã loại Open=0 và Sales=0); mô hình không nhận `Open` làm đầu vào | Trung bình | Ghi vào hạn chế; web cảnh báo "giả định cửa hàng mở" |
| 9 | XGBoost thắng Val nhưng Test RMSPE (0,1485) chưa thấp hơn Baseline A (0,1475); MAE, RMSE vẫn tốt hơn | Thông tin | Báo cáo đã diễn giải đúng; giữ nguyên, không tuyên bố vượt mọi baseline |
| 10 | Checklist: ô "Tôi là" = Hân, trong khi người dùng là Huyên | Thông tin | Chờ Huyên xác nhận; không tự sửa |
| 11 | Checklist: 8 việc "Xong" nhưng cột Link trống (A1, A2, B4, B5, D4, E2, E5, E6) trái quy tắc "không có link = chưa xong" | Thông tin | Nhóm bổ sung link; không tự sửa trạng thái |
