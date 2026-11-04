# Data Dictionary & Data Issue Log (B2)

## train.csv
| Cột | Kiểu | Ý nghĩa | Ghi chú |
|---|---|---|---|
| Store | int | Mã cửa hàng 1–1115 | khóa nối với store.csv |
| DayOfWeek | int | Thứ trong tuần (1 = Thứ Hai … 7 = Chủ Nhật) | pipeline tính lại từ Date |
| Date | date | Ngày | 2013-01-01 → 2015-07-31 |
| Sales | int | Doanh thu ngày (€) – **biến mục tiêu** | huấn luyện trên log1p(Sales) |
| Customers | int | Số khách | **không dùng làm đầu vào** (không có trong test.csv) |
| Open | 0/1 | Cửa hàng mở cửa | 0 = đóng |
| Promo | 0/1 | Có khuyến mãi trong ngày | biến chính của What-if |
| StateHoliday | str | 0 = không; a = public holiday; b = Easter; c = Christmas | trong dữ liệu có đủ 4 mã `0, a, b, c` |
| SchoolHoliday | 0/1 | Ngày nghỉ học ảnh hưởng cửa hàng | |

## store.csv
| Cột | Ý nghĩa | Thiếu |
|---|---|---|
| StoreType (a,b,c,d), Assortment (a,b,c) | Loại cửa hàng, mức hàng hóa | 0 |
| CompetitionDistance | Khoảng cách đối thủ gần nhất (m) | 3 |
| CompetitionOpenSinceMonth/Year | Thời điểm đối thủ mở | 354 |
| Promo2, Promo2SinceWeek/Year, PromoInterval | Chương trình khuyến mãi kéo dài | 544 (Since*, Interval) |

## Bảng vấn đề dữ liệu → cách xử lý
| Vấn đề | Số liệu (đã kiểm tra) | Cách xử lý (trong `src/features/pipeline.py`) |
|---|---|---|
| Trùng lặp train.csv | 0 dòng | Không cần xử lý |
| Thiếu CompetitionDistance | 3 cửa hàng | Điền median (2325,0) |
| Thiếu CompetitionOpenSince* | 354 | Điền -1 và cờ HasCompetition |
| Thiếu Promo2Since*, PromoInterval | 544 | Điền 0 và 'None' |
| Ngày đóng cửa | 172.817 dòng Open=0, Sales=0 | Loại khỏi huấn luyện/đánh giá |
| Open=1 nhưng Sales=0 | 54 dòng | Loại (Sales > 0) → còn 844.338 dòng |
| Open thiếu ở test.csv | 11 dòng | Điền 1 (như notebook) |
| StateHoliday kiểu hỗn hợp | 0/'0'/a/b/c | Ép về chuỗi '0','a','b','c' |
| Ngày lễ có rất ít dòng có doanh thu | a=694, b=145, c=71 (trên 844.338) | Giữ nguyên; ghi hạn chế cho What-if ngày lễ |
