# Nguồn dữ liệu (B1)

- **Bộ dữ liệu:** Rossmann Store Sales (Kaggle, cuộc thi do Rossmann tổ chức).
- **URL:** https://www.kaggle.com/c/rossmann-store-sales/data
- **File:** train.csv (1.017.209 dòng, 9 cột), store.csv (1.115 dòng, 10 cột), test.csv (41.088 dòng, 8 cột), sample_submission.csv.
- **Giai đoạn:** train 01/01/2013 – 31/07/2015; test.csv 01/08/2015 – 17/09/2015 (không có nhãn Sales).
- **Cách có dữ liệu:** tải `rossmann-store-sales.zip` từ trang trên (cần tài khoản Kaggle và chấp nhận điều khoản cuộc thi), đặt vào `data/raw/`, rồi chạy `python scripts/prepare_data.py`.
- **Giấy phép/điều khoản:** CHƯA XÁC MINH trong tài liệu này. Dữ liệu thuộc điều khoản (Rules) của cuộc thi Kaggle; nhóm cần đọc mục Rules/Data Access trên trang cuộc thi và ghi nguyên văn điều kiện sử dụng vào đây trước khi nộp. Do chưa xác minh, repo KHÔNG đưa dữ liệu gốc lên GitHub (data/raw nằm trong .gitignore). `data/sample/` chỉ là trích nhỏ 12 cửa hàng để demo; nếu repo để public, hãy kiểm tra điều khoản trước.
