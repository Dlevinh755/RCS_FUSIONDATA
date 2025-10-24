RCS_FUSIONDATA


1. Tải Mã Nguồn (Clone Repository)

Bash

!git clone https://github.com/Dlevinh755/RCS_FUSIONDATA.git
2. Cài Đặt Các Thư Viện Phụ Thuộc (Dependencies)

Bash

!pip install -r RCS_FUSIONDATA/requirements.txt
3. Khởi Chạy Chương Trình Chính

Bash

!python /kaggle/working/RCS_FUSIONDATA/main.py \
--epochs 5 \
--batch_size 32 \
--learning_rate 1e-4

📁 Cấu Trúc Dự Án (Giả định)

RCS_FUSIONDATA/
├── main.py             # Script chính để huấn luyện/chạy mô hình
├── requirements.txt    # Danh sách các thư viện cần cài đặt
└── README.md           # File này
