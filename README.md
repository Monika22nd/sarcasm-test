# LSTM Sarcasm Detector

Web demo nhận diện câu tiếng Anh có tính mỉa mai (sarcasm) bằng mô hình **Bidirectional LSTM**.

Mô hình ban đầu được huấn luyện bằng TensorFlow trên Google Colab. Bản demo trong repository này đã xuất trọng số sang NumPy để chạy dự đoán, vì vậy **không cần tải TensorFlow, Streamlit, Flask hay Django**.

## Chạy nhanh bằng PowerShell

```powershell
git clone https://github.com/Monika22nd/sarcasm-test.git
cd sarcasm-test
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1
```

Lần đầu, script tạo thư mục `.venv` và chỉ cài **NumPy**. Giao diện web dùng thư viện có sẵn của Python nên không kéo thêm framework. Những lần chạy sau sẽ dùng lại môi trường này, không tải lại thư viện.

Sau khi thấy dòng `Local URL`, mở địa chỉ được in ra, thường là:

```text
http://127.0.0.1:8501
```

Nhấn `Ctrl + C` trong PowerShell để tắt web.

> Nếu cổng 8501 đang được dùng, `run.ps1` sẽ tự chọn cổng tiếp theo.

## Chạy thủ công

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python server.py
```

## Cấu trúc cần thiết

```text
sarcasm-test/
├── server.py                 # web server bằng thư viện chuẩn Python
├── inference.py              # suy luận LSTM bằng NumPy
├── requirements.txt          # chỉ NumPy
├── run.ps1                   # tự tạo môi trường và chạy web
├── web/
│   └── index.html            # giao diện HTML/CSS/JavaScript
└── artifacts/
    ├── config.json           # thông số mô hình
    ├── tokenizer_min.json    # từ điển tokenizer tối giản
    └── weights.npz           # trọng số đã nén
```

Repository không chứa file CSV huấn luyện vì web chỉ cần tokenizer và trọng số để dự đoán. `weights.npz` khoảng 10 MB, nhỏ hơn file TensorFlow gốc khoảng 32 MB. Toàn bộ repository chỉ khoảng 11 MB.

## Thông số mô hình

| Thành phần | Giá trị |
|---|---:|
| Kiến trúc | Embedding → SpatialDropout → BiLSTM → Dense → Sigmoid |
| Vocabulary | 20.000 từ |
| Độ dài câu tối đa | 40 token |
| Embedding dimension | 128 |
| LSTM units | 64 mỗi chiều |
| Số tham số | 2.667.137 |
| Ngưỡng mặc định | 0.5 |

## Lưu ý

- Mô hình được huấn luyện cho **câu tiếng Anh**; câu tiếng Việt có thể cho kết quả không đáng tin cậy.
- Đây là demo học tập. Kết quả là xác suất của mô hình, không phải đánh giá chắc chắn về ý định của người viết.
- Xóa `.venv` sẽ mất các thư viện đã cài trong môi trường đó, nhưng source code và mô hình trong `artifacts/` vẫn còn. Chạy lại `run.ps1` sẽ tạo lại môi trường.
