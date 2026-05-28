# 📋 Phân công công việc

## Thông tin nhóm

| Họ và tên | Mã sinh viên |
|---|---|
| Nguyễn Thị Thanh Huyền | 23020381 |
| Nguyễn Thị Minh Ly | 23020399 |
| Đặng Minh Nguyệt | 23020407 |

---

# 📌 Nội dung phân công

## 1. Nguyễn Thị Thanh Huyền — 23020381

### Phụ trách:
- Thu thập và tiền xử lý dữ liệu
- Xây dựng pipeline chunking
- Chuẩn hóa dữ liệu đầu vào cho retrieval
- Thiết kế cấu trúc dữ liệu chunks
- Tối ưu preprocessing cho tiếng Việt

### Công việc chi tiết:
- Xây dựng `crawler.py`
- Xử lý dữ liệu crawl từ website và HuggingFace
- Làm sạch dữ liệu:
  - loại bỏ ký tự thừa
  - chuẩn hóa khoảng trắng
  - xử lý dữ liệu trùng lặp
- Thiết kế và cài đặt:
  - `chunking.py`
  - chunk overlap
  - title extraction
- Tạo tập dữ liệu `chunks.json`
- Kiểm tra chất lượng chunking và độ bao phủ dữ liệu

---

## 2. Nguyễn Thị Minh Ly — 23020399

### Phụ trách:
- Xây dựng hệ thống retrieval
- Dense retrieval và hybrid retrieval
- Query rewriting và RRF
- Tích hợp pipeline chính

### Công việc chi tiết:
- Xây dựng:
  - `indexer.py`
  - `retriever.py`
  - `query_rewriter.py`
  - `rrf.py`
- Tích hợp:
  - BM25
  - FAISS
  - Sentence Transformer
- Thiết kế hybrid retrieval:
  - sparse + dense
  - score fusion
- Xây dựng query preprocessing:
  - mở rộng từ viết tắt
  - chuẩn hóa truy vấn
- Tích hợp reciprocal rank fusion (RRF)
- Xây dựng pipeline tổng thể trong `main.py`
- Tối ưu hệ thống chạy trên CPU

---

## 3. Đặng Minh Nguyệt — 23020407

### Phụ trách:
- Generative Reader
- Đánh giá hệ thống
- Thử nghiệm và báo cáo kết quả

### Công việc chi tiết:
- Xây dựng:
  - `generative_reader.py`
  - `evaluate.py`
  - `reranker.py`
- Tích hợp mô hình:
  - Qwen2.5-3B-Instruct
- Thiết kế prompt cho generative QA
- Xử lý trường hợp:
  - hallucination
  - câu hỏi ngoài miền
  - trả về UNKNOWN
- Đánh giá hệ thống:
  - Exact Match (EM)
  - F1 Score
  - Recall@K
- Thực hiện thực nghiệm:
  - baseline BM25
  - dense retrieval
  - advanced pipeline
- Tổng hợp kết quả và viết báo cáo đánh giá

---

# 🤝 Công việc chung của cả nhóm

- Thảo luận hướng tiếp cận hệ thống RAG
- Thiết kế kiến trúc tổng thể
- Kiểm thử và sửa lỗi pipeline
- Thảo luận lựa chọn mô hình phù hợp
- Hoàn thiện tài liệu README
```