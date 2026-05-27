# XAF_Assignment-End-to-End--NLP-System-Building
# Hệ thống RAG Nâng cao cho miền Trường học (ĐHQGHN, UET)

Dự án xây dựng một pipeline **Retrieval-Augmented Generation (RAG)** tối ưu cho CPU, hỗ trợ trả lời các câu hỏi về Đại học Quốc gia Hà Nội (VNU), Trường Đại học Công nghệ (UET) và các quy định đào tạo liên quan.

Hệ thống tích hợp nhiều kỹ thuật nâng cao:

- Hybrid Retrieval (BM25 + Dense Retrieval)
- Query Rewriting bằng LLM
- Reciprocal Rank Fusion (RRF)
- Generative Reader dựa trên Qwen

---

# 🚀 Kiến trúc tổng quan

```text
Câu hỏi đầu vào
│
▼
[1. Query Preprocessing]
(mở rộng từ viết tắt, chuẩn hóa)
│
▼
[2. Query Rewriting]
(dùng Qwen sinh 3 cách diễn đạt khác nhau)
│
▼
[3. Multi-Retrieval]
(với mỗi câu hỏi đã viết lại)
├── BM25 (sparse) → danh sách rank
└── Dense (FAISS + multilingual-embedder) → danh sách rank
│
▼
[4. Reciprocal Rank Fusion (RRF)]
kết hợp 2 × 3 = 6 danh sách rank
│
▼
[5. Chọn Top-5 chunk]
→ Generative Reader (Qwen2.5-3B)
│
▼
[6. Trả lời]
(hoặc "UNKNOWN" nếu không liên quan)
```

---

# 📁 Cấu trúc thư mục

```text
.
├── config.yaml
├── requirements.txt
├── run_all.bat
├── contributions.md
├── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   └── test/
│
├── system_outputs/
│
└── src/
    ├── crawler.py
    ├── chunking.py
    ├── query_processing.py
    ├── indexer.py
    ├── retriever.py
    ├── reranker.py
    ├── generative_reader.py
    ├── query_rewriter.py
    ├── rrf.py
    ├── evaluate.py
    └── main.py
```

---

# 📌 Mô tả thành phần

| File | Chức năng |
|---|---|
| `crawler.py` | Thu thập dữ liệu |
| `chunking.py` | Chia văn bản thành chunks |
| `query_processing.py` | Chuẩn hóa câu hỏi, mở rộng từ viết tắt |
| `indexer.py` | Xây dựng BM25 + FAISS index |
| `retriever.py` | Hybrid retrieval |
| `reranker.py` | Cross-encoder reranker |
| `generative_reader.py` | Sinh câu trả lời bằng Qwen |
| `query_rewriter.py` | Sinh nhiều phiên bản câu hỏi |
| `rrf.py` | Reciprocal Rank Fusion |
| `evaluate.py` | Tính EM và F1 |
| `main.py` | Pipeline chính |

---

# ⚙️ Cài đặt và chạy

## 1. Tạo môi trường ảo

### Windows

```bash
python -m venv rag_env
rag_env\Scripts\activate
```

### Linux / MacOS

```bash
python -m venv rag_env
source rag_env/bin/activate
```

---

## 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

# 🤖 Tải mô hình

Các mô hình không được push lên GitHub do kích thước lớn.

Tạo thư mục:

```text
models/
```

Sau đó tải các model bằng:

## Embedding Model

```bash
huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
--local-dir models/paraphrase-multilingual-MiniLM-L12-v2
```

## Reader Model

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
--local-dir models/Qwen2.5-3B-Instruct
```

---

# 📚 Chuẩn bị dữ liệu

## Cách tự động

```bash
python src/crawler.py --output data/raw
python src/chunking.py
```

## Cách thủ công

Tạo file:

```text
data/raw/raw_docs.json
```

sau đó chạy:

```bash
python src/chunking.py
```

---

# ⚙️ Cấu hình mô hình

Chỉnh sửa file `config.yaml`

```yaml
project_setup:
  num_threads: 4

models:
  dense_embedder: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  reader_qa: "Qwen/Qwen2.5-3B-Instruct"

hyperparameters:
  hybrid_alpha: 0.5
  top_k_hybrid: 15
  top_k_rerank: 5
  use_reranker: false
  max_new_tokens: 150
```

---

# ▶️ Chạy pipeline

## Baseline BM25

```bash
python src/main.py \
--mode baseline0 \
--input data/test/questions.txt \
--output system_outputs/system_output_3.txt
```

## Dense Retrieval

```bash
python src/main.py \
--mode dense_only \
--input data/test/questions.txt \
--output system_outputs/system_output_2.txt
```

## Advanced Full Pipeline

```bash
python src/main.py \
--mode advanced_full \
--input data/test/questions.txt \
--output system_outputs/system_output_1.txt
```

Hoặc chạy tự động:

```bash
run_all.bat
```

---

# 📊 Đánh giá kết quả

```bash
python src/evaluate.py \
--pred system_outputs/system_output_1.txt \
--gold data/test/reference_answers.txt
```

## Các chỉ số đánh giá

### Exact Match (EM)

Tỉ lệ câu trả lời trùng khớp hoàn toàn sau khi chuẩn hóa.

### F1 Score

Độ phủ từ khóa giữa câu trả lời dự đoán và đáp án tham chiếu.

---

# 🧪 Thử nghiệm và cải tiến

- Retrieval đạt khoảng **Recall@5 ≈ 71%** với bộ dữ liệu ban đầu.
- Generative Reader giúp tăng:
  - từ **F1 ≈ 0.23**
  - lên **F1 ≈ 0.41**
- Query Rewriting + RRF cải thiện khả năng:
  - xử lý câu hỏi ngắn
  - xử lý từ khóa mơ hồ
  - tăng semantic recall

---

---

# 📄 Tài liệu liên quan

| Tài liệu | Mô tả |
|---|---|
| [contributions.md](contributions.md) | Phân công công việc giữa các thành viên |

# 👥 Nhóm thực hiện

- Nguyễn Thị Thanh Huyền — 23020381
- Nguyễn Thị Minh Ly — 23020399
- Đặng Minh Nguyệt — 23020407