<p align="center">

<img src="https://readme-typing-svg.herokuapp.com?size=24&center=true&vCenter=true&width=900&lines=🚀+Hệ+thống+RAG+cho+UET+%28VNU%29;BM25+%2B+FAISS+Hybrid+Retrieval;Qwen2.5-3B+cho+hệ+thống+QA;XAF_Assignment-End-to-End--NLP-System-Building" />

</p>

# Hệ thống RAG Nâng cao cho miền Trường học (ĐHQGHN, UET)

Dự án xây dựng một pipeline **Retrieval-Augmented Generation (RAG)** cho miền dữ liệu giáo dục, hỗ trợ trả lời các câu hỏi về Đại học Quốc gia Hà Nội (VNU), Trường Đại học Công nghệ (UET) và các quy chế đào tạo.

Hệ thống tích hợp các kỹ thuật hiện đại:

* Hybrid Retrieval (BM25 + Dense Retrieval)
* FAISS Vector Search
* Query Rewriting bằng LLM
* Reciprocal Rank Fusion (RRF)
* Generative Reader (Qwen2.5-3B)
* Evaluation đầy đủ cho Retrieval + QA

---

# 🚀 Kiến trúc hệ thống

```text
Câu hỏi đầu vào
│
▼
[1. Query Preprocessing]
(Chuẩn hóa + mở rộng từ viết tắt)
│
▼
[2. Query Rewriting]
(Sinh nhiều biến thể câu hỏi)
│
▼
[3. Multi-Retrieval]
├── BM25 Retrieval
└── Dense Retrieval (FAISS + Embedding)
│
▼
[4. Reciprocal Rank Fusion (RRF)]
│
▼
[5. Hybrid Ranking (alpha = 0.5)]
│
▼
[6. Top-5 Chunks Selection]
│
▼
[7. Generative Reader (Qwen2.5-3B)]
│
▼
[8. Final Answer]
(hoặc "UNKNOWN")
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
│   ├── raw/                  # (chưa crawl / dữ liệu gốc)
│   ├── processed/
│   │   └── chunks.json
│   │
│   └── test/
│       ├── questions.txt
│       ├── reference_answers.txt
│       └── retrieval_ground_truth.json
│
├── models/
│
├── system_outputs/
│   ├── QA_benchmark.json
│   ├── retrieval_benchmark.json
│   ├── system_output_1.txt
│   ├── system_output_2.txt
│   ├── system_output_3.txt
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
    ├── eval_retrieval.py
    └── main.py
```

---

# ⚙️ Cài đặt

## 1. Tạo môi trường

```bash
python -m venv rag_env
rag_env\Scripts\activate
```

## 2. Cài dependencies

```bash
pip install -r requirements.txt
```

---

# 🤖 Mô hình sử dụng

## Embedding model

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## Reader model

```text
Qwen/Qwen2.5-3B-Instruct (quantized GGUF)
```

---

# 📚 Chuẩn bị dữ liệu

## Dữ liệu gốc

```text
data/raw/
```

## Tạo chunks

```bash
python src/chunking.py
```

Output:

```text
data/processed/chunks.json
```

---

# ▶️ Chạy hệ thống

## Dense-only baseline

```bash
python src/main.py \
--mode dense_only \
--input data/test/questions.txt \
--output system_outputs/system_output_2.txt
```

---

## Advanced full pipeline

```bash
python src/main.py \
--mode advanced_full \
--input data/test/questions.txt \
--output system_outputs/system_output_1.txt
```

---

## Baseline BM25

```bash
python src/main.py \
--mode baseline0 \
--input data/test/questions.txt \
--output system_outputs/system_output_3.txt
```

---

# 📊 Evaluation QA (EM / F1)

```bash
python src/evaluate.py \
--pred system_outputs/system_output_1.txt \
--gold data/test/reference_answers.txt
```

## Kết quả QA

| System        | EM         | F1         |
| ------------- | ---------- | ---------- |
| FAISS Only    | 0.1800     | 0.3925     |
| Dense Only    | 0.2200     | 0.5250     |
| Advanced Full | **0.2400** | **0.6037** |

---

# 🔍 Retrieval Benchmark

## Chạy evaluation retrieval

```bash
python src/eval_retrieval.py
```

## Debug miss retrieval

```bash
python src/eval_retrieval.py --debug_miss
```

## Theo mode

```bash
python src/eval_retrieval.py --mode dense_only
python src/eval_retrieval.py --mode advanced_full
```

---

## 📊 Kết quả Retrieval

| System        | Recall@1 | Recall@3 | Recall@5 |
| ------------- | -------- | -------- | -------- |
| Dense Only    | 0.38     | 0.52     | 0.54     |
| Advanced Full | **0.64** | **0.84** | **0.92** |

---

# 🧠 Phân tích kết quả

* Dense retrieval cho semantic matching tốt hơn BM25
* Hybrid retrieval + RRF cải thiện mạnh Recall@K
* Query rewriting giúp tăng recall với câu hỏi mơ hồ
* Advanced pipeline đạt:

  * Recall@5 = **0.92**
  * F1 = **0.6037**

---

# 🧪 Output hệ thống

## QA outputs

* `system_output_1.txt` → Advanced Full
* `system_output_2.txt` → Dense Only
* `system_output_3.txt` → BM25 / baseline

## Benchmark outputs

* `QA_benchmark.json` → kết quả EM/F1
* `retrieval_benchmark.json` → Recall@K

---

# 📄 Evaluation dataset

## Test set

* `questions.txt` → câu hỏi đầu vào
* `reference_answers.txt` → ground truth QA
* `retrieval_ground_truth.json` → ground truth retrieval

---

# 📌 Thành phần hệ thống

| File                 | Vai trò                 |
| -------------------- | ----------------------- |
| crawler.py           | Thu thập dữ liệu        |
| chunking.py          | Tạo chunks              |
| query_processing.py  | Tiền xử lý query        |
| indexer.py           | BM25 + FAISS            |
| retriever.py         | Hybrid retrieval        |
| reranker.py          | Cross-encoder reranking |
| query_rewriter.py    | Query expansion         |
| rrf.py               | Fusion ranking          |
| generative_reader.py | Sinh câu trả lời        |
| eval_retrieval.py    | Đánh giá Recall@K       |
| evaluate.py          | Đánh giá QA             |
| main.py              | Pipeline chính          |

---

# 📄 Tài liệu liên quan

| Tài liệu           | Mô tả                                   |
| ------------------ | --------------------------------------- |
| `contributions.md` | Phân công công việc giữa các thành viên |

---

# 👥 Nhóm thực hiện

* Nguyễn Thị Thanh Huyền — 23020381
* Nguyễn Thị Minh Ly — 23020399
* Đặng Minh Nguyệt — 23020407
