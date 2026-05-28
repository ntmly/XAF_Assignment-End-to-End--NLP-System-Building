<p align="center">

<img src="https://readme-typing-svg.herokuapp.com?size=24&center=true&vCenter=true&width=900&lines=🚀+Hệ+thống+RAG+cho+UET+%28VNU%29;BM25+%2B+FAISS+Hybrid+Retrieval;Qwen2.5-3B+cho+hệ+thống+QA;Đồ+án+NLP+End-to-End" />

</p>

<p align="center">
  <img src="https://img.shields.io/badge/RAG-Hệ_thống-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/FAISS-Dense-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/BM25-Sparse-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Qwen-LLM-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/NLP-End--to--End-purple?style=for-the-badge">
</p>

---

# 🚀 Hệ thống RAG Nâng cao cho UET (VNU)

Dự án xây dựng một hệ thống **Retrieval-Augmented Generation (RAG)** cho miền dữ liệu giáo dục, hỗ trợ trả lời câu hỏi về Đại học Quốc gia Hà Nội (VNU), Trường Đại học Công nghệ (UET) và quy chế đào tạo.

Hệ thống kết hợp các kỹ thuật hiện đại:

* 🔍 BM25 Sparse Retrieval
* 🧠 Dense Retrieval (FAISS + Embedding)
* 🔁 Query Rewriting bằng LLM
* ⚡ Reciprocal Rank Fusion (RRF)
* 🤖 Generative Reader (Qwen2.5-3B)

---

# ⚙️ Kiến trúc hệ thống

```text
Câu hỏi đầu vào
   │
   ▼
Tiền xử lý câu hỏi
   │
   ▼
Query Rewriting (LLM)
   │
   ▼
Hybrid Retrieval
   ├── BM25
   └── FAISS Dense Search
   │
   ▼
RRF Fusion + Ranking
   │
   ▼
Top-K Chunks (K=5)
   │
   ▼
Qwen2.5-3B Reader
   │
   ▼
Câu trả lời cuối cùng
```

---

# 📁 Cấu trúc dự án

```
.
├── data/
│   ├── raw/                  # dữ liệu gốc
│   ├── processed/
│   │   └── chunks.json
│   └── test/
│       ├── questions.txt
│       ├── reference_answers.txt
│       └── retrieval_ground_truth.json
│
├── models/
├── system_outputs/
│   ├── QA_benchmark.json
│   ├── retrieval_benchmark.json
│   ├── system_output_1.txt
│   ├── system_output_2.txt
│   └── system_output_3.txt
│
└── src/
    ├── crawler.py
    ├── chunking.py
    ├── indexer.py
    ├── retriever.py
    ├── reranker.py
    ├── query_rewriter.py
    ├── generative_reader.py
    ├── eval_retrieval.py
    ├── evaluate.py
    └── main.py
```

---

# 🚀 Cài đặt

## 1. Tạo môi trường

```bash
python -m venv rag_env
rag_env\Scripts\activate
```

---

## 2. Cài thư viện

```bash
pip install -r requirements.txt
```

---

# 🤖 Mô hình sử dụng

* Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* Reader: `Qwen2.5-3B-Instruct (GGUF quantized)`

---

# 📚 Chuẩn bị dữ liệu

```bash
python src/chunking.py
```

Output:

```
data/processed/chunks.json
```

---

# ▶️ Chạy hệ thống

## 🔹 Advanced pipeline

```bash
python src/main.py --mode advanced_full
```

## 🔹 Dense baseline

```bash
python src/main.py --mode dense_only
```

## 🔹 BM25 baseline

```bash
python src/main.py --mode baseline0
```

---

# 📊 Đánh giá QA (EM / F1)

```bash
python src/evaluate.py
```

## 📌 Kết quả

| System        | EM            | F1            |
| ------------- | ------------- | ------------- |
| FAISS Only    | 0.1800        | 0.3925        |
| Dense Only    | 0.2200        | 0.5250        |
| Advanced Full | 🚀 **0.2400** | 🚀 **0.6037** |

---

# 🔍 Benchmark Retrieval

## Chạy đánh giá

```bash
python src/eval_retrieval.py
```

## Debug lỗi retrieval

```bash
python src/eval_retrieval.py --debug_miss
```

## Theo mode

```bash
python src/eval_retrieval.py --mode dense_only
python src/eval_retrieval.py --mode advanced_full
```

---

## 📌 Kết quả Retrieval

| System        | Recall@1    | Recall@3    | Recall@5    |
| ------------- | ----------- | ----------- | ----------- |
| Dense Only    | 0.38        | 0.52        | 0.54        |
| Advanced Full | 🚀 **0.64** | 🚀 **0.84** | 🚀 **0.92** |

---

# 🧠 Nhận xét

* Hybrid Retrieval cải thiện mạnh Recall@K
* Query Rewriting giúp xử lý câu hỏi mơ hồ
* RRF giúp ổn định ranking
* Advanced system đạt:

  * Recall@5 = **0.92**
  * F1 = **0.6037**

---

# 📦 Output hệ thống

* `system_output_1.txt` → Advanced Full

* `system_output_2.txt` → Dense Only

* `system_output_3.txt` → BM25 baseline

* `QA_benchmark.json` → kết quả QA

* `retrieval_benchmark.json` → kết quả retrieval

---

# 📄 Dataset

* `questions.txt`
* `reference_answers.txt`
* `retrieval_ground_truth.json`

---

# 👥 Nhóm thực hiện

* Nguyễn Thị Thanh Huyền — 23020381
* Nguyễn Thị Minh Ly — 23020399
* Đặng Minh Nguyệt — 23020407

---

# ⭐ Kết luận

Hệ thống thể hiện một pipeline **RAG hoàn chỉnh từ retrieval đến generation**, kết hợp IR cổ điển và LLM hiện đại.
