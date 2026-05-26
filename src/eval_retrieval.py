import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ==================== CẤU HÌNH ====================
CHUNKS_FILE = "chunks/uet_rag_chunks_dataset.json"
GT_FILE = "data/test/retrieval_ground_truth.json"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5
# ==================================================

# 1. Kiểm tra file đầu vào
if not os.path.exists(CHUNKS_FILE):
    raise FileNotFoundError(f"Không tìm thấy file chunks: {CHUNKS_FILE}")
if not os.path.exists(GT_FILE):
    raise FileNotFoundError(f"Không tìm thấy file ground truth: {GT_FILE}")

# 2. Đọc chunks (bắt buộc encoding='utf-8')
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]
# Lấy chunk_id, nếu không có thì dùng index
chunk_ids = [c.get("chunk_id", f"idx_{i}") for i, c in enumerate(chunks)]

print(f"Loaded {len(chunks)} chunks.")

# 3. Khởi tạo embedding model
print(f"Loading embedder: {EMBED_MODEL} ...")
embedder = SentenceTransformer(EMBED_MODEL)

# 4. Tạo embedding cho toàn bộ chunks
print("Encoding chunks...")
embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# 5. Xây dựng FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

# 6. Đọc ground truth (cũng cần encoding utf-8)
with open(GT_FILE, "r", encoding="utf-8") as f:
    gt = json.load(f)   # List of {"question": "...", "ground_truth_chunk_id": "..."}

total = len(gt)
if total == 0:
    print("Ground truth trống. Thoát.")
    exit()

# 7. Đánh giá retrieval
hits = 0
for item in gt:
    q = item["question"]
    target_id = item["ground_truth_chunk_id"]
    
    q_emb = embedder.encode([q], convert_to_numpy=True)
    distances, indices = index.search(q_emb, TOP_K)
    retrieved_ids = [chunk_ids[i] for i in indices[0]]
    
    if target_id in retrieved_ids:
        hits += 1

recall = hits / total
print(f"Recall@{TOP_K}: {recall:.2%} ({hits}/{total})")