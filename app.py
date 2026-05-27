"""
Flask backend cho VNU-UET RAG Demo
Chạy: python app.py
Truy cập: http://localhost:5000
"""

import sys
import os
import json
import yaml
import numpy as np
import torch

from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sentence_transformers import SentenceTransformer
from query_processing import preprocess_query
from indexer import DualIndexer
from retriever import HybridRetriever
from generative_reader import GenerativeReader

app = Flask(__name__, static_folder=".")

# ── Load pipeline một lần khi khởi động ─────────────────────────────────────

print("=" * 50)
print("Khởi động VNU-UET RAG Demo...")
print("=" * 50)

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

torch.set_num_threads(config["project_setup"]["num_threads"])

with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

knowledge_base = [
    f"{c.get('title', '') or ''}. {c['text']}".strip(". ")
    for c in chunks
]

print("Loading embedder...")

embedder = SentenceTransformer(config["models"]["dense_embedder"])

print("Building dual index...")
indexer = DualIndexer(knowledge_base, embedder)
retriever = HybridRetriever(
    indexer, embedder,
    alpha=config["hyperparameters"]["hybrid_alpha"]
)

print("Loading generative reader...")
reader = GenerativeReader(
    model_path=config["models"]["reader_qa"],
    max_new_tokens=config["hyperparameters"]["max_new_tokens"]
)

print("=" * 50)
print("Sẵn sàng! Truy cập: http://localhost:5000")
print("=" * 50)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "demo.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query_raw = data.get("query", "").strip()
    mode      = data.get("mode", "advanced_full")

    if not query_raw:
        return jsonify({"error": "Câu hỏi trống"}), 400

    try:
        processed_q = preprocess_query(query_raw)
        top_k = config["hyperparameters"]["top_k_hybrid"]

        # ── Retrieval theo mode ──────────────────────────────────────────
        if mode == "baseline0":
            scores   = indexer.get_bm25_scores(processed_q)
            indices  = np.argsort(scores)[::-1][:top_k]
            chunks_retrieved = [knowledge_base[i] for i in indices if scores[i] > 0][:3]
            retrieved_info   = [
                {"index": int(i), "text": knowledge_base[i][:150], "score": float(scores[i])}
                for i in indices[:3] if scores[i] > 0
            ]

        elif mode == "dense_only":
            q_emb = embedder.encode([processed_q], convert_to_tensor=False)[0].astype("float32")
            _, indices = indexer.search_faiss(q_emb, top_k)
            chunks_retrieved = [knowledge_base[i] for i in indices if i != -1][:3]
            retrieved_info   = [
                {"index": int(i), "text": knowledge_base[i][:150], "score": None}
                for i in indices[:3] if i != -1
            ]

        else:  # advanced_full
            chunks_retrieved = retriever.retrieve(processed_q, top_k=top_k)[:3]
            retrieved_info   = [
                {"index": None, "text": c[:150], "score": None}
                for c in chunks_retrieved
            ]

        # ── Generate answer ──────────────────────────────────────────────
        answer = reader.answer(processed_q, chunks_retrieved)

        return jsonify({
            "answer":    answer,
            "mode":      mode,
            "query_processed": processed_q,
            "retrieved": retrieved_info,
            "num_chunks": len(chunks_retrieved)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
