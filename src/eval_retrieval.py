"""
Retrieval Benchmark — Phase 1
Đo Recall@1, Recall@3, Recall@5 cho 2 mode:
  - dense_only  : FAISS only
  - advanced_full: BM25 + FAISS hybrid

Chạy:
    python src/eval_retrieval.py
    python src/eval_retrieval.py --debug_miss        # in chi tiết các câu miss
    python src/eval_retrieval.py --mode advanced_full --debug_miss
"""

import json
import yaml
import argparse
import os
import numpy as np
from sentence_transformers import SentenceTransformer

from indexer import DualIndexer
from retriever import HybridRetriever
from query_processing import preprocess_query


# ─── Helpers ────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_chunks(path: str = "data/processed/chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    knowledge_base = [
        f"{c.get('title', '') or ''}. {c['text']}".strip(". ")
        for c in chunks
    ]
    return chunks, knowledge_base


def load_ground_truth(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("items", data.get("questions", []))


# ─── Recall@K ───────────────────────────────────────────────────────────────

def recall_at_k(retrieved_indices: list, relevant_ids: list, k: int) -> float:
    top_k = set(retrieved_indices[:k])
    return 1.0 if top_k & set(relevant_ids) else 0.0


# ─── Retrieve ───────────────────────────────────────────────────────────────

def retrieve_dense(query, embedder, indexer, top_k) -> list:
    q_emb = embedder.encode([query], convert_to_tensor=False)[0].astype("float32")
    _, indices = indexer.search_faiss(q_emb, top_k)
    return [int(i) for i in indices if i != -1]


def retrieve_hybrid(query, embedder, indexer, retriever, knowledge_base, top_k) -> list:
    retrieved_texts = retriever.retrieve(query, top_k=top_k)
    text_to_idx = {text: idx for idx, text in enumerate(knowledge_base)}
    return [text_to_idx[t] for t in retrieved_texts if t in text_to_idx]


# ─── Benchmark ──────────────────────────────────────────────────────────────

def run_benchmark(mode, ground_truth, embedder, indexer,
                  retriever, knowledge_base, chunks, top_k_max,
                  debug_miss=False):

    ks = [1, 3, 5]
    scores = {k: [] for k in ks}
    miss_details = []  

    for item in ground_truth:
        query_raw   = item["question"]
        relevant_ids = item["relevant_chunk_ids"]
        query = preprocess_query(query_raw)

        if mode == "dense_only":
            indices = retrieve_dense(query, embedder, indexer, top_k_max)
        else:
            indices = retrieve_hybrid(query, embedder, indexer,
                                      retriever, knowledge_base, top_k_max)

        hit_at5 = recall_at_k(indices, relevant_ids, 5)
        for k in ks:
            scores[k].append(recall_at_k(indices, relevant_ids, k))

        # Thu thập miss@5 để phân tích
        if hit_at5 == 0.0:
            retrieved_texts = [knowledge_base[i] for i in indices[:5]]
            relevant_texts  = [chunks[i]["text"][:120] for i in relevant_ids
                               if i < len(chunks)]
            miss_details.append({
                "question":       query_raw,
                "query_processed": query,
                "relevant_ids":   relevant_ids,
                "relevant_texts": relevant_texts,
                "top5_retrieved_ids": indices[:5],
                "top5_retrieved_texts": [t[:120] for t in retrieved_texts],
            })

    metrics = {f"Recall@{k}": round(np.mean(scores[k]), 4) for k in ks}

    # ── In debug miss ────────────────────────────────────────────────────────
    if debug_miss and miss_details:
        print(f"\n{'='*60}")
        print(f"[{mode}] MISS ANALYSIS — {len(miss_details)} câu miss@5 "
              f"/ {len(ground_truth)} tổng")
        print(f"{'='*60}")
        for i, m in enumerate(miss_details):
            print(f"\n── Miss #{i+1} ──────────────────────────────────────")
            print(f"  Q (gốc)   : {m['question']}")
            print(f"  Q (xử lý) : {m['query_processed']}")
            print(f"  Relevant  : idx={m['relevant_ids']}")
            for rt in m["relevant_texts"]:
                print(f"    → \"{rt}\"")
            print(f"  Top-5 retrieved: idx={m['top5_retrieved_ids']}")
            for rt in m["top5_retrieved_texts"]:
                print(f"    → \"{rt}\"")

        # ── Phân loại nguyên nhân miss tự động ──────────────────────────────
        print(f"\n{'='*60}")
        print("[Phân loại nguyên nhân miss — gợi ý]")
        print(f"{'='*60}")

        keyword_miss = 0
        boundary_miss = 0

        for m in miss_details:
            q_words = set(m["query_processed"].lower().split())
            rel_words = set(" ".join(m["relevant_texts"]).lower().split())
            overlap = len(q_words & rel_words) / max(len(q_words), 1)

            if overlap < 0.3:
                keyword_miss += 1
                print(f"  [KEYWORD MISS] overlap={overlap:.2f} — "
                      f"\"{m['question'][:60]}\"")
            else:
                boundary_miss += 1
                print(f"  [BOUNDARY/INDEX MISS] overlap={overlap:.2f} — "
                      f"\"{m['question'][:60]}\"")

        print(f"\n  Tổng KEYWORD MISS  : {keyword_miss} "
              f"→ cần thêm synonym vào query_processing.py")
        print(f"  Tổng BOUNDARY MISS : {boundary_miss} "
              f"→ cần kiểm tra chunk_overlap hoặc ground truth index")

        # Lưu miss details ra file
        miss_path = f"system_outputs/miss_{mode}.json"
        os.makedirs("system_outputs", exist_ok=True)
        with open(miss_path, "w", encoding="utf-8") as f:
            json.dump(miss_details, f, ensure_ascii=False, indent=2)
        print(f"\n  Chi tiết miss đã lưu: {miss_path}")

    return metrics


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retrieval Benchmark")
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--chunks",       default="data/processed/chunks.json")
    parser.add_argument("--ground_truth", default="data/test/retrieval_ground_truth.json")
    parser.add_argument("--mode",         default="all",
                        choices=["all", "dense_only", "advanced_full"])
    parser.add_argument("--debug_miss",   action="store_true",
                        help="In chi tiết các câu bị miss@5")
    args = parser.parse_args()

    config       = load_config(args.config)
    chunks, knowledge_base = load_chunks(args.chunks)
    ground_truth = load_ground_truth(args.ground_truth)

    print(f"Loaded {len(knowledge_base)} chunks, {len(ground_truth)} test queries\n")

    print("Loading embedder...")
    embedder = SentenceTransformer(config["models"]["dense_embedder"])

    print("Building dual index...")
    indexer = DualIndexer(knowledge_base, embedder)

    retriever = HybridRetriever(
        indexer, embedder,
        alpha=config["hyperparameters"]["hybrid_alpha"]
    )

    top_k_max = max(config["hyperparameters"].get("top_k_hybrid", 10), 5)

    modes = ["dense_only", "advanced_full"] if args.mode == "all" else [args.mode]

    results = {}
    for mode in modes:
        print(f"\nBenchmarking [{mode}]...")
        results[mode] = run_benchmark(
            mode, ground_truth, embedder, indexer,
            retriever, knowledge_base, chunks, top_k_max,
            debug_miss=args.debug_miss
        )

    # ── Bảng kết quả ────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"{'Mode':<22} {'R@1':>6} {'R@3':>6} {'R@5':>6}")
    print("-" * 50)
    for mode, metrics in results.items():
        print(f"{mode:<22} "
              f"{metrics['Recall@1']:>6.4f} "
              f"{metrics['Recall@3']:>6.4f} "
              f"{metrics['Recall@5']:>6.4f}")
    print("=" * 50)

    if len(results) == 2:
        print("\n[So sánh Hybrid vs Dense]")
        d, h = results["dense_only"], results["advanced_full"]
        for k in [1, 3, 5]:
            key  = f"Recall@{k}"
            diff = h[key] - d[key]
            sign = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            print(f"  Recall@{k}: Hybrid {sign} so với Dense")

    if "advanced_full" in results:
        r5 = results["advanced_full"]["Recall@5"]
        if r5 >= 0.88:
            print(f"\n[✓] Recall@5 = {r5:.4f} — Xuất sắc, chạy QA benchmark thôi.")
        elif r5 >= 0.80:
            print(f"\n[✓] Recall@5 = {r5:.4f} — Ổn, đủ để chạy QA benchmark.")
            print("    Muốn cải thiện: chạy --debug_miss để xem câu nào đang miss.")
        else:
            print(f"\n[!] Recall@5 = {r5:.4f} — Cần cải thiện retrieval trước.")

    # Lưu kết quả
    os.makedirs("system_outputs", exist_ok=True)
    out_path = "system_outputs/retrieval_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nKết quả đã lưu: {out_path}")


if __name__ == "__main__":
    main()