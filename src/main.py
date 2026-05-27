import yaml
import numpy as np
import argparse
import torch
import json
import os

from sentence_transformers import SentenceTransformer

from query_processing import preprocess_query
from indexer import DualIndexer
from retriever import HybridRetriever
from reranker import CPUReranker
from generative_reader import GenerativeReader


def load_knowledge_base_fallback() -> list:
    return [
        "Vietnam National University, Hanoi was established in 1993.",
        "The University of Engineering and Technology is a member of VNU.",
        "Carnegie Mellon University was founded in 1900.",
    ]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="advanced_full",
        choices=["baseline0", "dense_only", "advanced_full"]
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/test/questions.txt"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="system_outputs/system_output_1.txt"
    )

    parser.add_argument(
        "--debug",
        action="store_true"
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    torch.set_num_threads(
        config["project_setup"]["num_threads"]
    )

    chunks_path = "data/processed/chunks.json"

    if os.path.exists(chunks_path):

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        knowledge_base = [
            f"{chunk.get('title', '')}. {chunk['text']}"
            for chunk in chunks
        ]

        print(f"Loaded {len(knowledge_base)} chunks")

    else:

        knowledge_base = load_knowledge_base_fallback()

        print(f"Using fallback KB with {len(knowledge_base)} docs")

    print("Loading embedder...")

    embedder = SentenceTransformer(
        config["models"]["dense_embedder"]
    )

    print("Building dual index...")

    indexer = DualIndexer(
        knowledge_base,
        embedder
    )

    retriever = HybridRetriever(
        indexer,
        embedder,
        alpha=config["hyperparameters"]["hybrid_alpha"]
    )

    use_reranker = config["hyperparameters"].get(
        "use_reranker",
        False
    )

    reranker = None

    if use_reranker:
        reranker = CPUReranker(
            config["models"]["reranker"]
        )

    reader = GenerativeReader(
        model_path=config["models"]["reader_qa"],
        max_new_tokens=config["hyperparameters"]["max_new_tokens"]
    )

    with open(args.input, "r", encoding="utf-8") as f:
        questions = [
            line.strip()
            for line in f
            if line.strip()
        ]

    answers = []

    for q in questions:

        processed_q = preprocess_query(q)

        # ── baseline0: BM25 only ─────────────────────────────────────────
        if args.mode == "baseline0":

            top_k = config["hyperparameters"]["top_k_hybrid"]
            bm25_scores = indexer.get_bm25_scores(processed_q)
            top_indices = np.argsort(bm25_scores)[::-1][:top_k]

            final_chunks = [
                knowledge_base[i]
                for i in top_indices
                if bm25_scores[i] > 0
            ][:3]

            ans = reader.answer(processed_q, final_chunks)

        # ── dense_only: FAISS only ───────────────────────────────────────
        elif args.mode == "dense_only":

            q_emb = embedder.encode(
                [processed_q],
                convert_to_tensor=False
            )[0].astype("float32")

            _, indices = indexer.search_faiss(
                q_emb,
                config["hyperparameters"]["top_k_hybrid"]
            )

            retrieved_chunks = [
                knowledge_base[i]
                for i in indices
                if i != -1
            ]

            final_chunks = retrieved_chunks[:3]

            ans = reader.answer(processed_q, final_chunks)

        # ── advanced_full: BM25 + FAISS hybrid (+ optional reranker) ────
        else:

            retrieved_chunks = retriever.retrieve(
                processed_q,
                top_k=config["hyperparameters"]["top_k_hybrid"]
            )

            if args.debug:
                print(f"\n[DEBUG] Query: {processed_q}")
                print(f"[DEBUG] Retrieved {len(retrieved_chunks)} chunks")
                for i, chunk in enumerate(retrieved_chunks):
                    print(f"\n[{i}]")
                    print(chunk[:300])

            if use_reranker and reranker is not None:

                reranked = reranker.rerank(
                    processed_q,
                    retrieved_chunks,
                    top_k=config["hyperparameters"]["top_k_rerank"]
                )

                final_chunks = reranked[:3]

            else:

                final_chunks = retrieved_chunks[:3]

            ans = reader.answer(processed_q, final_chunks)

        answers.append(ans)

        print(f"\nQ: {q}")
        print(f"A: {ans}")

    os.makedirs(
        os.path.dirname(args.output),
        exist_ok=True
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for ans in answers:
            f.write(ans + "\n")

    print(f"\nDone! Saved to: {args.output}")


if __name__ == "__main__":
    main()