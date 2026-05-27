# main.py

import yaml
import numpy as np
import argparse
import torch
import json
import os

from sentence_transformers import SentenceTransformer

from query_processing import preprocess_query
from indexer import DualIndexer
from generative_reader import GenerativeReader

from query_rewriter import QueryRewriter
from rrf import reciprocal_rank_fusion


def load_knowledge_base_fallback():

    return [
        "Vietnam National University, Hanoi was established in 1993.",
        "The University of Engineering and Technology is a member of VNU.",
        "Carnegie Mellon University was founded in 1900.",
    ]


def verify_relevance(query, chunks):

    if not chunks:
        return False

    query_tokens = query.lower().split()

    top_chunk = chunks[0].lower()

    overlap = sum(
        1
        for token in query_tokens
        if token in top_chunk
    )

    return overlap > 0


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="advanced_full",
        choices=[
            "baseline0",
            "dense_only",
            "advanced_full"
        ]
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

    # ─────────────────────────────────────────────
    # Load Knowledge Base
    # ─────────────────────────────────────────────

    if os.path.exists(chunks_path):

        with open(chunks_path, "r", encoding="utf-8") as f:

            chunks = json.load(f)

        knowledge_base = []

        for chunk in chunks:

            text = (
                f"{chunk.get('title', '')}. "
                f"{chunk['text']}"
            ).strip()

            knowledge_base.append(text)

    else:

        knowledge_base = load_knowledge_base_fallback()

    # ─────────────────────────────────────────────
    # Device
    # ─────────────────────────────────────────────

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ─────────────────────────────────────────────
    # Embedder
    # ─────────────────────────────────────────────

    print("Loading embedder...")

    embedder = SentenceTransformer(
        config["models"]["dense_embedder"],
        device=device
    )

    # ─────────────────────────────────────────────
    # Index
    # ─────────────────────────────────────────────

    print("Building dual index...")

    indexer = DualIndexer(
        knowledge_base,
        embedder
    )

    # ─────────────────────────────────────────────
    # Reader
    # ─────────────────────────────────────────────

    reader = GenerativeReader(
        model_path=config["models"]["reader_qa"],
        max_new_tokens=config["hyperparameters"]["max_new_tokens"],
        device=device
    )

    # ─────────────────────────────────────────────
    # Query Rewriter
    # ─────────────────────────────────────────────

    query_rewriter = QueryRewriter(
        reader.llm
    )

    # ─────────────────────────────────────────────
    # Questions
    # ─────────────────────────────────────────────

    with open(args.input, "r", encoding="utf-8") as f:

        questions = [
            line.strip()
            for line in f
            if line.strip()
        ]

    answers = []

    # ─────────────────────────────────────────────
    # Main Loop
    # ─────────────────────────────────────────────

    for q in questions:

        processed_q = preprocess_query(q)

        # ───────────────── baseline0
        if args.mode == "baseline0":

            top_k = config["hyperparameters"]["top_k_hybrid"]

            bm25_scores = indexer.get_bm25_scores(
                processed_q
            )

            top_indices = np.argsort(
                bm25_scores
            )[::-1][:top_k]

            final_chunks = [
                knowledge_base[i]
                for i in top_indices
                if bm25_scores[i] > 0
            ][:5]

            if verify_relevance(
                processed_q,
                final_chunks
            ):

                ans = reader.answer(
                    processed_q,
                    final_chunks
                )

            else:

                ans = "UNKNOWN"

        # ───────────────── dense_only
        elif args.mode == "dense_only":

            q_emb = embedder.encode(
                [processed_q],
                convert_to_tensor=False
            )[0].astype("float32")

            _, indices = indexer.search_faiss(
                q_emb,
                config["hyperparameters"]["top_k_hybrid"]
            )

            final_chunks = [
                knowledge_base[i]
                for i in indices
                if i != -1
            ][:5]

            if verify_relevance(
                processed_q,
                final_chunks
            ):

                ans = reader.answer(
                    processed_q,
                    final_chunks
                )

            else:

                ans = "UNKNOWN"

        # ───────────────── advanced_full
        else:

            # ── Query Rewrite ───────────────────

            try:

                rewritten_queries = (
                    query_rewriter.rewrite(
                        processed_q
                    )
                )

            except Exception:

                rewritten_queries = [
                    processed_q
                ]

            if args.debug:

                print("\n[DEBUG] Rewritten Queries:")

                for rq in rewritten_queries:

                    print("-", rq)

            bm25_rank_lists = []

            dense_rank_lists = []

            # ────────────────────────────────
            # Multi Retrieval
            # ────────────────────────────────

            for rq in rewritten_queries:

                # ── BM25 ─────────────────

                bm25_scores = (
                    indexer.get_bm25_scores(rq)
                )

                bm25_indices = np.argsort(
                    bm25_scores
                )[::-1][
                    :config["hyperparameters"]["top_k_hybrid"]
                ]

                bm25_indices = [
                    i
                    for i in bm25_indices
                    if bm25_scores[i] > 0
                ]

                bm25_rank_lists.append(
                    list(bm25_indices)
                )

                # ── Dense Retrieval ─────

                q_emb = embedder.encode(
                    [rq],
                    convert_to_tensor=False
                )[0].astype("float32")

                _, dense_indices = (
                    indexer.search_faiss(
                        q_emb,
                        config["hyperparameters"]["top_k_hybrid"]
                    )
                )

                dense_indices = [
                    i
                    for i in dense_indices
                    if i != -1
                ]

                dense_rank_lists.append(
                    dense_indices
                )

            # ────────────────────────────────
            # RRF Fusion
            # ────────────────────────────────

            fused_indices = reciprocal_rank_fusion(
                bm25_rank_lists +
                dense_rank_lists
            )

            # ────────────────────────────────
            # Final Chunks
            # ────────────────────────────────

            final_chunks = []

            used = set()

            for idx in fused_indices:

                if idx not in used:

                    used.add(idx)

                    final_chunks.append(
                        knowledge_base[idx]
                    )

                if len(final_chunks) >= 5:
                    break

            if args.debug:

                print("\n[DEBUG] Final Chunks:")

                for i, chunk in enumerate(
                    final_chunks
                ):

                    print(f"\n[{i}]")
                    print(chunk[:300])

            # ────────────────────────────────
            # Relevance Verification
            # ────────────────────────────────

            if verify_relevance(
                processed_q,
                final_chunks
            ):

                ans = reader.answer(
                    processed_q,
                    final_chunks
                )

            else:

                ans = "UNKNOWN"

        answers.append(ans)

        print(f"\nQ: {q}")
        print(f"A: {ans}")

    # ─────────────────────────────────────────────
    # Save Outputs
    # ─────────────────────────────────────────────

    os.makedirs(
        os.path.dirname(args.output),
        exist_ok=True
    )

    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as f:

        for ans in answers:

            f.write(ans + "\n")

    print(
        f"\nDone! Saved to: {args.output}"
    )


if __name__ == "__main__":

    main()