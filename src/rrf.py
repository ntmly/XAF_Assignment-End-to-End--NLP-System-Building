# rrf.py

from collections import defaultdict


def reciprocal_rank_fusion(rank_lists, k=60):

    scores = defaultdict(float)

    for rank_list in rank_lists:

        for rank, doc_id in enumerate(rank_list):

            scores[doc_id] += 1.0 / (k + rank + 1)

    sorted_docs = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc_id for doc_id, _ in sorted_docs]