import numpy as np
from typing import List

def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    s_min, s_max = np.min(scores), np.max(scores)
    if s_max == s_min:
        return np.ones_like(scores)
    return (scores - s_min) / (s_max - s_min)

class HybridRetriever:
    def __init__(self, indexer, embedder_model, alpha: float = 0.6):
        self.indexer = indexer
        self.embedder = embedder_model
        self.alpha = alpha
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        # BM25 scores
        bm25_scores = self.indexer.get_bm25_scores(query)
        
        # FAISS dense scores
        q_emb = self.embedder.encode([query], convert_to_tensor=False)[0].astype('float32')
        distances, indices = self.indexer.search_faiss(q_emb, len(self.indexer.get_all_documents()))
        
        # Chuyển distance L2 thành similarity
        faiss_scores = np.zeros(len(self.indexer.get_all_documents()))
        for dist, idx in zip(distances, indices):
            if idx != -1:
                faiss_scores[idx] = 1.0 / (1.0 + dist)
        
        # Normalize
        bm25_norm = min_max_normalize(bm25_scores)
        faiss_norm = min_max_normalize(faiss_scores)
        
        # Fusion
        final_scores = self.alpha * faiss_norm + (1.0 - self.alpha) * bm25_norm
        
        # Top-K
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        retrieved_docs = [self.indexer.get_all_documents()[i] for i in top_indices]
        
        return retrieved_docs