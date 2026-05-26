import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict
import json

class DualIndexer:
    def __init__(self, documents: List[str], embedder_model):
        self.documents = documents
        self.embedder = embedder_model
        
        # 1. BM25 Sparse Index
        tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. FAISS Dense Index
        print("Đang tạo embeddings cho FAISS...")
        embeddings = embedder_model.encode(documents, convert_to_tensor=False, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        self.dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.faiss_index.add(embeddings)
        print(f"FAISS index: {self.faiss_index.ntotal} vectors, dimension {self.dimension}")
    
    def get_bm25_scores(self, query: str) -> np.ndarray:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        return np.array(scores)
    
    def search_faiss(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Trả về (distances, indices)"""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        return distances[0], indices[0]
    
    def get_all_documents(self) -> List[str]:
        return self.documents

def build_index_from_chunks(chunks_path: str, embedder_model) -> DualIndexer:
    """Xây dựng index từ file chunks.json"""
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    documents = [chunk["text"] for chunk in chunks]
    return DualIndexer(documents, embedder_model)