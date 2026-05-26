import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

class CPUReranker:
    def __init__(self, model_name: str):
        print(f"Loading reranker {model_name} on CPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to("cpu")
        self.model.eval()
    
    def rerank(self, query: str, documents: List[str], top_k: int = 2) -> List[str]:
        if not documents:
            return []
        
        # Tạo pairs
        pairs = [[query, doc] for doc in documents]
        
        # Tokenize batch
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits[:, 0].cpu().numpy()
        
        # Sort
        ranked_indices = np.argsort(scores)[::-1]
        reranked_docs = [documents[i] for i in ranked_indices[:top_k]]
        
        return reranked_docs