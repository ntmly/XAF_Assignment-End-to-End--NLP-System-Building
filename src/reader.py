from transformers import pipeline
from typing import List

class CPUReader:
    def __init__(self, model_name: str):
        print(f"Loading QA pipeline {model_name} on CPU...")
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=-1  # CPU
        )
    
    def answer(self, query: str, retrieved_chunks: List[str]) -> str:
        if not retrieved_chunks:
            return "UNKNOWN"
        
        context = " ".join(retrieved_chunks)
        
        try:
            result = self.qa_pipeline(question=query, context=context)
            print(f"DEBUG: score={result['score']:.4f}, answer='{result['answer']}'")
            answer = result['answer'].strip()
            score = result['score']
            
            # Lọc câu trả lời kém chất lượng
            if result['score'] < 0.05:  
                return "UNKNOWN"
            return answer
        except Exception as e:
            print(f"QA error: {e}")
            return "UNKNOWN"