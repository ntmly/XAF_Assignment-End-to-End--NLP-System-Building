import re
from typing import List, Dict
import json
import os

class SemanticChunker:
    def __init__(self, chunk_size: int = 150, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split_by_sentences(self, text: str) -> List[str]:
        """Chia văn bản thành câu"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str) -> List[str]:
        """Chia văn bản thành các đoạn nhỏ với overlap"""
        sentences = self.split_by_sentences(text)
        chunks = []
        current_chunk = []
        current_len = 0
        
        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len <= self.chunk_size:
                current_chunk.append(sent)
                current_len += sent_len
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                # Overlap: giữ lại các câu cuối của chunk trước
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    s_len = len(s.split())
                    if overlap_len + s_len <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += s_len
                    else:
                        break
                current_chunk = overlap_sentences + [sent]
                current_len = overlap_len + sent_len
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def process_documents(self, documents: List[str]) -> List[Dict]:
        """Xử lý danh sách tài liệu thành các chunk"""
        chunked_docs = []
        for idx, doc in enumerate(documents):
            chunks = self.chunk_text(doc)
            for chunk_idx, chunk in enumerate(chunks):
                chunked_docs.append({
                    "doc_id": idx,
                    "chunk_id": chunk_idx,
                    "text": chunk
                })
        return chunked_docs

def main():
    # Đọc raw documents
    with open("data/raw/raw_docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    
    chunker = SemanticChunker(chunk_size=150, chunk_overlap=20)
    chunks = chunker.process_documents(docs)
    
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"Đã tạo {len(chunks)} chunks từ {len(docs)} tài liệu")

if __name__ == "__main__":
    main()