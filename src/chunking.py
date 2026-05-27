import os
import json
import re
from typing import List, Dict

def split_text_by_sentences(text: str, max_len: int, overlap_len: int) -> List[str]:
    """Chia văn bản thành các đoạn nhỏ dựa trên câu, với overlap."""
    sentences = re.split(r'(?<=[.!?;:])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len <= max_len:
            current_chunk.append(sent)
            current_len += sent_len
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            overlap_sents = []
            overlap_len_sofar = 0
            for s in reversed(current_chunk):
                if overlap_len_sofar + len(s) <= overlap_len:
                    overlap_sents.insert(0, s)
                    overlap_len_sofar += len(s)
                else:
                    break
            current_chunk = overlap_sents + [sent]
            current_len = overlap_len_sofar + sent_len
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def process_chunks(input_file: str, output_file: str, chunk_size: int, chunk_overlap: int):
    with open(input_file, 'r', encoding='utf-8') as f:
        old_chunks = json.load(f)
    
    new_chunks = []
    for old in old_chunks:
        title = old.get('title', '')
        text = old['text']
        sub_texts = split_text_by_sentences(text, chunk_size, chunk_overlap)
        for i, sub in enumerate(sub_texts):
            new_id = f"{old['chunk_id']}_sub_{i}"
            new_chunks.append({
                "chunk_id": new_id,
                "title": title,
                "text": sub.strip()
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_chunks, f, ensure_ascii=False, indent=2)
    print(f"Đã tạo {len(new_chunks)} chunks từ {len(old_chunks)} chunks cũ. Lưu tại {output_file}")

if __name__ == "__main__":
    import yaml
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    chunk_size = config['data']['chunk_size'] 
    chunk_overlap = config['data']['chunk_overlap'] 
    process_chunks('data/processed/ff_chunks.json', 'data/processed/chunks.json', chunk_size, chunk_overlap)