import json
import os

def merge_chunks_strictly(file_path_1, file_path_2, output_path):
    with open(file_path_1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    with open(file_path_2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
        
    merged_data = []
    
    for item in data1:
        merged_data.append({
            "chunk_id": item.get("chunk_id"),
            "title": item.get("title"), 
            "text": item.get("text")    
        })

    for item in data2:
        merged_data.append({
            "chunk_id": f"DOC_{item.get('doc_id')}_CHUNK_{item.get('chunk_id')}",
            "title": item.get("title"),          
            "text": item.get("text") 
        })
        
    # 3. Xuất ra file JSON tổng hợp
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
    print(f"Đã gộp và đồng bộ tên trường thành công cho {len(merged_data)} chunks!")

if __name__ == "__main__":
    merge_chunks_strictly(
        file_path_1="data/processed/f_chunks.json", 
        file_path_2="chunks/uet_rag_chunks_dataset.json", 
        output_path="data/processed/chunks.json"
    )