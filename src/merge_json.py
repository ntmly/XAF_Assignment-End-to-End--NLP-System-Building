import json
import os

def merge_chunks_strictly(file_path_1, file_path_2, output_path):
    # 1. Đọc file JSON thứ nhất (đã có sẵn chunk_id, title, text)
    with open(file_path_1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
        
    # 2. Đọc file JSON thứ hai (đang dùng doc_id, chunk_id số, text)
    with open(file_path_2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
        
    merged_data = []
    
    # Giữ nguyên vẹn nội dung file 1, đưa vào danh sách tổng
    for item in data1:
        merged_data.append({
            "chunk_id": item.get("chunk_id"),
            "title": item.get("title"), # Giữ nguyên title của file 1
            "text": item.get("text")    # Giữ nguyên văn bản của file 1
        })
        
    # Đổi tên trường của file 2 cho đồng bộ rồi đưa vào danh sách tổng
    for item in data2:
        merged_data.append({
            # Đổi doc_id và chunk_id số thành chuỗi để làm mã định danh duy nhất
            "chunk_id": f"DOC_{item.get('doc_id')}_CHUNK_{item.get('chunk_id')}",
            "title": item.get("title"),          
            "text": item.get("text")    # Giữ nguyên vẹn 100% văn bản thô bên trong
        })
        
    # 3. Xuất ra file JSON tổng hợp
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False để không bị lỗi font tiếng Việt, indent=2 để dễ đọc
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
    print(f"Đã gộp và đồng bộ tên trường thành công cho {len(merged_data)} chunks!")

if __name__ == "__main__":
    # Thay đường dẫn file thực tế của bạn vào đây để chạy
    merge_chunks_strictly(
        file_path_1="data/processed/f_chunks.json", 
        file_path_2="chunks/uet_rag_chunks_dataset.json", 
        output_path="data/processed/chunks.json"
    )